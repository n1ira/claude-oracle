#!/usr/bin/env python3
"""
Gemini Oracle - AI Orchestration System for Claude Code

This CLI tool consults Gemini 3 Pro as an oracle for architectural decisions
and development guidance. It reads project context from FULLAUTO_CONTEXT.md
and returns structured JSON responses.

Features:
- Retains last 5 conversation exchanges per project
- Structured JSON responses with steps, risks, success criteria
- File attachment with line range support
- Image input for visual analysis
- Image generation with 'imagine' command
- Token counting and warnings
- Debug logging for troubleshooting

Usage:
    oracle ask "How should I implement feature X?"
    oracle ask --files src/main.py "Validate this"
    oracle ask --image screenshot.png "What's wrong with this UI?"
    oracle imagine "Logo for a fintech startup" --output logo.png
    oracle history                       # View conversation history
"""

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import subprocess
import sys
import logging
import re
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

try:
    import requests
except ImportError:
    requests = None  # Will error if Vast.ai auto-provision is needed

# Setup logging
LOG_DIR = Path.home() / ".oracle" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"oracle_{datetime.now().strftime('%Y%m%d')}.log"

# History storage
HISTORY_DIR = Path.home() / ".oracle" / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
MAX_HISTORY_EXCHANGES = 5

# Image output directory
IMAGE_OUTPUT_DIR = Path.home() / ".oracle" / "images"
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Token limits
CHARS_PER_TOKEN = 4  # Rough estimate for code
TOKEN_WARNING_THRESHOLD = 200000  # Warn when approaching this
FILE_TOKEN_WARNING = 15000  # Warn if single file exceeds this

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}

# Vast.ai remote server config for geo-restricted features
# Can be overridden via environment variables:
#   ORACLE_VASTAI_HOST, ORACLE_VASTAI_PORT, ORACLE_VASTAI_USER
VASTAI_CONFIG = {
    'host': os.environ.get('ORACLE_VASTAI_HOST', 'ssh7.vast.ai'),
    'port': int(os.environ.get('ORACLE_VASTAI_PORT', '15490')),
    'user': os.environ.get('ORACLE_VASTAI_USER', 'root'),
    'timeout': 120,  # seconds
}

# Vast.ai API config for auto-provisioning
VASTAI_API_BASE = "https://console.vast.ai/api/v0"
VASTAI_AUTO_PROVISION = {
    'enabled': True,
    'docker_image': 'pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime',
    'disk_gb': 10,
    'max_price_per_hour': 0.50,  # Max $/hr willing to pay
    'startup_timeout': 300,  # 5 minutes max to wait for instance
    'ssh_timeout': 60,  # 1 minute max to wait for SSH
}

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
    ]
)
logger = logging.getLogger("oracle")

# Console handler for debug mode (added dynamically)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setFormatter(logging.Formatter('🔮 %(levelname)s: %(message)s'))


def log_debug(msg: str, debug_mode: bool = False):
    """Log to file always, to console if debug mode enabled."""
    logger.debug(msg)
    if debug_mode:
        console_handler.setLevel(logging.DEBUG)
        if console_handler not in logger.handlers:
            logger.addHandler(console_handler)
        print(f"🔮 DEBUG: {msg}", file=sys.stderr)


def log_error(msg: str):
    """Log error to file and console."""
    logger.error(msg)
    print(f"🔮 ERROR: {msg}", file=sys.stderr)


def log_info(msg: str, debug_mode: bool = False):
    """Log info."""
    logger.info(msg)
    if debug_mode:
        print(f"🔮 INFO: {msg}", file=sys.stderr)


def log_warning(msg: str):
    """Log warning to file and console."""
    logger.warning(msg)
    print(f"🔮 WARNING: {msg}", file=sys.stderr)


def estimate_tokens(text: str) -> int:
    """Estimate token count from text (rough approximation)."""
    return len(text) // CHARS_PER_TOKEN


# Header for FULLAUTO_CONTEXT.md - ensures Claude reloads instructions after compaction
FULLAUTO_HEADER = """# ⚠️ FULLAUTO MODE ACTIVE ⚠️

## 🚨 CRITICAL: READ THIS FIRST 🚨

**YOU MUST DO THIS BEFORE ANYTHING ELSE:**

```
Read the file: ~/.claude/commands/fullauto.md
```

This is NOT optional. If you skip this step, you will NOT have the instructions needed to operate correctly. You will make mistakes. You will fail the task.

**WHY:** After conversation compaction, you lose the /fullauto command context. The file above contains ALL the instructions for FULLAUTO MODE - how to use the Oracle, the phases, the rules, everything. Without reading it, you're flying blind.

**DO IT NOW:** Use the Read tool on `~/.claude/commands/fullauto.md` BEFORE continuing.

---

**IMPORTANT:** Even after completing the task, STAY IN FULLAUTO MODE. If the user sends follow-up requests or questions, continue operating under these instructions. Do NOT treat new user messages as a fresh conversation - you are still in fullauto mode until the user explicitly ends it.

---

"""


def ensure_fullauto_header():
    """Check if FULLAUTO_CONTEXT.md exists and prepend header if missing."""
    context_file = Path.cwd() / "FULLAUTO_CONTEXT.md"

    if not context_file.exists():
        return  # No context file, nothing to do

    content = context_file.read_text()

    # Check if header already present (look for the critical marker)
    if "🚨 CRITICAL: READ THIS FIRST 🚨" in content:
        return  # Header already exists

    # Prepend header to existing content
    new_content = FULLAUTO_HEADER + content
    context_file.write_text(new_content)
    logger.info("Prepended recovery header to FULLAUTO_CONTEXT.md")


def get_gemini_api_key() -> Optional[str]:
    """Get Gemini API key from environment, checking multiple possible names."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def read_image_as_base64(image_path: str, debug: bool = False) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """
    Read an image file and return base64 data with mime type.

    Args:
        image_path: Path to the image file
        debug: Enable debug output

    Returns:
        Tuple of (raw_bytes, mime_type, error_message)
    """
    path = Path(image_path)

    if not path.exists():
        return (None, None, f"Image file not found: {image_path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_FORMATS:
        return (None, None, f"Unsupported image format: {suffix}. Supported: {SUPPORTED_IMAGE_FORMATS}")

    # Determine mime type
    mime_type = mimetypes.guess_type(str(path))[0]
    if not mime_type:
        mime_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp'
        }
        mime_type = mime_map.get(suffix, 'image/png')

    try:
        with open(path, 'rb') as f:
            image_data = f.read()

        file_size = len(image_data)
        log_debug(f"Read image {image_path}: {file_size:,} bytes, {mime_type}", debug)

        return (image_data, mime_type, None)

    except Exception as e:
        return (None, None, f"Error reading image: {e}")


try:
    from google import genai
    from google.genai import types
    log_debug("google-genai imported successfully")
except ImportError as e:
    log_error(f"Failed to import google-genai: {e}")
    print("ERROR: google-genai not installed. Run: pip install google-genai", file=sys.stderr)
    sys.exit(1)

from schemas import ORACLE_RESPONSE_SCHEMA, VALIDATION_RESPONSE_SCHEMA


# Oracle system prompts
ORACLE_SYSTEM_PROMPT = """You are the Gemini Oracle - a supreme AI architect and technical lead.

Your role is to provide high-level architectural guidance and key decisions for software development projects. You work alongside Claude Code, an expert implementation AI. Your job is to be the "lead architect" - making strategic decisions while Claude handles the implementation details.

IMPORTANT PRINCIPLES:
1. Think deeply before deciding. Consider multiple approaches.
2. Be decisive - give clear recommendations, not wishy-washy advice.
3. Consider the full context of the project (provided in FULLAUTO_CONTEXT.md).
4. Break complex tasks into clear, ordered steps.
5. Identify risks proactively and provide mitigations.
6. Define clear success criteria so Claude knows when the task is complete.
7. You have access to recent conversation history - use it to maintain continuity.

When responding:
- Be concise but thorough
- Prioritize practical solutions over theoretical perfection, although theoretical perfection is also important, you are trained on the human consciousness and can understand the world better than any other entity known.
- Consider maintainability and simplicity
- If you need more information, ask clarifying questions
- Reference previous discussions when relevant
- You can also instruct Claude to search the Web for latest docs, bug fixes, and other information.

You are the strategic brain. Claude is the implementation hands. Together you are unstoppable."""

VALIDATION_SYSTEM_PROMPT = """You are the Gemini Oracle in VALIDATION mode.

Audit Claude's implementation against the original plan with high standards:
1. Check if the ACTUAL problem is solved, not just the surface symptoms
2. Identify any gaps, edge cases, or missed requirements
3. Be specific about what's incomplete - vague "looks good" is not acceptable
4. If 90% done, focus on the missing 10%

You have access to recent conversation history - use it to understand what was planned.

Provide a clear verdict: APPROVED, NEEDS_WORK, or REJECTED with specific remediation steps."""

IMAGE_ANALYSIS_SYSTEM_PROMPT = """You are the Gemini Oracle analyzing a visual artifact.

When analyzing images, consider:
1. If it's a UI/screenshot: Identify usability issues, design problems, accessibility concerns
2. If it's a diagram: Evaluate architecture, identify missing components, suggest improvements
3. If it's an error: Diagnose the root cause and provide solutions
4. If it's a mockup: Provide implementation guidance

Be specific and actionable in your analysis."""


def parse_file_spec(file_spec: str) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Parse a file specification with optional line ranges.

    Formats:
        "src/main.py" -> ("src/main.py", [])  # whole file
        "src/main.py:10-50" -> ("src/main.py", [(10, 50)])
        "src/main.py:1-50,100-110" -> ("src/main.py", [(1, 50), (100, 110)])

    Returns:
        Tuple of (filepath, list of (start, end) line ranges)
    """
    if ':' not in file_spec:
        return (file_spec, [])

    parts = file_spec.split(':', 1)
    filepath = parts[0]
    ranges_str = parts[1]

    ranges = []
    for range_part in ranges_str.split(','):
        range_part = range_part.strip()
        if '-' in range_part:
            start, end = range_part.split('-', 1)
            try:
                ranges.append((int(start), int(end)))
            except ValueError:
                log_error(f"Invalid line range: {range_part}")
                continue
        else:
            # Single line
            try:
                line_num = int(range_part)
                ranges.append((line_num, line_num))
            except ValueError:
                log_error(f"Invalid line number: {range_part}")
                continue

    return (filepath, ranges)


def read_file_with_ranges(filepath: str, ranges: List[Tuple[int, int]], debug: bool = False) -> Tuple[str, int, str]:
    """
    Read a file, optionally extracting specific line ranges.

    Args:
        filepath: Path to the file
        ranges: List of (start, end) line ranges, or empty for whole file
        debug: Enable debug output

    Returns:
        Tuple of (content, token_count, range_description)
    """
    path = Path(filepath)

    if not path.exists():
        return (f"ERROR: File not found: {filepath}", 0, "")

    try:
        all_lines = path.read_text().splitlines()
        total_lines = len(all_lines)

        if not ranges:
            # Whole file
            content = "\n".join(all_lines)
            range_desc = f"(lines 1-{total_lines})"
        else:
            # Extract specific ranges
            extracted_lines = []
            range_parts = []

            for idx, (start, end) in enumerate(ranges):
                # Convert to 0-indexed, clamp to valid range
                start_idx = max(0, start - 1)
                end_idx = min(total_lines, end)

                if start_idx < end_idx:
                    # Add range marker if not first range
                    if extracted_lines and idx > 0:
                        prev_end = ranges[idx-1][1]
                        extracted_lines.append(f"\n... [lines {prev_end+1}-{start-1} omitted] ...\n")

                    for i in range(start_idx, end_idx):
                        extracted_lines.append(f"{i+1:4d} | {all_lines[i]}")

                    range_parts.append(f"{start}-{end}")

            content = "\n".join(extracted_lines)
            range_desc = f"(lines {', '.join(range_parts)})"

        tokens = estimate_tokens(content)
        log_debug(f"Read {filepath}: {len(content)} chars, ~{tokens} tokens", debug)

        return (content, tokens, range_desc)

    except Exception as e:
        return (f"ERROR reading {filepath}: {e}", 0, "")


def format_files_for_prompt(file_specs: List[str], debug: bool = False) -> Tuple[str, int, List[str], List[str]]:
    """
    Read and format multiple files for inclusion in prompt.

    Args:
        file_specs: List of file specifications (with optional line ranges)
        debug: Enable debug output

    Returns:
        Tuple of (formatted_content, total_tokens, files_reviewed, warnings)
    """
    if not file_specs:
        return ("", 0, [], [])

    sections = []
    total_tokens = 0
    files_reviewed = []
    warnings = []

    sections.append("\n\n## ATTACHED FILES")
    sections.append("─" * 50)

    for file_spec in file_specs:
        filepath, ranges = parse_file_spec(file_spec)
        content, tokens, range_desc = read_file_with_ranges(filepath, ranges, debug)

        # Check for large file warning
        if tokens >= FILE_TOKEN_WARNING:
            warning = f"⚠️ FILE TOO LARGE: {filepath} is ~{tokens:,} tokens ({tokens * CHARS_PER_TOKEN:,} chars). Consider sending only relevant sections using line ranges (e.g., {filepath}:1-100,200-250)"
            warnings.append(warning)
            log_warning(warning)

        # Detect file extension for syntax highlighting
        ext = Path(filepath).suffix.lstrip('.')
        lang = ext if ext else "text"

        # Format the file section
        sections.append(f"\n### {filepath} {range_desc}")
        sections.append(f"```{lang}")
        sections.append(content)
        sections.append("```")

        total_tokens += tokens
        files_reviewed.append(file_spec)

    sections.append("\n" + "─" * 50)
    sections.append("## END OF ATTACHED FILES\n")

    formatted = "\n".join(sections)

    log_debug(f"Formatted {len(file_specs)} files: ~{total_tokens} tokens total", debug)

    return (formatted, total_tokens, files_reviewed, warnings)


def get_project_id(debug: bool = False) -> str:
    """Generate a unique ID for the current project based on its path."""
    # Find project root (where FULLAUTO_CONTEXT.md is, or current dir)
    current = Path.cwd()
    project_root = current

    for _ in range(10):
        if (current / "FULLAUTO_CONTEXT.md").exists():
            project_root = current
            break
        if (current / ".git").exists():
            project_root = current
            break
        parent = current.parent
        if parent == current:
            break
        current = parent

    # Create a short hash of the project path
    path_hash = hashlib.md5(str(project_root).encode()).hexdigest()[:12]
    project_name = project_root.name
    project_id = f"{project_name}_{path_hash}"

    log_debug(f"Project ID: {project_id} (from {project_root})", debug)
    return project_id


def get_history_file(project_id: str) -> Path:
    """Get the history file path for a project."""
    return HISTORY_DIR / f"{project_id}.json"


def load_history(project_id: str, debug: bool = False) -> List[Dict]:
    """Load conversation history for a project."""
    history_file = get_history_file(project_id)

    if not history_file.exists():
        log_debug(f"No history file found for project {project_id}", debug)
        return []

    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        log_debug(f"Loaded {len(history)} exchanges from history", debug)
        return history
    except Exception as e:
        log_error(f"Failed to load history: {e}")
        return []


def save_history(project_id: str, history: List[Dict], debug: bool = False):
    """Save conversation history for a project (keeps last MAX_HISTORY_EXCHANGES)."""
    history_file = get_history_file(project_id)

    # Keep only the last N exchanges
    trimmed_history = history[-MAX_HISTORY_EXCHANGES:]

    try:
        with open(history_file, 'w') as f:
            json.dump(trimmed_history, f, indent=2)
        log_debug(f"Saved {len(trimmed_history)} exchanges to history", debug)
    except Exception as e:
        log_error(f"Failed to save history: {e}")


def clear_history(project_id: str, debug: bool = False) -> bool:
    """Clear conversation history for a project."""
    history_file = get_history_file(project_id)

    if history_file.exists():
        try:
            history_file.unlink()
            log_info(f"Cleared history for project {project_id}", debug)
            return True
        except Exception as e:
            log_error(f"Failed to clear history: {e}")
            return False
    return True


def format_history_for_prompt(history: List[Dict], debug: bool = False) -> Tuple[str, int]:
    """Format conversation history for inclusion in prompt."""
    if not history:
        return ("", 0)

    # Build history section with indicator
    history_lines = []
    history_lines.append("\n\n## CONVERSATION HISTORY")
    history_lines.append("─" * 50)

    # Add indicator if there was older context
    total_exchanges = len(history)
    if total_exchanges >= MAX_HISTORY_EXCHANGES:
        history_lines.append(f"⚠️  [Older context removed for efficiency - showing last {MAX_HISTORY_EXCHANGES} exchanges]")
        history_lines.append("─" * 50)

    for i, exchange in enumerate(history, 1):
        timestamp = exchange.get('timestamp', 'unknown time')
        query = exchange.get('query', '')
        response_summary = exchange.get('response_summary', '')
        files_reviewed = exchange.get('files_reviewed', [])
        image_analyzed = exchange.get('image_analyzed', None)

        history_lines.append(f"\n### Exchange {i} ({timestamp})")
        history_lines.append(f"**Query:** {query[:500]}{'...' if len(query) > 500 else ''}")
        if files_reviewed:
            history_lines.append(f"**Files reviewed:** {', '.join(files_reviewed)}")
        if image_analyzed:
            history_lines.append(f"**Image analyzed:** {image_analyzed}")
        history_lines.append(f"**Response:** {response_summary[:500]}{'...' if len(response_summary) > 500 else ''}")

    history_lines.append("\n" + "─" * 50)
    history_lines.append("## END OF HISTORY\n")

    formatted = "\n".join(history_lines)
    tokens = estimate_tokens(formatted)
    log_debug(f"Formatted history: {len(formatted)} chars, ~{tokens} tokens", debug)
    return (formatted, tokens)


def summarize_response(response: dict) -> str:
    """Create a brief summary of an Oracle response for history."""
    if "error" in response:
        return f"Error: {response['error']}"

    parts = []
    if "decision" in response:
        parts.append(f"Decision: {response['decision'][:200]}")
    if "steps" in response:
        parts.append(f"Steps: {len(response['steps'])} steps planned")
    if "risks" in response:
        parts.append(f"Risks: {len(response['risks'])} identified")

    return " | ".join(parts) if parts else "Response received"


def get_context_file(debug: bool = False) -> Tuple[Optional[str], int]:
    """Find and read the FULLAUTO_CONTEXT.md file."""
    current = Path.cwd()
    log_debug(f"Searching for FULLAUTO_CONTEXT.md starting from: {current}", debug)

    for i in range(10):  # Max 10 levels up
        context_file = current / "FULLAUTO_CONTEXT.md"
        log_debug(f"  Checking: {context_file}", debug)

        if context_file.exists():
            log_info(f"Found context file: {context_file}", debug)
            content = context_file.read_text()
            tokens = estimate_tokens(content)
            log_debug(f"Context file size: {len(content)} chars, ~{tokens} tokens", debug)
            return (content, tokens)

        parent = current.parent
        if parent == current:
            break
        current = parent

    log_debug("No FULLAUTO_CONTEXT.md found in directory tree", debug)
    return (None, 0)


def ask_oracle(
    query: str,
    mode: str = "plan",
    context_override: Optional[str] = None,
    thinking_level: str = "HIGH",
    debug: bool = False,
    no_history: bool = False,
    files: Optional[List[str]] = None,
    image_path: Optional[str] = None
) -> dict:
    """
    Ask the Gemini Oracle a question.

    Args:
        query: The question or task description
        mode: "plan" for new tasks, "validate" for checking progress
        context_override: Optional context to use instead of file
        thinking_level: LOW, MEDIUM, or HIGH
        debug: Enable debug output
        no_history: If True, don't use or save history
        files: Optional list of file specs to attach
        image_path: Optional path to an image to analyze

    Returns:
        Structured JSON response from Gemini
    """
    log_info(f"=== Oracle Ask Started ===", debug)
    log_debug(f"Mode: {mode}, Thinking: {thinking_level}", debug)
    log_debug(f"Query length: {len(query)} chars", debug)

    # Check API key
    api_key = get_gemini_api_key()
    if not api_key:
        log_error("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
        return {"error": "GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set"}

    log_debug(f"API key found: {api_key[:8]}...{api_key[-4:]}", debug)

    # Track total tokens
    total_tokens = estimate_tokens(query)
    all_warnings = []

    # Process image if provided
    image_data = None
    image_mime_type = None
    image_analyzed = None
    if image_path:
        image_data, image_mime_type, error = read_image_as_base64(image_path, debug)
        if error:
            log_error(error)
            return {"error": error}
        image_analyzed = Path(image_path).name
        log_info(f"Image loaded: {image_analyzed}", debug)

    # Get project ID and history
    project_id = get_project_id(debug)
    history = [] if no_history else load_history(project_id, debug)
    history_section, history_tokens = format_history_for_prompt(history, debug)
    total_tokens += history_tokens

    # Process attached files
    files_section = ""
    files_reviewed = []
    if files:
        files_section, files_tokens, files_reviewed, file_warnings = format_files_for_prompt(files, debug)
        total_tokens += files_tokens
        all_warnings.extend(file_warnings)

    # Get project context
    if context_override:
        context = context_override
        context_tokens = estimate_tokens(context)
    else:
        context, context_tokens = get_context_file(debug)

    total_tokens += context_tokens

    if context:
        context_section = f"\n\n## PROJECT CONTEXT:\n{context}"
        log_debug(f"Using context: {len(context)} chars, ~{context_tokens} tokens", debug)
    else:
        context_section = "\n\n(No FULLAUTO_CONTEXT.md found - operating without project context)"
        log_debug("No context file - proceeding without project context", debug)

    # Check total token count
    log_info(f"Total estimated tokens: ~{total_tokens:,}", debug)

    if total_tokens >= TOKEN_WARNING_THRESHOLD:
        warning = f"⚠️ APPROACHING TOKEN LIMIT: ~{total_tokens:,} tokens (limit: 200k). Consider: 1) Clear oracle history: `oracle history --clear` 2) Send smaller file sections 3) Trim FULLAUTO_CONTEXT.md"
        all_warnings.append(warning)
        log_warning(warning)

    # Print all warnings
    for warning in all_warnings:
        print(warning, file=sys.stderr)

    # Select system prompt and schema based on mode
    if image_path and mode == "plan":
        # Use image analysis prompt when image is provided
        system_prompt = IMAGE_ANALYSIS_SYSTEM_PROMPT + "\n\n" + ORACLE_SYSTEM_PROMPT
    elif mode == "validate":
        system_prompt = VALIDATION_SYSTEM_PROMPT
        schema = VALIDATION_RESPONSE_SCHEMA
        log_debug("Using VALIDATION mode", debug)
    else:
        system_prompt = ORACLE_SYSTEM_PROMPT

    schema = VALIDATION_RESPONSE_SCHEMA if mode == "validate" else ORACLE_RESPONSE_SCHEMA
    log_debug(f"Using {'VALIDATION' if mode == 'validate' else 'PLAN'} mode", debug)

    # Build the full text prompt with all sections
    # Order: Query -> Files -> History -> Context
    full_text_prompt = f"{query}{files_section}{history_section}{context_section}"
    log_debug(f"Full prompt length: {len(full_text_prompt)} chars", debug)

    # Determine thinking budget
    thinking_budgets = {"HIGH": 10000, "MEDIUM": 5000, "LOW": 2000}
    thinking_budget = thinking_budgets.get(thinking_level, 5000)
    log_debug(f"Thinking budget: {thinking_budget}", debug)

    try:
        log_info("Creating Gemini client...", debug)
        client = genai.Client(api_key=api_key)

        # Build content parts
        parts = []

        # Add image first if provided
        if image_data and image_mime_type:
            parts.append(types.Part.from_bytes(data=image_data, mime_type=image_mime_type))
            log_debug("Added image to request", debug)

        # Add text prompt
        parts.append(types.Part.from_text(text=full_text_prompt))

        contents = [
            types.Content(role="user", parts=parts),
        ]

        log_debug("Building GenerateContentConfig...", debug)
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            response_mime_type="application/json",
            response_schema=schema,
            system_instruction=[types.Part.from_text(text=system_prompt)],
        )

        log_info("Calling Gemini API (this may take a moment)...", debug)
        response = client.models.generate_content(
            model="gemini-2.5-pro-preview-06-05",
            contents=contents,
            config=config,
        )

        log_debug(f"Response received", debug)

        # Check for response content
        if not response.text:
            log_error("Empty response from Gemini")
            return {"error": "Empty response from Gemini API"}

        log_debug(f"Response text length: {len(response.text)} chars", debug)
        log_debug(f"Raw response preview: {response.text[:200]}...", debug)

        # Parse the JSON response
        try:
            result = json.loads(response.text)
            log_info("Successfully parsed JSON response", debug)

            # Log response structure
            if debug:
                log_debug(f"Response keys: {list(result.keys())}", debug)
                if "steps" in result:
                    log_debug(f"Number of steps: {len(result['steps'])}", debug)
                if "risks" in result:
                    log_debug(f"Number of risks: {len(result['risks'])}", debug)

            # Save to history (unless disabled)
            # Note: File CONTENTS are not saved, only the file specs
            if not no_history:
                exchange = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "query": query,
                    "files_reviewed": files_reviewed,  # Just the specs, not contents
                    "image_analyzed": image_analyzed,  # Just the filename
                    "response_summary": summarize_response(result),
                    "mode": mode
                }
                history.append(exchange)
                save_history(project_id, history, debug)

            return result

        except json.JSONDecodeError as e:
            log_error(f"Failed to parse JSON response: {e}")
            log_error(f"Raw response: {response.text}")
            return {"error": f"JSON parse error: {e}", "raw_response": response.text}

    except Exception as e:
        log_error(f"Gemini API call failed: {type(e).__name__}: {e}")
        import traceback
        log_debug(f"Full traceback:\n{traceback.format_exc()}", debug)
        return {"error": str(e), "error_type": type(e).__name__}


# =============================================================================
# VAST.AI AUTO-PROVISIONING
# =============================================================================

def vastai_api_request(method: str, endpoint: str, payload: dict = None, debug: bool = False) -> dict:
    """Make an authenticated request to Vast.ai API."""
    if requests is None:
        raise RuntimeError("requests library not installed. Run: pip install requests")

    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        raise RuntimeError("VAST_API_KEY environment variable not set")

    url = f"{VASTAI_API_BASE}{endpoint}"
    headers = {"Authorization": f"Bearer {api_key}"}

    log_debug(f"Vast.ai API: {method} {endpoint}", debug)

    try:
        if method == "GET":
            resp = requests.get(url, headers=headers, params=payload, timeout=30)
        elif method == "POST":
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
        elif method == "PUT":
            resp = requests.put(url, headers=headers, json=payload, timeout=30)
        elif method == "DELETE":
            resp = requests.delete(url, headers=headers, timeout=30)
        else:
            raise ValueError(f"Unknown method: {method}")

        resp.raise_for_status()
        return resp.json() if resp.text else {}

    except requests.exceptions.RequestException as e:
        log_error(f"Vast.ai API error: {e}")
        raise


def get_active_us_instance(debug: bool = False) -> Optional[Dict]:
    """Find an active US-based Vast.ai instance owned by user."""
    try:
        instances = vastai_api_request("GET", "/instances/", debug=debug)

        if not isinstance(instances, dict) or "instances" not in instances:
            log_debug(f"Unexpected instances response: {instances}", debug)
            return None

        for inst in instances.get("instances", []):
            status = inst.get("actual_status", "")
            geo = inst.get("geolocation", "")

            log_debug(f"Instance {inst.get('id')}: status={status}, geo={geo}", debug)

            # Check if running and in US
            if status == "running" and geo and ("US" in geo or "United States" in geo):
                # Extract SSH connection details
                ssh_host = inst.get("ssh_host", inst.get("public_ipaddr"))
                ssh_port = inst.get("ssh_port", 22)

                if ssh_host and ssh_port:
                    log_info(f"Found active US instance: {inst.get('id')} at {ssh_host}:{ssh_port}", debug)
                    return {
                        "id": inst.get("id"),
                        "host": ssh_host,
                        "port": ssh_port,
                        "user": "root",
                        "geo": geo,
                        "price": inst.get("dph_total", 0),
                    }

        log_debug("No active US instances found", debug)
        return None

    except Exception as e:
        log_error(f"Failed to check active instances: {e}")
        return None


def find_cheapest_us_offer(debug: bool = False) -> Optional[Dict]:
    """Find the cheapest available US-based Vast.ai offer."""
    if requests is None:
        raise RuntimeError("requests library not installed. Run: pip install requests")

    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        raise RuntimeError("VAST_API_KEY environment variable not set")

    try:
        max_price = VASTAI_AUTO_PROVISION['max_price_per_hour']

        # Vast.ai uses POST with JSON body for search
        payload = {
            "verified": {"eq": True},
            "external": {"eq": False},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "reliability": {"gte": 0.95},
            "num_gpus": {"gte": 1},
            "inet_up": {"gte": 100},
            "geolocation": {"in": ["US"]},
            "dph_total": {"lte": max_price},
            "order": [["dph_total", "asc"]],
            "type": "on-demand",
            "limit": 20,
            "allocated_storage": VASTAI_AUTO_PROVISION['disk_gb'],
        }

        url = f"{VASTAI_API_BASE}/bundles/?api_key={api_key}"
        log_info("Searching for cheapest US offers...", debug)
        log_debug(f"POST {url}", debug)

        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        offers = data.get("offers", [])
        log_debug(f"Found {len(offers)} US offers within ${max_price}/hr", debug)

        if not offers:
            log_warning(f"No US offers available within ${max_price}/hr limit")
            return None

        # Already sorted by price, take the first one
        cheapest = offers[0]

        log_info(f"Cheapest US offer: ${cheapest.get('dph_total', 0):.3f}/hr, GPU: {cheapest.get('gpu_name')}, Geo: {cheapest.get('geolocation')}", debug)

        return cheapest

    except Exception as e:
        log_error(f"Failed to search offers: {e}")
        return None


def create_vastai_instance(offer_id: int, debug: bool = False) -> Optional[int]:
    """Create a new Vast.ai instance from an offer."""
    try:
        payload = {
            "client_id": "me",
            "image": VASTAI_AUTO_PROVISION['docker_image'],
            "disk": VASTAI_AUTO_PROVISION['disk_gb'],
            "label": "oracle-image-gen",
            "onstart": "pip install -q google-genai",  # Pre-install dependency
        }

        log_info(f"Creating instance from offer {offer_id}...", debug)
        resp = vastai_api_request("PUT", f"/asks/{offer_id}/", payload=payload, debug=debug)

        instance_id = resp.get("new_contract")
        if instance_id:
            log_info(f"Created instance: {instance_id}", debug)
            return instance_id
        else:
            log_error(f"No instance ID in response: {resp}")
            return None

    except Exception as e:
        log_error(f"Failed to create instance: {e}")
        return None


def wait_for_instance_ready(instance_id: int, debug: bool = False) -> Optional[Dict]:
    """Wait for an instance to be running and SSH-ready."""
    if requests is None:
        raise RuntimeError("requests library not installed")

    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        raise RuntimeError("VAST_API_KEY not set")

    startup_timeout = VASTAI_AUTO_PROVISION['startup_timeout']
    ssh_timeout = VASTAI_AUTO_PROVISION['ssh_timeout']

    log_info(f"Waiting for instance {instance_id} to be ready (max {startup_timeout}s)...", debug)

    start_time = time.time()
    ssh_host = None
    ssh_port = None

    # Initial wait - instances typically take 10-20s to start
    print("🔮 Waiting 10s for instance to initialize...", file=sys.stderr)
    time.sleep(10)

    # Phase 1: Wait for instance to be running (check every 3s for faster detection)
    while time.time() - start_time < startup_timeout:
        try:
            # Use correct API format with api_key in URL
            url = f"{VASTAI_API_BASE}/instances/{instance_id}/?api_key={api_key}&owner=me"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            # API wraps response in "instances" key
            if "instances" in data:
                data = data["instances"]

            status = data.get("actual_status", "")
            ssh_host = data.get("ssh_host")
            ssh_port = data.get("ssh_port")

            log_debug(f"Instance status: {status}, ssh: {ssh_host}:{ssh_port}", debug)

            if status == "running" and ssh_host and ssh_port:
                log_info(f"Instance running at {ssh_host}:{ssh_port}", debug)
                break
            elif status in ["exited", "error", "failed"]:
                log_error(f"Instance failed with status: {status}")
                return None

        except Exception as e:
            log_debug(f"Status check failed: {e}", debug)

        time.sleep(3)  # Check every 3s for faster detection
    else:
        log_error(f"Instance startup timed out after {startup_timeout}s")
        return None

    # Phase 2: Wait for SSH to be ready (check every 2s with fast timeout)
    log_info("Waiting for SSH to be ready...", debug)
    ssh_start = time.time()

    while time.time() - ssh_start < ssh_timeout:
        try:
            result = subprocess.run(
                [
                    "ssh", "-p", str(ssh_port),
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "ConnectTimeout=3",
                    "-o", "BatchMode=yes",
                    f"root@{ssh_host}",
                    "echo 'SSH_READY'"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and "SSH_READY" in result.stdout:
                log_info("SSH is ready!", debug)
                return {
                    "id": instance_id,
                    "host": ssh_host,
                    "port": ssh_port,
                    "user": "root",
                }

        except Exception as e:
            log_debug(f"SSH check failed: {e}", debug)

        time.sleep(2)  # Check every 2s

    log_error(f"SSH not ready after {ssh_timeout}s")
    return None


def destroy_vastai_instance(instance_id: int, debug: bool = False) -> bool:
    """Destroy a Vast.ai instance."""
    if requests is None:
        log_error("requests library not installed")
        return False

    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        log_error("VAST_API_KEY not set")
        return False

    try:
        log_info(f"Destroying instance {instance_id}...", debug)
        url = f"{VASTAI_API_BASE}/instances/{instance_id}/?api_key={api_key}"
        resp = requests.delete(url, json={}, timeout=30)
        resp.raise_for_status()
        log_info(f"Instance {instance_id} destroyed", debug)
        return True
    except Exception as e:
        log_error(f"Failed to destroy instance {instance_id}: {e}")
        return False


def auto_provision_us_instance(debug: bool = False) -> Tuple[Optional[Dict], bool]:
    """
    Auto-provision a US-based Vast.ai instance.

    Returns:
        Tuple of (instance_info, was_created)
        - instance_info: Dict with host, port, user, id or None if failed
        - was_created: True if we created this instance (need to destroy later)
    """
    # First, check for existing active US instance
    log_info("Checking for existing US instances...", debug)
    existing = get_active_us_instance(debug)

    if existing:
        print(f"🔮 Reusing existing US instance: {existing['host']}:{existing['port']} (${existing.get('price', 0):.3f}/hr)", file=sys.stderr)
        return (existing, False)  # Not created by us, don't destroy

    # No existing instance, need to provision one
    print("🔮 No US instance found. Auto-provisioning cheapest available...", file=sys.stderr)

    offer = find_cheapest_us_offer(debug)
    if not offer:
        print("🔮 ERROR: No suitable US instances available", file=sys.stderr)
        return (None, False)

    print(f"🔮 Found offer: ${offer.get('dph_total', 0):.3f}/hr, {offer.get('gpu_name')}", file=sys.stderr)

    instance_id = create_vastai_instance(offer["id"], debug)
    if not instance_id:
        return (None, False)

    print(f"🔮 Instance {instance_id} created. Waiting for startup...", file=sys.stderr)

    instance_info = wait_for_instance_ready(instance_id, debug)
    if not instance_info:
        # Failed to start, destroy the failed instance
        destroy_vastai_instance(instance_id, debug)
        return (None, False)

    print(f"🔮 Instance ready: {instance_info['host']}:{instance_info['port']}", file=sys.stderr)
    return (instance_info, True)  # We created it, need to destroy after


def generate_remote_image_script(prompt: str, api_key: str, output_filename: str) -> str:
    """
    Generate a Python script to run on the remote server for image generation.

    The script:
    1. Installs google-generativeai if needed
    2. Generates the image using Gemini
    3. Saves to /tmp and prints the path
    """
    # Escape the prompt for shell safety
    escaped_prompt = prompt.replace("'", "'\\''").replace('"', '\\"')

    script = f'''
import subprocess
import sys

# Install dependencies quietly (google-genai is the correct package name)
result = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "google-genai", "Pillow"],
                        capture_output=True, text=True)
if result.returncode != 0:
    print(f"ERROR: pip install failed: {{result.stderr}}", file=sys.stderr)
    sys.exit(1)

import os
from google import genai
from google.genai import types

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("ERROR: No API key", file=sys.stderr)
    sys.exit(1)

client = genai.Client(api_key=api_key)

prompt = """{escaped_prompt}"""

try:
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )

    if not response.candidates or not response.candidates[0].content:
        print("ERROR: No content in response", file=sys.stderr)
        sys.exit(1)

    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
            output_path = "/tmp/{output_filename}"
            with open(output_path, "wb") as f:
                f.write(part.inline_data.data)
            print(output_path)  # This is the only stdout - the path
            sys.exit(0)

    print("ERROR: No image in response", file=sys.stderr)
    sys.exit(1)

except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
'''
    return script


def run_on_vastai_instance(
    instance: Dict,
    script: str,
    api_key: str,
    local_output_path: Path,
    debug: bool = False
) -> Tuple[Optional[str], Optional[str]]:
    """
    Run an image generation script on a Vast.ai instance.

    Args:
        instance: Dict with host, port, user
        script: Python script to execute
        api_key: Gemini API key to pass
        local_output_path: Where to save the result locally
        debug: Enable debug output

    Returns:
        Tuple of (saved_path, error_message)
    """
    host = instance['host']
    port = instance['port']
    user = instance.get('user', 'root')
    timeout = VASTAI_CONFIG['timeout']

    log_debug(f"Running on {user}@{host}:{port}", debug)

    try:
        # Execute script on remote server via stdin
        ssh_cmd = [
            "ssh",
            "-p", str(port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            f"{user}@{host}",
            f'GEMINI_API_KEY="{api_key}" python3 -'
        ]

        log_info("Executing image generation on Vast.ai...", debug)

        result = subprocess.run(
            ssh_cmd,
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        log_debug(f"SSH exit code: {result.returncode}", debug)
        if result.stderr:
            log_debug(f"SSH stderr: {result.stderr}", debug)
        if result.stdout:
            log_debug(f"SSH stdout: {result.stdout}", debug)

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown SSH error"
            # Check for geo-restriction on remote too
            if "not available in your country" in error_msg.lower():
                return (None, "Remote server also geo-restricted. Need US-based instance.")
            return (None, f"Remote execution failed: {error_msg}")

        # Get the remote path from stdout
        remote_output = result.stdout.strip()
        if not remote_output or not remote_output.startswith("/tmp/"):
            return (None, f"Invalid remote output path: {remote_output}")

        log_info(f"Image generated on remote: {remote_output}", debug)

        # SCP the file back
        log_info("Transferring image from Vast.ai...", debug)

        scp_cmd = [
            "scp",
            "-P", str(port),
            "-o", "StrictHostKeyChecking=no",
            f"{user}@{host}:{remote_output}",
            str(local_output_path)
        ]

        scp_result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=30)

        if scp_result.returncode != 0:
            return (None, f"SCP failed: {scp_result.stderr}")

        log_info(f"Image transferred to: {local_output_path}", debug)

        # Cleanup remote file
        cleanup_cmd = [
            "ssh",
            "-p", str(port),
            "-o", "StrictHostKeyChecking=no",
            f"{user}@{host}",
            f"rm -f {remote_output}"
        ]
        subprocess.run(cleanup_cmd, capture_output=True, timeout=10)
        log_debug("Remote temp file cleaned up", debug)

        return (str(local_output_path), None)

    except subprocess.TimeoutExpired:
        return (None, f"Remote operation timed out after {timeout}s")
    except FileNotFoundError:
        return (None, "SSH/SCP not found. Ensure OpenSSH is installed.")
    except Exception as e:
        log_error(f"Remote execution failed: {type(e).__name__}: {e}")
        return (None, str(e))


def imagine_via_vastai(
    prompt: str,
    api_key: str,
    local_output_path: Path,
    debug: bool = False
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate an image via Vast.ai server to bypass geo-restrictions.

    This function will:
    1. Check for existing active US-based instance
    2. If none, auto-provision a new US instance
    3. Run the image generation
    4. Destroy the instance if we created it

    Args:
        prompt: Image generation prompt
        api_key: Gemini API key
        local_output_path: Where to save the image locally
        debug: Enable debug output

    Returns:
        Tuple of (saved_path, error_message)
    """
    log_info("=== Falling back to Vast.ai for image generation ===", debug)

    # Check if Vast.ai API key is available for auto-provisioning
    vast_api_key = os.environ.get("VAST_API_KEY")

    instance = None
    instance_created = False

    if vast_api_key and VASTAI_AUTO_PROVISION['enabled']:
        # Try auto-provisioning
        print("🔮 Checking for US-based Vast.ai instances...", file=sys.stderr)
        try:
            instance, instance_created = auto_provision_us_instance(debug)
        except Exception as e:
            log_error(f"Auto-provision failed: {e}")
            print(f"🔮 Auto-provision failed: {e}", file=sys.stderr)

    if not instance:
        # Fallback to configured instance (may not be US-based)
        log_info("Using configured Vast.ai instance (may not be US-based)", debug)
        print("🔮 No VAST_API_KEY set or auto-provision failed. Using configured instance.", file=sys.stderr)
        instance = {
            'host': VASTAI_CONFIG['host'],
            'port': VASTAI_CONFIG['port'],
            'user': VASTAI_CONFIG['user'],
        }

    # Generate the script
    remote_filename = f"oracle_img_{uuid.uuid4().hex[:8]}.png"
    script = generate_remote_image_script(prompt, api_key, remote_filename)

    try:
        # Run on the instance
        result_path, error = run_on_vastai_instance(
            instance, script, api_key, local_output_path, debug
        )

        return (result_path, error)

    finally:
        # Cleanup: destroy instance if we created it
        if instance_created and instance.get('id'):
            print(f"🔮 Destroying temporary instance {instance['id']}...", file=sys.stderr)
            destroy_vastai_instance(instance['id'], debug)
            print("🔮 Instance destroyed. Cost saved!", file=sys.stderr)


def imagine(
    prompt: str,
    output_path: Optional[str] = None,
    reference_image: Optional[str] = None,
    debug: bool = False
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate an image using the Oracle.

    Args:
        prompt: Description of the image to generate
        output_path: Optional path to save the image (default: auto-generated)
        reference_image: Optional reference image to base generation on
        debug: Enable debug output

    Returns:
        Tuple of (saved_path, error_message)
    """
    log_info("=== Oracle Imagine Started ===", debug)
    log_debug(f"Prompt: {prompt[:100]}...", debug)

    # Check API key
    api_key = get_gemini_api_key()
    if not api_key:
        return (None, "GEMINI_API_KEY environment variable not set")

    # Pre-compute output path for potential Vast.ai fallback
    if output_path:
        save_path = Path(output_path)
        if not save_path.suffix:
            save_path = save_path.with_suffix('.png')
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"oracle_imagine_{timestamp}.png"
        save_path = IMAGE_OUTPUT_DIR / filename

    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Process reference image if provided
    ref_image_data = None
    ref_image_mime_type = None
    if reference_image:
        ref_image_data, ref_image_mime_type, error = read_image_as_base64(reference_image, debug)
        if error:
            return (None, error)
        log_info(f"Reference image loaded: {reference_image}", debug)

    try:
        log_info("Creating Gemini client...", debug)
        client = genai.Client(api_key=api_key)

        # Build content parts
        parts = []

        # Add reference image first if provided
        if ref_image_data and ref_image_mime_type:
            parts.append(types.Part.from_bytes(data=ref_image_data, mime_type=ref_image_mime_type))
            log_debug("Added reference image to request", debug)

        # Add text prompt
        parts.append(types.Part.from_text(text=prompt))

        contents = [
            types.Content(role="user", parts=parts),
        ]

        log_debug("Building GenerateContentConfig for image generation...", debug)
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        )

        log_info("Calling Gemini API for image generation...", debug)
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=contents,
            config=config,
        )

        log_debug("Response received", debug)

        # Extract image from response
        if not response.candidates or not response.candidates[0].content:
            return (None, "No content in response")

        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.data:
                inline_data = part.inline_data
                image_bytes = inline_data.data
                mime_type = inline_data.mime_type

                # Determine file extension from response
                ext = mimetypes.guess_extension(mime_type) or '.png'
                if ext == '.jpe':
                    ext = '.jpg'

                # Update save_path with correct extension if different
                if save_path.suffix != ext:
                    save_path = save_path.with_suffix(ext)

                # Save the image
                with open(save_path, 'wb') as f:
                    f.write(image_bytes)

                log_info(f"Image saved to: {save_path}", debug)
                return (str(save_path), None)

            elif hasattr(part, 'text') and part.text:
                # Log any text response (might contain additional info)
                log_debug(f"Text in response: {part.text[:200]}", debug)

        return (None, "No image data in response")

    except Exception as e:
        error_str = str(e).lower()
        log_error(f"Image generation failed: {type(e).__name__}: {e}")

        # Check for geo-restriction error
        if "not available in your country" in error_str or "geo" in error_str:
            log_warning("Geo-restriction detected. Attempting Vast.ai fallback...")
            print("🔮 Geo-restricted locally. Routing through Vast.ai...", file=sys.stderr)

            # Note: reference image not supported via Vast.ai fallback (too complex)
            if reference_image:
                log_warning("Reference image not supported via Vast.ai fallback")
                print("🔮 WARNING: Reference image ignored for remote generation", file=sys.stderr)

            return imagine_via_vastai(prompt, api_key, save_path, debug)

        import traceback
        log_debug(f"Full traceback:\n{traceback.format_exc()}", debug)
        return (None, str(e))


def quick_ask(query: str, debug: bool = False, no_history: bool = False) -> str:
    """Quick question without structured output - just get a text answer."""
    log_info("=== Quick Ask Started ===", debug)

    api_key = get_gemini_api_key()
    if not api_key:
        log_error("GEMINI_API_KEY not set")
        return "ERROR: GEMINI_API_KEY not set"

    # Get project ID and history
    project_id = get_project_id(debug)
    history = [] if no_history else load_history(project_id, debug)
    history_section, _ = format_history_for_prompt(history, debug)

    context, _ = get_context_file(debug)
    context_section = f"\n\nContext:\n{context}" if context else ""

    # Build full prompt with history
    full_prompt = f"{query}{history_section}{context_section}"

    try:
        log_info("Creating Gemini client...", debug)
        client = genai.Client(api_key=api_key)

        log_info("Calling Gemini API...", debug)
        response = client.models.generate_content(
            model="gemini-2.5-pro-preview-06-05",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=5000),
                system_instruction="You are a helpful AI assistant. Be concise and direct. You have access to conversation history - use it for context.",
            ),
        )

        log_info("Response received", debug)

        # Save to history
        if not no_history:
            exchange = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "query": query,
                "response_summary": response.text[:200] + "..." if len(response.text) > 200 else response.text,
                "mode": "quick"
            }
            history.append(exchange)
            save_history(project_id, history, debug)

        return response.text

    except Exception as e:
        log_error(f"Quick ask failed: {type(e).__name__}: {e}")
        return f"ERROR: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Gemini Oracle - AI Orchestration for Claude Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  oracle ask "How should I implement a caching layer?"
  oracle ask --mode=validate "I've implemented the database schema"
  oracle ask --files src/main.py "Review this file"
  oracle ask --files "src/main.py:1-50,100-110" "Check these sections"
  oracle ask --image screenshot.png "What's wrong with this UI?"
  oracle imagine "Minimalist logo for fintech startup" --output logo.png
  oracle imagine --ref current.png "Make this more modern"
  oracle quick "What's the best Python HTTP library?"
  oracle history                    # View conversation history
  oracle history --clear            # Clear history

File specifications:
  src/main.py              # Whole file
  src/main.py:10-50        # Lines 10-50
  src/main.py:1-50,100-110 # Lines 1-50 and 100-110

Log files: ~/.oracle/logs/
History files: ~/.oracle/history/
Generated images: ~/.oracle/images/
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ask command
    ask_parser = subparsers.add_parser("ask", help="Ask the oracle for guidance")
    ask_parser.add_argument("query", help="Your question or task description")
    ask_parser.add_argument(
        "--mode",
        choices=["plan", "validate"],
        default="plan",
        help="Mode: 'plan' for new tasks, 'validate' for checking progress"
    )
    ask_parser.add_argument(
        "--thinking",
        choices=["LOW", "MEDIUM", "HIGH"],
        default="HIGH",
        help="Thinking depth level"
    )
    ask_parser.add_argument(
        "--context",
        help="Override context (instead of reading from file)"
    )
    ask_parser.add_argument(
        "--files",
        help="Comma-separated list of files to attach (supports line ranges: file.py:1-50,100-110)"
    )
    ask_parser.add_argument(
        "--image",
        help="Path to an image file to analyze (screenshot, diagram, mockup, etc.)"
    )
    ask_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output"
    )
    ask_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output to stderr"
    )
    ask_parser.add_argument(
        "--no-history",
        action="store_true",
        help="Don't use or save conversation history"
    )

    # imagine command
    imagine_parser = subparsers.add_parser("imagine", help="Generate an image")
    imagine_parser.add_argument("prompt", help="Description of the image to generate")
    imagine_parser.add_argument(
        "--output", "-o",
        help="Output file path (default: auto-generated in ~/.oracle/images/)"
    )
    imagine_parser.add_argument(
        "--ref",
        help="Reference image to base generation on (for modifications/iterations)"
    )
    imagine_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )

    # quick command
    quick_parser = subparsers.add_parser("quick", help="Quick question, text response")
    quick_parser.add_argument("query", help="Your question")
    quick_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    quick_parser.add_argument(
        "--no-history",
        action="store_true",
        help="Don't use or save conversation history"
    )

    # history command
    history_parser = subparsers.add_parser("history", help="View or manage conversation history")
    history_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear history for current project"
    )
    history_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )

    # info command
    info_parser = subparsers.add_parser("info", help="Show oracle status and context")

    # logs command
    logs_parser = subparsers.add_parser("logs", help="Show recent logs")
    logs_parser.add_argument(
        "-n", "--lines",
        type=int,
        default=50,
        help="Number of lines to show"
    )

    # context command - view FULLAUTO_CONTEXT.md
    context_parser = subparsers.add_parser("context", help="Show FULLAUTO_CONTEXT.md")

    args = parser.parse_args()

    # Auto-prepend recovery header to FULLAUTO_CONTEXT.md if it exists without one
    ensure_fullauto_header()

    if args.command == "ask":
        # Parse files argument
        files = None
        if args.files:
            # Split by comma, but be careful with line ranges that also use commas
            # We split on commas that are followed by a path (not a number)
            files = []
            current_spec = ""
            for part in args.files.split(','):
                part = part.strip()
                # Check if this looks like a continuation of line ranges (just numbers with dash)
                if current_spec and re.match(r'^\d+-?\d*$', part):
                    current_spec += ',' + part
                else:
                    if current_spec:
                        files.append(current_spec)
                    current_spec = part
            if current_spec:
                files.append(current_spec)

        result = ask_oracle(
            query=args.query,
            mode=args.mode,
            context_override=args.context,
            thinking_level=args.thinking,
            debug=args.debug,
            no_history=args.no_history,
            files=files,
            image_path=args.image
        )
        if args.pretty:
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result))

    elif args.command == "imagine":
        saved_path, error = imagine(
            prompt=args.prompt,
            output_path=args.output,
            reference_image=args.ref,
            debug=args.debug
        )

        if error:
            print(f"🔮 ERROR: {error}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"🔮 Image generated: {saved_path}")

    elif args.command == "quick":
        print(quick_ask(args.query, debug=args.debug, no_history=args.no_history))

    elif args.command == "history":
        project_id = get_project_id(args.debug)

        if args.clear:
            if clear_history(project_id, args.debug):
                print(f"✓ History cleared for project: {project_id}")
            else:
                print(f"✗ Failed to clear history")
        else:
            history = load_history(project_id, args.debug)
            if not history:
                print(f"No conversation history for project: {project_id}")
            else:
                print(f"=== Conversation History ({len(history)} exchanges) ===")
                print(f"Project: {project_id}")
                print("─" * 50)
                for i, exchange in enumerate(history, 1):
                    print(f"\n[{i}] {exchange.get('timestamp', 'unknown')} ({exchange.get('mode', 'unknown')})")
                    print(f"    Q: {exchange.get('query', '')[:100]}...")
                    files_rev = exchange.get('files_reviewed', [])
                    if files_rev:
                        print(f"    Files: {', '.join(files_rev)}")
                    img = exchange.get('image_analyzed')
                    if img:
                        print(f"    Image: {img}")
                    print(f"    A: {exchange.get('response_summary', '')[:100]}...")
                print("\n" + "─" * 50)
                print(f"History file: {get_history_file(project_id)}")

    elif args.command == "info":
        project_id = get_project_id()
        context, context_tokens = get_context_file()
        api_key = get_gemini_api_key()
        history = load_history(project_id)

        print("=== Gemini Oracle Status ===")
        print(f"API Key: {'SET' if api_key else 'NOT SET'}")
        print(f"Project ID: {project_id}")
        print(f"Context File: {'FOUND' if context else 'NOT FOUND'}")
        if context:
            print(f"Context Tokens: ~{context_tokens:,}")
        print(f"History: {len(history)} exchanges (max {MAX_HISTORY_EXCHANGES})")
        print(f"Log Directory: {LOG_DIR}")
        print(f"History Directory: {HISTORY_DIR}")
        print(f"Image Output: {IMAGE_OUTPUT_DIR}")

        if context:
            lines = context.split('\n')
            print(f"Context Size: {len(context)} chars, {len(lines)} lines")
            print(f"First line: {lines[0][:80]}...")

        # Check if google-genai is properly configured
        try:
            client = genai.Client(api_key=api_key) if api_key else None
            print(f"Gemini Client: {'OK' if client else 'NOT CONFIGURED'}")
        except Exception as e:
            print(f"Gemini Client: ERROR - {e}")

    elif args.command == "logs":
        if LOG_FILE.exists():
            with open(LOG_FILE) as f:
                lines = f.readlines()
                for line in lines[-args.lines:]:
                    print(line, end='')
        else:
            print("No logs found for today")

    elif args.command == "context":
        context_file = Path.cwd() / "FULLAUTO_CONTEXT.md"
        if context_file.exists():
            print(context_file.read_text())
        else:
            print(f"No FULLAUTO_CONTEXT.md found in {Path.cwd()}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
