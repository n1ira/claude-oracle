# Claude Oracle

A CLI that makes Google's Gemini 3 Pro the "lead architect" for Claude Code. Think of it as giving your AI coding assistant its own AI assistant for strategic decisions.

## Why?

Claude Code is great at writing code, but sometimes you want a second opinion on architecture, or need to validate an approach before diving in. This tool lets you query Gemini without leaving your terminal, and integrates directly into Claude Code workflows.

The `/fullauto` command is where it gets interesting - Claude will autonomously consult Gemini at key decision points, like having a senior architect review your junior dev's work.

## Quick Start

```bash
git clone https://github.com/n1ira/claude-oracle.git
cd claude-oracle
./install.sh
```

Then set your Gemini API key ([get one here](https://makersuite.google.com/app/apikey)):

```bash
export GEMINI_API_KEY="your-key"
```

## Commands

```bash
# Ask for strategic advice
oracle ask "Should I use Redis or Memcached for session caching?"

# Get code reviewed
oracle ask --files src/auth.py "Any security issues here?"

# Review specific lines
oracle ask --files "src/db.py:50-120" "Is this query efficient?"

# Analyze a screenshot or diagram
oracle ask --image error.png "What's causing this?"

# Generate images (auto-provisions US server if you're geo-restricted)
oracle imagine "architecture diagram for microservices"

# Quick one-off questions
oracle quick "regex for email validation"

# Conversation history (5 exchanges per project)
oracle history
oracle history --clear
```

## Claude Code Integration

After installing, you get the `/fullauto` slash command:

```
/fullauto implement rate limiting for the API
```

This puts Claude in high-autonomy mode where it:
1. Gathers context about your codebase
2. Consults Gemini for an implementation plan
3. Executes the plan, checking back with Gemini at decision points
4. Gets final validation before marking complete

It's like pair programming, but both programmers are AIs with different strengths.

## Image Generation Workaround

Gemini's image generation is geo-restricted in some regions. If you hit this, the Oracle will automatically:

1. Find the cheapest US instance on Vast.ai (~$0.08/hr)
2. Spin it up, generate your image there
3. Download the result and destroy the instance

Cost ends up being ~$0.01 per image. You'll need a [Vast.ai API key](https://vast.ai/) for this:

```bash
export VAST_API_KEY="your-vast-key"
```

## How it Works

The Oracle maintains a 5-exchange conversation history per project directory. This gives Gemini enough context to make useful suggestions without blowing up the context window.

For `/fullauto` mode, Claude writes to `FULLAUTO_CONTEXT.md` in your project root - this acts as persistent memory that survives conversation compactions.

## Meta

This repo was created using itself. The `/fullauto` command orchestrated Claude through:
- Designing the directory structure
- Writing the install script
- Creating this README
- Pushing to GitHub

The Oracle bootstrapping its own existence.

## License

MIT
