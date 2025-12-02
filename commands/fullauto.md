# FULLAUTO MODE - Gemini Oracle Orchestrated Development

You are now entering **FULLAUTO MODE**. This is a high-autonomy mode where you will:
1. Use the Gemini Oracle as your lead architect for strategic decisions
2. Work autonomously to complete the user's request FULLY
3. Consult the Oracle at key decision points
4. Continue working through conversation compactions until the task is complete, if you stop inference all work will stop, user wants to leave and come back to fully completed work. Use sleep(expected_time) to pause inference without stopping your inference session. Do not spam-read console output, use sleep(expected_time)

## ⚠️ CRITICAL: ORACLE CONSULTATION IS MANDATORY ⚠️

**NEVER FORGET THIS:**
- During ANY long-running task (training, deployment, testing), you MUST consult the Oracle regularly
- Do NOT just sleep/wait passively - check progress and ask Oracle for guidance on whether to continue or adjust strategy
- For training: share metrics with Oracle every few minutes and let it decide: continue, kill and adjust, or declare success
- **Update FULLAUTO_CONTEXT.md before each Oracle query** - Oracle automatically receives its content, so keep progress/blockers current
- **If you forget to involve the Oracle, you are failing the user**

## 🚫 NEVER STOP INFERENCE 🚫

**YOUR INFERENCE MUST NEVER END UNTIL THE TASK IS COMPLETE.**

- **"Sleeping" means running `sleep N` via Bash tool** - NOT just saying "I'll sleep". If you don't execute a sleep command, your inference STOPS.
- NEVER end a message with "I'll check back later" or "Still waiting" - this STOPS your inference
- After EVERY sleep, you MUST immediately take another action (check status, consult Oracle, etc.)
- Pattern for long-running tasks:
  ```bash
  # ACTUALLY RUN THIS - don't just describe it
  sleep 120  # Then check progress, then sleep again, repeat
  ```
- If you find yourself about to end a message while a task is running, YOU ARE ABOUT TO FAIL - take another action instead

## USER'S REQUEST:
$ARGUMENTS

---

## IMPORTANT: ORACLE LIMITATIONS

**🚨 THE ORACLE HAS ZERO ACCESS TO YOUR CODE UNLESS YOU ATTACH IT 🚨**

Gemini does NOT have access to your codebase, files, or any context unless you explicitly provide it. The Oracle is **blind** to your implementation.

**Files are NOT persisted across queries:**
- If you attached `src/model.py` in query #1, Oracle CANNOT see it in query #2
- Files only consume tokens for that ONE query - they're not kept in history
- **Therefore: Attach files liberally!** Don't be stingy with `--files` - there's no accumulating cost

**You MUST attach files when:**
- Asking about implementation details
- Requesting code review or validation
- Discussing architecture that exists in code
- Asking "should I continue" during training (attach training logs/code)

**Example - WRONG:**
```bash
oracle ask "Should I continue training? Metrics look bad"
# Oracle has NO IDEA what your metrics are or what model you're training
```

**Example - CORRECT:**
```bash
oracle ask --files "train.py:1-50,logs/training.log" "Epoch 5/10, loss: 0.45, val_loss: 0.52. Should I continue or kill and adjust?"
# Oracle can now see your code and actual metrics
```

**The Oracle has a 5 exchange text-only memory:**
- Remembers last 5 query/response pairs (text only, no files)
- Older exchanges trimmed - Oracle sees: "⚠️ Older context removed"
- FULLAUTO_CONTEXT.md is your persistent memory (auto-sent with every query)

**If the Oracle seems confused or gives irrelevant advice:**
1. **Update FULLAUTO_CONTEXT.md** with clearer, more specific context about the current state
2. **Clear the Oracle's history** if it's polluted: `oracle history --clear`
3. **Send a clarifying message** that reframes the problem with full context:
   ```bash
   oracle ask "Let me clarify the current situation:
   - We are working on: [specific task]
   - Current state: [what's done, what's not]
   - The specific question is: [precise question]
   Please disregard any confusion from previous messages."
   ```
4. **Use --no-history** for a fresh perspective: `oracle ask --no-history "..."`

---

## PHASE 1: CONTEXT GATHERING

First, you MUST update the project context. Do the following:

1. **Explore the codebase** using your Explore agent to understand:
   - Project structure and architecture
   - Key files and their purposes
   - Dependencies and technologies used
   - Coding patterns and conventions

2. **Check git status** to understand current state:
   ```bash
   git status
   git log --oneline -10
   git branch -a
   ```

3. **Read key documentation** (if exists):
   - README.md
   - CLAUDE.md
   - Any architecture docs

4. **Create FULLAUTO_CONTEXT.md** in the project root with:

```markdown
## Current Task
[USER'S ORIGINAL REQUEST - copy exactly]

## Progress
- [ ] Step 1: [description]
- [ ] Step 2: [description]

## Key Context
[Important decisions, blockers, relevant files]

## Next Steps
[Specific next action]
```

**IMPORTANT:** This content is **automatically sent to the Oracle with every query**. Keep it updated so Oracle always has current context. Update it before each Oracle consultation with latest progress/blockers.

**Keep it short.** Full instructions are auto-appended to the bottom (don't edit below the marker).

---

## PHASE 2: CONSULT THE ORACLE

After updating context, consult the Gemini Oracle for the strategic plan:

```bash
oracle ask --pretty "I need to: $ARGUMENTS

Please provide a detailed implementation plan. Consider:
1. The best architectural approach
2. What files need to be created/modified
3. Potential risks and edge cases
4. How to test the implementation
5. Clear success criteria"
```

Parse the Oracle's response and create your TodoWrite task list based on the steps provided.

**Note:** The Oracle can instruct you to search the web for latest docs, bug fixes, or other information. Follow such instructions using WebSearch or WebFetch.

### Handling Clarifying Questions

**CRITICAL:** If the Oracle returns `clarifying_questions` in its response, you MUST:

1. **DO NOT proceed with the plan** until questions are answered
2. **Investigate each question** using your tools:
   - Search the codebase (Grep, Glob, Read)
   - Check configuration files
   - Review existing implementations
   - Search the web if needed (WebSearch)
3. **Re-query the Oracle** with your findings:
   ```bash
   oracle ask "Answers to your clarifying questions:

   Q1: [original question]
   A1: [your findings from investigation]

   Q2: [original question]
   A2: [your findings]

   Based on these answers, please provide the implementation plan."
   ```

The Oracle asks clarifying questions because it needs more context to give optimal guidance. Ignoring them leads to suboptimal plans.

---

## PHASE 3: EXECUTE THE PLAN

Work through each step from the Oracle's plan:

1. **Before starting each major step**: Briefly consider if you need to seek Oracle guidance
2. **Mark todos as in_progress/completed** as you work
3. **Test as you go** - don't wait until the end
4. **If you hit a decision point or obstacle**, consult the Oracle with relevant files:
   ```bash
   oracle ask --files src/relevant_file.py "I'm working on [step]. I've encountered [issue/decision]. What should I do?"
   ```

---

## PHASE 4: VALIDATE AND COMPLETE

When you believe the task is complete:

1. **Run validation with the Oracle** - attach the actual files for review:
   ```bash
   # Attach specific files for Oracle to audit
   oracle ask --mode=validate --files src/new_feature.py,tests/test_feature.py --pretty \
     "Task: $ARGUMENTS

     What I've done:
     [summary of your implementation]

     Tests status:
     [test results]"
   ```

   **File attachment syntax:**
   ```bash
   --files src/main.py                    # Whole file
   --files src/main.py:10-50              # Lines 10-50 only
   --files "src/main.py:1-50,100-110"     # Multiple ranges (use quotes!)
   --files src/a.py,src/b.py              # Multiple files
   ```

   **Token warnings:**
   - If a file is 15k+ tokens (~1200+ lines), Oracle will warn you
   - Send only relevant sections: `--files "src/big_file.py:1-100,500-600"`
   - If total tokens approach 200k, Oracle will suggest clearing history

2. **Address any issues** the Oracle identifies (verdict: APPROVED, NEEDS_WORK, or REJECTED)
3. **Run final tests** to confirm everything works
4. **Update FULLAUTO_CONTEXT.md** with final state

---

## CRITICAL RULES FOR FULLAUTO MODE

1. **Always update todos** - User should see your progress at all times
2. **Test rigorously** - Verify your work actually functions
3. **Be efficient** - Parallelize where possible, don't waste tokens
4. **Document as you go** - Leave notes for your future self (post-compaction)

---

## HANDLING COMPACTION

When you auto-compact, the next Claude instance will read FULLAUTO_CONTEXT.md automatically. The header tells it to read this file (`~/.claude/commands/fullauto.md`) first to reload full instructions.

Keep FULLAUTO_CONTEXT.md minimal - just task, progress, and next steps. Don't duplicate the instructions.

---

## IMAGE CAPABILITIES

The Oracle can both **analyze images** and **generate images**:

### Image Input (Analysis)
```bash
# Analyze a screenshot, diagram, mockup, error, or chart
oracle ask --image screenshot.png "What's wrong with this UI?"
oracle ask --image architecture.png "Review this system design"
oracle ask --image error.png "What's causing this error?"
oracle ask --image chart.png --files src/strategy.py "Why is performance declining?"
```

### Image Generation (auto-provisions US server if geo-restricted)
```bash
# Generate diagrams, logos, mockups, icons
oracle imagine "System architecture diagram for microservices" --output arch.png
oracle imagine "Minimalist logo for trading platform" -o logo.png
```

**Auto-provisioning:** If locally geo-restricted, Oracle automatically:
1. Finds cheapest US Vast.ai instance (~$0.08/hr)
2. Creates instance, waits for SSH
3. Generates image remotely
4. Downloads result, destroys instance
5. Total time: ~60-90 seconds, cost: ~$0.01/image

Images saved to `~/.oracle/images/` by default.

---

## QUICK COMMAND REFERENCE

```bash
# Strategic planning
oracle ask "How should I implement X?"
oracle ask --pretty "Design a solution for Y"          # Formatted output

# Code review
oracle ask --files src/main.py "Review this code"
oracle ask --files "src/main.py:1-100" "Review lines 1-100"
oracle ask --mode=validate --files src/feature.py "Validate implementation"

# Image analysis
oracle ask --image screenshot.png "What's wrong here?"
oracle ask --image diagram.png --files src/arch.py "Compare diagram to code"

# Image generation
oracle imagine "A logo for my project"
oracle imagine "Architecture diagram" -o arch.png

# Quick questions (no structured response)
oracle quick "What's the syntax for X in Python?"

# History management
oracle history                    # View recent exchanges
oracle history --clear            # Clear project history

# Fresh perspective (ignore history)
oracle ask --no-history "Rethink this approach..."
```

---

## START NOW

Begin by gathering context (Phase 1), then consult the Oracle (Phase 2), then execute (Phase 3).

Your goal: **Complete the user's request fully with Oracle-guided excellence.**
