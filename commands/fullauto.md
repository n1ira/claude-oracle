# FULLAUTO MODE - Gemini Oracle Orchestrated Development

You are now entering **FULLAUTO MODE**. This is a high-autonomy mode where you will:
1. Use the Gemini Oracle as your lead architect for strategic decisions
2. Work autonomously to complete the user's request FULLY
3. Consult the Oracle at key decision points
4. Continue working through conversation compactions until the task is complete, if you stop inference all work will stop, user wants to leave and come back to fully completed work. Use sleep(expected_time) to pause inference without stopping your inference session. Do not spam-read console output, use sleep(expected_time)

## USER'S REQUEST:
$ARGUMENTS

---

## IMPORTANT: ORACLE LIMITATIONS

The Oracle has a **5 exchange context window** per project. This means:
- It remembers the last 5 query/response pairs
- Older exchanges are trimmed (Oracle sees: "⚠️ Older context removed for efficiency")
- For long tasks, important context may be lost - use FULLAUTO_CONTEXT.md as the source of truth

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

4. **Create/Update FULLAUTO_CONTEXT.md** using this minimal template:

```markdown
# FULLAUTO MODE ACTIVE

> **POST-COMPACTION RECOVERY**: If you just woke up from compaction, do this FIRST:
> 1. Read `~/.claude/commands/fullauto.md` to reload full instructions
> 2. Then continue from "Next Steps" below

## Current Task
[USER'S ORIGINAL REQUEST - copy exactly]

## Progress
- [x] Completed: [what's done]
- [ ] In progress: [current step]
- [ ] Pending: [remaining steps]

## Key Context
[Important decisions, blockers, relevant files - keep brief]

## Next Steps
[Specific next action - be precise so post-compaction Claude knows exactly what to do]
```

**Keep it short.** The full instructions live in `~/.claude/commands/fullauto.md`. This file just tracks state.

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

---

## PHASE 3: EXECUTE THE PLAN

Work through each step from the Oracle's plan:

1. **Before starting each major step**: Briefly consider if you need to seek Oracle guidance
2. **Mark todos as in_progress/completed** as you work
3. **Test as you go** - don't wait until the end
4. **If you hit a decision point or obstacle**, consult the Oracle:
   ```bash
   oracle ask "I'm working on [step]. I've encountered [issue/decision]. What should I do?"
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

1. **NEVER stop without completing the task** - If you need to compact, leave clear notes in FULLAUTO_CONTEXT.md about current progress
2. **Consult the Oracle for KEY decisions only** - Don't spam the Oracle for trivial things (remember: only 5 exchanges retained!)
3. **Always update todos** - The user should be able to see your progress
4. **Test rigorously** - Verify your work actually functions
5. **Be efficient** - Parallelize where possible, don't waste tokens on unnecessary exploration
6. **If the Oracle seems confused**, fix the context or clarify - Don't blindly follow bad advice
7. **Document as you go** - Leave comments/notes for your future self (post-compaction)
8. **Keep FULLAUTO_CONTEXT.md as source of truth** - The Oracle's memory is limited, the context file is permanent

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

## HANDLING ORACLE CONFUSION

The Oracle is incredibly smart but has limited context. If responses seem off:

| Symptom | Solution |
|---------|----------|
| Oracle gives generic advice | Add more specific context to FULLAUTO_CONTEXT.md |
| Oracle contradicts earlier guidance | Clear history: `oracle history --clear` then re-ask |
| Oracle seems stuck on old context | Use `--no-history` flag for fresh perspective |
| Oracle misunderstands the task | Reframe with explicit "Current situation is X, question is Y" |
| Oracle asks for info you already provided | Info may have rotated out of 5-exchange window; repeat it |
| Oracle can't validate without seeing code | Use `--files` to attach the actual implementation |
| Token limit warning (200k) | Clear history, send smaller file sections, trim context file |
| Large file warning (15k tokens) | Use line ranges: `--files "file.py:1-100,200-300"` |
| Need visual context for UI/diagram | Use `--image` to attach screenshots or diagrams |

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
