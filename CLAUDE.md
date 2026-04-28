# CLAUDE.md

## Project Overview

**Rs-stat** (slug: `rs-stat`) — Rs-stat — Rust statistical distributions library (fitting, CDF, PDF, discrete/continuous).

## Project Orchestrator — Mandatory Agent Workflow

**This project is tracked by the Project Orchestrator (MCP).** All work MUST follow this protocol.

### Before ANY code change

1. **Check existing plans**: `plan(action: "list")` — is there already a plan for this work?
2. **Load context**: `note(action: "search_semantic", query)` + `decision(action: "search_semantic", query)` — check past knowledge
3. **Create or resume a plan**: `plan(action: "create")` with tasks and steps if none exists
4. **Mark task in_progress**: `task(action: "update", task_id, status: "in_progress")`

### During work

5. **Update steps in real-time**: `step(action: "update", step_id, status: "in_progress")` then `"completed"`
6. **Record decisions**: `decision(action: "add")` for any non-trivial architectural choice
7. **Capture knowledge**: `note(action: "create")` for every gotcha, pattern, or convention discovered

### After each commit

8. **Register the commit**: `commit(action: "create", sha, message, author, files_changed, project_id)`
9. **Link to task**: `commit(action: "link_to_task", task_id, commit_sha)`
10. **Link to plan**: `commit(action: "link_to_plan", plan_id, commit_sha)`

### On task completion

11. **Verify acceptance criteria** before marking completed
12. **Update task**: `task(action: "update", task_id, status: "completed")`
13. **Check plan progress**: if all tasks done, mark plan completed

### NEVER

- Write code without an active plan and task
- Commit without linking to a task
- Mark a task completed without verifying acceptance criteria
- Skip step status updates
- Forget to capture knowledge (notes) from debugging sessions
