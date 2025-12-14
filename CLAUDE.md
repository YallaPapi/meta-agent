# Claude Code Instructions

## IMPORTANT: Always Use Taskmaster
**You MUST use Taskmaster for ALL task management in this project. This is MANDATORY.**

### Task Management (Always Use)
- `task-master list` - See all current tasks with status
- `task-master next` - Get the next available task to work on
- `task-master show <id>` - View detailed task information
- `task-master set-status --id=<id> --status=<status>` - Update task status (pending, in-progress, done, deferred, cancelled, blocked)

### Research & Analysis (Use for ALL Research)
- `task-master add-task --prompt="description" --research` - Add new task WITH research
- `task-master expand --id=<id> --research --force` - Break task into subtasks WITH research
- `task-master update-task --id=<id> --prompt="changes" --research` - Update task WITH research
- `task-master update --from=<id> --prompt="changes" --research` - Update multiple tasks WITH research
- `task-master analyze-complexity --research` - Analyze task complexity with AI
- **ALWAYS use `--research` flag when you need to gather information or make informed decisions**

### Progress Tracking
- `task-master update-subtask --id=<id> --prompt="notes"` - Log implementation progress and notes
- `task-master complexity-report` - View complexity analysis report

### Task Organization
- `task-master add-dependency --id=<id> --depends-on=<id>` - Add task dependencies
- `task-master move --from=<id> --to=<id>` - Reorganize task hierarchy
- `task-master validate-dependencies` - Check for dependency issues
- `task-master expand --all --research` - Expand all eligible tasks with research

### Configuration
- `task-master models --setup` - Configure AI models interactively
- `task-master models` - View current model configuration

### Key Rules
1. **NEVER manage tasks manually** - always use Taskmaster commands or MCP tools
2. **ALWAYS use `--research` flag** when doing research, analysis, or making decisions
3. **Mark tasks in-progress** before starting work
4. **Mark tasks complete immediately** after finishing
5. **Use `update-subtask`** to log implementation notes during development

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
