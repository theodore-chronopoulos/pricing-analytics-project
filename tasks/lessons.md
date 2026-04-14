# Lessons

Capture patterns after user corrections or repeated mistakes.

## Entries

- Date: 2026-04-08
  Pattern: Established default workflow for this project.
  Rule: For any non-trivial task, write a checkable plan in `tasks/todo.md`, verify before completion, and record lessons here after corrections.

- Date: 2026-04-14
  Pattern: Temporary solver artifacts should not clutter the project root.
  Rule: When a tool needs intermediate files or temp directories, write them under `results/` or `/tmp`, not alongside the main project files, unless the user explicitly asks otherwise.

- Date: 2026-04-14
  Pattern: Repository docs should not contain local absolute-path links or references to unrelated local helper files.
  Rule: When preparing code for GitHub, use relative README links only, and exclude unrelated local example files from the tracked repo unless the user explicitly asks to publish them.
