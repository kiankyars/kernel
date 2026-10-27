# Standing user preferences

- After completing implementation or documentation work, create focused, atomic
  Git commits and push them to the configured remote. The user explicitly wants
  this done as part of the work, without another routine confirmation.
- Each commit should contain one coherent change and the tests or supporting
  files it needs. Keep every commit usable; do not mix unrelated changes.
- Validate the affected behavior before pushing. Preserve unrelated user changes,
  exclude credentials and generated caches, and never force-push or rewrite
  published history unless explicitly requested.
- Verify the remote contains the pushed commits and report the branch and commit
  IDs. If pushing is blocked, report the concrete blocker rather than presenting
  unpushed work as complete.

These preferences were explicitly requested by the user on 2026-10-27 and apply
to future tasks in this repository.
