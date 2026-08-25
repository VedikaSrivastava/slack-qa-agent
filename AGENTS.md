# Coding Agent Instructions

- Work on a feature branch. Do not commit implementation work directly to `main`.
- Keep agent reasoning and retrieval logic independent from Slack and background-job adapters.
- Do not create, modify, rewrite, or suggest prose for `DESIGN.md`; the candidate must write it personally.
- Never commit secrets, Slack tokens, OpenAI keys, Inngest keys, or the shared 1Password link.
- Prefer bounded, testable workflows over open-ended agent loops.
- Add or update tests with every behavior change.
