# Notes for Claude

- **Show important graphs in chat.** When a result figure matters (an accuracy curve, a seed comparison, a training curve that decides something), send it with SendUserFile (`display: render`), with a one-line caption and the key numbers in the reply. The user often follows remotely, so a path on the pod is not enough. Check the figure for clipped labels or overlaps first. Skip routine redraws.
- **Discuss new directions, then run without asking.** Discuss a genuinely new direction or a rule exception before launching it. Once a direction is agreed, run its follow-ups (diagnostics, replicates, the next variant, queueing work so both GPUs stay busy) without asking again, and report what started.
- **Plain English.** Explain results, recipes and plans in everyday words. Keep config flags, run names and coined terms out of user-facing explanations (they belong in logistics.md and job files); if a term is unavoidable, say what it means.
- **Times in Pacific time only.** The pod clock and logs are UTC; convert before quoting a time to the user (`TZ=America/Los_Angeles date`).
