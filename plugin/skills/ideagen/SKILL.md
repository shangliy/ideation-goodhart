---
name: ideagen
description: Generate testable conditional hypotheses about an experiment grid by letting the model design and read cheap experiments before it proposes, then check each hypothesis on held-out runs or fresh paired runs. Use when the user has a table of finished runs (CSV of configs and a metric), a training script that maps a config to a metric, or asks "what interactions / hypotheses does my sweep support", "generate hypotheses from my experiments", "从实验结果里找假设", "这些超参数之间有什么交互". Not for literature-level ideation from papers (that is the research-idea skill).
---

# ideagen

`ideagen` is a command-line tool shipped in the same repository as this plugin
(`pip install git+https://github.com/shangliy/ideation-goodhart`). It
implements the arm that worked in the ideation-goodhart study: before proposing
hypotheses, the model designs about 100 single-edit experiments and reads their
noisy results. Hypotheses are conditional interactions: "setting j = b raises
the metric more (or less) when k is at level c than otherwise".

## Pick the mode from what the user has

| The user has | Command |
|---|---|
| nothing yet, wants to see it work | `ideagen demo` (synthetic grid, 3 LLM calls, compares direct vs measure) |
| a CSV of finished runs, one row per run | `ideagen table runs.csv --metric <col> [--group dataset] [--minimize]` |
| a script that reads a config as JSON on stdin and prints the metric | `ideagen run --space space.json --run-cmd "python train.py" [--verify-n 8] [--jobs 4]` |

Useful flags: `--describe "<what the experiments are>"` (the model sees it, so
give real context), `--mode measure|direct|both`, `--probes 100`, `--n 10`,
`--model sonnet`, `--prior-check` (only propose claims that the experiments and the model's own knowledge both support; use it when the experiments are exact but come from few configurations), `--prior haiku,opus` (blind panel: a supported hypothesis the
panel rated below 50 is a surprise worth reading), `--llm-cmd "<any CLI that
reads a prompt on stdin>"` for non-Claude models, `--dry-run` to print the
first prompt without calling anything.

## Rules

- **Say the cost before running.** `table` and `demo` use 1 call for `direct`
  and 2 for `measure`, plus 1 per prior model per arm. `run` also executes
  2 × probes training runs, plus 4 × N × hypotheses with `--verify-n N`. Those
  can be GPU-hours: give the user the count and get a yes first.
- Run `--dry-run` first on a new dataset and show the user the prompt, so they
  can fix parameter names, levels and the description.
- In `table` mode the CSV is split in half: probes are answered from one half
  and verdicts come from the other, so a verdict is a held-out check, not the
  data the model saw. Report precision next to the random base rate printed in
  `hypotheses.md`.
- Treat verdicts as evidence about the grid that was tested. A hypothesis
  supported on a 3-seed grid is a lead for a confirmatory run, not a result.
- Outputs land in `--out` (default `ideagen_out/`): `hypotheses.md`, `run.json`
  with every probe and result, and `cache/` with every prompt and answer.
