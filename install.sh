#!/usr/bin/env bash
# One-step install: the ideagen CLI, plus the research-idea and ideagen skills for Claude Code.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 -m pip install "$HERE"
if command -v claude >/dev/null 2>&1; then
  claude plugin marketplace add shangliy/ideation-goodhart
  claude plugin install research-ideation@ideation-goodhart
else
  echo "Claude Code not found: installed the ideagen CLI only. Use --llm-cmd to point it at another LLM CLI."
fi
echo "Try: ideagen demo --dry-run   (prints the prompt, no LLM calls), then: ideagen demo"
