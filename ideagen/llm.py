"""LLM calls through a command-line client. The prompt goes to stdin, the answer comes back on stdout.

The default uses the Claude Code CLI with tools disabled. Any other client works through --llm-cmd, for example
  --llm-cmd "codex exec --model gpt-5 -"   or   --llm-cmd "llm -m gemini-2.5-pro"
`{model}` in the template is replaced by --model.
"""
import hashlib, json, os, re, shlex, subprocess, sys, tempfile

DEFAULT_CMD = "claude -p --model {model} --tools '' --strict-mcp-config"


class LLM:
    def __init__(self, model='sonnet', cmd=None, cache_dir='.ideagen_cache', timeout=900, dry_run=False, log=sys.stderr):
        self.model, self.timeout, self.dry_run, self.log = model, timeout, dry_run, log
        self.cmd = (cmd or os.environ.get('IDEAGEN_LLM_CMD') or DEFAULT_CMD).replace('{model}', model)
        self.cache_dir, self.calls, self.shown = cache_dir, 0, False
        os.makedirs(cache_dir, exist_ok=True)

    def ask(self, prompt, tag=''):
        key = hashlib.sha256((self.cmd + '\0' + prompt).encode()).hexdigest()[:24]
        f = os.path.join(self.cache_dir, key + '.json')
        if os.path.exists(f):
            return json.load(open(f))['answer']
        if not self.shown or self.dry_run:
            print(f'\n----- prompt ({tag}) -----\n{prompt}\n----- end of prompt -----\n', file=self.log)
            self.shown = True
        if self.dry_run:
            raise DryRun(tag)
        print(f'[ideagen] calling: {self.cmd}  ({tag})', file=self.log)
        with tempfile.TemporaryDirectory() as d:  # empty cwd so the client cannot read the project
            r = subprocess.run(shlex.split(self.cmd), input=prompt, capture_output=True, text=True, timeout=self.timeout, cwd=d)
        if r.returncode != 0 or not r.stdout.strip():
            raise RuntimeError(f'LLM command failed ({r.returncode}): {r.stderr[-500:] or r.stdout[-500:]}')
        self.calls += 1
        json.dump({'cmd': self.cmd, 'tag': tag, 'prompt': prompt, 'answer': r.stdout}, open(f, 'w'))
        return r.stdout


class DryRun(Exception):
    pass


def last_json(text, key):
    """The value of `key` in the last JSON object in `text` that starts with that key."""
    for m in reversed(list(re.finditer(r'\{\s*"' + re.escape(key) + r'"\s*:', text))):
        s, depth, instr, esc = text[m.start():], 0, False, False
        for i, ch in enumerate(s):
            if instr:
                esc, instr = (not esc and ch == '\\'), instr and not (ch == '"' and not esc)
                continue
            if ch == '"':
                instr = True
            depth += (ch == '{') - (ch == '}')
            if depth == 0:
                try:
                    return json.loads(s[:i + 1])[key]
                except (json.JSONDecodeError, KeyError):
                    break
    return None
