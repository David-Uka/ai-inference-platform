# Contributing

## Branching

- `main` is protected and should only receive changes through pull requests.
- Use short-lived branches for changes.
- Recommended prefixes:
  - `feature/` for new functionality
  - `fix/` for bug fixes
  - `chore/` for maintenance
  - `codex/` for Codex-driven repo work

## Local Checks

Run these before opening a pull request:

```bash
python3 -m py_compile api/app.py api/metrics.py api/model.py api/queue.py api/worker.py
ruby -e 'require "yaml"; Dir["k8s/**/*.yaml"].sort.each { |f| YAML.load_stream(File.read(f)); puts "ok #{f}" }'
```

## Pull Requests

- Open pull requests against `main`.
- Keep pull requests focused and small when possible.
- Update documentation when behavior, configuration, or operations change.
- Do not merge code that has not passed CI.
