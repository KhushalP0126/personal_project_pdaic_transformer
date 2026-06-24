# Test Log

Date: 2026-06-24

## Commands

```bash
make test
./.venv/bin/python -m unittest discover -s tests
./.venv/bin/python -m py_compile scripts/run_ip_characterization_study.py
```

## Results

- `make test`: failed in offline mode because `setup` re-entered editable install and attempted to download build dependencies from PyPI.
- `./.venv/bin/python -m unittest discover -s tests`: passed, `108` tests, `OK`.
- `./.venv/bin/python -m py_compile scripts/run_ip_characterization_study.py`: passed.

## Note

For this repo, the offline-safe verification path is the direct venv test command rather than `make test` when network access is unavailable.
