# Data Snippet

This project has two data paths: the controlled synthetic IP-prefix study and an optional open-dataset BETH sanity check.

## BETH CSV layout

Place the BETH CSV files under:

```text
data/beth/
```

The loader accepts files named like `labelled_*.csv` or any `.csv` file in that directory. It reads process-local event streams, maps frequent `eventId` values into a compact vocabulary, converts event ids into truncated p-adic digits, and builds sliding windows labelled anomalous if any event in the window is marked malicious.

## Minimal Python snippet

```python
from pathlib import Path

from scripts.run_open_dataset import load_beth

windows, labels, stats, families = load_beth(
    data_dir=Path("data/beth"),
    p=3,
    r=8,
    window_size=32,
    stride=4,
)

print(windows.shape)   # [num_windows, window_size, r]
print(labels.shape)    # [num_windows]
print(stats)
print(families[:5])
```

## CPU 3-seed BETH command

```bash
.venv/bin/python scripts/run_open_dataset.py \
  --dataset beth \
  --data-dir ./data/beth \
  --device cpu \
  --p 3 \
  --r 8 \
  --window-size 32 \
  --stride 4 \
  --d-model 128 \
  --n-heads 4 \
  --n-layers 2 \
  --epochs 5 \
  --batch-size 256 \
  --seeds 3 \
  --no-download
```

This writes:

```text
results/open_dataset_report.json
results/open_dataset_report.md
```

Use this as an external-data sanity check. It is not the same as the five-variant synthetic characterization study in `make ip-study-cpu`.
