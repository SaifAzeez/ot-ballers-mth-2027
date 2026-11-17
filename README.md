# Old Trafford Ballers — More Than A Hack 2027, ball possession

Our entry to the More Than A Hack 2027 possession challenge (FC Barcelona v Athletic Club).
For every video frame it says who has the ball: `H`, `A` or `D` (dead ball).

The method is rule-based, and nothing in it is trained. It takes the Metrica starter's
nearest-player labels and turns them into whole possessions with a 2-second controlled-possession
rule. Short tracker dropouts are filled, and Smart Tagging set pieces count only as evidence that
the ball was dead. The technical document is [Old_Trafford_Ballers.md](Old_Trafford_Ballers.md).

This repository started from the **Metrica Sports starter kit** for the challenge, which provides
`games.py`, `starter/` (including `baseline.py`, the nearest-player baseline we build on) and
`evaluation/` (the official scorer and validator). Credit for all of that goes to Metrica Sports
and the challenge organisers. The starter's original README is kept as
[README_STARTER.md](README_STARTER.md).

## What we added

| file | what it does |
|---|---|
| [possession_filter.py](possession_filter.py) | raw nearest-player labels → submission. Applies the 2 s control rule, fills dead-ball blips under 1 s, gives restarts a clear possessor, and with `--use-tags` fills same-team gaps under 30 s that have no set-piece tag |
| [test_possession_filter.py](test_possession_filter.py) | tests for every rule, on hand-made label strings (no match data) |
| [gap_viewer.py](gap_viewer.py) | lists long missing-ball gaps alongside the Smart Tagging and renders sample frames (for understanding only) |
| [review_player.py](review_player.py) | video player showing labels, tracking, tags, a pitch inset and a seek bar (for understanding only) |

The review tools never write labels, and nothing they show was used to set a parameter.

## Reproduce the submission

You need Python 3.12 (3.8+ should also work for the pipeline) and the challenge data bundle,
unpacked into `data/` as described in [DATA.md](DATA.md), or pointed to with `MTAH_DATA_DIR`.

```bash
pip install -r requirements.txt

# 1. Raw labels: team of the player nearest the ball, D where the ball is not tracked.
#    Writes evaluation/results/baseline_raw.csv (identical to the committed copy).
python starter/baseline.py --window 0

# 2. Possession rules + Smart Tagging stoppage evidence -> the submission file.
python possession_filter.py evaluation/results/baseline_raw.csv Old_Trafford_Ballers.csv --use-tags

# 3. Check the format.
python evaluation/validate_submission.py Old_Trafford_Ballers.csv
```

The output is deterministic. Its SHA-1 should be `f24e04c771ec9fc2c13fd392d2ba7b4d22f36a96`:

```bash
shasum Old_Trafford_Ballers.csv
```

Leaving out `--use-tags` gives our v1 (SHA-1 `4843c922fd0ab2f894e9f207e9a4441f0208bb5a`).

Tests (these need no data):

```bash
python test_possession_filter.py
```

Review tools (these need the video and `opencv-python`, `matplotlib`):

```bash
python gap_viewer.py list              # writes gap_report.csv
python gap_viewer.py sample --n 10     # PNGs in gap_views/
python review_player.py                # interactive player; keys are listed in its docstring
```

## What is not in this repository

There is no match data of any kind: no tracking, video, tags, ground truth or submission CSVs.
`.gitignore` excludes `data/`, every top-level `*.csv`, `evaluation/results/` (apart from the
starter's own baselines), `gap_views/`, `gap_report.csv` and `review_cache/`.
