# Old Trafford Ballers — technical document

More Than A Hack 2027, ball possession · Code: <https://github.com/SaifAzeez/ot-ballers-mth-2027>

## Approach

We wanted a small system we could explain completely, with no learned parameters. The ranking
metric rewards recovering whole possessions, so we put the effort into the structure of
possession rather than per-frame accuracy. The pipeline has two steps:

1. **Raw labels.** For every frame where the tracking sees the ball, the team of the nearest
   tracked player has it, and `D` where the ball is not tracked. This is the Metrica starter's
   `baseline.py --window 0`, unchanged.
2. **Possession rules** (`possession_filter.py --use-tags`), described below.

We used Claude (an LLM) as a coding and reasoning assistant. It wrote and reviewed code and
helped design the rules. It produced no labels and was given no data about this match beyond the
provided assets.

The output has 319 possessions during play (mean 12.8 s, median 7.7 s), and 5.6% of them are
shorter than 2 s. The smoothed starter baseline has 584 possessions (median 4.6 s, 31.5% under
2 s). The flicker it produces is what our rules are designed to remove.

## Possession rules

- **2 s controlled possession.** A team keeps the ball until the other team has been nearest to it
  for 2 s without a break. The change is then dated back to the start of those 2 s, so it is not
  reported late. Challenges, deflections and players running past the ball are shorter than 2 s
  and do not change possession.
- **Dead-ball blips.** A missing-ball run shorter than 1 s, with the ball seen on both sides, is a
  tracker dropout. It is filled with the current possessor. (A second rule fills 1–3 s gaps
  between the same team, but it changes nothing on this match, because no such gaps exist.)
- **Restarts.** After a stoppage, nothing carries over from before it. The first team near the
  ball when it reappears is the possessor.
- **Tag-based gap fill (v2, `--use-tags`).** A missing-ball run with the same team on both sides,
  shorter than 30 s and with **no** Smart Tagging set piece overlapping it or within 2 s, is
  treated as live play and filled with that team. Open play rarely loses the ball for 30 s, and
  long untagged stoppages tend to be injuries, substitutions or goals. The tags are used **only
  as evidence that the ball was dead, never to decide which team has it.** This filled 68 gaps
  (12,103 frames).

## What we learned about the data

- **The tracker's ball gaps fall into two groups.** Of the 184 gaps inside the match, 43 last
  under 1 s, exactly one lasts 1–5 s (26 frames), and the other 140 last 5 s or more. Length
  alone cannot separate stoppages from long dropouts in open play, which is why we turned to the
  tags.
- **Smart Tagging set pieces are good at locating stoppages, but not at saying who restarts.**
  51% of gaps of 5 s or more have a set piece within 2 s. However, the restart type and team are
  unreliable: we saw goal kicks tagged as corners for the other team.
- **Error review.** We built a video review tool (`review_player.py`) and sorted 31 errors from
  about 12 minutes of the first half, plus spot checks through the second half. This was **for
  understanding only**: no frame labels were written, and none were used for tuning or
  validation. Missed stoppages accounted for 12, live play called dead for 8, the wrong team in
  mid-possession for 7, the keeper holding the ball for 2, and one each was a late or an early
  handover. Where v1 and v2 disagreed, each was right 3 times.

## Assets

- **Used:**
  - ATD tracking (ball and player positions), with the ATD metadata mapping each track to its
    team.
  - Smart Tagging `SET PIECES` intervals, as stoppage evidence only.
  - The tactical video, only for looking at errors and never as an input to the pipeline.
- **Dropped:**
  - The Smart Tagging team and restart-type fields, which were unreliable as described above.
  - The Smart Tagging phase tags (Build Up, Progression and so on), which describe phases of play,
    not possessions.
  - Homography and the broadcast video, which we did not need for this approach.

## Training data

**None.** There is no labelled data, from this game or any other. Every parameter was chosen from
the written definition of possession and from football reasoning: 2 s of control, a 1 s blip,
30 s and 2 s for tags, 25 fps. There was no validation set. We ran the public 5-minute ground
truth once, on v1, to check the file scored without errors, and changed no parameter as a result.

## Limitations

- **v2 can fill real stoppages.** When an untagged free kick or VAR check has the same team on
  both sides, v2 labels it as possession.
- **The ball goes untracked in the goalkeeper's hands**, so keeper possession often becomes `D`.
- **Pressing opponents create false possession changes.** An opponent closing down can be nearest
  to the ball for 2 s, which produces a run for the wrong team.
- **Possibly a camera-distance effect.** Tracking near the far touchline may be weaker, but we
  have not confirmed this.
- **There is no dedicated dead-ball detection.** Stoppages are inferred only from missing ball
  tracking, which covers 43% of frames in the raw labels.

## Next steps

- Detect stoppages from player speed (everyone walking) and from the ball leaving the pitch. A
  first try at speed-based detection used a fixed walking-pace cut-off (mean speed under
  1.4 m/s for 3 s) and flagged about a third of play, so it needs a threshold relative to the
  match's normal speed instead.
- Handle the goalkeeper specifically.
- Plot where on the pitch gaps start, to test the camera-distance idea.

## Reproduction

```bash
pip install -r requirements.txt
python starter/baseline.py --window 0
python possession_filter.py evaluation/results/baseline_raw.csv Old_Trafford_Ballers.csv --use-tags
python evaluation/validate_submission.py Old_Trafford_Ballers.csv   # SHA-1 f24e04c7…
```

The data goes in `data/` (see DATA.md). The output is deterministic. Without `--use-tags` the
pipeline gives v1.
