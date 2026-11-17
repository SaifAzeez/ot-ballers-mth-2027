"""Turn raw nearest-player labels into controlled possessions.

    python possession_filter.py <raw.csv> <out.csv>

Input: a frame,label CSV of per-frame raw labels (H / A = team of the player nearest the ball,
D = ball not located), as written by `python starter/baseline.py --window 0`.
Output: a frame,label CSV in the submission format.

Rules (parameters fixed from the written definition of possession, not fitted to any labels):

  1. Dead-ball blips. A run of D shorter than BLIP_FRAMES (1 s) with ball evidence on both sides
     is a tracking dropout, not a stoppage. It is bridged: it carries no evidence, and the frames
     take the label of whoever has possession at that point. A run of D shorter than MIN_DEAD
     (3 s) is bridged the same way when the same team is nearest the ball on both sides: real
     restarts take 5 s or more, while tracker dropouts in open play are short. Every other run
     of D lasting 1 s or more is a stoppage and stays D.
  2. Restarts. Each live stretch between stoppages starts with a clear possessor: the team
     nearest the ball when it reappears. There is no carry-over from before the stoppage.
  3. Controlled possession. Inside a live stretch the possessor keeps the ball until the other
     team is nearest for CONTROL_FRAMES (2 s) without interruption (bridged blips do not
     interrupt). Possession then changes, dated back to the start of that 2 s run, so a change
     is not reported late. Shorter runs by the other team -- challenges, deflections, a player
     running past the ball -- do not change possession.

Optional, with --use-tags (applied to the raw labels before rules 1-3):

  0. Untagged dropouts. A run of D with the same team nearest on both sides, shorter than
     TAG_MAX_GAP (30 s), and with no Smart Tagging SET PIECES event overlapping it or within
     TAG_MARGIN (2 s) of it, is filled with that team. Open play rarely loses the ball for 30 s,
     and long stoppages without a set-piece tag are injuries, substitutions or goals. The tags
     are used only as evidence that the ball was dead -- never for which team restarts, which
     they get wrong too often to trust.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

FPS = 25
CONTROL_FRAMES = 2 * FPS  # 2 s of uninterrupted control to win the ball
BLIP_FRAMES = 1 * FPS     # dead-ball runs shorter than 1 s are dropouts, not stoppages
MIN_DEAD = 3 * FPS        # ... and so are runs under 3 s with the same team on both sides
TAG_MAX_GAP = 30 * FPS    # --use-tags: untagged same-team gaps shorter than this are filled
TAG_MARGIN = 2 * FPS      # --use-tags: a set piece this close to a gap marks it as a stoppage

HERE = os.path.dirname(os.path.abspath(__file__))


def runs(values):
    """(start, end_exclusive, value) for each maximal run of equal values."""
    n = len(values)
    if n == 0:
        return []
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    return [(int(s), int(e), values[s]) for s, e in zip(starts, ends)]


def filter_live(raw):
    """Possession labels for one live stretch: no long D inside, starts and ends with evidence.

    `raw` holds 'H', 'A' and short-blip 'D'. Returns an array of 'H'/'A' of the same length.
    """
    n = len(raw)
    out = np.empty(n, dtype="<U1")

    # Team runs with blips bridged: drop D frames, find runs of the remaining team labels, and
    # map each run back to the frame span it covers (bridged D inside a run belongs to it).
    seen = np.flatnonzero(raw != "D")
    teams = raw[seen]
    team_runs = [(int(seen[s]), int(seen[e - 1]) + 1, t) for s, e, t in runs(teams)]

    possessor = team_runs[0][2]  # rule 2: whoever is nearest when the ball comes back
    cursor = 0
    for start, end, team in team_runs:
        if team != possessor and end - start >= CONTROL_FRAMES:
            # rule 3: the challenger had 2 s of control; possession changes at the run start
            out[cursor:start] = possessor
            possessor, cursor = team, start
    out[cursor:] = possessor
    return out


def fill_untagged_gaps(raw, set_pieces):
    """Rule 0. `set_pieces` is a list of (start_frame, end_frame_exclusive) video-frame intervals.

    Returns a copy of `raw` with each qualifying D gap filled by the team on both sides.
    """
    raw = np.array(raw, dtype="<U1")
    out = raw.copy()
    for start, end, label in runs(raw):
        if label != "D" or start == 0 or end == len(raw):
            continue
        team = raw[start - 1]
        if raw[end] != team or end - start >= TAG_MAX_GAP:
            continue
        tagged = any(sp_end > start - TAG_MARGIN and sp_start < end + TAG_MARGIN
                     for sp_start, sp_end in set_pieces)
        if not tagged:
            out[start:end] = team
    return out


def load_set_pieces():
    """SET PIECES intervals from the Smart Tagging, via the Metrica starter loader."""
    sys.path.insert(0, os.path.join(HERE, "starter"))
    import load_smart_tagging
    return [(e["start_frame"], e["end_frame"]) for e in load_smart_tagging.load()
            if (e["code"] or "").strip().lower() == "set pieces"]


def possession_filter(raw, set_pieces=None):
    """Apply the rules to a full-match array of raw 'H'/'A'/'D' labels.

    With `set_pieces` (a list of intervals, see fill_untagged_gaps), rule 0 runs first.
    """
    raw = np.asarray(raw, dtype="<U1")
    if set_pieces is not None:
        raw = fill_untagged_gaps(raw, set_pieces)
    out = np.full(len(raw), "D", dtype="<U1")

    # Stoppages: D runs of at least 1 s, unless shorter than MIN_DEAD with the same team nearest
    # on both sides. A D run at either end of the file has no evidence on one side, so it cannot
    # be bridged -- treat it as a stoppage too.
    stoppage = np.zeros(len(raw), dtype=bool)
    for start, end, label in runs(raw):
        if label != "D":
            continue
        length = end - start
        if start == 0 or end == len(raw):
            is_stoppage = True
        elif length < BLIP_FRAMES:
            is_stoppage = False
        elif length < MIN_DEAD and raw[start - 1] == raw[end]:
            is_stoppage = False
        else:
            is_stoppage = True
        stoppage[start:end] = is_stoppage

    # Live stretches are the maximal runs of non-stoppage frames.
    for start, end, is_stoppage in runs(stoppage):
        if not is_stoppage:
            out[start:end] = filter_live(raw[start:end])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw", help="raw labels from starter/baseline.py --window 0")
    parser.add_argument("out", help="submission CSV to write")
    parser.add_argument("--use-tags", action="store_true",
                        help="fill untagged same-team D gaps under 30 s (needs the Smart Tagging)")
    args = parser.parse_args()

    raw = pd.read_csv(args.raw)
    labels = raw["label"].astype(str).str.strip().str.upper()
    if (labels == "X").any():
        sys.exit("input has X labels; expected raw H/A/D from baseline.py --window 0")
    if not np.array_equal(raw["frame"].to_numpy(), np.arange(len(raw))):
        sys.exit("input frames are not 0..N-1 in order")

    set_pieces = load_set_pieces() if args.use_tags else None
    out = possession_filter(labels.to_numpy(), set_pieces)
    pd.DataFrame({"frame": np.arange(len(out)), "label": out}).to_csv(args.out, index=False)

    n_poss = sum(1 for _, _, v in runs(out) if v in ("H", "A"))
    counts = {k: int((out == k).sum()) for k in "HAD"}
    print("{} -> {}: {} frames, {} possessions, H {} A {} D {}{}".format(
        args.raw, args.out, len(out), n_poss, counts["H"], counts["A"], counts["D"],
        "  ({} set-piece tags used)".format(len(set_pieces)) if args.use_tags else ""))


if __name__ == "__main__":
    main()
