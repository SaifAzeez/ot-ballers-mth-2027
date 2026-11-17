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

Optional, with --stoppages (applied after rule 0, before rules 1-3; forced frames become D and
any D run containing them is a stoppage, whatever its length):

  S1. Ball out of play. Tracked ball more than OUT_MARGIN (0.5 m) beyond a line -> D, until it
      is tracked back inside and moving faster than RESTART_SPEED (2 m/s).
  S2. Players stopped. Mean frame-to-frame player speed (moves over 12 m/s ignored as tracking
      jumps) below walking pace, STOP_SPEED (1.4 m/s), for at least 3 s -> D. Skipped when the
      ball is tracked in a penalty area with that end's goalkeeper nearest (a keeper holding the
      ball is possession). With no position data, the goalkeeper is the team's deepest player.
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

# --stoppages (design choices from the pitch lines and walking pace, not fitted to labels)
OUT_MARGIN = 0.5          # m beyond the lines before the ball counts as out of play
RESTART_SPEED = 2.0       # m/s: ball back inside and moving at least this fast = play restarted
STOP_SPEED = 1.4          # m/s: walking pace; mean player speed below this = players stopped
STOP_FRAMES = 3 * FPS     # ... for at least 3 s
MAX_PLAYER_SPEED = 12.0   # m/s: faster frame-to-frame moves are tracking jumps, ignored
BOX_DEPTH, BOX_WIDTH = 16.5, 40.3  # penalty area, m

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


def ball_out_of_play(ball, length, width):
    """Rule S1. Bool mask: tracked ball beyond the lines (+ OUT_MARGIN) -> dead, and it stays
    dead until the ball is tracked back inside and moving faster than RESTART_SPEED.

    `ball` is (N, 2) in metres, NaN where not tracked. Speed is frame to frame, only between two
    consecutive tracked frames.
    """
    n = len(ball)
    x, y = ball[:, 0], ball[:, 1]
    tracked = ~np.isnan(x)
    outside = tracked & ((x < -OUT_MARGIN) | (x > length + OUT_MARGIN) |
                         (y < -OUT_MARGIN) | (y > width + OUT_MARGIN))
    speed = np.full(n, np.nan)
    speed[1:] = np.hypot(np.diff(x), np.diff(y)) * FPS  # NaN unless both frames tracked
    restarted = tracked & ~outside & (speed > RESTART_SPEED)

    dead = np.zeros(n, dtype=bool)
    out = False
    for f in range(n):
        if outside[f]:
            out = True
        elif out and restarted[f]:
            out = False
        dead[f] = out
    return dead


def mean_player_speed(players):
    """(N,) mean frame-to-frame speed of tracked players, ignoring jumps over MAX_PLAYER_SPEED.
    NaN where no player has a valid speed."""
    step = np.full(players.shape[:2], np.nan)
    step[1:] = np.hypot(*np.moveaxis(np.diff(players, axis=0), -1, 0)) * FPS
    step[step > MAX_PLAYER_SPEED] = np.nan
    valid = ~np.isnan(step)
    count = valid.sum(axis=1)
    total = np.where(valid, step, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def left_defenders(players, teams, blocks):
    """(N,) team (0/1) defending the x = 0 end at each frame, -1 outside the halves.

    Per half, the team whose players' mean x is lower defends the x = 0 end.
    """
    out = np.full(len(players), -1, dtype=np.int8)
    for first, last in blocks:
        xs = players[first:last + 1, :, 0]
        mean0 = np.nanmean(xs[:, teams == 0])
        mean1 = np.nanmean(xs[:, teams == 1])
        out[first:last + 1] = 0 if mean0 < mean1 else 1
    return out


def keeper_on_ball_in_box(players, ball, teams, length, width, frame, left_team):
    """Ball tracked inside a penalty area and the nearest player is that end's goalkeeper.

    There is no position data (track slots are tracklets), so the goalkeeper is taken to be the
    deepest tracked player of the team defending that end (`left_team`, see left_defenders).
    """
    bx, by = ball[frame]
    if np.isnan(bx):
        return False
    in_y = abs(by - width / 2) <= BOX_WIDTH / 2
    left, right = bx <= BOX_DEPTH, bx >= length - BOX_DEPTH
    if not (in_y and (left or right)):
        return False
    pos = players[frame]
    seen = ~np.isnan(pos[:, 0])
    if not seen.any():
        return False
    dist = np.where(seen, np.hypot(pos[:, 0] - bx, pos[:, 1] - by), np.inf)
    k = int(np.argmin(dist))
    defending = left_team[frame] if left else 1 - left_team[frame]
    if left_team[frame] < 0 or teams[k] != defending:
        return False
    mates = seen & (teams == teams[k])
    deepest = pos[mates, 0].min() if left else pos[mates, 0].max()
    return pos[k, 0] == deepest


def players_stopped(players, ball, teams, length, width, left_team):
    """Rule S2. Bool mask: mean player speed below STOP_SPEED for at least STOP_FRAMES -> dead.
    A run is skipped if on any of its frames the goalkeeper has the ball in his penalty area
    (a keeper holding the ball is possession, not a stoppage)."""
    slow = mean_player_speed(players) < STOP_SPEED  # NaN compares False
    dead = np.zeros(len(slow), dtype=bool)
    for start, end, is_slow in runs(slow):
        if not is_slow or end - start < STOP_FRAMES:
            continue
        if any(keeper_on_ball_in_box(players, ball, teams, length, width, f, left_team)
               for f in range(start, end)):
            continue
        dead[start:end] = True
    return dead


def load_tracking():
    """Players (N, slots, 2) and ball (N, 2) in metres, team per slot, from the raw ATD.

    Same format and conventions as starter/baseline.py: frame_id = video frame + 1, normalised
    coordinates scaled by the pitch size.
    """
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, "starter"))
    from games import GAME, GAMES, require, source_frames, half_video_blocks
    import baseline

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    teams = baseline.team_of_each_track(meta_path)
    n, n_slots = source_frames("tactical", GAME), len(teams)
    length, width = GAMES[GAME]["pitch"]
    scale = np.array([length, width])
    players = np.full((n, n_slots, 2), np.nan)
    ball = np.full((n, 2), np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(baseline.FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n:
                continue
            vals = np.array(parts[1].replace(baseline.GROUP_SEP, ",").split(","),
                            dtype=float).reshape(-1, 3)[:n_slots]
            players[frame, :len(vals)] = vals[:, :2] * scale
            if parts[2][0] != baseline.MISSING:
                ball[frame] = np.array(parts[2].split(",")[:2], dtype=float) * scale
    blocks = [(first, last) for _, first, last in half_video_blocks(GAME)]
    return players, ball, teams, length, width, left_defenders(players, teams, blocks)


def stoppage_masks():
    """(ball_out, players_stopped) masks for the match."""
    players, ball, teams, length, width, left_team = load_tracking()
    return (ball_out_of_play(ball, length, width),
            players_stopped(players, ball, teams, length, width, left_team))


def possession_filter(raw, set_pieces=None, forced_dead=None):
    """Apply the rules to a full-match array of raw 'H'/'A'/'D' labels.

    With `set_pieces` (a list of intervals, see fill_untagged_gaps), rule 0 runs first.
    With `forced_dead` (a bool mask from the --stoppages rules), those frames become D after
    rule 0 and before rules 1-3, and any D run containing them is a stoppage whatever its length.
    """
    raw = np.asarray(raw, dtype="<U1")
    if set_pieces is not None:
        raw = fill_untagged_gaps(raw, set_pieces)
    if forced_dead is not None:
        raw = np.where(forced_dead, "D", raw)
    out = np.full(len(raw), "D", dtype="<U1")

    # Stoppages: D runs of at least 1 s, unless shorter than MIN_DEAD with the same team nearest
    # on both sides. A D run at either end of the file has no evidence on one side, so it cannot
    # be bridged -- treat it as a stoppage too.
    stoppage = np.zeros(len(raw), dtype=bool)
    for start, end, label in runs(raw):
        if label != "D":
            continue
        length = end - start
        if forced_dead is not None and forced_dead[start:end].any():
            is_stoppage = True
        elif start == 0 or end == len(raw):
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
    parser.add_argument("--stoppages", action="store_true",
                        help="ball out of play and players-stopped rules (needs the ATD)")
    parser.add_argument("--ball-out", action="store_true",
                        help="ball out of play rule only (S1; needs the ATD)")
    args = parser.parse_args()

    raw = pd.read_csv(args.raw)
    labels = raw["label"].astype(str).str.strip().str.upper()
    if (labels == "X").any():
        sys.exit("input has X labels; expected raw H/A/D from baseline.py --window 0")
    if not np.array_equal(raw["frame"].to_numpy(), np.arange(len(raw))):
        sys.exit("input frames are not 0..N-1 in order")

    set_pieces = load_set_pieces() if args.use_tags else None
    forced_dead = None
    if args.stoppages:
        out_mask, stop_mask = stoppage_masks()
        forced_dead = out_mask | stop_mask
        print("stoppages: ball out of play on {} frames, players stopped on {} frames".format(
            int(out_mask.sum()), int(stop_mask.sum())))
    elif args.ball_out:
        _, ball, _, length, width, _ = load_tracking()
        forced_dead = ball_out_of_play(ball, length, width)
        print("ball out of play on {} frames".format(int(forced_dead.sum())))
    out = possession_filter(labels.to_numpy(), set_pieces, forced_dead)
    pd.DataFrame({"frame": np.arange(len(out)), "label": out}).to_csv(args.out, index=False)

    n_poss = sum(1 for _, _, v in runs(out) if v in ("H", "A"))
    counts = {k: int((out == k).sum()) for k in "HAD"}
    print("{} -> {}: {} frames, {} possessions, H {} A {} D {}{}".format(
        args.raw, args.out, len(out), n_poss, counts["H"], counts["A"], counts["D"],
        "  ({} set-piece tags used)".format(len(set_pieces)) if args.use_tags else ""))


if __name__ == "__main__":
    main()
