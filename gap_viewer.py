"""Gap viewer: look at the long missing-ball gaps and what the smart tagging says about them.

For UNDERSTANDING only. Do not hand-label frames from this and do not pick thresholds from what you
see (RULES.md bans hand labels on this match). Any rule built from it should use what the tags ARE
(set piece vs not), not numbers fitted to these moments.

Run from the repo root (needs the data in data/ and opencv-python + matplotlib):

  python gap_viewer.py list                  # every D gap >= 5 s, with tags; writes gap_report.csv
  python gap_viewer.py sample --n 10         # 10 random gaps (seed 0) -> PNGs in gap_views/
  python gap_viewer.py show 54321            # one frame -> PNG in gap_views/

Each PNG shows: the video frame, a pitch plot of tracked players (blue = home, red = away) and the
ball, the label from baseline_raw at that frame, and smart-tagging events within +/- 5 s.
"""
import argparse
import csv
import os
import random
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "starter"))
from games import GAME, GAMES, require  # noqa: E402
import load_smart_tagging  # noqa: E402

FPS = GAMES[GAME]["fps"]
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
RAW_LABELS = os.path.join(HERE, "evaluation", "results", "baseline_raw.csv")
OUT_DIR = os.path.join(HERE, "gap_views")
MIN_GAP_S = 5
TAG_WINDOW_S = 5


# ---------- data ----------

def read_labels(path=RAW_LABELS):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [r["label"].strip().upper() for r in rows]


def find_gaps(labels, min_frames):
    """D runs at least min_frames long: (start, end_exclusive, team_before, team_after)."""
    gaps, i, n = [], 0, len(labels)
    while i < n:
        if labels[i] != "D":
            i += 1
            continue
        j = i
        while j < n and labels[j] == "D":
            j += 1
        if j - i >= min_frames:
            before = labels[i - 1] if i > 0 else "-"
            after = labels[j] if j < n else "-"
            gaps.append((i, j, before, after))
        i = j
    return gaps


def tags_near(events, start, end, pad_frames):
    return [e for e in events
            if e["end_frame"] >= start - pad_frames and e["start_frame"] <= end + pad_frames]


def describe(e):
    t = e["tags"]
    bits = [e["code"]]
    if t.get("Type"):
        bits.append(t["Type"])
    if t.get("Team"):
        bits.append(t["Team"])
    return " / ".join(bits)


def is_set_piece(e):
    return (e["code"] or "").strip().lower() == "set pieces"


def team_of_slots():
    meta_path, _ = require(["atd_meta", "atd_raw"])
    root = ET.parse(meta_path).getroot()
    home_id = [t.get("id") for t in root.iter("Team")][0]
    return [0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")]


def positions_for(frames):
    """{video_frame: (players[(x, y, team)], ball (x, y) or None)} for the requested frames."""
    _, raw_path = require(["atd_meta", "atd_raw"])
    wanted = {f + 1 for f in frames}  # ATD frame_id counts from 1
    slots = team_of_slots()
    out = {}
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3 or int(parts[0]) not in wanted:
                continue
            players = []
            for slot, group in enumerate(parts[1].split(";")):
                if not group or group[0] == "N":
                    continue
                x, y = group.split(",")[:2]
                players.append((float(x) * PITCH_L, float(y) * PITCH_W, slots[slot]))
            ball = None
            if parts[2][0] != "N":
                bx, by = parts[2].split(",")[:2]
                ball = (float(bx) * PITCH_L, float(by) * PITCH_W)
            out[int(parts[0]) - 1] = (players, ball)
            if len(out) == len(wanted):
                break
    return out


def video_frame(frame):
    import cv2
    cap = cv2.VideoCapture(require(["video"])[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if ok else None


# ---------- drawing ----------

def render(frame, labels, events, pos, title_extra=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils import draw_pitch

    fig, (ax_v, ax_p) = plt.subplots(1, 2, figsize=(18, 6.5))
    img = video_frame(frame)
    if img is not None:
        ax_v.imshow(img)
    ax_v.set_axis_off()
    ax_v.set_title("video frame {}  ({:.1f} s)".format(frame, frame / FPS))

    draw_pitch(ax_p, PITCH_L, PITCH_W)
    players, ball = pos.get(frame, ([], None))
    for x, y, team in players:
        ax_p.scatter(x, y, s=70, c="tab:blue" if team == 0 else "tab:red", edgecolors="k", zorder=3)
    if ball:
        ax_p.scatter(*ball, s=50, c="white", edgecolors="k", zorder=4)
    else:
        ax_p.text(PITCH_L / 2, -3, "BALL NOT TRACKED", ha="center", color="crimson")
    near = tags_near(events, frame, frame, TAG_WINDOW_S * FPS)
    tag_text = "\n".join("{:>7.1f}-{:<7.1f} {}".format(e["start_s"], e["end_s"], describe(e))
                         for e in near) or "(no tags within +/-5 s)"
    ax_p.set_title("label (baseline_raw): {}   {}".format(labels[frame], title_extra))
    fig.text(0.52, 0.01, tag_text, family="monospace", fontsize=8, va="bottom")
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "frame_{:06d}.png".format(frame))
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------- commands ----------

def cmd_list(labels, events):
    gaps = find_gaps(labels, MIN_GAP_S * FPS)
    rows, with_sp = [], 0
    for s, e, before, after in gaps:
        near = tags_near(events, s, e, 2 * FPS)
        sp = [describe(x) for x in near if is_set_piece(x)]
        phases = [describe(x) for x in near if not is_set_piece(x)]
        with_sp += bool(sp)
        rows.append({"start_frame": s, "end_frame": e, "start_s": round(s / FPS, 1),
                     "length_s": round((e - s) / FPS, 1), "before": before, "after": after,
                     "same_team": before == after and before in "HA",
                     "set_pieces": " | ".join(sp), "phases": " | ".join(phases)})
    with open("gap_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    same = sum(r["same_team"] for r in rows)
    same_sp = sum(r["same_team"] and bool(r["set_pieces"]) for r in rows)
    print("{} gaps >= {} s  |  {} have a set-piece tag within 2 s ({:.0f}%)".format(
        len(rows), MIN_GAP_S, with_sp, 100 * with_sp / max(1, len(rows))))
    print("{} have the same team on both sides; of those, {} have a set-piece tag".format(same, same_sp))
    print("wrote gap_report.csv")
    return gaps


def cmd_sample(labels, events, n, seed):
    gaps = find_gaps(labels, MIN_GAP_S * FPS)
    pick = sorted(random.Random(seed).sample(gaps, min(n, len(gaps))))
    frames = []
    for s, e, *_ in pick:
        frames += [max(0, s - FPS), s + (e - s) // 2, min(len(labels) - 1, e + FPS)]
    pos = positions_for(frames)
    for k, (s, e, before, after) in enumerate(pick, 1):
        sp = [describe(x) for x in tags_near(events, s, e, 2 * FPS) if is_set_piece(x)]
        info = "gap {}/{}: {:.1f}s long, {} -> {}, set piece: {}".format(
            k, len(pick), (e - s) / FPS, before, after, "; ".join(sp) or "none")
        print(info)
        for f, when in zip([max(0, s - FPS), s + (e - s) // 2, min(len(labels) - 1, e + FPS)],
                           ["1s before", "middle", "1s after"]):
            print("   ", when, render(f, labels, events, pos, "[gap {} {}]".format(k, when)))


def cmd_show(labels, events, frame):
    print(render(frame, labels, events, positions_for([frame])))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    sh = sub.add_parser("show")
    sh.add_argument("frame", type=int)
    a = p.parse_args()

    labels = read_labels()
    events = load_smart_tagging.load()
    if a.cmd == "list":
        cmd_list(labels, events)
    elif a.cmd == "sample":
        cmd_sample(labels, events, a.n, a.seed)
    else:
        cmd_show(labels, events, a.frame)


if __name__ == "__main__":
    main()
