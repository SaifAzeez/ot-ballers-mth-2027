"""Review player: watch the tactical video with v1 / v2 labels, tracking and tags overlaid.

For UNDERSTANDING errors only. It writes no labels, reads no ground truth and changes no
submission file. Do not hand-label from it (RULES.md).

    python review_player.py                 # start at the first-half kick-off
    python review_player.py --start 54321   # start at a video frame
    python review_player.py --width 1600    # bigger window (default 1280)

Controls (click the video window first so it has keyboard focus):

    mouse        click or drag the whole-match bar (bottom) to seek, hover it to see the time;
                 click or drag the +/- 30 s strip to move within it
    space        pause / play
    left/right   step 1 frame   (also , and .)
    shift+arrow  -/+ 5 s  (macOS OpenCV does not report shift: use a / d, or < / >)
    j  l         -/+ 10 s
    0-9          jump to 0%, 10% ... 90% of the video
    [  ]         previous / next missing-ball gap (raw D run of 1 s or more)
    n            next place where v1 and v2 differ
    +  -         playback speed: x0.25, x0.5, x1, x2, x3, x4, x8
                 (or click the x0.5 / x1 / x2 / x3 / x4 buttons under the seek bar)
    g            go to frame: type digits, Enter to jump, Esc to cancel
    h            hide / show the data overlays (info panel, pitch inset) to see the full video;
                 the timeline and seek bar stay, below the picture
    q            quit

Overlay: frame and match clock, v1 / v2 label (H blue, A red, D grey), ball tracked or not, the
raw nearest-player team (baseline_raw), Smart Tagging events active at the frame, and the mean
speed of tracked players. Top right: pitch inset, players by team and the ball (drawn y-up, as in
the starter's plots). Bottom: v1 and v2 for +/- 30 s, current frame marked.

Startup parses the raw ATD once (about a minute) and caches it in review_cache/ (gitignored --
it is match data). Later starts load the cache in a second or two.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402
import baseline  # noqa: E402  -- the starter's raw ATD format and team-of-slot mapping
import load_smart_tagging  # noqa: E402
import possession_filter  # noqa: E402

FPS = GAMES[GAME]["fps"]
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
N_FRAMES = source_frames("tactical", GAME)
RAW_CSV = os.path.join(HERE, "evaluation", "results", "baseline_raw.csv")
V1_CSV = os.path.join(HERE, "Old_Trafford_Ballers_v1.csv")
V2_CSV = os.path.join(HERE, "Old_Trafford_Ballers_v2.csv")
CACHE = os.path.join(HERE, "review_cache", "{}_tracking.npz".format(GAME))

KICKOFF_1, END_1 = 11185, 90651     # first half: kick-off .. last tracked frame (DATA.md)
KICKOFF_2, END_2 = 113742, 190197   # second half
GAP_MIN = possession_filter.BLIP_FRAMES
TIMELINE_S = 30
SPEEDS = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0)   # +/- cycle through these
SPEED_BUTTONS = (0.5, 1.0, 2.0, 3.0, 4.0)       # clickable buttons under the seek bar

# BGR. Codes: 0 H, 1 A, 2 D, 3 anything else / outside the video.
CODE = {"H": 0, "A": 1, "D": 2}
PALETTE = np.array([(235, 130, 30), (40, 40, 225), (150, 150, 150), (25, 25, 25)], np.uint8)
TEAM_BGR = [tuple(int(c) for c in PALETTE[0]), tuple(int(c) for c in PALETTE[1])]
WHITE, BLACK, YELLOW = (255, 255, 255), (0, 0, 0), (0, 220, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------- precompute ----------

def parse_atd():
    """Players (N, slots, 2) and ball (N, 2) in metres, mean speed (N,), tracked count (N,)."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    teams = baseline.team_of_each_track(meta_path)
    n_slots = len(teams)
    players = np.full((N_FRAMES, n_slots, 2), np.nan, np.float32)
    ball = np.full((N_FRAMES, 2), np.nan, np.float32)
    mean_speed = np.full(N_FRAMES, np.nan, np.float32)
    n_tracked = np.zeros(N_FRAMES, np.int16)
    scale = np.array([PITCH_L, PITCH_W], np.float32)

    print("parsing {} (once; cached afterwards) ...".format(os.path.basename(raw_path)))
    t0 = time.time()
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(baseline.FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id counts from 1, video frames from 0
            if not 0 <= frame < N_FRAMES:
                continue
            vals = np.array(parts[1].replace(baseline.GROUP_SEP, ",").split(","),
                            dtype=np.float32).reshape(-1, 3)[:n_slots]
            players[frame, :len(vals)] = vals[:, :2] * scale
            seen = ~np.isnan(vals[:, 0])
            n_tracked[frame] = seen.sum()
            speeds = vals[seen, 2]
            speeds = speeds[~np.isnan(speeds)]
            if len(speeds):
                mean_speed[frame] = speeds.mean()
            if parts[2][0] != baseline.MISSING:
                ball[frame] = np.array(parts[2].split(",")[:2], np.float32) * scale
    print("  done in {:.0f} s".format(time.time() - t0))
    return dict(players=players, ball=ball, mean_speed=mean_speed, n_tracked=n_tracked,
                teams=teams)


def load_tracking():
    if os.path.exists(CACHE):
        with np.load(CACHE) as z:
            return {k: z[k] for k in z.files}
    data = parse_atd()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, **data)
    return data


def read_labels(path):
    labels = pd.read_csv(path)["label"].astype(str).str.strip().str.upper().to_numpy()
    if len(labels) != N_FRAMES:
        sys.exit("{}: {} rows, expected {}".format(path, len(labels), N_FRAMES))
    return labels


def to_codes(labels):
    return np.array([CODE.get(x, 3) for x in labels], np.uint8)


def load_labels(events):
    """raw, v1, v2 label arrays. v1/v2 from their files; computed in memory if a file is missing."""
    raw = read_labels(RAW_CSV)
    set_pieces = [(e["start_frame"], e["end_frame"]) for e in events
                  if (e["code"] or "").strip().lower() == "set pieces"]
    v1 = read_labels(V1_CSV) if os.path.exists(V1_CSV) else possession_filter.possession_filter(raw)
    v2 = (read_labels(V2_CSV) if os.path.exists(V2_CSV)
          else possession_filter.possession_filter(raw, set_pieces))
    return raw, v1, v2


def run_starts(mask):
    """Frames where a run of True begins."""
    prev = np.concatenate(([False], mask[:-1]))
    return np.flatnonzero(mask & ~prev)


def gap_starts(raw):
    out = []
    for start, end, label in possession_filter.runs(raw):
        if label == "D" and end - start >= GAP_MIN:
            out.append(start)
    return np.array(out, np.int64)


def match_clock(frame):
    if frame < KICKOFF_1:
        return "pre-match"
    if frame <= END_1:
        s = (frame - KICKOFF_1) / FPS
        return "1H {:02d}:{:04.1f}".format(int(s // 60), s % 60)
    if frame < KICKOFF_2:
        return "half-time"
    if frame <= END_2:
        s = 45 * 60 + (frame - KICKOFF_2) / FPS
        return "2H {:02d}:{:04.1f}".format(int(s // 60), s % 60)
    return "full-time"


# ---------- video ----------

class Video:
    """Sequential reads where possible, seeks only for jumps; remembers the last frame."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            sys.exit("cannot open video " + path)
        self.next_pos = 0
        self.last = (None, None)
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get(self, frame):
        if self.last[0] == frame:
            return self.last[1]
        if frame < self.next_pos or frame - self.next_pos > 2 * FPS:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            self.next_pos = frame
        while self.next_pos < frame:
            self.cap.grab()
            self.next_pos += 1
        ok, img = self.cap.read()
        self.next_pos += 1
        if not ok:
            img = np.zeros((self.h, self.w, 3), np.uint8)
        self.last = (frame, img)
        return img


# ---------- drawing ----------

def text(img, x, y, s, color=WHITE, scale=0.55, thick=1):
    cv2.putText(img, s, (x, y), FONT, scale, BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    return x + cv2.getTextSize(s, FONT, scale, thick)[0][0]


def label_color(label):
    return tuple(int(c) for c in PALETTE[CODE.get(label, 3)]) if label in CODE else WHITE


def darken(img, x0, y0, x1, y1, keep=0.12):
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * keep).astype(np.uint8)


class Inset:
    PX_PER_M = 3.0

    def __init__(self):
        k = self.PX_PER_M
        self.w, self.h = int(PITCH_L * k), int(PITCH_W * k)
        bg = np.full((self.h, self.w, 3), (60, 130, 60), np.uint8)
        line = (230, 230, 230)
        cv2.rectangle(bg, (0, 0), (self.w - 1, self.h - 1), line, 1)
        cv2.line(bg, (self.w // 2, 0), (self.w // 2, self.h), line, 1)
        cv2.circle(bg, (self.w // 2, self.h // 2), int(9.15 * k), line, 1)
        for depth, box_w in ((16.5, 40.3), (5.5, 18.3)):
            y0, y1 = int((PITCH_W - box_w) / 2 * k), int((PITCH_W + box_w) / 2 * k)
            cv2.rectangle(bg, (0, y0), (int(depth * k), y1), line, 1)
            cv2.rectangle(bg, (self.w - 1 - int(depth * k), y0), (self.w - 1, y1), line, 1)
        self.bg = bg

    def xy(self, x, y):
        return int(round(x * self.PX_PER_M)), int(round(self.h - y * self.PX_PER_M))  # y up

    def draw(self, players, teams, ball):
        img = self.bg.copy()
        for (x, y), team in zip(players, teams):
            if np.isnan(x):
                continue
            p = self.xy(x, y)
            cv2.circle(img, p, 4, TEAM_BGR[int(team)], -1, cv2.LINE_AA)
            cv2.circle(img, p, 4, BLACK, 1, cv2.LINE_AA)
        if not np.isnan(ball[0]):
            p = self.xy(*ball)
            cv2.circle(img, p, 3, WHITE, -1, cv2.LINE_AA)
            cv2.circle(img, p, 3, BLACK, 1, cv2.LINE_AA)
        else:
            text(img, 6, self.h - 6, "ball not tracked", (80, 80, 255), 0.4)
        return img


def timeline(width, frame, v1c, v2c):
    """Two bars (v1, v2) for +/- TIMELINE_S around `frame`, ticks every 10 s, centre marker."""
    margin, bar_h, top = BAR_X0, 18, 8
    h = top + 2 * bar_h + 6 + 20
    img = np.full((h, width, 3), 20, np.uint8)
    bar_w = bar_width(width)
    half = TIMELINE_S * FPS
    idx = frame + np.round(np.linspace(-half, half, bar_w)).astype(np.int64)
    valid = (idx >= 0) & (idx < N_FRAMES)
    clipped = np.clip(idx, 0, N_FRAMES - 1)
    for row, codes in enumerate((v1c, v2c)):
        colors = PALETTE[codes[clipped]]
        colors[~valid] = PALETTE[3]
        y0 = top + row * (bar_h + 3)
        img[y0:y0 + bar_h, margin:margin + bar_w] = colors[None]
        text(img, 6, y0 + bar_h - 4, ("v1", "v2")[row], WHITE, 0.45)
    y_ticks = top + 2 * bar_h + 6
    for sec in range(-TIMELINE_S, TIMELINE_S + 1, 10):
        x = margin + int((sec * FPS + half) / (2 * half) * (bar_w - 1))
        cv2.line(img, (x, top), (x, y_ticks), (90, 90, 90), 1)
        text(img, min(x - 10, width - 34), y_ticks + 13, "{:+d}s".format(sec) if sec else "0",
             (180, 180, 180), 0.38)
    xc = margin + bar_w // 2
    cv2.line(img, (xc, top - 4), (xc, y_ticks + 2), WHITE, 2)
    return img


# Shared bar geometry: bars run from BAR_X0 to width - BAR_RIGHT, in the timeline and scrub bar.
BAR_X0, BAR_RIGHT = 40, 10
SCRUB_H = 46


def bar_width(width):
    return width - BAR_X0 - BAR_RIGHT


def x_to_frac(x, width):
    return float(np.clip((x - BAR_X0) / (bar_width(width) - 1), 0.0, 1.0))


def clock_label(frame):
    s = frame / FPS
    return "{}  ({:d}:{:02d})".format(match_clock(frame), int(s // 60), int(s % 60))


class ScrubBar:
    """Whole-video seek bar: v1 and v2 overview, half markers, playhead, hover time."""

    def __init__(self, width, v1c, v2c):
        self.width = width
        cols = np.round(np.linspace(0, N_FRAMES - 1, bar_width(width))).astype(np.int64)
        self.rows = [PALETTE[v1c[cols]], PALETTE[v2c[cols]]]

    def frame_at(self, x):
        return int(round(x_to_frac(x, self.width) * (N_FRAMES - 1)))

    def x_of(self, frame):
        return BAR_X0 + int(frame / (N_FRAMES - 1) * (bar_width(self.width) - 1))

    def draw(self, frame, hover_x=None):
        img = np.full((SCRUB_H, self.width, 3), 20, np.uint8)
        bw = bar_width(self.width)
        y0 = 20
        for row, colors in enumerate(self.rows):
            y = y0 + row * 9
            img[y:y + 8, BAR_X0:BAR_X0 + bw] = colors[None]
        text(img, 4, y0 + 15, "all", (170, 170, 170), 0.4)
        for f in (KICKOFF_1, END_1, KICKOFF_2, END_2):  # half boundaries
            x = self.x_of(f)
            cv2.line(img, (x, y0 - 3), (x, y0 + 19), (230, 230, 230), 1)
        xp = self.x_of(frame)
        cv2.rectangle(img, (BAR_X0, y0 - 6), (xp, y0 - 4), (60, 60, 230), -1)  # progress
        cv2.circle(img, (xp, y0 + 8), 7, WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, (xp, y0 + 8), 7, BLACK, 1, cv2.LINE_AA)
        if hover_x is not None:
            hf = self.frame_at(hover_x)
            label = "{}  {}".format(hf, clock_label(hf))
            tw = cv2.getTextSize(label, FONT, 0.4, 1)[0][0]
            tx = int(np.clip(hover_x - tw // 2, 0, self.width - tw - 2))
            cv2.line(img, (hover_x, y0 - 2), (hover_x, y0 + 17), YELLOW, 1)
            text(img, tx, 10, label, YELLOW, 0.4)
        return img


BTN_H, BTN_W, BTN_GAP, BTN_X0 = 28, 52, 8, 70


def speed_button_at(x):
    """The speed of the button under x, or None."""
    for i, s in enumerate(SPEED_BUTTONS):
        x0 = BTN_X0 + i * (BTN_W + BTN_GAP)
        if x0 <= x < x0 + BTN_W:
            return s
    return None


def speed_row(width, speed):
    img = np.full((BTN_H, width, 3), 20, np.uint8)
    text(img, 8, 19, "speed", (170, 170, 170), 0.45)
    for i, s in enumerate(SPEED_BUTTONS):
        x0 = BTN_X0 + i * (BTN_W + BTN_GAP)
        active = s == speed
        cv2.rectangle(img, (x0, 3), (x0 + BTN_W - 1, BTN_H - 4),
                      WHITE if active else (70, 70, 70), -1 if active else 1)
        label = "x{:g}".format(s)
        tw = cv2.getTextSize(label, FONT, 0.5, 1)[0][0]
        cv2.putText(img, label, (x0 + (BTN_W - tw) // 2, 19), FONT, 0.5,
                    BLACK if active else (210, 210, 210), 1, cv2.LINE_AA)
    if speed not in SPEED_BUTTONS:  # x0.25 or x8 from the keyboard
        text(img, BTN_X0 + len(SPEED_BUTTONS) * (BTN_W + BTN_GAP) + 4, 19,
             "x{:g}".format(speed), YELLOW, 0.5)
    return img


HELP = ("space play  click/drag bars to seek  <-/-> or ,/. 1 frame  a/d 5 s  j/l 10 s  0-9 jump %  "
        "[ ] gaps  n v1!=v2  +/- speed  g goto  h hide data  q quit")


# ---------- keys ----------

LEFT, RIGHT = (63234, 65361), (63235, 65363)      # macOS Cocoa, GTK
WIN_LEFT, WIN_RIGHT = 2424832, 2555904            # Windows


def decode_key(k):
    """('left'|'right', shifted) for arrows, else (character or None, False)."""
    if k == WIN_LEFT:
        return "left", False
    if k == WIN_RIGHT:
        return "right", False
    low, shifted = k & 0xFFFF, bool((k >> 16) & 1)  # GTK packs modifiers above bit 16
    if low in LEFT:
        return "left", shifted
    if low in RIGHT:
        return "right", shifted
    if low < 128:
        return chr(low), False
    return None, False


# ---------- main loop ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", type=int, default=KICKOFF_1, help="video frame to start at")
    ap.add_argument("--width", type=int, default=1280, help="window width in pixels")
    args = ap.parse_args()

    events = load_smart_tagging.load()
    raw, v1, v2 = load_labels(events)
    trk = load_tracking()
    v1c, v2c = to_codes(v1), to_codes(v2)
    diff_starts = run_starts(v1 != v2)
    gaps = gap_starts(raw)
    ev_start = np.array([e["start_frame"] for e in events])
    ev_end = np.array([e["end_frame"] for e in events])
    ev_text = [" / ".join(x for x in (e["code"], e["tags"].get("Type"), e["tags"].get("Team"))
                          if x) for e in events]
    print("{} gaps >= {} s, {} places where v1 and v2 differ".format(
        len(gaps), GAP_MIN // FPS, len(diff_starts)))
    print(HELP)

    video = Video(require(["video"])[0])
    dw = args.width
    dh = int(round(video.h * dw / video.w))
    inset = Inset()
    scrub = ScrubBar(dw, v1c, v2c)
    win = "review_player"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    frame = int(np.clip(args.start, 0, N_FRAMES - 1))

    # Mouse: the callback only records where to seek; the loop does one seek per redraw, so
    # dragging stays responsive however fast the mouse moves.
    timeline_h = timeline(dw, 0, v1c, v2c).shape[0]
    mouse = {"seek": None, "drag": None, "hover": None, "speed": None}
    buttons_y0 = dh + timeline_h + SCRUB_H

    def on_mouse(event, x, y, flags, _param):
        in_timeline = dh <= y < dh + timeline_h
        in_scrub = dh + timeline_h <= y < buttons_y0
        mouse["hover"] = x if in_scrub else None
        if event == cv2.EVENT_LBUTTONDOWN and buttons_y0 <= y < buttons_y0 + BTN_H:
            mouse["speed"] = speed_button_at(x)
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["drag"] = "scrub" if in_scrub else "timeline" if in_timeline else None
            mouse["anchor"] = frame  # timeline drags are relative to where the click started
        elif event == cv2.EVENT_LBUTTONUP:
            mouse["drag"] = None
            return
        elif not (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON):
            return
        if mouse["drag"] == "scrub":
            mouse["seek"] = scrub.frame_at(x)
            mouse["hover"] = x
        elif mouse["drag"] == "timeline":
            offset = (x_to_frac(x, dw) * 2 - 1) * TIMELINE_S * FPS
            mouse["seek"] = int(np.clip(mouse["anchor"] + round(offset), 0, N_FRAMES - 1))

    cv2.setMouseCallback(win, on_mouse)
    playing, speed_i = False, SPEEDS.index(1.0)
    show_data = True
    typing, typed = False, ""
    message, message_until = "", 0.0

    def flash(s):
        nonlocal message, message_until
        message, message_until = s, time.time() + 2.0

    while True:
        t0 = time.time()
        if mouse["seek"] is not None:
            frame, mouse["seek"] = mouse["seek"], None
        if mouse["speed"] is not None:
            speed_i, mouse["speed"] = SPEEDS.index(mouse["speed"]), None
        img = cv2.resize(video.get(frame), (dw, dh), interpolation=cv2.INTER_AREA)

        clean = img.copy()  # the bare video frame, shown instead when the data is hidden (h)

        # info panel
        tags = [ev_text[i] for i in np.flatnonzero((ev_start <= frame) & (ev_end > frame))]
        panel_h = 128 + 18 * min(len(tags), 5)
        darken(img, 0, 0, 520, panel_h)
        y = 22
        text(img, 10, y, "frame {}   {}   x{:g} {}".format(
            frame, match_clock(frame), SPEEDS[speed_i], "PLAYING" if playing else "paused"))
        y += 24
        x = text(img, 10, y, "v1 ", WHITE, 0.7, 2)
        x = text(img, x, y, v1[frame], label_color(v1[frame]), 0.7, 2)
        x = text(img, x + 20, y, "v2 ", WHITE, 0.7, 2)
        x = text(img, x, y, v2[frame], label_color(v2[frame]), 0.7, 2)
        if v1[frame] != v2[frame]:
            text(img, x + 16, y, "differ", YELLOW, 0.5)
        y += 24
        tracked = not np.isnan(trk["ball"][frame, 0])
        x = text(img, 10, y, "ball: ")
        x = text(img, x, y, "tracked" if tracked else "NOT TRACKED",
                 (120, 230, 120) if tracked else (80, 80, 255))
        x = text(img, x + 20, y, "raw nearest: ")
        text(img, x, y, raw[frame] if tracked else "-", label_color(raw[frame]))
        y += 22
        ms = trk["mean_speed"][frame]
        text(img, 10, y, "mean player speed: {} ({} tracked)".format(
            "{:.2f} m/s".format(ms) if not np.isnan(ms) else "-", int(trk["n_tracked"][frame])))
        y += 22
        text(img, 10, y, "tags:" if tags else "tags: none", (200, 200, 200), 0.5)
        for t in tags[:5]:
            y += 18
            text(img, 20, y, t, (200, 200, 200), 0.45)

        # pitch inset, top right
        pitch = inset.draw(trk["players"][frame], trk["teams"], trk["ball"][frame])
        x0, y0 = dw - inset.w - 10, 10
        img[y0:y0 + inset.h, x0:x0 + inset.w] = pitch

        if not show_data:
            img = clean
        msg_y = panel_h + 26 if show_data else 30
        if typing:
            text(img, 10, msg_y, "go to frame: {}_".format(typed), YELLOW, 0.7, 2)
        elif message and time.time() < message_until:
            text(img, 10, msg_y, message, YELLOW, 0.6)

        strip = timeline(dw, frame, v1c, v2c)
        help_row = np.full((22, dw, 3), 20, np.uint8)
        text(help_row, 8, 16, HELP, (170, 170, 170), 0.42)
        cv2.imshow(win, np.vstack([img, strip, scrub.draw(frame, mouse["hover"]),
                                   speed_row(dw, SPEEDS[speed_i]), help_row]))

        # timing and keys
        speed = SPEEDS[speed_i]
        step = max(1, int(speed))
        if playing:
            target = step / (FPS * speed)
            wait = max(1, int((target - (time.time() - t0)) * 1000))
        else:
            wait = 30
        k = cv2.waitKeyEx(wait)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

        if k == -1:
            if playing and mouse["drag"] is None and mouse["seek"] is None:
                frame = min(frame + step, N_FRAMES - 1)
                if frame == N_FRAMES - 1:
                    playing = False
            continue

        key, shifted = decode_key(k)
        if typing:
            if key and key.isdigit():
                typed += key
            elif k in (8, 127) or key in ("\b", "\x7f"):
                typed = typed[:-1]
            elif key in ("\r", "\n"):
                if typed and 0 <= int(typed) < N_FRAMES:
                    frame = int(typed)
                else:
                    flash("frame must be 0..{}".format(N_FRAMES - 1))
                typing, typed = False, ""
            elif key == "\x1b":
                typing, typed = False, ""
            continue

        if key == "q":
            break
        elif key == " ":
            playing = not playing
        elif key in ("left", "right"):
            delta = (5 * FPS if shifted else 1) * (-1 if key == "left" else 1)
            frame = int(np.clip(frame + delta, 0, N_FRAMES - 1))
        elif key in (",", "."):
            frame = int(np.clip(frame + (1 if key == "." else -1), 0, N_FRAMES - 1))
        elif key in ("a", "<"):
            frame = max(0, frame - 5 * FPS)
        elif key in ("d", ">"):
            frame = min(N_FRAMES - 1, frame + 5 * FPS)
        elif key in ("j", "l"):
            frame = int(np.clip(frame + (10 * FPS if key == "l" else -10 * FPS), 0, N_FRAMES - 1))
        elif key and key.isdigit():
            frame = int(int(key) / 10 * (N_FRAMES - 1))
            flash("jumped to {}0%".format(key) if key != "0" else "jumped to start")
        elif key == "]":
            nxt = gaps[gaps > frame]
            frame = int(nxt[0]) if len(nxt) else frame
            flash("gap at {}".format(frame) if len(nxt) else "no later gap")
        elif key == "[":
            prv = gaps[gaps < frame]
            frame = int(prv[-1]) if len(prv) else frame
            flash("gap at {}".format(frame) if len(prv) else "no earlier gap")
        elif key == "n":
            nxt = diff_starts[diff_starts > frame]
            frame = int(nxt[0]) if len(nxt) else frame
            flash("v1/v2 differ from {}".format(frame) if len(nxt) else "no later difference")
        elif key in ("+", "="):
            speed_i = min(speed_i + 1, len(SPEEDS) - 1)
        elif key in ("-", "_"):
            speed_i = max(speed_i - 1, 0)
        elif key == "g":
            typing, typed, playing = True, "", False
        elif key == "h":
            show_data = not show_data
            flash("data shown" if show_data else "data hidden - press h to show")

    video.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
