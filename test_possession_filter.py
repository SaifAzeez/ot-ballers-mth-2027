"""Tests for possession_filter.py on hand-made label strings (no match data needed).

    python test_possession_filter.py      # or: python -m pytest test_possession_filter.py
"""
from possession_filter import FPS, possession_filter


def run(labels, set_pieces=None):
    return "".join(possession_filter(list(labels), set_pieces))


# -- rules 1-3 ----------------------------------------------------------------------------------

def test_short_blip_bridged_and_short_challenge_ignored():
    x = "H" * 30 + "D" * 10 + "H" * 30 + "A" * 20 + "H" * 30
    assert run(x) == "H" * len(x)


def test_two_second_challenger_wins_from_run_start():
    assert run("H" * 30 + "A" * 50 + "H" * 5) == "H" * 30 + "A" * 55


def test_restart_possessor_is_first_team_seen():
    assert run("H" * 30 + "D" * 25 + "A" * 3 + "H" * 10) == "H" * 30 + "D" * 25 + "A" * 13


def test_file_edge_dead_ball_kept():
    assert run("DDD" + "H" * 5 + "DD") == "DDDHHHHHDD"


def test_min_dead_same_team_bridged_other_kept():
    assert run("H" * 30 + "D" * 60 + "H" * 30) == "H" * 120
    assert run("H" * 30 + "D" * 75 + "H" * 30) == "H" * 30 + "D" * 75 + "H" * 30
    assert run("H" * 30 + "D" * 60 + "A" * 30) == "H" * 30 + "D" * 60 + "A" * 30


# -- rule 0 (--use-tags) ------------------------------------------------------------------------

GAP = 10 * FPS  # a 10 s gap, long enough to be a stoppage without tags


def test_tags_untagged_same_team_gap_filled():
    x = "H" * 60 + "D" * GAP + "H" * 60
    assert run(x) == "H" * 60 + "D" * GAP + "H" * 60      # without tags: stoppage
    assert run(x, set_pieces=[]) == "H" * len(x)            # with tags, none near: filled


def test_tags_tagged_gap_kept():
    x = "H" * 200 + "D" * GAP + "H" * 200
    inside = [(300, 310)]                       # overlaps the gap
    near_after = [(200 + GAP + 40, 200 + GAP + 60)]  # starts 1.6 s after the gap ends
    for sp in (inside, near_after):
        assert run(x, set_pieces=sp) == "H" * 200 + "D" * GAP + "H" * 200
    far = [(200 + GAP + 60, 200 + GAP + 80)]    # starts 2.4 s after: not evidence
    assert run(x, set_pieces=far) == "H" * len(x)


def test_tags_30s_gap_kept():
    long_gap = 30 * FPS
    x = "H" * 60 + "D" * long_gap + "H" * 60
    assert run(x, set_pieces=[]) == x
    y = "H" * 60 + "D" * (long_gap - 1) + "H" * 60
    assert run(y, set_pieces=[]) == "H" * len(y)


def test_tags_different_teams_kept():
    x = "H" * 60 + "D" * GAP + "A" * 60
    assert run(x, set_pieces=[]) == x


# -- --stoppages rules --------------------------------------------------------------------------

import numpy as np  # noqa: E402

from possession_filter import (ball_out_of_play, players_stopped,  # noqa: E402
                               STOP_FRAMES)

L, W = 105.0, 68.0


def ball_path(xs, y=34.0):
    return np.column_stack([np.asarray(xs, float), np.full(len(xs), y)])


def test_ball_out_dead_until_back_inside_and_moving():
    # inside, out over the end line (x > L + 0.5), lost for a frame, back inside but still,
    # then kicked (>2 m/s)
    xs = ([50.0] * 5 + [L + 2.0] * 4 + [np.nan] + [L - 1.0] * 5
          + [L - 1.0 - 0.2 * i for i in range(1, 6)])
    dead = ball_out_of_play(ball_path(xs), L, W)
    assert not dead[:5].any()
    assert dead[5:15].all()          # out, then back in but not moving (0 m/s): still dead
    assert not dead[15:].any()       # moving 0.2 m/frame = 5 m/s: play restarted


def test_ball_out_margin_and_untracked():
    xs = [L + 0.4] * 5 + [np.nan] * 5 + [50.0] * 5
    assert not ball_out_of_play(ball_path(xs), L, W).any()   # within 0.5 m: still in
    xs = [L + 1.0] * 3 + [np.nan] * 5 + [50.0] * 5
    dead = ball_out_of_play(ball_path(xs), L, W)
    assert dead[:8].all()            # stays dead while the ball is not tracked
    assert dead[8:].all()            # tracked inside but no speed yet (first frame) / still


def walkers(n_frames, speed, n_players=10, start_x=60.0):
    """Players moving along x at `speed` m/s; team 0 and 1 alternate."""
    t = np.arange(n_frames)[:, None] / FPS
    x = start_x + speed * t + np.zeros((1, n_players))
    y = np.linspace(10, 58, n_players)[None, :] + np.zeros((n_frames, 1))
    return np.stack([x, y], axis=-1), np.arange(n_players) % 2


def test_players_stopped_three_seconds_is_dead():
    players, teams = walkers(STOP_FRAMES + 10, speed=1.0)
    ball = np.full((len(players), 2), np.nan)
    dead = players_stopped(players, ball, teams, L, W, np.zeros(len(players), np.int8))
    assert dead[1:].all() and not dead[0]   # frame 0 has no speed


def test_players_stopped_short_or_fast_is_live():
    players, teams = walkers(STOP_FRAMES - 5, speed=1.0)      # under 3 s
    ball = np.full((len(players), 2), np.nan)
    assert not players_stopped(players, ball, teams, L, W, np.zeros(len(players), np.int8)).any()
    players, teams = walkers(STOP_FRAMES + 10, speed=2.0)     # jogging
    assert not players_stopped(players, ball, teams, L, W, np.zeros(len(players), np.int8)).any()


def test_players_stopped_ignores_tracking_jumps():
    players, teams = walkers(STOP_FRAMES + 10, speed=1.0)
    players[40, 0, 0] += 5.0                                  # one 125 m/s jump: ignored
    ball = np.full((len(players), 2), np.nan)
    assert players_stopped(players, ball, teams, L, W, np.zeros(len(players), np.int8))[1:].all()


def test_players_stopped_keeper_holding_ball_is_kept():
    players, teams = walkers(STOP_FRAMES + 10, speed=0.0, start_x=30.0)
    players[:, 0] = [3.0, 34.0]                               # team 0's deepest: the keeper
    ball = np.tile([3.5, 34.0], (len(players), 1))            # in his box, he is nearest
    assert not players_stopped(players, ball, teams, L, W, np.zeros(len(players), np.int8)).any()
    ball[:] = [3.5, 34.0]
    players[:, 1] = [3.2, 34.0]                               # opponent (team 1) nearest instead
    assert players_stopped(players, ball, teams, L, W, np.zeros(len(players), np.int8))[1:].all()


def test_forced_dead_short_run_is_a_stoppage():
    raw = np.array(list("H" * 60 + "A" * 60))
    forced = np.zeros(len(raw), bool)
    forced[58:62] = True                                      # 4 frames, under the 1 s blip
    out = "".join(possession_filter(raw, None, forced))
    assert out == "H" * 58 + "D" * 4 + "A" * 58


if __name__ == "__main__":
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith("test_")]
    for name, f in tests:
        f()
        print("ok  ", name)
    print("{} tests passed".format(len(tests)))
