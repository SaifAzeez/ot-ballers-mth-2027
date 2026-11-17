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


if __name__ == "__main__":
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith("test_")]
    for name, f in tests:
        f()
        print("ok  ", name)
    print("{} tests passed".format(len(tests)))
