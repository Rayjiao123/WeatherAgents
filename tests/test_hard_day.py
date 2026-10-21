import pytest

from weatheragents import hard_day as hd


def test_rank_correlation_handles_ties_and_direction():
    assert hd.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert hd.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert hd.spearman([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0  # a constant sign says nothing
    assert hd.ranks([5, 5, 1]) == [2.5, 2.5, 1.0]


def test_signs_read_the_logged_prediction_in_celsius():
    p = {"unit": "F", "brackets": ["a", "b"], "ensemble": {"std_dev": 1.8, "model_summaries": {"x": {"mean": 70.0}, "y": {"mean": 73.6}}},
         "nowcast": {"drift_vs_3d": -3.6, "lead_days": 1}, "alerts": {"flags": ["nws_heat"]}, "ml": {"sd": 2.7}}
    s = hd.signs(p)
    assert s["ensemble spread (119 members)"] == pytest.approx(1.0)  # 1.8 F is 1.0 C
    assert s["disagreement between ensemble systems"] == pytest.approx(1.0)  # 3.6 F apart, population sd 1.8 F
    assert s["forecast drift over 3 days"] == pytest.approx(2.0) and s["active alerts"] == 1 and s["days ahead"] == 1
    assert s["the model's own stated spread"] == pytest.approx(1.5)


def test_a_sign_that_tracks_the_miss_ranks_above_one_that_does_not():
    sizes = list(range(40))
    rows = [(float(i) * 0.1, {"informative": float(i), "noise": float((i * 7) % 5)}) for i in sizes]
    got = {name: hd.spearman([s[name] for _, s in rows], [e for e, _ in rows]) for name in ("informative", "noise")}
    assert got["informative"] == pytest.approx(1.0) and abs(got["noise"]) < 0.3
