import math

from weatheragents import postmortem as pm
from weatheragents import rt_obs


def rep(h, t=20.0, dwp=12.0, drct=90.0, sknt=8.0, sky=0.0, vsby=6.0, alti=29.92, p01i=0.0, wx=""):
    return {"h": h, "tmpc": t, "dwpc": dwp, "drct": drct, "sknt": sknt, "sky": sky, "vsby": vsby, "alti": alti, "p01i": p01i, "wx": wx}


PATH = [15 + min(h, 15) * 0.6 for h in range(24)]  # warms to 24 C at 15:00


def test_diverge_hour_finds_when_the_station_left_the_forecast_in_the_direction_of_the_miss():
    obs = {h: PATH[h] - (4.0 if h >= 11 else 0.0) for h in range(0, 18)}  # 4 C colder than forecast from 11:00
    assert pm.diverge_hour(obs, PATH, -1) == 11
    assert pm.diverge_hour(obs, PATH, +1) is None  # it never ran warm
    assert pm.hourly_obs([rep(10.0, 10.0), rep(10.5, 12.0), rep(11.0, 20.0)]) == {10: 11.0, 11: 20.0}


def test_a_front_is_recognized_and_flagged_as_visible_ahead_of_time():
    rows = []
    for h2 in range(0, 30):
        h = h2 / 2
        started = h >= 9.0  # pressure falls and the wind swings from 09:00; the temperature only drops at 12:00
        rows.append(rep(h, t=22.0 - (6 if h >= 12 else 0), alti=29.92 - (0.07 if started else 0), drct=270.0 if started else 90.0,
                        dwp=12.0 + (4 if h >= 12 else 0), wx="RA" if h >= 12 else "", sky=1.0 if h >= 12 else 0.0))
    sig = pm.signature(rows, 13.0, -5.0)
    assert sig["label"] == "front or rain"
    assert sig["early_warning"] is True  # the pressure fall and wind swing were on the record by 11:00, before the 12:00 drop
    quiet = pm.signature([rep(h / 2) for h in range(0, 30)], 13.0, -5.0)
    assert quiet["label"] == "no clear sign" and quiet["early_warning"] is False
