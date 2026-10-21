import httpx

from weatheragents import alerts_analyst as al


def test_flags_combine_cyclone_distance_and_us_alert_wording():
    assert al.flags_from(500, ["Heat Advisory", "Dense Fog Advisory"]) == ["cyclone_within_800km", "nws_heat", "nws_fog"]
    assert al.flags_from(2500, ["Flood Watch"]) == ["nws_rain"]  # too far to matter
    assert al.flags_from(None, []) == []
    assert "nws_cold" in al.flags_from(None, ["Winter Weather Advisory"])


def test_active_cyclones_keeps_only_current_storms_and_reads_lon_lat_order():
    feats = [
        {"geometry": {"type": "Point", "coordinates": [146.3, 18.8]}, "properties": {"eventtype": "TC", "iscurrent": True, "name": "Typhoon X", "alertlevel": "Orange", "severitydata": {"severity": 200.0}}},
        {"geometry": {"type": "Point", "coordinates": [10.0, 10.0]}, "properties": {"eventtype": "EQ", "iscurrent": True, "name": "quake"}},
        {"geometry": {"type": "Point", "coordinates": [100.0, 5.0]}, "properties": {"eventtype": "TC", "iscurrent": False, "name": "old storm"}},
    ]
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"features": feats})))
    al._CYCLONES.update(at=0.0, data=[])
    out = al.active_cyclones(client)
    assert [c["name"] for c in out] == ["Typhoon X"]
    assert out[0]["lat"] == 18.8 and out[0]["lon"] == 146.3 and out[0]["wind_kmh"] == 200.0
