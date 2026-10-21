"""One place for time: everything stored or shown is Thailand time (ICT, UTC+7, no DST).

Timestamps are timezone-aware, so each carries its +07:00 offset and still compares correctly
against older records that were written in UTC. Calculations that need a real clock (a station's
local "today", filtering by hours left) keep using UTC internally; only stored and displayed
values come through here.
"""

from datetime import datetime, timedelta, timezone

ICT = timezone(timedelta(hours=7), "ICT")


def now() -> datetime:
    return datetime.now(ICT)


def to_ict(dt: datetime) -> datetime:
    return dt.astimezone(ICT)


def stamp() -> str:
    """Sortable file/run stamp in ICT, e.g. 20261007T231500."""
    return now().strftime("%Y%m%dT%H%M%S")


def fmt(dt: datetime) -> str:
    return to_ict(dt).strftime("%m-%d %H:%M")
