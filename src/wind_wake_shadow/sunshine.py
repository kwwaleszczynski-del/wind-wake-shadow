"""Monthly sunshine fraction from Met Office historic station data.

Station: Oxford (Radcliffe Observatory, 51.76 N 1.26 W), about 75 km south-west of Kelmarsh; the closer stations
with sunshine records (Sutton Bonington, Cambridge NIAB) have no sunshine data for most of 2000-2020.
Source: https://www.metoffice.gov.uk/research/climate/maps-and-data/historic-station-data,
Contains Met Office data (c) Crown copyright, Open Government Licence v3.0.

Sunshine fraction = mean monthly sunshine hours (1991-2020 normal) / astronomically possible daylight hours.

Usage:
    path = download("oxford", Path("data"))
    monthly = read_station(path)                      # year, month, sun_hours
    fraction = sunshine_fraction(monthly, lat=51.761)  # 12 values, January first
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from wind_wake_shadow.shadow import daylight_hours, declination

URL = "https://www.metoffice.gov.uk/pub/data/weather/uk/climate/stationdata/{station}data.txt"
STATION = "oxford"
STATION_LAT = 51.761
NORMAL = (1991, 2020)
ROW = re.compile(r"^\s*(\d{4})\s+(\d{1,2})\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)")
log = logging.getLogger(__name__)


def download(station: str, data_dir: Path) -> Path:
    dest = data_dir / "raw" / f"metoffice_{station}.txt"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(URL.format(station=station), timeout=60,
                     headers={"User-Agent": "wind-wake-shadow (open-source analysis)"})
    r.raise_for_status()
    dest.write_text(r.text, encoding="utf-8")
    log.info("downloaded %s (%d bytes)", dest.name, len(r.content))
    return dest


def parse_station(text: str) -> pd.DataFrame:
    """Monthly rows of a station file: year, month, sun_hours (NaN where missing; '*' and '#' flags dropped)."""
    rows = []
    for line in text.splitlines():
        m = ROW.match(line)
        if not m:
            continue
        sun = m.group(7).rstrip("*#")
        rows.append({"year": int(m.group(1)), "month": int(m.group(2)),
                     "sun_hours": float(sun) if re.fullmatch(r"\d+(\.\d+)?", sun) else np.nan})
    return pd.DataFrame(rows, columns=["year", "month", "sun_hours"])


def read_station(path: Path) -> pd.DataFrame:
    return parse_station(path.read_text(encoding="utf-8"))


def possible_hours(lat: float, year: int = 2021) -> np.ndarray:
    """Astronomical daylight hours in each month of ``year``."""
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    hours = daylight_hours(declination(days.dayofyear.to_numpy(), 12.0, len(days)), lat)
    return pd.Series(hours).groupby(days.month.to_numpy()).sum().to_numpy()


def sunshine_fraction(monthly: pd.DataFrame, lat: float = STATION_LAT, years: tuple[int, int] = NORMAL,
                      min_years: int = 20) -> np.ndarray:
    """Mean sunshine hours per calendar month over ``years`` divided by the possible daylight hours."""
    sel = monthly[monthly["year"].between(*years)].dropna(subset=["sun_hours"])
    stats = sel.groupby("month")["sun_hours"].agg(["mean", "count"]).reindex(range(1, 13))
    if (stats["count"].fillna(0) < min_years).any():
        raise ValueError(f"fewer than {min_years} years of sunshine data in some months: {stats['count'].tolist()}")
    return stats["mean"].to_numpy() / possible_hours(lat)
