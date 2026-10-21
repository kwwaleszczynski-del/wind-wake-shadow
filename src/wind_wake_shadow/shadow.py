"""Shadow flicker on a regular grid around the farm: astronomical worst case and an expected case.

Geometry: flat terrain, receptor points 2 m above ground, each rotor a disc of radius R at hub height.
Worst case (astronomical maximum): the sun always shines, rotors always turn and always face the sun. A point is
in flicker while the centre of the sun is behind a rotor disc, the sun is at least 3 deg above the horizon and
the point lies within 10 rotor diameters of the turbine (the distance used in UK shadow flicker guidance).

Method: during one day the sun moves on a circle of (nearly) constant declination at 15 deg of hour angle per
hour. Seen from a receptor, a rotor facing the sun covers a cap of angular radius atan(R / d) around the hub
direction, so the time the sun spends behind it has a closed form:

    cos(theta) = sin(dec) sin(dec_hub) + cos(dec) cos(dec_hub) cos(h - h_hub) >= cos(rho)

gives an interval of hour angle centred on the hub's hour angle, clipped to the part of the day with the sun
above 3 deg. Each (receptor, turbine, day) is a handful of array operations instead of 1440 sun positions;
``brute_force_minutes`` does the 1-minute time stepping and the tests check that both agree.

Expected case: worst-case minutes per turbine and month x sunshine fraction (station sunshine hours / possible
daylight hours) x share of daylight time the rotor turns x mean |cos| of the angle between the rotor axis and
the line receptor-turbine (a rotor seen edge-on casts a thin shadow), the last two from SCADA.

Usage:
    grid = Grid.around(x, y, half_width_m=1600.0, step_m=10.0)
    wc = worst_case(grid, turbines, lat=52.4, lon=-0.94, year=2021)
    wc["hours"], wc["max_minutes_day"]                          # (ny, nx) maps
    hours = expected_hours(wc, grid, turbines, sunshine, operating, orientation)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_ELEVATION_DEG = 3.0
MAX_DISTANCE_D = 10.0
RECEPTOR_HEIGHT_M = 2.0
MINUTES_PER_RADIAN = 1440.0 / (2 * np.pi)
EMPTY = -10.0  # start = end = EMPTY marks "no shadow"; sorts before every real hour angle


@dataclass
class Grid:
    xs: np.ndarray
    ys: np.ndarray

    @classmethod
    def around(cls, x, y, half_width_m: float, step_m: float) -> "Grid":
        """Square grid centred on the middle of the turbine positions."""
        cx, cy = (np.max(x) + np.min(x)) / 2, (np.max(y) + np.min(y)) / 2
        n = int(round(half_width_m / step_m))
        offsets = np.arange(-n, n + 1) * step_m
        return cls(cx + offsets, cy + offsets)

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        gx, gy = np.meshgrid(self.xs, self.ys)
        return gx.ravel(), gy.ravel()

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.ys), len(self.xs)


@dataclass
class Turbines:
    x: np.ndarray
    y: np.ndarray
    hub_height: np.ndarray
    rotor_radius: np.ndarray

    def __len__(self) -> int:
        return len(self.x)


# --- sun ------------------------------------------------------------------------------------------------------

def _gamma(doy, hour, days_in_year):
    return 2 * np.pi / days_in_year * (np.asarray(doy, dtype=float) - 1 + (np.asarray(hour, dtype=float) - 12) / 24)


def declination(doy, hour=12.0, days_in_year: int = 365):
    """Solar declination [rad] (NOAA / Spencer series, accurate to about 0.03 deg)."""
    g = _gamma(doy, hour, days_in_year)
    return (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))


def equation_of_time(doy, hour=12.0, days_in_year: int = 365):
    """Equation of time [minutes]: apparent solar time minus mean solar time."""
    g = _gamma(doy, hour, days_in_year)
    return 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                     - 0.040849 * np.sin(2 * g))


def hour_angle(times_utc: pd.DatetimeIndex, lon: float):
    """Solar hour angle [rad] (0 at solar noon, positive in the afternoon) for UTC timestamps."""
    t = pd.DatetimeIndex(times_utc)
    doy, hour = t.dayofyear.to_numpy(), (t.hour + t.minute / 60 + t.second / 3600).to_numpy()
    days = np.where(t.is_leap_year, 366, 365)
    solar_minutes = hour * 60 + 4 * lon + equation_of_time(doy, hour, days)
    return np.deg2rad(solar_minutes / 4 - 180.0)


def sun_elevation(times_utc: pd.DatetimeIndex, lat: float, lon: float):
    """Sun elevation [deg] without refraction."""
    t = pd.DatetimeIndex(times_utc)
    hour = (t.hour + t.minute / 60).to_numpy()
    dec = declination(t.dayofyear.to_numpy(), hour, np.where(t.is_leap_year, 366, 365))
    phi = np.deg2rad(lat)
    s = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(hour_angle(t, lon))
    return np.rad2deg(np.arcsin(np.clip(s, -1, 1)))


def limit_hour_angle(dec, lat: float, elevation_deg: float = MIN_ELEVATION_DEG):
    """Hour angle [rad] at which the sun is at ``elevation_deg``; the sun is higher for |h| below it."""
    phi = np.deg2rad(lat)
    c = (np.sin(np.deg2rad(elevation_deg)) - np.sin(phi) * np.sin(dec)) / (np.cos(phi) * np.cos(dec))
    return np.arccos(np.clip(c, -1.0, 1.0))


def daylight_hours(dec, lat: float):
    """Astronomical day length [h] (sun centre above -0.833 deg: refraction and the solar disc)."""
    return 2 * np.rad2deg(limit_hour_angle(dec, lat, -0.833)) / 15.0


# --- geometry -------------------------------------------------------------------------------------------------

def hub_directions(px, py, turbines: Turbines, lat: float, receptor_height: float = RECEPTOR_HEIGHT_M) -> dict:
    """Equatorial coordinates of each hub seen from each receptor, angular rotor radius and horizontal distance.

    Arrays have shape (n_points, n_turbines). Local frame: x east, y north, z up.
    """
    phi = np.deg2rad(lat)
    vx = turbines.x[None, :] - np.asarray(px)[:, None]
    vy = turbines.y[None, :] - np.asarray(py)[:, None]
    vz = np.broadcast_to(turbines.hub_height[None, :] - receptor_height, vx.shape)
    d = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    to_pole = (vy * np.cos(phi) + vz * np.sin(phi)) / d          # sin(dec)
    to_meridian = (-vy * np.sin(phi) + vz * np.cos(phi)) / d     # cos(dec) cos(h)
    to_west = -vx / d                                            # cos(dec) sin(h)
    return {
        "dec": np.arcsin(np.clip(to_pole, -1, 1)),
        "h": np.arctan2(to_west, to_meridian),
        "rho": np.arctan(turbines.rotor_radius[None, :] / d),
        "horizontal": np.hypot(vx, vy),
        "azimuth": np.rad2deg(np.arctan2(vx, vy)) % 360.0,       # direction receptor -> turbine
    }


def shadow_intervals(hub: dict, dec_sun: float, h_limit: float, max_distance) -> tuple[np.ndarray, np.ndarray]:
    """Hour-angle interval [start, end] of shadow for one day, per (receptor, turbine); EMPTY where none."""
    k = ((np.cos(hub["rho"]) - np.sin(dec_sun) * np.sin(hub["dec"]))
         / (np.cos(dec_sun) * np.cos(hub["dec"])))
    half = np.arccos(np.clip(k, -1.0, 1.0))
    start = np.maximum(hub["h"] - half, -h_limit)
    end = np.minimum(hub["h"] + half, h_limit)
    none = (k > 1.0) | (end <= start) | (hub["horizontal"] > max_distance)
    return np.where(none, EMPTY, start), np.where(none, EMPTY, end)


def union_length(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Total length of the union of intervals along the last axis (overlapping shadows count once)."""
    order = np.argsort(start, axis=-1)
    s, e = np.take_along_axis(start, order, -1), np.take_along_axis(end, order, -1)
    prev_end = np.maximum.accumulate(e, axis=-1)
    prev_end = np.concatenate([np.full(s.shape[:-1] + (1,), -np.inf), prev_end[..., :-1]], axis=-1)
    return np.clip(e - np.maximum(s, prev_end), 0.0, None).sum(axis=-1)


def worst_case(grid: Grid, turbines: Turbines, lat: float, year: int = 2021,
               max_distance_d: float = MAX_DISTANCE_D) -> dict:
    """Astronomical worst case on the grid for one calendar year.

    Returns ``hours`` (all turbines, overlaps counted once), ``max_minutes_day`` and ``days`` (days with any
    flicker) as (ny, nx) maps, ``monthly_minutes`` per turbine as (n_turbines, 12, ny, nx) and the hub geometry.
    """
    px, py = grid.points()
    hub = hub_directions(px, py, turbines, lat)
    max_distance = max_distance_d * 2 * turbines.rotor_radius[None, :]
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    n_days = len(days)
    total = np.zeros(len(px))
    max_day = np.zeros(len(px))
    n_with = np.zeros(len(px), dtype=int)
    monthly = np.zeros((len(turbines), 12, len(px)), dtype=np.float32)
    for doy, month in zip(days.dayofyear, days.month):
        dec = float(declination(doy, 12.0, n_days))
        start, end = shadow_intervals(hub, dec, float(limit_hour_angle(dec, lat)), max_distance)
        per_turbine = (end - start) * MINUTES_PER_RADIAN
        monthly[:, month - 1] += per_turbine.T.astype(np.float32)
        day_minutes = union_length(start, end) * MINUTES_PER_RADIAN
        total += day_minutes
        max_day = np.maximum(max_day, day_minutes)
        n_with += day_minutes > 0
    shape = grid.shape
    return {
        "hours": (total / 60).reshape(shape),
        "max_minutes_day": max_day.reshape(shape),
        "days": n_with.reshape(shape),
        "monthly_minutes": monthly.reshape((len(turbines), 12) + shape),
        "azimuth": hub["azimuth"],
    }


def brute_force_minutes(px, py, turbines: Turbines, lat: float, lon: float, times_utc: pd.DatetimeIndex,
                        max_distance_d: float = MAX_DISTANCE_D) -> np.ndarray:
    """Reference implementation: count time steps with the sun behind a sun-facing rotor (per point, turbine).

    Each timestamp counts as one step, so pass a 1-minute index to get minutes.
    """
    t = pd.DatetimeIndex(times_utc)
    hour = (t.hour + t.minute / 60).to_numpy()
    dec = declination(t.dayofyear.to_numpy(), hour, np.where(t.is_leap_year, 366, 365))
    h = hour_angle(t, lon)
    phi = np.deg2rad(lat)
    elev = np.arcsin(np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(h))
    up = elev >= np.deg2rad(MIN_ELEVATION_DEG)
    hub = hub_directions(px, py, turbines, lat)
    cos_theta = (np.sin(dec[up])[:, None, None] * np.sin(hub["dec"])[None]
                 + np.cos(dec[up])[:, None, None] * np.cos(hub["dec"])[None]
                 * np.cos(h[up][:, None, None] - hub["h"][None]))
    inside = cos_theta >= np.cos(hub["rho"])[None]
    inside &= (hub["horizontal"] <= max_distance_d * 2 * turbines.rotor_radius[None, :])[None]
    return inside.sum(axis=0)


def calendar(px: float, py: float, turbines: Turbines, lat: float, lon: float, year: int = 2021,
             max_distance_d: float = MAX_DISTANCE_D) -> pd.DataFrame:
    """Worst-case flicker periods at one point: date, turbine, start and end time (UTC) for every day."""
    hub = hub_directions(np.array([px]), np.array([py]), turbines, lat)
    max_distance = max_distance_d * 2 * turbines.rotor_radius[None, :]
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D", tz="UTC")
    rows = []
    for day in days:
        doy = day.dayofyear
        dec = float(declination(doy, 12.0, len(days)))
        start, end = shadow_intervals(hub, dec, float(limit_hour_angle(dec, lat)), max_distance)
        # apparent solar time = UTC + 4 min/deg x longitude + equation of time; hour angle 0 at 12:00 solar time
        noon_utc = day + pd.Timedelta(minutes=720 - 4 * lon - float(equation_of_time(doy, 12.0, len(days))))
        for t in np.flatnonzero(end[0] > start[0]):
            rows.append({"date": day.date(), "turbine": t,
                         "start": noon_utc + pd.Timedelta(minutes=4 * np.rad2deg(start[0, t])),
                         "end": noon_utc + pd.Timedelta(minutes=4 * np.rad2deg(end[0, t]))})
    return pd.DataFrame(rows, columns=["date", "turbine", "start", "end"])


# --- expected case --------------------------------------------------------------------------------------------

def operating_statistics(nacelle: pd.DataFrame, rpm: pd.DataFrame, lat: float, lon: float,
                         min_rpm: float = 1.0) -> tuple[pd.DataFrame, np.ndarray]:
    """From 10-minute SCADA (timestamp x turbine tables): rotor turning share and nacelle direction histogram.

    Only daylight records (sun at least 3 deg up at the middle of the interval) count. Returns the turning
    share as a (month x turbine) table and the direction histogram of turning records, normalised per turbine
    and month, as an array (n_turbines, 12, 360).
    """
    mid = nacelle.index + pd.Timedelta(minutes=5)
    day = sun_elevation(mid, lat, lon) >= MIN_ELEVATION_DEG
    month = nacelle.index.month.to_numpy()
    turning = rpm.gt(min_rpm).astype(float).where(rpm.notna())
    share = turning[day].groupby(month[day]).mean()
    share.index.name = "month"
    hist = np.zeros((nacelle.shape[1], 12, 360))
    for j, col in enumerate(nacelle.columns):
        sel = day & turning[col].eq(1.0).to_numpy() & nacelle[col].notna().to_numpy()
        direction = np.floor(nacelle[col].to_numpy()[sel]).astype(int) % 360
        np.add.at(hist[j], (month[sel] - 1, direction), 1.0)
    hist /= np.maximum(hist.sum(axis=2, keepdims=True), 1.0)
    return share, hist


def orientation_factor(hist: np.ndarray) -> np.ndarray:
    """Mean |cos(rotor axis - beta)| for every line direction beta (1 deg steps), from direction histograms.

    A rotor whose axis points along the line receptor-turbine shows its full disc; seen edge-on it casts
    almost no shadow. Shape follows ``hist``: (..., 360).
    """
    beta = np.arange(360)
    kernel = np.abs(np.cos(np.deg2rad(beta[:, None] - beta[None, :])))     # [rotor axis, beta]
    return hist @ kernel


def expected_hours(wc: dict, sunshine: np.ndarray, operating: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Expected flicker hours per year on the grid.

    sunshine: (12,) sunshine fraction per month; operating: (n_turbines, 12) rotor turning share;
    orientation: (n_turbines, 12, 360) from ``orientation_factor``. Overlaps between turbines are removed with
    the worst-case ratio of union to sum at each point.
    """
    monthly = wc["monthly_minutes"]                               # (n_t, 12, ny, nx)
    n_t, _, ny, nx = monthly.shape
    beta = np.floor(wc["azimuth"]).astype(int) % 360              # (n_points, n_t)
    minutes = np.zeros(ny * nx)
    for t in range(n_t):
        g = orientation[t][:, beta[:, t]]                         # (12, n_points)
        factor = sunshine[:, None] * operating[t][:, None] * g
        minutes += (monthly[t].reshape(12, -1) * factor).sum(axis=0)
    total = monthly.reshape(n_t, 12, -1).sum(axis=(0, 1))
    overlap = np.divide(wc["hours"].ravel() * 60, total, out=np.ones_like(total, dtype=float), where=total > 0)
    return (minutes * overlap / 60).reshape(ny, nx)
