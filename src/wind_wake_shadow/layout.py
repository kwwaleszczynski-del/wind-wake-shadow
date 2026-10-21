"""Turbine positions in a local east/north frame and the geometry used by the wake and shadow models.

An equirectangular projection around the farm centre is accurate to well under a metre over a few kilometres.
Directions follow the meteorological convention: degrees clockwise from north, wind direction = where the wind
comes from.

Usage:
    x, y = to_local(static["lat"], static["lon"])           # metres east / north of the farm centre
    bearing_deg(x[0], y[0], x[1], y[1])                      # direction from turbine 1 to turbine 2
    free = free_stream(x, y, wd=np.arange(360), diameter_m=92.0)   # (360, n_turbines) bool, IEC 61400-12-1
"""
from __future__ import annotations

import warnings

import numpy as np

EARTH_RADIUS_M = 6371008.8
MAX_DISTURBANCE_D = 20.0  # IEC 61400-12-1 Annex A: neighbours further away than 20 D are ignored


def to_local(lat, lon, lat0: float | None = None, lon0: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Metres east and north of (lat0, lon0), by default the mean position."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    lat0 = lat.mean() if lat0 is None else lat0
    lon0 = lon.mean() if lon0 is None else lon0
    x = np.deg2rad(lon - lon0) * EARTH_RADIUS_M * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * EARTH_RADIUS_M
    return x, y


def bearing_deg(x0, y0, x1, y1):
    """Direction from point 0 to point 1, degrees clockwise from north."""
    return np.rad2deg(np.arctan2(np.asarray(x1) - x0, np.asarray(y1) - y0)) % 360.0


def angle_diff(a, b):
    """Signed smallest difference a - b in degrees, in [-180, 180)."""
    return (np.asarray(a) - b + 180.0) % 360.0 - 180.0


def circular_mean_deg(angles, axis=None):
    """Mean direction in degrees [0, 360), ignoring NaN."""
    rad = np.deg2rad(np.asarray(angles, dtype=float))
    with warnings.catch_warnings():  # all-NaN rows give NaN, which is what callers expect
        warnings.simplefilter("ignore", RuntimeWarning)
        s, c = np.nanmean(np.sin(rad), axis=axis), np.nanmean(np.cos(rad), axis=axis)
    return np.rad2deg(np.arctan2(s, c)) % 360.0


def disturbed_sector_deg(distance_m, diameter_m: float):
    """Full width of the sector disturbed by an operating neighbour (IEC 61400-12-1 Annex A).

    alpha = 1.3 arctan(2.5 D / L + 0.15) + 10 degrees, for a neighbour of rotor diameter D at distance L.
    """
    return 1.3 * np.rad2deg(np.arctan(2.5 * diameter_m / np.asarray(distance_m, dtype=float) + 0.15)) + 10.0


def free_stream(x, y, wd, diameter_m: float) -> np.ndarray:
    """True where a turbine sees undisturbed inflow: no neighbour within 20 D lies in its disturbed sector.

    Returns a (len(wd), n_turbines) array. The most upstream turbine is always free, so every direction has at
    least one reference turbine.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    wd = np.atleast_1d(np.asarray(wd, dtype=float))
    dist = np.hypot(x[None, :] - x[:, None], y[None, :] - y[:, None])        # [i, j]: from i to j
    bearing = bearing_deg(x[:, None], y[:, None], x[None, :], y[None, :])
    half = disturbed_sector_deg(np.where(dist > 0, dist, np.inf), diameter_m) / 2.0
    near = (dist > 0) & (dist <= MAX_DISTURBANCE_D * diameter_m)
    inside = np.abs(angle_diff(wd[:, None, None], bearing[None])) < half[None]   # [wd, i, j]
    return ~(inside & near[None]).any(axis=2)
