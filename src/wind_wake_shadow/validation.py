"""Wake models against SCADA: normalised power of each turbine as a function of wind direction.

Records: 10-minute intervals in which all turbines run in partial load (power > 20 kW, blade pitch < 5 deg, so
no stops, derating or curtailment) and report a wind direction. The farm direction is the circular mean of the
turbines' wind direction signals (nacelle position + vane); they agree within a few degrees.

Reference: for each direction the free-stream turbines follow from the IEC 61400-12-1 Annex A disturbed
sectors. Each turbine's power is first divided by a site factor (its power relative to the other free turbines
when it is itself free), which removes hub height, terrain and power curve differences; the reference power is
the mean normalised power of the free turbines and the reference wind speed its inverse on the catalogue curve.

Normalised power of turbine i in a direction bin = sum(P_i / c_i) / sum(P_ref) over the records in the bin: 1
when free, < 1 in a wake. The models are run for every record at its reference wind speed and direction and
processed in exactly the same way, optionally averaged over a Gaussian spread of direction (sigma) to mimic
10-minute averaging and wake meandering.

Usage:
    rec = select_records(scada, static, x, y)
    meas = normalise(rec["power"], rec["free"])
    ws_ref = reference_wind_speed(meas["ref"])
    measured = binned_ratio(meas, rec["wd"], days=pd.DatetimeIndex(rec["time"]).normalize())
    table = wake.power_table(model, x, y, types, WD_GRID, WS_GRID)
    modelled = binned_ratio(normalise(model_records(table, rec["wd"], ws_ref), rec["free"]), rec["wd"])
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from wind_wake_shadow.layout import circular_mean_deg, free_stream
from wind_wake_shadow.turbine import DIAMETER_M, ws_from_power

MIN_POWER_KW = 20.0
MAX_PITCH_DEG = 5.0
WS_RANGE = (5.0, 10.0)
WD_GRID = np.arange(360.0)
WS_GRID = np.arange(4.0, 11.01, 0.25)
BIN_DEG = 2.0
MIN_RECORDS = 30


def select_records(scada: pd.DataFrame, static: pd.DataFrame, x, y) -> dict:
    """Wide arrays of the records where every turbine runs in partial load: power, direction, free-stream mask."""
    power = scada["power_kw"].unstack("turbine").sort_index()
    pitch = scada["pitch"].unstack("turbine").reindex(power.index)
    wdir = scada["wind_dir"].unstack("turbine").reindex(power.index)
    after_cod = power.index >= static["cod"].max()
    ok = (after_cod & (power > MIN_POWER_KW).all(axis=1) & (pitch < MAX_PITCH_DEG).all(axis=1)
          & wdir.notna().all(axis=1))
    wd = circular_mean_deg(wdir[ok].to_numpy(), axis=1)
    free_lut = free_stream(x, y, WD_GRID, DIAMETER_M)
    return {
        "time": power.index[ok],
        "turbines": power.columns.to_numpy(),
        "power": power[ok].to_numpy(dtype=float),
        "wd": wd,
        "free": free_lut[np.round(wd).astype(int) % 360],
    }


def site_factors(power: np.ndarray, free: np.ndarray, max_iter: int = 100, tol: float = 1e-10) -> np.ndarray:
    """Per turbine: its power relative to the other free turbines, over records where it is free (mean 1).

    The other turbines carry their own site factors, so they are divided by theirs first; the factors are
    iterated to a fixed point.
    """
    n = power.shape[1]
    c = np.ones(n)
    for _ in range(max_iter):
        p = power / c
        new = np.ones(n)
        for i in range(n):
            others = free.copy()
            others[:, i] = False
            sel = free[:, i] & others.any(axis=1)
            ref = np.where(others[sel], p[sel], 0.0).sum(axis=1) / others[sel].sum(axis=1)
            new[i] = power[sel, i].sum() / ref.sum()
        new /= new.mean()
        if np.max(np.abs(new - c)) < tol:
            return new
        c = new
    return c


def normalise(power: np.ndarray, free: np.ndarray) -> dict:
    """Site-factor corrected power and the free-stream reference power of each record."""
    c = site_factors(power, free)
    p = power / c
    ref = np.where(free, p, 0.0).sum(axis=1) / free.sum(axis=1)
    return {"c": c, "p": p, "ref": ref}


def reference_wind_speed(ref_power: np.ndarray) -> np.ndarray:
    return ws_from_power(ref_power)


def smooth_directions(table: np.ndarray, sigma_deg: float) -> np.ndarray:
    """Average a (wd, ...) table over a Gaussian distribution of direction (periodic, 1 deg grid)."""
    if sigma_deg <= 0:
        return table
    return gaussian_filter1d(table, sigma_deg / (WD_GRID[1] - WD_GRID[0]), axis=0, mode="wrap")


def model_records(table: np.ndarray, wd, ws, sigma_deg: float = 0.0) -> np.ndarray:
    """Model power per record by bilinear interpolation of a (WD_GRID, WS_GRID, turbine) table."""
    t = smooth_directions(table, sigma_deg)
    wd = np.asarray(wd, dtype=float) % 360.0
    ws = np.clip(np.asarray(ws, dtype=float), WS_GRID[0], WS_GRID[-1])
    i0 = np.floor(wd).astype(int) % 360
    i1 = (i0 + 1) % 360
    fw = (wd - np.floor(wd))[:, None]
    j = np.clip(np.searchsorted(WS_GRID, ws, side="right") - 1, 0, len(WS_GRID) - 2)
    fs = ((ws - WS_GRID[j]) / (WS_GRID[j + 1] - WS_GRID[j]))[:, None]
    lo = t[i0, j] * (1 - fw) + t[i1, j] * fw
    hi = t[i0, j + 1] * (1 - fw) + t[i1, j + 1] * fw
    return lo * (1 - fs) + hi * fs


def direction_bins(wd, width: float = BIN_DEG) -> np.ndarray:
    return (np.round(np.asarray(wd) / width) * width) % 360.0


def binned_ratio(norm: dict, wd, days=None, width: float = BIN_DEG) -> pd.DataFrame:
    """Normalised power per direction bin and turbine (ratio of sums) with a 95 % interval and counts.

    The interval uses the spread of the per-record ratios and the number of distinct days in the bin as the
    sample size, since 10-minute records within a day are strongly correlated.
    """
    b = direction_bins(wd, width)
    p, ref = norm["p"], norm["ref"]
    rows = []
    for i in range(p.shape[1]):
        df = pd.DataFrame({"bin": b, "p": p[:, i], "ref": ref, "r": p[:, i] / ref})
        if days is not None:
            df["day"] = days
        g = df.groupby("bin")
        out = pd.DataFrame({"ratio": g["p"].sum() / g["ref"].sum(), "n": g.size(), "std": g["r"].std()})
        n_eff = g["day"].nunique() if days is not None else out["n"]
        out["ci95"] = 1.96 * out["std"] / np.sqrt(n_eff)
        out["turbine"] = i
        rows.append(out.drop(columns="std"))
    return pd.concat(rows).reset_index().set_index(["turbine", "bin"]).sort_index()


def farm_efficiency(norm: dict, wd=None, width: float = BIN_DEG):
    """Sum of normalised power over all turbines / (n turbines x reference power); per bin if ``wd`` is given."""
    total, ref = norm["p"].sum(axis=1), norm["ref"] * norm["p"].shape[1]
    if wd is None:
        return total.sum() / ref.sum()
    b = direction_bins(wd, width)
    return pd.Series(total).groupby(b).sum() / pd.Series(ref).groupby(b).sum()


def waked_bins(free: np.ndarray, wd, width: float = BIN_DEG) -> pd.DataFrame:
    """(turbine, bin) pairs where the turbine is inside a disturbed sector for most records."""
    b = direction_bins(wd, width)
    rows = [pd.Series(~free[:, i]).groupby(b).mean().rename("waked_share").to_frame().assign(turbine=i)
            for i in range(free.shape[1])]
    out = pd.concat(rows).reset_index(names="bin").set_index(["turbine", "bin"]).sort_index()
    return out["waked_share"] > 0.5


def rmse(measured: pd.DataFrame, modelled: pd.DataFrame, mask: pd.Series, min_records: int = MIN_RECORDS) -> float:
    """Root mean square difference of normalised power over the (turbine, bin) pairs selected by ``mask``."""
    sel = mask.reindex(measured.index, fill_value=False) & (measured["n"] >= min_records)
    d = measured.loc[sel, "ratio"] - modelled.loc[sel[sel].index, "ratio"]
    return float(np.sqrt(np.mean(d.to_numpy() ** 2)))


def best_direction_shift(measured: pd.DataFrame, modelled: pd.DataFrame, shifts=np.arange(-10, 11, 2),
                         min_records: int = MIN_RECORDS) -> pd.Series:
    """RMSE between measured and modelled curves when the measured directions are shifted by each offset.

    A minimum away from 0 would point to a north calibration error of the SCADA wind direction.
    """
    out = {}
    for s in shifts:
        shifted = measured.copy()
        shifted.index = pd.MultiIndex.from_arrays([shifted.index.get_level_values("turbine"),
                                                   (shifted.index.get_level_values("bin") + s) % 360.0],
                                                  names=["turbine", "bin"])
        common = shifted.index.intersection(modelled.index)
        m = shifted.loc[common]
        m = m[m["n"] >= min_records]
        out[int(s)] = float(np.sqrt(np.mean((m["ratio"] - modelled.loc[m.index, "ratio"]) ** 2)))
    return pd.Series(out, name="rmse").rename_axis("shift_deg")


def pair_deficits(measured: pd.DataFrame, modelled: dict[str, pd.DataFrame], pairs, window_deg: float = 15.0,
                  min_records: int = MIN_RECORDS) -> pd.DataFrame:
    """Deepest normalised power of the downstream turbine within +-window of each pair's alignment direction.

    ``pairs`` holds (upstream index, downstream index, distance in D, wind direction of full alignment).
    """
    rows = []
    for up, down, dist_d, wd0 in pairs:
        row = {"upstream": up + 1, "downstream": down + 1, "spacing_D": round(dist_d, 1), "wd_deg": round(wd0)}
        for name, curves in {"measured": measured, **modelled}.items():
            c = curves.loc[down]
            c = c[(np.abs((c.index - wd0 + 180) % 360 - 180) <= window_deg)]
            if "n" in c:
                c = c[c["n"] >= min_records]
            row[name] = float(c["ratio"].min()) if len(c) else np.nan
            if name == "measured" and len(c):
                row["measured_at_deg"] = float(c["ratio"].idxmin())
        rows.append(row)
    return pd.DataFrame(rows)
