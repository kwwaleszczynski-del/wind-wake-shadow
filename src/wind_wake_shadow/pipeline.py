"""Analysis steps: farm set-up, wind climate, wake model runs, SCADA comparison and shadow flicker.

Each step returns plain arrays / DataFrames; the CLI writes them to results/, the charts and the web page.

Usage:
    farm = load_farm(Path("data"))
    scada = kelmarsh.load_scada(list(range(2016, 2022)), Path("data"))
    climate, ti = wind_climate_from_scada(scada, farm)
    wake_res = run_wake(farm, climate, ti)
    val = run_validation(scada, farm, climate, ti)
    shade = run_shadow(farm, scada, sunshine_fraction)
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wind_wake_shadow import kelmarsh, shadow, validation, wake
from wind_wake_shadow.layout import bearing_deg, circular_mean_deg, free_stream, to_local
from wind_wake_shadow.turbine import DIAMETER_M, farm_turbines

FLOW_WS = 8.0
FLOW_STEP_M = 20.0
FLOW_HALF_X, FLOW_HALF_Y = 1500.0, 1300.0
MAP_DIRECTIONS = (240.0, 302.0, 143.0, 60.0)       # prevailing SW, along row T3-T5-T6, T1 -> T2, NE
ANIMATION_STEP_DEG = 5.0
SIGMAS = (0.0, 2.5, 5.0, 7.5)
TI_RANGE_WS = (7.5, 8.5)
SHADOW_HALF_WIDTH_M = 1600.0
SHADOW_STEP_M = 10.0
SHADOW_YEAR = 2021
log = logging.getLogger(__name__)


@dataclass
class Farm:
    static: pd.DataFrame
    x: np.ndarray
    y: np.ndarray
    lat: float
    lon: float

    @property
    def hub_heights(self) -> np.ndarray:
        return self.static["hub_height_m"].to_numpy()

    def turbines(self):
        return farm_turbines(self.hub_heights)

    def shadow_turbines(self) -> shadow.Turbines:
        return shadow.Turbines(self.x, self.y, self.hub_heights, self.static["rotor_diameter_m"].to_numpy() / 2)

    @property
    def labels(self) -> list[str]:
        return [f"T{t}" for t in self.static.index]


def load_farm(data_dir: Path) -> Farm:
    static = kelmarsh.read_static(data_dir)
    x, y = to_local(static["lat"], static["lon"])
    return Farm(static, x, y, float(static["lat"].mean()), float(static["lon"].mean()))


def close_pairs(farm: Farm, max_spacing_d: float = 6.0) -> list[tuple[int, int, float, float]]:
    """(upstream, downstream, spacing in D, wind direction that aligns them) for turbine pairs closer than the limit."""
    pairs = []
    for i in range(len(farm.x)):
        for j in range(len(farm.x)):
            dist = np.hypot(farm.x[j] - farm.x[i], farm.y[j] - farm.y[i]) / DIAMETER_M
            if i != j and dist <= max_spacing_d:
                pairs.append((i, j, float(dist), float(bearing_deg(farm.x[j], farm.y[j], farm.x[i], farm.y[i]))))
    return sorted(pairs, key=lambda p: p[2])


# --- wind climate ---------------------------------------------------------------------------------------------

def wind_climate_from_scada(scada: pd.DataFrame, farm: Farm) -> tuple[pd.DataFrame, float]:
    """Sector frequencies + Weibull fits of the free-stream wind speed at 78.5 m, and the ambient TI.

    Free-stream wind speed: mean of the two nacelle anemometers of turbines outside every disturbed sector,
    scaled to 78.5 m with the power-law shear. Ambient TI: median std/mean of those anemometers at 7.5-8.5 m/s.
    """
    after = scada.index.get_level_values("timestamp") >= farm.static["cod"].max()
    s = scada[after]
    ws = s[["ws1", "ws2"]].mean(axis=1, skipna=False).fillna(s["ws"]).unstack("turbine")
    ti = (s["ws_std"] / s["ws"]).unstack("turbine").reindex(ws.index)
    wdir = s["wind_dir"].unstack("turbine").reindex(ws.index)
    wd = circular_mean_deg(wdir.to_numpy(), axis=1)
    ok = np.isfinite(wd)
    free = free_stream(farm.x, farm.y, np.arange(360), DIAMETER_M)[np.round(wd[ok]).astype(int) % 360]
    scale = (wake.REF_HEIGHT_M / farm.hub_heights) ** wake.SHEAR_ALPHA
    ws_free = np.where(free, ws.to_numpy()[ok] * scale, np.nan)
    with warnings.catch_warnings():  # records without any free-stream reading stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        ws_ref = np.nanmean(ws_free, axis=1)
    climate = wake.wind_climate(ws_ref, wd[ok])
    ti_free = np.where(free & (ws_free >= TI_RANGE_WS[0]) & (ws_free <= TI_RANGE_WS[1]), ti.to_numpy()[ok], np.nan)
    ambient_ti = float(np.nanmedian(ti_free))
    log.info("wind climate from %d records: mean free-stream speed %.2f m/s, ambient TI %.3f",
             np.isfinite(ws_ref).sum(), np.nanmean(ws_ref), ambient_ti)
    return climate, ambient_ti


# --- wake models ----------------------------------------------------------------------------------------------

def flow_axes() -> tuple[np.ndarray, np.ndarray]:
    return (np.arange(-FLOW_HALF_X, FLOW_HALF_X + 1, FLOW_STEP_M),
            np.arange(-FLOW_HALF_Y, FLOW_HALF_Y + 1, FLOW_STEP_M))


def run_wake(farm: Farm, climate: pd.DataFrame, ti: float) -> dict:
    """AEP with/without wakes and per-sector losses for every model; flow maps for the figures and animation."""
    wts, types = farm.turbines()
    site = wake.make_site(climate, ti=ti)
    xs, ys = flow_axes()
    anim_wd = np.arange(0.0, 360.0, ANIMATION_STEP_DEG)
    out = {"aep": {}, "sectors": {}, "maps": {}, "xs": xs, "ys": ys, "anim_wd": anim_wd}
    for name in wake.MODELS:
        model = wake.make_model(name, site, wts)
        sim = model(farm.x, farm.y, type=types)
        out["aep"][name] = wake.aep_by_turbine(sim)
        out["sectors"][name] = wake.loss_by_sector(sim)
        out["maps"][name] = wake.flow_field(model, farm.x, farm.y, types, MAP_DIRECTIONS, FLOW_WS, xs, ys)
        log.info("%s: farm wake loss %.2f %%", name, out["aep"][name].loc["farm", "wake_loss_pct"])
    main = wake.make_model("gaussian", site, wts)
    out["animation"] = wake.flow_field(main, farm.x, farm.y, types, anim_wd, FLOW_WS, xs, ys)
    return out


# --- SCADA comparison -----------------------------------------------------------------------------------------

def run_validation(scada: pd.DataFrame, farm: Farm, climate: pd.DataFrame, ti: float) -> dict:
    """Measured vs modelled normalised power per direction bin for every model and direction spread."""
    rec = validation.select_records(scada, farm.static, farm.x, farm.y)
    ws_ref = validation.reference_wind_speed(validation.normalise(rec["power"], rec["free"])["ref"])
    keep = (ws_ref >= validation.WS_RANGE[0]) & (ws_ref <= validation.WS_RANGE[1])
    rec = {k: (v if k == "turbines" else v[keep]) for k, v in rec.items()}
    ws_ref = ws_ref[keep]
    days = pd.DatetimeIndex(rec["time"]).normalize()
    meas = validation.normalise(rec["power"], rec["free"])
    measured = validation.binned_ratio(meas, rec["wd"], days)
    mask = validation.waked_bins(rec["free"], rec["wd"])
    log.info("validation: %d records (%d days), measured farm efficiency %.4f", len(ws_ref), days.nunique(),
             validation.farm_efficiency(meas))

    wts, types = farm.turbines()
    site = wake.make_site(climate, ti=ti)  # only TI matters here: every record is run at its own wd and ws
    rows, curves, efficiency = [], {}, {"measured": validation.farm_efficiency(meas, rec["wd"])}
    for name in wake.MODELS:
        table = wake.power_table(wake.make_model(name, site, wts), farm.x, farm.y, types,
                                 validation.WD_GRID, validation.WS_GRID)
        for sigma in SIGMAS:
            norm = validation.normalise(validation.model_records(table, rec["wd"], ws_ref, sigma), rec["free"])
            c = validation.binned_ratio(norm, rec["wd"])
            curves[(name, sigma)] = c
            if sigma == 0.0:
                efficiency[name] = validation.farm_efficiency(norm, rec["wd"])
            rows.append({"model": name, "sigma_deg": sigma,
                         "wake_loss_pct": 100 * (1 - validation.farm_efficiency(norm)),
                         "rmse_waked": validation.rmse(measured, c, mask)})
    summary = pd.DataFrame(rows)
    measured_loss = 100 * (1 - validation.farm_efficiency(meas))
    summary.insert(3, "measured_wake_loss_pct", measured_loss)
    pairs = close_pairs(farm)
    deficits = validation.pair_deficits(measured, {name: curves[(name, 0.0)] for name in wake.MODELS}, pairs)
    shift = validation.best_direction_shift(measured, curves[("gaussian", 0.0)], shifts=np.arange(-8, 9, 2))
    return {
        "measured": measured, "curves": curves, "summary": summary, "pairs": deficits, "shift": shift,
        "efficiency": pd.DataFrame(efficiency), "waked": mask, "site_factors": meas["c"],
        "n_records": int(len(ws_ref)), "n_days": int(days.nunique()),
        "period": (str(pd.DatetimeIndex(rec["time"]).min().date()), str(pd.DatetimeIndex(rec["time"]).max().date())),
    }


# --- shadow flicker -------------------------------------------------------------------------------------------

def run_shadow(farm: Farm, scada: pd.DataFrame, sunshine_fraction: np.ndarray,
               step_m: float = SHADOW_STEP_M) -> dict:
    """Worst-case and expected flicker maps plus the SCADA operating statistics behind the expected case."""
    tb = farm.shadow_turbines()
    grid = shadow.Grid.around(farm.x, farm.y, SHADOW_HALF_WIDTH_M, step_m)
    wc = shadow.worst_case(grid, tb, farm.lat, SHADOW_YEAR)
    after = scada.index.get_level_values("timestamp") >= farm.static["cod"].max()
    nacelle = scada.loc[after, "nacelle_pos"].unstack("turbine")
    rpm = scada.loc[after, "rotor_rpm"].unstack("turbine").reindex(nacelle.index)
    share, hist = shadow.operating_statistics(nacelle, rpm, farm.lat, farm.lon)
    orientation = shadow.orientation_factor(hist)
    expected = shadow.expected_hours(wc, sunshine_fraction, share.to_numpy().T, orientation)
    log.info("shadow: worst case max %.0f h/yr, expected max %.0f h/yr", wc["hours"].max(), expected.max())
    return {"grid": grid, "worst": wc, "expected": expected, "operating_share": share, "orientation": orientation,
            "sunshine": sunshine_fraction}


def shadow_summary(res: dict, farm: Farm) -> dict:
    """Headline numbers: maxima, areas above common reference values and the furthest distance with flicker."""
    grid, wc, exp = res["grid"], res["worst"], res["expected"]
    cell_km2 = (grid.xs[1] - grid.xs[0]) * (grid.ys[1] - grid.ys[0]) / 1e6
    gx, gy = np.meshgrid(grid.xs, grid.ys)
    nearest = np.min(np.hypot(gx[..., None] - farm.x, gy[..., None] - farm.y), axis=-1)
    return {
        "worst_case_max_h": float(wc["hours"].max()),
        "worst_case_max_min_per_day": float(wc["max_minutes_day"].max()),
        "expected_max_h": float(exp.max()),
        "area_worst_gt_30h_km2": float((wc["hours"] > 30).sum() * cell_km2),
        "area_worst_gt_30min_day_km2": float((wc["max_minutes_day"] > 30).sum() * cell_km2),
        "area_expected_gt_8h_km2": float((exp > 8).sum() * cell_km2),
        "area_any_flicker_km2": float((wc["hours"] > 0).sum() * cell_km2),
        "max_distance_worst_gt_30h_m": float(nearest[wc["hours"] > 30].max()),
        "max_distance_expected_gt_8h_m": float(nearest[exp > 8].max()) if (exp > 8).any() else 0.0,
        "expected_to_worst_ratio": float(exp.sum() / wc["hours"].sum()),
    }
