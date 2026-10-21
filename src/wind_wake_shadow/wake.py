"""Engineering wake models (PyWake) for the Kelmarsh layout: site climate, AEP with and without wakes, flow maps.

Three model families are compared:

    jensen     Jensen / N.O. Jensen (1983) top-hat wake, expansion k = 0.075 (the usual onshore value)
    gaussian   Niayifar & Porte-Agel (2016) Gaussian wake, expansion tied to local turbulence intensity
               (Crespo-Hernandez added turbulence); the main model for the maps
    turbopark  TurbOPark (Nygaard et al. 2022), Gaussian wake with turbulence-driven expansion

The wind climate (direction frequencies and a Weibull distribution per sector) comes from SCADA: the wind speed
of turbines that are in free stream for the direction (IEC 61400-12-1 disturbed sectors), at 78.5 m.

Usage:
    site = make_site(climate, ti=0.10)
    model = make_model("gaussian", site, wts)
    sim = model(x, y, type=types)
    table = aep_by_turbine(sim)               # gross / net AEP and wake loss per turbine
    field = flow_field(model, x, y, types, wd=[240], ws=8.0, xs=xs, ys=ys, h=78.5)   # U / U_free
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from py_wake import HorizontalGrid
from py_wake.literature import Jensen_1983, Nygaard_2022
from py_wake.literature.gaussian_models import Niayifar_PorteAgel_2016
from py_wake.site import UniformWeibullSite
from py_wake.site.shear import PowerShear
from scipy import stats

MODELS = {
    "jensen": "Jensen (k = 0.075)",
    "gaussian": "Niayifar & Porte-Agel Gaussian",
    "turbopark": "TurbOPark",
}
JENSEN_K = 0.075
SHEAR_ALPHA = 0.2   # power-law shear exponent, typical for farmland with hedges and woods; not measured on site
REF_HEIGHT_M = 78.5


def make_model(name: str, site, wts):
    if name == "jensen":
        return Jensen_1983(site, wts, k=JENSEN_K)
    if name == "gaussian":
        return Niayifar_PorteAgel_2016(site, wts)
    if name == "turbopark":
        return Nygaard_2022(site, wts)
    raise ValueError(f"unknown wake model {name!r}; choose from {list(MODELS)}")


def wind_climate(ws, wd, n_sectors: int = 12, min_ws: float = 0.5) -> pd.DataFrame:
    """Direction frequency and Weibull A, k per sector from paired wind speed / direction samples."""
    ws, wd = np.asarray(ws, dtype=float), np.asarray(wd, dtype=float) % 360.0
    ok = np.isfinite(ws) & np.isfinite(wd) & (ws >= min_ws)
    ws, wd = ws[ok], wd[ok]
    width = 360.0 / n_sectors
    sector = np.floor(((wd + width / 2) % 360.0) / width).astype(int)
    rows = []
    for s in range(n_sectors):
        v = ws[sector == s]
        k, _, a = stats.weibull_min.fit(v, floc=0.0)
        rows.append({"sector_deg": s * width, "freq": len(v) / len(ws), "A": a, "k": k,
                     "mean_ws": v.mean(), "n": len(v)})
    return pd.DataFrame(rows).set_index("sector_deg")


def make_site(climate: pd.DataFrame, ti: float, shear_alpha: float = SHEAR_ALPHA,
              h_ref: float = REF_HEIGHT_M) -> UniformWeibullSite:
    return UniformWeibullSite(p_wd=climate["freq"].to_numpy(), a=climate["A"].to_numpy(),
                              k=climate["k"].to_numpy(), ti=ti, shear=PowerShear(h_ref=h_ref, alpha=shear_alpha))


def aep_by_turbine(sim) -> pd.DataFrame:
    """Gross (no wake) and net AEP in GWh and the wake loss per turbine, plus a farm total row."""
    gross = sim.aep(with_wake_loss=False).sum(["wd", "ws"]).values
    net = sim.aep().sum(["wd", "ws"]).values
    out = pd.DataFrame({"gross_gwh": gross, "net_gwh": net}, index=pd.Index(sim.wt.values + 1, name="turbine"))
    out.loc["farm"] = out.sum()
    out["wake_loss_pct"] = 100 * (1 - out["net_gwh"] / out["gross_gwh"])
    return out


def loss_by_sector(sim, width: float = 10.0) -> pd.DataFrame:
    """Farm energy share and wake loss per direction sector (sectors centred on multiples of ``width``)."""
    gross = sim.aep(with_wake_loss=False).sum(["wt", "ws"]).to_series()
    net = sim.aep().sum(["wt", "ws"]).to_series()
    sector = (np.floor(((gross.index.to_numpy() + width / 2) % 360.0) / width) * width).astype(int)
    g, n = gross.groupby(sector).sum(), net.groupby(sector).sum()
    out = pd.DataFrame({"gross_gwh": g, "net_gwh": n})
    out.index.name = "sector_deg"
    out["energy_share"] = out["gross_gwh"] / out["gross_gwh"].sum()
    out["wake_loss_pct"] = 100 * (1 - out["net_gwh"] / out["gross_gwh"])
    return out


def power_table(model, x, y, types, wd, ws) -> np.ndarray:
    """Power [kW] of every turbine for each (wd, ws) pair of the grid, shape (len(wd), len(ws), n_turbines)."""
    sim = model(x, y, type=types, wd=np.asarray(wd, dtype=float), ws=np.asarray(ws, dtype=float))
    return sim.Power.transpose("wd", "ws", "wt").values / 1e3


def flow_field(model, x, y, types, wd, ws: float, xs, ys, h: float = REF_HEIGHT_M) -> np.ndarray:
    """Effective wind speed relative to the free stream at height ``h``, shape (len(wd), len(ys), len(xs))."""
    wd = np.atleast_1d(np.asarray(wd, dtype=float))
    sim = model(x, y, type=types, wd=wd, ws=[ws])
    fm = sim.flow_map(HorizontalGrid(x=np.asarray(xs), y=np.asarray(ys), h=h), wd=wd, ws=[ws])
    ratio = (fm.WS_eff / fm.WS).squeeze(["h", "ws"], drop=True)
    return ratio.transpose("wd", "y", "x").values
