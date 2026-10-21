"""Senvion MM92 for PyWake: catalogue power curve and a generic thrust curve.

Power: the manufacturer's catalogue curve at 1.225 kg/m3 (Open Energy Platform turbine library, as distributed
with windpowerlib; originally windenergie-im-binnenland.de). It is a sales curve, not a warranted one.

Thrust: no Ct curve for the MM92 is public. Ct comes from PyWake's GenericWindTurbine (DTU) for the same rotor
and rating: Ct = 0.8 in partial load, then the momentum-theory value that delivers rated power. The generic
model's own power curve stays within 70 kW (3.5 % of rated) of the catalogue curve (see ``generic_power_kw``), so the thrust
curve is consistent with the power curve used.

Usage:
    wts, types = farm_turbines(hub_heights=[78.5, 78.5, 68.5])   # PyWake WindTurbines + type index per turbine
    power_kw(8.0)                                                 # 991 kW
    ws_from_power(991.2)                                          # 8.0 m/s (partial load only)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from py_wake.wind_turbines import WindTurbine, WindTurbines
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

NAME = "Senvion MM92"
DIAMETER_M = 92.0
RATED_KW = 2050.0
CUT_IN, CUT_OUT = 3.0, 25.0
MM92_POWER_KW = pd.Series(
    [0.0, 22.0, 93.1, 207.2, 388.2, 642.7, 991.2, 1355.7, 1780.0, 2012.6, 2048.0] + [2055.0] * 13,
    index=np.arange(2.0, 26.0),
    name="power_kw",
).rename_axis("ws")


def _generic(hub_height: float = 78.5) -> GenericWindTurbine:
    return GenericWindTurbine(NAME, DIAMETER_M, hub_height, RATED_KW)


def ct_curve(ws=MM92_POWER_KW.index) -> np.ndarray:
    """Thrust coefficient of the generic DTU turbine with the MM92 rotor and rating."""
    return np.asarray(_generic().ct(np.asarray(ws, dtype=float)), dtype=float)


def generic_power_kw(ws=MM92_POWER_KW.index) -> np.ndarray:
    """Power curve of the generic DTU turbine, only used to check consistency with the catalogue curve."""
    return np.asarray(_generic().power(np.asarray(ws, dtype=float)), dtype=float) / 1e3


def power_kw(ws):
    """Catalogue power at the given wind speed(s), linear between the 1 m/s points, 0 outside 3-25 m/s."""
    ws = np.asarray(ws, dtype=float)
    p = np.interp(ws, MM92_POWER_KW.index, MM92_POWER_KW.values)
    return np.where((ws >= CUT_IN) & (ws <= CUT_OUT), p, 0.0)


def ws_from_power(p_kw):
    """Invert the catalogue curve in partial load (3-11 m/s, where it rises steeply); NaN outside that range."""
    curve = MM92_POWER_KW.loc[CUT_IN:11.0]
    p = np.asarray(p_kw, dtype=float)
    ws = np.interp(p, curve.values, curve.index)
    return np.where((p > curve.values[0]) & (p < curve.values[-1]), ws, np.nan)


def mm92(hub_height: float) -> WindTurbine:
    ws = MM92_POWER_KW.index.to_numpy(dtype=float)
    return WindTurbine(
        name=f"MM92 hub {hub_height:g} m", diameter=DIAMETER_M, hub_height=hub_height,
        powerCtFunction=PowerCtTabular(ws, MM92_POWER_KW.to_numpy(), "kW", ct_curve(ws),
                                       ws_cutin=CUT_IN, ws_cutout=CUT_OUT),
    )


def farm_turbines(hub_heights) -> tuple[WindTurbines, np.ndarray]:
    """One PyWake turbine type per distinct hub height, and the type index of each turbine."""
    heights = sorted(set(float(h) for h in hub_heights), reverse=True)
    wts = WindTurbines.from_WindTurbine_lst([mm92(h) for h in heights])
    types = np.array([heights.index(float(h)) for h in hub_heights])
    return wts, types
