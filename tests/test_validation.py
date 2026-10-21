import numpy as np
import pandas as pd
import pytest

from wind_wake_shadow import validation as v


def synthetic(n=20000, seed=0):
    """Three turbines; turbine 3 produces 10 % more (site effect) and loses 50 % for wind from 80-100 deg."""
    rng = np.random.default_rng(seed)
    wd = rng.uniform(0, 360, n)
    base = rng.uniform(300, 1500, n)
    power = np.column_stack([base, base, 1.1 * base])
    waked = (wd > 80) & (wd < 100)
    power[waked, 2] *= 0.5
    free = np.ones((n, 3), dtype=bool)
    free[waked, 2] = False
    return power, free, wd


def test_site_factors_and_normalised_ratio():
    power, free, wd = synthetic()
    norm = v.normalise(power, free)
    assert norm["c"][2] / norm["c"][0] == pytest.approx(1.1, rel=1e-6)
    curves = v.binned_ratio(norm, wd, width=10.0)
    t3 = curves.loc[2]
    assert t3.loc[90.0, "ratio"] == pytest.approx(0.5, rel=1e-6)
    assert t3.loc[180.0, "ratio"] == pytest.approx(1.0, rel=1e-6)
    assert curves.loc[0, "ratio"].to_numpy() == pytest.approx(1.0)
    assert v.farm_efficiency(norm) < 1.0
    eff = v.farm_efficiency(norm, wd, width=10.0)
    assert eff.loc[90.0] == pytest.approx(2.5 / 3, rel=1e-6)


def test_confidence_interval_uses_days():
    power, free, wd = synthetic()
    power = power * np.random.default_rng(3).normal(1, 0.05, power.shape)
    norm = v.normalise(power, free)
    few_days = v.binned_ratio(norm, wd, days=np.arange(len(wd)) % 5, width=10.0)
    every_record = v.binned_ratio(norm, wd, width=10.0)
    assert (few_days["ci95"] > every_record["ci95"]).all()


def test_model_records_bilinear_and_periodic():
    wd_grid, ws_grid = v.WD_GRID, v.WS_GRID
    table = (wd_grid[:, None, None] + 10 * ws_grid[None, :, None]) * np.ones((1, 1, 2))
    out = v.model_records(table, wd=[10.5, 20.25], ws=[5.0, 6.1])
    assert out[:, 0] == pytest.approx([10.5 + 50.0, 20.25 + 61.0])
    wrap = v.model_records(table, wd=[359.5], ws=[5.0])
    assert wrap[0, 0] == pytest.approx(0.5 * 359 + 0.5 * 0 + 50.0)   # interpolates between 359 and 0 deg


def test_direction_smoothing_keeps_mean():
    table = np.zeros((360, 1, 1))
    table[0] = 1.0
    smooth = v.smooth_directions(table, 5.0)
    assert smooth.sum() == pytest.approx(1.0)
    assert smooth[355, 0, 0] == pytest.approx(smooth[5, 0, 0])
    assert v.smooth_directions(table, 0.0) is table


def _curves(centre):
    bins = np.arange(0.0, 360.0, 2.0)
    ratio = 1 - 0.5 * np.exp(-0.5 * (((bins - centre + 180) % 360 - 180) / 6.0) ** 2)
    idx = pd.MultiIndex.from_product([[0], bins], names=["turbine", "bin"])
    return pd.DataFrame({"ratio": ratio, "n": 100}, index=idx)


def test_best_direction_shift_recovers_offset():
    measured, modelled = _curves(104.0), _curves(100.0)
    shift = v.best_direction_shift(measured, modelled, shifts=np.arange(-8, 9, 2))
    assert shift.idxmin() == -4
    assert shift.min() == pytest.approx(0.0, abs=1e-12)


def test_pair_deficits_and_rmse():
    measured, model = _curves(100.0), _curves(102.0)
    pairs = [(1, 0, 3.0, 100.0)]
    out = v.pair_deficits(measured, {"m": model}, pairs, window_deg=10.0)
    assert out.loc[0, "measured"] == pytest.approx(0.5)
    assert out.loc[0, "measured_at_deg"] == pytest.approx(100.0)
    mask = pd.Series(True, index=measured.index)
    assert v.rmse(measured, measured, mask) == 0.0
    assert v.rmse(measured, model, mask) > 0.0


def make_scada(unit: str) -> pd.DataFrame:
    ts = pd.date_range("2017-03-01", periods=6, freq="10min", tz="UTC").as_unit(unit)
    frames = []
    for t in (1, 2):
        df = pd.DataFrame({"power_kw": [500.0, 500, 500, 10, 500, 500], "pitch": [0.0, 0, 0, 0, 12, 0],
                           "wind_dir": [270.0, 270, 270, 270, 270, np.nan]}, index=ts.rename("timestamp"))
        df.insert(0, "turbine", t)
        frames.append(df.set_index("turbine", append=True).swaplevel())
    return pd.concat(frames).sort_index()


@pytest.mark.parametrize("unit", ["ns", "us", "s"])
def test_select_records_filters_and_is_resolution_independent(unit):
    static = pd.DataFrame({"cod": [pd.Timestamp("2017-01-01", tz="UTC")] * 2}, index=[1, 2])
    rec = v.select_records(make_scada(unit), static, x=[0.0, 400.0], y=[0.0, 0.0])
    # record 4: turbine below 20 kW, record 5: pitched out, record 6: no direction
    assert len(rec["wd"]) == 3
    assert pd.DatetimeIndex(rec["time"]).as_unit("ns").asi8.tolist() == \
        pd.date_range("2017-03-01", periods=3, freq="10min", tz="UTC").as_unit("ns").asi8.tolist()
    assert rec["free"].tolist() == [[True, False]] * 3
