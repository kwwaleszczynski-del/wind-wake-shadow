import numpy as np
import pandas as pd
import pytest

from wind_wake_shadow import wake
from wind_wake_shadow.turbine import farm_turbines

X, Y = np.array([0.0, 400.0]), np.array([0.0, 0.0])   # two turbines on an east-west line, 4.3 D apart
UNIFORM = pd.DataFrame({"freq": np.full(12, 1 / 12), "A": 8.0, "k": 2.0}, index=np.arange(0, 360, 30))


@pytest.fixture(scope="module")
def setup():
    wts, types = farm_turbines([78.5, 78.5])
    return wake.make_site(UNIFORM, ti=0.1), wts, types


@pytest.mark.parametrize("name", list(wake.MODELS))
def test_downstream_turbine_loses_power(setup, name):
    site, wts, types = setup
    model = wake.make_model(name, site, wts)
    p = wake.power_table(model, X, Y, types, wd=[270.0, 0.0], ws=[8.0])
    assert p.shape == (2, 1, 2)
    assert p[0, 0, 0] == pytest.approx(991.2, rel=1e-3)          # upstream turbine sees the free stream
    assert p[0, 0, 1] < 0.8 * p[0, 0, 0]                          # wind from the west: turbine 2 waked
    assert p[1, 0, 0] == pytest.approx(p[1, 0, 1], rel=1e-6)      # wind from the north: no interaction


def test_unknown_model(setup):
    site, wts, _ = setup
    with pytest.raises(ValueError):
        wake.make_model("nope", site, wts)


def test_aep_and_sector_losses(setup):
    site, wts, types = setup
    sim = wake.make_model("gaussian", site, wts)(X, Y, type=types)
    aep = wake.aep_by_turbine(sim)
    assert list(aep.index) == [1, 2, "farm"]
    assert (aep["net_gwh"] <= aep["gross_gwh"] + 1e-9).all()
    assert aep.loc["farm", "gross_gwh"] == pytest.approx(aep.loc[[1, 2], "gross_gwh"].sum())
    sectors = wake.loss_by_sector(sim, width=10.0)
    assert len(sectors) == 36
    assert sectors["energy_share"].sum() == pytest.approx(1.0)
    # the line is east-west, so the 90 and 270 deg sectors lose most
    assert set(sectors["wake_loss_pct"].nlargest(2).index) == {90, 270}


def test_flow_field_is_free_upstream_and_slower_behind(setup):
    site, wts, types = setup
    model = wake.make_model("jensen", site, wts)
    xs, ys = np.arange(-600.0, 1201.0, 50.0), np.arange(-300.0, 301.0, 50.0)
    f = wake.flow_field(model, X, Y, types, wd=[270.0], ws=8.0, xs=xs, ys=ys)
    assert f.shape == (1, len(ys), len(xs))
    assert f[0, :, xs < -100].max() == pytest.approx(1.0)                    # upstream of everything
    behind = f[0, ys == 0.0, (xs > 600) & (xs < 1000)]
    assert behind.max() < 0.95


def test_wind_climate_recovers_weibull():
    rng = np.random.default_rng(1)
    n = 40000
    wd = rng.uniform(0, 360, n)
    ws = np.where((wd > 195) & (wd < 285), 9.0, 6.0) * rng.weibull(2.2, n)
    clim = wake.wind_climate(ws, wd, n_sectors=12, min_ws=0.0)
    assert clim["freq"].sum() == pytest.approx(1.0)
    assert clim.loc[240.0, "A"] == pytest.approx(9.0, rel=0.05)
    assert clim.loc[60.0, "A"] == pytest.approx(6.0, rel=0.05)
    assert clim["k"].to_numpy() == pytest.approx(2.2, rel=0.08)
