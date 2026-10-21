import numpy as np
import pytest

from wind_wake_shadow.turbine import (CUT_OUT, MM92_POWER_KW, RATED_KW, ct_curve, farm_turbines, generic_power_kw,
                                      power_kw, ws_from_power)


def test_catalogue_power_curve():
    assert power_kw(8.0) == pytest.approx(991.2)
    assert power_kw(8.5) == pytest.approx((991.2 + 1355.7) / 2)
    assert power_kw([2.5, 25.5]).tolist() == [0.0, 0.0]
    assert power_kw(CUT_OUT) == pytest.approx(2055.0)


def test_power_inversion_round_trip():
    ws = np.arange(3.5, 10.6, 0.25)
    assert ws_from_power(power_kw(ws)) == pytest.approx(ws)
    assert np.isnan(ws_from_power([0.0, 10.0, 2050.0])).all()


def test_thrust_curve_is_physical_and_consistent():
    ct = ct_curve()
    assert ((ct > 0) & (ct < 1)).all()
    assert ct_curve([6.0])[0] == pytest.approx(0.8, abs=0.01)
    assert ct_curve([20.0])[0] < 0.15
    ws = MM92_POWER_KW.index[MM92_POWER_KW.index >= 3]
    assert np.abs(generic_power_kw(ws) - MM92_POWER_KW.loc[ws].to_numpy()).max() < 0.035 * RATED_KW


def test_farm_turbines_one_type_per_hub_height():
    wts, types = farm_turbines([78.5, 78.5, 68.5, 78.5])
    assert types.tolist() == [0, 0, 1, 0]
    assert wts.hub_height(type=types).tolist() == [78.5, 78.5, 68.5, 78.5]
    assert wts.diameter(type=types) == pytest.approx([92.0] * 4)
