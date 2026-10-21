import numpy as np
import pytest

from wind_wake_shadow.layout import (EARTH_RADIUS_M, angle_diff, bearing_deg, circular_mean_deg,
                                     disturbed_sector_deg, free_stream, to_local)


def test_to_local_distances():
    dlat = np.rad2deg(1000.0 / EARTH_RADIUS_M)
    x, y = to_local([52.0, 52.0 + dlat], [-1.0, -1.0], lat0=52.0, lon0=-1.0)
    assert x == pytest.approx([0.0, 0.0], abs=1e-6)
    assert y == pytest.approx([0.0, 1000.0], rel=1e-9)


def test_bearing_and_angle_diff():
    assert bearing_deg(0, 0, 0, 10) == pytest.approx(0.0)
    assert bearing_deg(0, 0, 10, 0) == pytest.approx(90.0)
    assert bearing_deg(0, 0, -10, -10) == pytest.approx(225.0)
    assert angle_diff(10, 350) == pytest.approx(20.0)
    assert angle_diff(350, 10) == pytest.approx(-20.0)


def test_circular_mean_wraps_and_ignores_nan():
    assert circular_mean_deg([350.0, 10.0]) % 360 == pytest.approx(0.0, abs=1e-9)
    rows = circular_mean_deg(np.array([[350.0, 10.0, np.nan], [80.0, 100.0, 90.0], [np.nan] * 3]), axis=1)
    assert rows[0] % 360 == pytest.approx(0.0, abs=1e-9)
    assert rows[1] == pytest.approx(90.0)
    assert np.isnan(rows[2])


def test_iec_disturbed_sector():
    # IEC 61400-12-1 Annex A: alpha = 1.3 atan(2.5 D / L + 0.15) + 10
    assert disturbed_sector_deg(2 * 92.0, 92.0) == pytest.approx(1.3 * np.rad2deg(np.arctan(1.4)) + 10)
    assert disturbed_sector_deg(20 * 92.0, 92.0) < disturbed_sector_deg(4 * 92.0, 92.0)


def test_free_stream_two_turbines():
    x, y = np.array([0.0, 400.0]), np.array([0.0, 0.0])  # turbine 1 west of turbine 2, 4.3 D apart
    free = free_stream(x, y, [270.0, 90.0, 0.0, 300.0], 92.0)
    assert free[0].tolist() == [True, False]    # wind from west: turbine 2 in the wake
    assert free[1].tolist() == [False, True]    # wind from east
    assert free[2].tolist() == [True, True]     # across the line
    assert free[3].tolist() == [True, True]     # 30 deg off the line is outside the +-26 deg sector


def test_free_stream_ignores_far_neighbours():
    x, y = np.array([0.0, 30 * 92.0]), np.array([0.0, 0.0])
    assert free_stream(x, y, [270.0], 92.0).all()
