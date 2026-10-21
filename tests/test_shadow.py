import numpy as np
import pandas as pd
import pytest

from wind_wake_shadow import shadow as sh

LAT, LON = 52.4, -0.94
ONE = sh.Turbines(np.array([0.0]), np.array([0.0]), np.array([78.5]), np.array([46.0]))
TWO = sh.Turbines(np.array([0.0, 300.0]), np.array([0.0, 150.0]), np.array([78.5, 68.5]), np.array([46.0, 46.0]))


def point(x, y):
    return sh.Grid(np.array([float(x)]), np.array([float(y)]))


def test_declination_and_daylight():
    assert np.rad2deg(sh.declination(172)) == pytest.approx(23.44, abs=0.15)
    assert np.rad2deg(sh.declination(355)) == pytest.approx(-23.44, abs=0.15)
    assert np.rad2deg(sh.declination(80)) == pytest.approx(0.0, abs=0.6)
    days = np.arange(1, 366)
    total = sh.daylight_hours(sh.declination(days), LAT).sum()
    assert 4380 < total < 4520   # half the year plus refraction and the solar disc
    assert sh.daylight_hours(sh.declination(172), LAT) == pytest.approx(16.6, abs=0.3)


def test_sun_elevation_noon_and_resolution_independence():
    t = pd.DatetimeIndex(["2021-06-21 12:04", "2021-12-21 12:04"], tz="UTC")
    elev = sh.sun_elevation(t, LAT, LON)
    assert elev == pytest.approx([90 - LAT + 23.44, 90 - LAT - 23.44], abs=0.5)
    for unit in ("s", "us", "ns"):
        assert sh.sun_elevation(t.as_unit(unit), LAT, LON) == pytest.approx(elev)


def test_union_length():
    start = np.array([[0.0, 0.5, 2.0, sh.EMPTY]])
    end = np.array([[1.0, 1.5, 2.5, sh.EMPTY]])
    assert sh.union_length(start, end)[0] == pytest.approx(2.0)
    nested = sh.union_length(np.array([[0.0, 0.2]]), np.array([[1.0, 0.4]]))
    assert nested[0] == pytest.approx(1.0)


def test_no_shadow_south_of_a_turbine_shadow_north_in_winter():
    south = sh.worst_case(point(0, -300), ONE, LAT)
    north = sh.worst_case(point(0, 300), ONE, LAT)
    assert south["hours"][0, 0] == 0.0
    monthly = north["monthly_minutes"][0, :, 0, 0]
    assert monthly[[0, 11]].sum() > 0          # around noon in December and January
    assert monthly[4:8].sum() == 0             # the summer sun is too high
    assert sh.worst_case(point(0, 2000), ONE, LAT)["hours"][0, 0] == 0.0   # beyond 10 rotor diameters


@pytest.mark.parametrize("px,py", [(-400.0, 150.0), (600.0, 500.0), (150.0, 80.0), (-200.0, -40.0)])
def test_analytic_matches_one_minute_time_stepping(px, py):
    times = pd.date_range("2021-01-01", "2022-01-01", freq="1min", inclusive="left", tz="UTC")
    brute = sh.brute_force_minutes(np.array([px]), np.array([py]), TWO, LAT, LON, times)[0]
    wc = sh.worst_case(point(px, py), TWO, LAT, year=2021)
    analytic = wc["monthly_minutes"].sum(axis=1)[:, 0, 0]
    assert brute.sum() > 0
    assert analytic == pytest.approx(brute, rel=0.01, abs=10)


def test_calendar_adds_up_to_worst_case():
    px, py = 600.0, 500.0
    cal = sh.calendar(px, py, TWO, LAT, LON, year=2021)
    wc = sh.worst_case(point(px, py), TWO, LAT, year=2021)
    hours = (cal["end"] - cal["start"]).dt.total_seconds().sum() / 3600
    assert hours == pytest.approx(wc["monthly_minutes"].sum() / 60, rel=1e-6)
    # receptor north-east of the turbines: shadows come with a south-westerly afternoon / evening sun
    assert (cal["start"].dt.hour >= 12).all()


def test_orientation_factor():
    hist = np.zeros((1, 1, 360))
    hist[0, 0, 0] = 1.0                               # rotor axis always north-south
    g = sh.orientation_factor(hist)[0, 0]
    assert g[0] == pytest.approx(1.0) and g[180] == pytest.approx(1.0)
    assert g[90] == pytest.approx(0.0, abs=1e-12)
    uniform = sh.orientation_factor(np.full((1, 1, 360), 1 / 360))[0, 0]
    assert uniform == pytest.approx(2 / np.pi, rel=1e-3)


def test_expected_hours_scaling():
    grid = sh.Grid(np.array([-400.0, 600.0]), np.array([150.0, 500.0]))
    wc = sh.worst_case(grid, TWO, LAT)
    ones = np.ones((2, 12))
    full = sh.expected_hours(wc, np.ones(12), ones, np.ones((2, 12, 360)))
    assert full == pytest.approx(wc["hours"])         # every factor 1 gives back the worst case
    half = sh.expected_hours(wc, np.full(12, 0.5), ones, np.ones((2, 12, 360)))
    assert half == pytest.approx(0.5 * wc["hours"])


@pytest.mark.parametrize("unit", ["s", "us", "ns"])
def test_operating_statistics(unit):
    ts = pd.date_range("2021-06-01", periods=144, freq="10min", tz="UTC").as_unit(unit)
    nacelle = pd.DataFrame({1: np.full(144, 90.0), 2: np.full(144, 180.0)}, index=ts)
    rpm = pd.DataFrame({1: np.full(144, 12.0), 2: np.r_[np.zeros(72), np.full(72, 12.0)]}, index=ts)
    share, hist = sh.operating_statistics(nacelle, rpm, LAT, LON)
    assert share.loc[6, 1] == pytest.approx(1.0)
    assert 0.3 < share.loc[6, 2] < 0.7                 # stopped all morning, running all afternoon
    assert hist[0, 5, 90] == pytest.approx(1.0) and hist[1, 5, 180] == pytest.approx(1.0)
