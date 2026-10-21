"""Readers for the Kelmarsh (Greenbyte) exports and the Met Office station file, on small synthetic files."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from wind_wake_shadow import kelmarsh, sunshine

GREENBYTE = """# This file was exported by Greenbyte at 2022-01-01
# Turbine: Kelmarsh 2
# Time zone: UTC
#
# Date and time,Wind speed (m/s),Wind speed Sensor 1 (m/s),Wind speed Sensor 2 (m/s),Wind direction (°),Nacelle position (°),Power (kW),Blade angle (pitch position) A (°),Blade angle (pitch position) B (°),Blade angle (pitch position) C (°),Rotor speed (RPM),Some other signal
2017-01-01 00:00:00,7.5,7.4,7.6,250.0,248.0,640.0,0.0,0.0,0.3,12.1,1
2017-01-01 00:10:00,NaN,7.0,7.2,251.0,249.0,600.0,0.1,0.1,0.1,12.0,1
"""

STATIC = """﻿Wind Farm,Title,Alternative Title,Identity,Manufacturer,Model,Rated power (kW),Hub Height (m),Rotor Diameter (m),Latitude,Longitude,Elevation (m),Country,Commercial Operations Date
Kelmarsh,Kelmarsh 1,KWF1,X,Senvion,MM92,2050,78.5,92,52.4,-0.95,145.6,UK,15/04/2016
Kelmarsh,Kelmarsh 3,KWF3,Y,Senvion,MM92,2050,68.5,92,52.41,-0.94,153.5,UK,15/04/2016
"""

METOFFICE = """Oxford
Location: 450900E 207200N, Lat 51.761 Lon -1.262, 63 metres amsl
   yyyy  mm   tmax    tmin      af    rain     sun
              degC    degC    days      mm   hours
   1995   1    8.0     2.0       5    80.0    55.5
   1995   2    9.0*    2.5       2    60.0   70.1#
   1995   3   10.0     3.0       1    40.0     ---
   2026   9   20.8    12.2       0    24.2    ---    Provisional
"""


def test_read_turbine_csv_keeps_signals_and_averages_pitch():
    df = kelmarsh.read_turbine_csv(io.StringIO(GREENBYTE), turbine=2)
    assert df.index.names == ["turbine", "timestamp"]
    assert df.index.get_level_values("turbine").unique().tolist() == [2]
    assert str(df.index.get_level_values("timestamp").tz) == "UTC"
    row = df.iloc[0]
    assert row["power_kw"] == pytest.approx(640.0)
    assert row["pitch"] == pytest.approx(0.1)
    assert np.isnan(df.iloc[1]["ws"]) and np.isnan(row["ws_std"])   # missing value and missing column
    assert set(kelmarsh.SIGNALS) <= set(df.columns)


def test_read_static(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / kelmarsh.STATIC_FILE).write_text(STATIC, encoding="utf-8")
    st = kelmarsh.read_static(tmp_path)
    assert st.index.tolist() == [1, 3]
    assert st.loc[3, "hub_height_m"] == 68.5
    assert st.loc[1, "cod"] == pd.Timestamp("2016-04-15", tz="UTC")


def test_load_scada_from_zip_then_cache(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / "Kelmarsh_SCADA_2017_3083.zip", "w") as z:
        z.writestr("Turbine_Data_Kelmarsh_2_2017-01-01_-_2018-01-01_229.csv", GREENBYTE)
        z.writestr("Status_Kelmarsh_2_2017-01-01_-_2018-01-01_229.csv", "ignored")
    first = kelmarsh.load_scada([2017], tmp_path)
    assert (tmp_path / "processed" / "wake_scada_2017.parquet").exists()
    assert list(first.columns) == kelmarsh.SIGNALS
    second = kelmarsh.load_scada([2017], tmp_path)
    pd.testing.assert_frame_equal(first, second)


def test_load_scada_missing_archive(tmp_path):
    with pytest.raises(FileNotFoundError):
        kelmarsh.load_scada([2019], tmp_path)


def test_select_files():
    files = {"Kelmarsh_WT_static.csv": {}, "Kelmarsh_SCADA_2017_3083.zip": {}, "Kelmarsh_SCADA_2018_3084.zip": {}}
    assert kelmarsh.select_files(files, [2018]) == ["Kelmarsh_WT_static.csv", "Kelmarsh_SCADA_2018_3084.zip"]
    with pytest.raises(ValueError):
        kelmarsh.select_files(files, [2016])


def test_parse_met_office_station_file():
    df = sunshine.parse_station(METOFFICE)
    assert df["year"].tolist() == [1995, 1995, 1995, 2026]
    assert df["sun_hours"].iloc[:2].tolist() == [55.5, 70.1]
    assert df["sun_hours"].iloc[2:].isna().all()


def test_sunshine_fraction():
    monthly = pd.DataFrame([{"year": y, "month": m, "sun_hours": 100.0} for y in range(1991, 2021) for m in range(1, 13)])
    frac = sunshine.sunshine_fraction(monthly, lat=52.0)
    assert frac.shape == (12,)
    assert frac == pytest.approx(100.0 / sunshine.possible_hours(52.0))
    assert frac[11] > frac[5]                     # same hours, shorter December days
    with pytest.raises(ValueError):
        sunshine.sunshine_fraction(monthly[monthly["year"] < 2000], lat=52.0)
