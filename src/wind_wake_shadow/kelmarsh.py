"""Open Kelmarsh wind farm data: turbine metadata and 10-minute SCADA.

Dataset: "Kelmarsh wind farm data", C. Plumley and R. Takeuchi, Cubico Sustainable Investments,
https://doi.org/10.5281/zenodo.16807551 (concept DOI 10.5281/zenodo.5841833), licence CC-BY-4.0.
Six Senvion MM92 turbines (2.05 MW, 92 m rotor) in Northamptonshire, UK.

Files live in ``<data_dir>/raw`` (Zenodo downloads); the signals used here are cached per year as Parquet in
``<data_dir>/processed``. A cache written by the companion wind-power-curve-kelmarsh project
(``scada_<year>.parquet``, a superset of these signals) is reused, so both projects can share one data folder.

Usage:
    static = read_static(Path("data"))                       # lat/lon, hub height, rotor diameter per turbine
    scada = load_scada([2017, 2018], Path("data"))           # (turbine, timestamp) indexed 10-minute records
    download([2017], Path("data"))                           # fetch the archives from Zenodo (~300 MB per year)
"""
from __future__ import annotations

import hashlib
import io
import logging
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RECORD_ID = "16807551"
RECORD_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
STATIC_FILE = "Kelmarsh_WT_static.csv"
SCADA_PATTERN = re.compile(r"Kelmarsh_SCADA_(\d{4})_\d+\.zip")
TURBINE_RE = re.compile(r"_Kelmarsh_(\d+)_")
TIME_COL = "# Date and time"
COLUMNS = {
    "Wind speed (m/s)": "ws",
    "Wind speed, Standard deviation (m/s)": "ws_std",
    "Wind speed Sensor 1 (m/s)": "ws1",
    "Wind speed Sensor 2 (m/s)": "ws2",
    "Wind direction (°)": "wind_dir",
    "Nacelle position (°)": "nacelle_pos",
    "Vane position 1+2 (°)": "vane",
    "Power (kW)": "power_kw",
    "Nacelle ambient temperature (°C)": "temp_c",
    "Blade angle (pitch position) A (°)": "pitch_a",
    "Blade angle (pitch position) B (°)": "pitch_b",
    "Blade angle (pitch position) C (°)": "pitch_c",
    "Rotor speed (RPM)": "rotor_rpm",
}
SIGNALS = ["ws", "ws_std", "ws1", "ws2", "wind_dir", "nacelle_pos", "power_kw", "pitch", "rotor_rpm"]
CHUNK = 1 << 20
log = logging.getLogger(__name__)


def read_static(data_dir: Path) -> pd.DataFrame:
    """Turbine metadata indexed by turbine number: lat, lon, hub height, rotor diameter, rating, ground elevation."""
    df = pd.read_csv(data_dir / "raw" / STATIC_FILE, encoding="utf-8-sig")
    out = pd.DataFrame({
        "turbine": df["Title"].str.extract(r"(\d+)$", expand=False).astype(int),
        "lat": df["Latitude"],
        "lon": df["Longitude"],
        "hub_height_m": df["Hub Height (m)"].astype(float),
        "rotor_diameter_m": df["Rotor Diameter (m)"].astype(float),
        "rated_kw": df["Rated power (kW)"].astype(float),
        "elevation_m": df["Elevation (m)"].astype(float),
        "model": df["Manufacturer"] + " " + df["Model"],
        "cod": pd.to_datetime(df["Commercial Operations Date"], dayfirst=True).dt.tz_localize("UTC"),
    })
    return out.set_index("turbine").sort_index()


def _header_row(text: str, marker: str) -> int:
    """Index of the column header line (exports start with a block of '#' comment lines)."""
    for i, line in enumerate(io.StringIO(text)):
        if line.startswith(marker):
            return i
        if i > 50:
            break
    raise ValueError(f"header line starting with {marker!r} not found")


def read_turbine_csv(fh: io.TextIOBase, turbine: int) -> pd.DataFrame:
    """Parse one Greenbyte 10-minute export; returns a frame indexed by (turbine, timestamp UTC)."""
    text = fh.read()
    skip = _header_row(text, TIME_COL)
    wanted = {TIME_COL, *COLUMNS}
    df = pd.read_csv(io.StringIO(text), skiprows=skip, usecols=lambda c: c in wanted, na_values=["NaN"])
    out = pd.DataFrame(index=pd.to_datetime(df[TIME_COL]).dt.tz_localize("UTC").rename("timestamp"))
    for src, dst in COLUMNS.items():
        out[dst] = df[src].to_numpy(dtype="float32") if src in df else np.float32(np.nan)
    out["pitch"] = out[["pitch_a", "pitch_b", "pitch_c"]].mean(axis=1)
    out = out.drop(columns=["pitch_a", "pitch_b", "pitch_c"])
    out.insert(0, "turbine", np.int8(turbine))
    return out.set_index("turbine", append=True).swaplevel().sort_index()


def read_zip(path: Path) -> pd.DataFrame:
    """All turbines' 10-minute data from one yearly archive (status logs inside are not needed here)."""
    frames = []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            base = name.rsplit("/", 1)[-1]
            m = TURBINE_RE.search(base)
            if m and base.startswith("Turbine_Data") and base.endswith(".csv"):
                with z.open(name) as raw:
                    frames.append(read_turbine_csv(io.TextIOWrapper(raw, encoding="utf-8-sig"), int(m.group(1))))
    if not frames:
        raise ValueError(f"{path.name}: no turbine data files found")
    return pd.concat(frames).sort_index()


def load_scada(years: list[int], data_dir: Path) -> pd.DataFrame:
    """10-minute SCADA for the given years, from the Parquet cache or (first time) from the Zenodo archives."""
    raw, processed = data_dir / "raw", data_dir / "processed"
    frames = []
    for year in years:
        shared, path = processed / f"scada_{year}.parquet", processed / f"wake_scada_{year}.parquet"
        cached = next((p for p in (shared, path) if p.exists()), None)
        if cached is not None:
            frames.append(pd.read_parquet(cached, columns=SIGNALS))
            continue
        archives = sorted(raw.glob(f"Kelmarsh_SCADA_{year}_*.zip"))
        if not archives:
            raise FileNotFoundError(f"no SCADA archive for {year} in {raw}; run the 'fetch' command first")
        df = read_zip(archives[0])
        processed.mkdir(parents=True, exist_ok=True)
        df[SIGNALS].to_parquet(path)
        log.info("%d: %d records cached in %s", year, len(df), path)
        frames.append(df[SIGNALS])
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="first")].sort_index()


def wide(scada: pd.DataFrame, column: str) -> pd.DataFrame:
    """One signal as a (timestamp x turbine) table."""
    return scada[column].unstack("turbine").sort_index()


# --- download -------------------------------------------------------------------------------------------------

def list_files(session: requests.Session) -> dict[str, dict]:
    """Return {file name: {"size", "md5", "url"}} for the Zenodo record."""
    r = session.get(RECORD_URL, timeout=60)
    r.raise_for_status()
    out = {}
    for f in r.json()["files"]:
        algo, _, digest = f["checksum"].partition(":")
        out[f["key"]] = {"size": f["size"], "md5": digest if algo == "md5" else None, "url": f["links"]["self"]}
    return out


def select_files(files: dict[str, dict], years: list[int]) -> list[str]:
    """The static metadata file plus the yearly SCADA archives for the requested years."""
    by_year = {int(m.group(1)): name for name in files if (m := SCADA_PATTERN.fullmatch(name))}
    missing = sorted(set(years) - set(by_year))
    if missing:
        raise ValueError(f"no SCADA archive for {missing}; available years: {sorted(by_year)}")
    return [STATIC_FILE] + [by_year[y] for y in sorted(years)]


def download_file(session: requests.Session, info: dict, dest: Path) -> Path:
    """Stream one file to ``dest`` (via a .part file) and check size and MD5."""
    if dest.exists() and dest.stat().st_size == info["size"]:
        log.info("cached %s", dest.name)
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s (%.1f MB)", dest.name, info["size"] / 1e6)
    h, done = hashlib.md5(), 0
    with session.get(info["url"], stream=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
                h.update(chunk)
                done += len(chunk)
    if done != info["size"] or (info["md5"] and h.hexdigest() != info["md5"]):
        tmp.unlink()
        raise IOError(f"{dest.name}: size/checksum mismatch ({done} bytes, md5 {h.hexdigest()})")
    tmp.replace(dest)
    return dest


def download(years: list[int], data_dir: Path, session: requests.Session | None = None) -> list[Path]:
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", "wind-wake-shadow (open-source analysis)")
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    files = list_files(session)
    return [download_file(session, files[name], raw / name) for name in select_files(files, years)]
