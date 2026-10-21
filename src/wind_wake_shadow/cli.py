"""Command line interface.

Usage:
    python -m wind_wake_shadow fetch --years 2016 2017 2018 2019 2020 2021   # Kelmarsh (Zenodo) + Met Office sunshine
    python -m wind_wake_shadow fetch --sunshine-only
    python -m wind_wake_shadow run                                           # everything below, ~4 minutes
    python -m wind_wake_shadow run --skip-shadow --skip-gif                  # faster iterations
    python -m wind_wake_shadow run --help

Kelmarsh data are read from ./data (override with --data-dir or KELMARSH_DATA_DIR, e.g. a folder shared with
the wind-power-curve-kelmarsh project), the sunshine file from ./data/raw; results go to ./results, charts to
./docs/img and the web page to ./docs/index.html.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from wind_wake_shadow import kelmarsh, pipeline, plots, shadow, sunshine, wake, web

CALENDAR_POINT = (1000.0, -300.0)   # an arbitrary grid point east-south-east of the farm, not a dwelling
log = logging.getLogger("wind_wake_shadow")


def cmd_fetch(args):
    if not args.sunshine_only:
        for path in kelmarsh.download(args.years, args.data_dir):
            log.info("ok %s", path)
    log.info("ok %s", sunshine.download(sunshine.STATION, args.sunshine_dir))


def _write(df, path: Path, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **kw)


def cmd_run(args):
    res_dir, img = args.results_dir, args.img_dir
    farm = pipeline.load_farm(args.data_dir)
    scada = kelmarsh.load_scada(sorted(args.years), args.data_dir)
    labels = wake.MODELS
    summary = {"years": sorted(args.years), "turbines": len(farm.x), "lat": round(farm.lat, 4), "lon": round(farm.lon, 4)}

    climate, ti = pipeline.wind_climate_from_scada(scada, farm)
    _write(climate, res_dir / "wind_climate.csv", float_format="%.4f")
    summary["ambient_ti"] = round(ti, 4)

    # wake models: AEP, sector losses, maps
    w = pipeline.run_wake(farm, climate, ti)
    aep = pd.concat(w["aep"], names=["model"])
    _write(aep, res_dir / "aep_by_turbine.csv", float_format="%.4f")
    _write(pd.concat(w["sectors"], names=["model"]), res_dir / "wake_loss_by_sector.csv", float_format="%.4f")
    summary["aep"] = {m: {"gross_gwh": round(a.loc["farm", "gross_gwh"], 3), "net_gwh": round(a.loc["farm", "net_gwh"], 3),
                          "wake_loss_pct": round(a.loc["farm", "wake_loss_pct"], 2)} for m, a in w["aep"].items()}
    plots.flow_maps(w["maps"]["gaussian"], w["xs"], w["ys"], pipeline.MAP_DIRECTIONS, farm, img / "flow_maps.png",
                    ws=pipeline.FLOW_WS)
    plots.model_maps(w["maps"], labels, w["xs"], w["ys"], 0, pipeline.MAP_DIRECTIONS[0], farm, img / "model_maps.png",
                     ws=pipeline.FLOW_WS)
    plots.wake_loss_rose(w["sectors"], labels, img / "wake_loss_rose.png")
    if not args.skip_gif:
        plots.flow_animation(w["animation"], w["anim_wd"], w["xs"], w["ys"], farm, img / "flow_animation.gif",
                             ws=pipeline.FLOW_WS)

    # SCADA comparison
    v = pipeline.run_validation(scada, farm, climate, ti)
    _write(v["summary"].set_index(["model", "sigma_deg"]), res_dir / "validation_summary.csv", float_format="%.4f")
    curves = pd.concat({"measured": v["measured"].drop(columns=["ci95", "n"])}
                       | {m: c.drop(columns=["ci95", "n"]) for (m, s), c in v["curves"].items() if s == 0.0},
                       names=["source"])
    curves = curves.join(v["measured"][["n", "ci95"]], on=["turbine", "bin"])
    curves = curves.rename(index=lambda t: t + 1 if isinstance(t, (int, np.integer)) else t, level="turbine")
    _write(curves, res_dir / "validation_curves.csv", float_format="%.4f")
    _write(v["pairs"].set_index(["upstream", "downstream"]), res_dir / "pair_deficits.csv", float_format="%.4f")
    _write(v["shift"].to_frame(), res_dir / "direction_shift.csv", float_format="%.4f")
    _write(v["efficiency"].rename_axis("wd_bin"), res_dir / "farm_efficiency_by_direction.csv", float_format="%.4f")
    _write(pd.Series(v["site_factors"], index=pd.Index(farm.static.index, name="turbine"), name="site_factor"),
           res_dir / "site_factors.csv", float_format="%.4f")
    s0 = v["summary"][v["summary"]["sigma_deg"].eq(0.0)].set_index("model")
    summary["validation"] = {
        "records": v["n_records"], "days": v["n_days"], "period": v["period"],
        "measured_wake_loss_pct": round(float(s0["measured_wake_loss_pct"].iloc[0]), 2),
        "model_wake_loss_pct": s0["wake_loss_pct"].round(2).to_dict(),
        "rmse_waked": s0["rmse_waked"].round(4).to_dict(),
        "best_direction_shift_deg": int(v["shift"].idxmin()),
    }
    plots.validation_panels(v["measured"], v["curves"], labels, farm, img / "validation_turbines.png")
    cases = [(1, 143.0, "T2 behind T1, 2.9 D"), (4, 304.0, "T5 behind T3, 3.3 D"),
             (5, 301.0, "T6 behind T5 and T3 (row)"), (2, 248.0, "T3 behind T2, 4.2 D")]
    plots.pair_profiles(v["measured"], v["curves"], labels, farm, cases, img / "pair_profiles.png")
    plots.farm_efficiency(v["efficiency"], labels, img / "farm_efficiency.png")

    # shadow flicker
    shade = None
    if not args.skip_shadow:
        sun_file = sunshine.download(sunshine.STATION, args.sunshine_dir)
        fraction = sunshine.sunshine_fraction(sunshine.read_station(sun_file))
        _write(pd.Series(fraction, index=pd.Index(range(1, 13), name="month"), name="sunshine_fraction"),
               res_dir / "sunshine_fraction.csv", float_format="%.4f")
        shade = pipeline.run_shadow(farm, scada, fraction, step_m=args.shadow_step)
        share = shade["operating_share"].rename(columns=lambda t: f"T{t}")
        _write(share, res_dir / "rotor_turning_share_daylight.csv", float_format="%.4f")
        summary["shadow"] = {k: round(val, 3) for k, val in pipeline.shadow_summary(shade, farm).items()}
        cal = shadow.calendar(*CALENDAR_POINT, farm.shadow_turbines(), farm.lat, farm.lon, pipeline.SHADOW_YEAR)
        _write(cal, res_dir / "shadow_calendar_point.csv", index=False)
        summary["shadow"]["calendar_point_hours"] = round(float((cal["end"] - cal["start"]).dt.total_seconds().sum() / 3600), 2)
        plots.shadow_maps(shade["grid"], shade["worst"]["hours"], shade["expected"], farm, img / "shadow_maps.png")
        plots.shadow_calendar(cal, CALENDAR_POINT, farm, img / "shadow_calendar.png")
        cache = res_dir / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache / "shadow_maps.npz", xs=shade["grid"].xs, ys=shade["grid"].ys,
                            worst=shade["worst"]["hours"], max_minutes_day=shade["worst"]["max_minutes_day"],
                            expected=shade["expected"])
        if not args.skip_site:
            (args.site_dir / "index.html").write_text(_page(farm, w, v, shade, cal, summary, labels), encoding="utf-8")
            log.info("web page written to %s", args.site_dir / "index.html")

    (res_dir / "run.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("results in %s, charts in %s", res_dir, img)


def _calendar_text(cal: pd.DataFrame, farm, hours: float, tz: str = "Europe/London") -> str:
    months = pd.to_datetime(cal["date"]).dt.strftime("%B")
    order = pd.to_datetime(cal["date"]).dt.month
    first, last = months[order.idxmin()], months[order.idxmax()]
    start, end = cal["start"].dt.tz_convert(tz), cal["end"].dt.tz_convert(tz)
    earliest = (start - start.dt.normalize()).min()   # time of day, not timestamp
    latest = (end - end.dt.normalize()).max()
    hhmm = lambda td: f"{int(td.total_seconds() // 3600):02d}:{int(td.total_seconds() % 3600 // 60):02d}"  # noqa: E731
    names = ", ".join(farm.labels[t] for t in sorted(cal["turbine"].unique()))
    x, y = CALENDAR_POINT
    return (f"When the worst-case flicker happens at one grid point {abs(x):.0f} m {'east' if x > 0 else 'west'} "
            f"and {abs(y):.0f} m {'north' if y > 0 else 'south'} of the farm centre: shadows of {names} from "
            f"{first} to {last}, between {hhmm(earliest)} and {hhmm(latest)} local time "
            f"({hours:.0f} h/year in the worst case).")


def _page(farm, w, v, shade, cal, summary, labels) -> str:
    va, sh = summary["validation"], summary["shadow"]
    main = summary["aep"]["gaussian"]
    tiles = [
        (f"{main['wake_loss_pct']:.1f} %", "modelled AEP wake loss (Gaussian model)"),
        (f"{va['measured_wake_loss_pct']:.1f} % vs {va['model_wake_loss_pct']['gaussian']:.1f} %",
         "wake loss in the SCADA sample: measured vs Gaussian model"),
        (f"{sh['worst_case_max_h']:.0f} h", "worst-case flicker at the most exposed grid point (per year)"),
        (f"{sh['area_worst_gt_30h_km2']:.1f} km²", "area above 30 h/year in the worst case"),
    ]
    aep = pd.concat({labels[m]: a.loc["farm"] for m, a in w["aep"].items()}, axis=1).T
    aep.index.name = "model"
    aep.columns = ["gross AEP (GWh)", "net AEP (GWh)", "wake loss (%)"]
    s0 = v["summary"][v["summary"]["sigma_deg"].eq(0.0)].set_index("model")
    vt = pd.DataFrame({
        "wake loss in sample (%)": s0["wake_loss_pct"].rename(index=labels),
        "RMSE in waked bins": s0["rmse_waked"].rename(index=labels),
    })
    vt.loc["SCADA (measured)"] = [va["measured_wake_loss_pct"], np.nan]
    vt.index.name = "source"
    sections = {
        "flow": "Hub-height wind speed relative to the free stream at 8 m/s, from the Niayifar &amp; Porte-Agel "
                "Gaussian model. Press Play or drag the slider to turn the wind; the closest pairs (T1-T2 at "
                "2.9 rotor diameters, T3-T5 at 3.3 D) shade each other strongly for north-west and south-east winds.",
        "rose": f"Annual energy production with and without wakes over the measured wind climate "
                f"(free-stream nacelle wind speed {summary['years'][0]}-{summary['years'][-1]}, ambient turbulence "
                f"intensity {summary['ambient_ti']:.3f}). The south-west sectors carry most of the energy; the "
                "highest sector losses are where turbines line up.",
        "validation": f"Power of each turbine divided by the power of the free-stream turbines, binned by farm "
                      f"wind direction (2 deg), from {va['records']:,} ten-minute records with all six turbines in "
                      f"partial load (5-10 m/s reference wind speed, {va['period'][0]} to {va['period'][1]}). "
                      "The models are run for every record at its own wind speed and direction. Pick a turbine "
                      "above the chart.",
        "efficiency": "Farm efficiency (total power over six times the free-stream turbine power) per direction "
                      "bin, for the same records.",
        "shadow": f"Hours of shadow flicker per year on a {pipeline.SHADOW_STEP_M:g} m grid, shown at "
                  f"{2 * pipeline.SHADOW_STEP_M:g} m. The worst case assumes permanent sunshine and rotors "
                  "that always turn and face the sun; the expected case scales it by the monthly sunshine "
                  "fraction (Met Office, Oxford, 1991-2020), the share of daylight time each rotor turned and its "
                  "measured orientation (SCADA). No buildings, terrain or screening are included.",
        "calendar": _calendar_text(cal, farm, sh["calendar_point_hours"]),
        "method": ("<ul>"
                   "<li>Wake models: PyWake 2.6 (DTU). Jensen top-hat (k = 0.075), Niayifar &amp; Porte-Agel "
                   "Gaussian with Crespo-Hernandez added turbulence, TurbOPark. Senvion MM92 catalogue power "
                   "curve; thrust curve from PyWake's generic turbine for the same rotor and rating (no public "
                   "Ct curve exists). Power-law shear 0.2 for the two 68.5 m hub heights.</li>"
                   "<li>Measured wake losses come from records where every turbine runs in partial load, so "
                   "they describe that sample, not the annual loss.</li>"
                   "<li>Shadow flicker: analytic sun-path model checked against 1-minute time stepping; "
                   "flat terrain, receptors 2 m above ground, sun at least 3 deg up, 10 rotor diameters limit. "
                   "The expected case treats sunshine, operation and wind direction as independent.</li>"
                   "</ul>"),
        "footer": ("Data: Kelmarsh wind farm data, Cubico Sustainable Investments (Plumley, Takeuchi), "
                   "<a href=\"https://doi.org/10.5281/zenodo.16807551\">doi:10.5281/zenodo.16807551</a>, CC-BY-4.0. "
                   "Sunshine: Met Office historic station data, Oxford, Contains Met Office data © Crown copyright, "
                   "Open Government Licence v3.0. Code: "
                   "<a href=\"https://github.com/kwwaleszczynski-del/wind-wake-shadow\">wind-wake-shadow</a> (MIT)."),
    }
    ctx = {
        "tiles": tiles, "sections": sections,
        "flow_div": web._div(web.flow_figure(w["animation"], w["anim_wd"], w["xs"], w["ys"], farm, pipeline.FLOW_WS), "flow"),
        "rose_div": web._div(web.rose_figure(w["sectors"], labels), "rose"),
        "aep_table": web.table_html(aep, {c: "{:.2f}" for c in aep.columns}),
        "validation_div": web._div(web.validation_figure(v["measured"], v["curves"], labels, farm), "validation"),
        "validation_table": web.table_html(vt, {"wake loss in sample (%)": "{:.1f}", "RMSE in waked bins": "{:.3f}"}),
        "efficiency_div": web._div(web.efficiency_figure(v["efficiency"], labels), "efficiency"),
        "shadow_div": web._div(web.shadow_figure(shade["grid"], shade["worst"]["hours"], shade["expected"], farm), "shadow"),
        "calendar_div": web._div(web.calendar_figure(cal, farm), "calendar"),
    }
    return web.build_page(ctx)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wind-wake-shadow", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "run"):
        p = sub.add_parser(name)
        p.add_argument("--years", type=int, nargs="+", default=list(range(2016, 2022)))
        p.add_argument("--data-dir", type=Path, default=Path(os.environ.get("KELMARSH_DATA_DIR", "data")),
                       help="Kelmarsh data folder (raw/ archives, processed/ Parquet cache)")
        p.add_argument("--sunshine-dir", type=Path, default=Path("data"), help="folder for the Met Office file")
        if name == "fetch":
            p.add_argument("--sunshine-only", action="store_true", help="only the Met Office sunshine file")
        else:
            p.add_argument("--results-dir", type=Path, default=Path("results"))
            p.add_argument("--img-dir", type=Path, default=Path("docs/img"))
            p.add_argument("--site-dir", type=Path, default=Path("docs"))
            p.add_argument("--shadow-step", type=float, default=pipeline.SHADOW_STEP_M, help="flicker grid step (m)")
            p.add_argument("--skip-shadow", action="store_true", help="no flicker maps (and no web page)")
            p.add_argument("--skip-gif", action="store_true", help="no flow animation GIF")
            p.add_argument("--skip-site", action="store_true", help="no web page")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    {"fetch": cmd_fetch, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
