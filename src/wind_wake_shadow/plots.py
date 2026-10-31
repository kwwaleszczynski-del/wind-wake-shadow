"""Static charts (PNG) and the flow animation (GIF) for the README.

Colour roles: a single-hue blue ramp for wind speed (dark = slow), a single-hue orange ramp for flicker hours,
fixed categorical colours per wake model, dark ink for measurements.

Usage:
    plots.flow_maps(fields, xs, ys, directions, farm, Path("docs/img/flow_maps.png"))
    plots.validation_panels(measured, curves, farm, Path("docs/img/validation_turbines.png"))
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODEL_COLORS = {"jensen": "#2a78d6", "gaussian": "#eb6834", "turbopark": "#1baf7a"}
TURBINE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "speed", ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb", "#f4f8fd"])
SPEED_RANGE = (0.4, 1.0)
FLICKER_BOUNDS = [0.0, 8.0, 30.0, 100.0, 300.0, 10000.0]
FLICKER_COLORS = ["#fde3d4", "#f8b994", "#f08a57", "#d4592a", "#9c3b15"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def _turbines(ax, farm, color=INK, labels=True, size=26):
    ax.scatter(farm.x, farm.y, s=size, facecolor="white", edgecolor=color, linewidth=1.2, zorder=5)
    if labels:
        for lx, ly, name in zip(farm.x, farm.y, farm.labels):
            ax.annotate(name, (lx, ly), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK, zorder=6)


def _wind_arrow(ax, wd: float, xs, ys):
    """Arrow in the upper-left corner pointing where the wind blows to."""
    rad = np.deg2rad(wd)
    span = (xs[-1] - xs[0]) * 0.07
    x0, y0 = xs[0] + (xs[-1] - xs[0]) * 0.1, ys[-1] - (ys[-1] - ys[0]) * 0.1
    ax.annotate("", xy=(x0 - span * np.sin(rad), y0 - span * np.cos(rad)),
                xytext=(x0 + span * np.sin(rad), y0 + span * np.cos(rad)),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.5})


def _map_axes(ax):
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    for s in ax.spines.values():
        s.set_visible(False)


def flow_maps(fields: np.ndarray, xs, ys, directions, farm, path: Path, ws: float = 8.0, title: str = ""):
    """Small multiples of U / U_free for several wind directions."""
    n = len(directions)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.4 * rows), squeeze=False)
    for ax, field, wd in zip(axes.flat, fields, directions):
        im = ax.pcolormesh(xs, ys, np.clip(field, *SPEED_RANGE), cmap=SPEED_CMAP, vmin=SPEED_RANGE[0],
                           vmax=SPEED_RANGE[1], shading="nearest", rasterized=True)
        _turbines(ax, farm)
        _wind_arrow(ax, wd, xs, ys)
        ax.set_title(f"wind from {wd:.0f} deg, {ws:g} m/s", loc="left")
        _map_axes(ax)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("wind speed / free stream at 78.5 m")
    cb.outline.set_visible(False)
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=11)
    _save(fig, path)


def model_maps(maps: dict, labels: dict, xs, ys, wd_index: int, wd: float, farm, path: Path, ws: float = 8.0):
    """The same direction in each wake model, side by side."""
    fig, axes = plt.subplots(1, len(maps), figsize=(4.6 * len(maps), 4.2))
    for ax, (name, fields) in zip(axes, maps.items()):
        im = ax.pcolormesh(xs, ys, np.clip(fields[wd_index], *SPEED_RANGE), cmap=SPEED_CMAP,
                           vmin=SPEED_RANGE[0], vmax=SPEED_RANGE[1], shading="nearest", rasterized=True)
        _turbines(ax, farm, labels=False)
        _wind_arrow(ax, wd, xs, ys)
        ax.set_title(labels[name], loc="left")
        _map_axes(ax)
    for ax in axes[1:]:  # same extent in every panel: one y axis, so "north (m)" no longer runs into the panel before
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
    cb.set_label("wind speed / free stream")
    cb.outline.set_visible(False)
    fig.suptitle(f"Wind from {wd:.0f} deg at {ws:g} m/s in three wake models", x=0.02, ha="left", fontsize=11)
    _save(fig, path)


def flow_animation(fields: np.ndarray, directions, xs, ys, farm, path: Path, ws: float = 8.0, fps: int = 6,
                   dpi: int = 72):
    """GIF of the Gaussian model flow map turning through all wind directions."""
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    mesh = ax.pcolormesh(xs, ys, np.clip(fields[0], *SPEED_RANGE), cmap=SPEED_CMAP, vmin=SPEED_RANGE[0],
                         vmax=SPEED_RANGE[1], shading="nearest")
    _turbines(ax, farm)
    _map_axes(ax)
    title = ax.set_title("", loc="left")
    cb = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("wind speed / free stream at 78.5 m")
    cb.outline.set_visible(False)
    arrow = []

    def draw(k):
        mesh.set_array(np.clip(fields[k], *SPEED_RANGE).ravel())
        title.set_text(f"Kelmarsh, Gaussian wake model: wind from {directions[k]:03.0f} deg at {ws:g} m/s")
        for a in arrow:
            a.remove()
        arrow.clear()
        rad = np.deg2rad(directions[k])
        span = (xs[-1] - xs[0]) * 0.07
        x0, y0 = xs[0] + (xs[-1] - xs[0]) * 0.1, ys[-1] - (ys[-1] - ys[0]) * 0.1
        arrow.append(ax.annotate("", xy=(x0 - span * np.sin(rad), y0 - span * np.cos(rad)),
                                 xytext=(x0 + span * np.sin(rad), y0 + span * np.cos(rad)),
                                 arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.5}))
        return [mesh, title]

    fig.tight_layout()
    anim = animation.FuncAnimation(fig, draw, frames=len(directions), blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def wake_loss_rose(sectors: dict, labels: dict, path: Path):
    """Energy share per direction sector (left) and the wake loss in each sector per model (right)."""
    fig = plt.figure(figsize=(10, 4.8))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    first = next(iter(sectors.values()))
    theta = np.deg2rad(first.index.to_numpy(dtype=float))
    width = np.deg2rad(np.diff(first.index.to_numpy(dtype=float))[0])
    ax1.bar(theta, 100 * first["energy_share"], width=width * 0.9, color="#2a78d6", edgecolor=SURFACE, linewidth=1)
    ax1.set_title("Share of gross energy by wind direction (%)", loc="left", pad=18)
    for name, s in sectors.items():
        t = np.r_[theta, theta[:1]]
        ax2.plot(t, np.r_[s["wake_loss_pct"], s["wake_loss_pct"].iloc[:1]], color=MODEL_COLORS[name], lw=2,
                 label=labels[name])
    ax2.set_title("Wake loss in each 10 deg sector (%)", loc="left", pad=18)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8)
    for ax in (ax1, ax2):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.grid(True, color=GRID)
        ax.spines["polar"].set_color(AXIS)
        ax.tick_params(labelsize=8)
    _save(fig, path)


def _measured_band(ax, m: pd.DataFrame, min_n: int):
    m = m[m["n"] >= min_n]
    ax.fill_between(m.index, m["ratio"] - m["ci95"], m["ratio"] + m["ci95"], color=GRID, lw=0)
    ax.plot(m.index, m["ratio"], "o", ms=2.6, color=INK, label="SCADA (95 % interval shaded)")


def validation_panels(measured: pd.DataFrame, curves: dict, labels: dict, farm, path: Path, sigma: float = 0.0,
                      min_n: int = 30):
    """Normalised power of every turbine vs wind direction: SCADA and the three models."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        _measured_band(ax, measured.loc[i], min_n)
        for name in labels:
            c = curves[(name, sigma)].loc[i]
            ax.plot(c.index, c["ratio"], color=MODEL_COLORS[name], lw=1.4, label=labels[name])
        ax.set_title(f"{farm.labels[i]}", loc="left")
        ax.axhline(1.0, color=AXIS, lw=0.8)
        ax.grid(True, axis="y")
        ax.set_xlim(0, 360)
        ax.set_xticks(np.arange(0, 361, 45))
        ax.set_ylim(0.15, 1.3)  # room for the highest SCADA points and their interval (T6: 1.275)
    for ax in axes[-1]:
        ax.set_xlabel("farm wind direction (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("power / free-stream power")
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(" ", fontsize=4)
    _save(fig, path)


def pair_profiles(measured: pd.DataFrame, curves: dict, labels: dict, farm, cases, path: Path,
                  sigma: float = 0.0, window: float = 40.0, min_n: int = 30):
    """Zoom on single and multiple wakes: ``cases`` = [(downstream index, centre direction, title), ...]."""
    fig, axes = plt.subplots(1, len(cases), figsize=(3.6 * len(cases), 3.6), sharey=True)
    for ax, (down, centre, title) in zip(np.atleast_1d(axes), cases):
        rel = lambda idx: (np.asarray(idx) - centre + 180) % 360 - 180  # noqa: E731
        m = measured.loc[down]
        m = m[(np.abs(rel(m.index)) <= window) & (m["n"] >= min_n)]
        order = np.argsort(rel(m.index))
        ax.errorbar(centre + rel(m.index)[order], m["ratio"].to_numpy()[order], yerr=m["ci95"].to_numpy()[order],
                    fmt="o", ms=3, color=INK, ecolor=MUTED, elinewidth=0.8, label="SCADA")
        for name in labels:
            c = curves[(name, sigma)].loc[down]
            c = c[np.abs(rel(c.index)) <= window]
            o = np.argsort(rel(c.index))
            ax.plot(centre + rel(c.index)[o], c["ratio"].to_numpy()[o], color=MODEL_COLORS[name], lw=1.6,
                    label=labels[name])
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel("wind direction (deg)")
        ax.grid(True, axis="y")
        ax.set_ylim(0.15, 1.3)  # the SCADA points outside the wake reach above 1.2 (T6)
    np.atleast_1d(axes)[0].set_ylabel("power / free-stream power")
    handles, names = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
    _save(fig, path)


def farm_efficiency(eff: pd.DataFrame, labels: dict, path: Path):
    """Farm efficiency (all turbines / free-stream reference) vs direction, SCADA vs models."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(eff.index, eff["measured"], "o", ms=2.6, color=INK, label="SCADA")
    for name in labels:
        ax.plot(eff.index, eff[name], color=MODEL_COLORS[name], lw=1.4, label=labels[name])
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xlabel("farm wind direction (deg)")
    ax.set_ylabel("farm efficiency")
    ax.grid(True, axis="y")
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))  # above the axes: TurbOPark's minima reached it
    _save(fig, path)


def _flicker_cmap():
    cmap = ListedColormap(FLICKER_COLORS)
    cmap.set_under(SURFACE)
    return cmap, BoundaryNorm(FLICKER_BOUNDS, cmap.N)


def shadow_maps(grid, worst: np.ndarray, expected: np.ndarray, farm, path: Path):
    """Worst-case and expected flicker hours per year on the grid, in the same classes."""
    cmap, norm = _flicker_cmap()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    titles = ["Astronomical worst case (h/year)", "Expected with sunshine, operation, rotor direction (h/year)"]
    for ax, data, title in zip(axes, (worst, expected), titles):
        shown = np.where(data > 0, data, np.nan)
        im = ax.pcolormesh(grid.xs, grid.ys, shown, cmap=cmap, norm=norm, shading="nearest", rasterized=True)
        _turbines(ax, farm, size=18)
        ax.set_title(title, loc="left")
        _map_axes(ax)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, ticks=FLICKER_BOUNDS[1:-1])
    cb.ax.set_yticklabels([f"{b:g} h" for b in FLICKER_BOUNDS[1:-1]])
    cb.set_label("flicker hours per year")
    cb.outline.set_visible(False)
    _save(fig, path)


def shadow_calendar(cal: pd.DataFrame, point: tuple[float, float], farm, path: Path, tz: str = "Europe/London"):
    """Days and local times of worst-case flicker at one grid point, one colour per turbine."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for k, (turbine, g) in enumerate(cal.groupby("turbine")):
        start, end = g["start"].dt.tz_convert(tz), g["end"].dt.tz_convert(tz)
        hours = lambda t: t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600  # noqa: E731
        ax.vlines(pd.to_datetime(g["date"]), hours(start), hours(end), color=TURBINE_COLORS[k % 3], lw=1.6,
                  label=farm.labels[turbine])
    local = pd.concat([cal["start"], cal["end"]]).dt.tz_convert(tz)
    hours_all = local.dt.hour + local.dt.minute / 60
    ax.set_ylabel(f"local time ({tz})")
    ax.set_ylim(np.floor(hours_all.min()) - 1, np.ceil(hours_all.max()) + 1)
    ax.set_xlim(pd.Timestamp(f"{cal['date'].iloc[0].year}-01-01"), pd.Timestamp(f"{cal['date'].iloc[0].year}-12-31"))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.grid(True, axis="y")
    ax.legend(title="shadow from", loc="lower left", ncol=3)
    ew, ns = ("east" if point[0] >= 0 else "west"), ("north" if point[1] >= 0 else "south")
    ax.set_title(f"Worst-case flicker calendar at a grid point {abs(point[0]):.0f} m {ew}, {abs(point[1]):.0f} m {ns} "
                 "of the farm centre", loc="left")
    _save(fig, path)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
