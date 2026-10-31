"""Static web page (GitHub Pages) with interactive Plotly charts.

One self-contained HTML file: the data are embedded as typed arrays, plotly.js comes from its CDN. The page
follows the viewer's light/dark preference.

Usage:
    html = build_page(ctx)          # ctx: dict with the results of the pipeline steps
    Path("docs/index.html").write_text(html, encoding="utf-8")
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs_version
from plotly.subplots import make_subplots

from wind_wake_shadow.plots import FLICKER_BOUNDS, FLICKER_COLORS, INK, MODEL_COLORS, TURBINE_COLORS

SPEED_SCALE = [[0.0, "#0d366b"], [0.15, "#184f95"], [0.3, "#256abf"], [0.45, "#3987e5"], [0.6, "#6da7ec"],
               [0.75, "#9ec5f4"], [0.9, "#cde2fb"], [1.0, "#f4f8fd"]]
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
LIGHT = {"ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
         "surface": "#fcfcfb", "band": "rgba(137,135,129,0.25)"}


def _base_layout(**kw) -> dict:
    layout = {"template": "none", "font": {"family": FONT, "size": 12, "color": LIGHT["ink2"]},
              "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
              "margin": {"l": 56, "r": 16, "t": 40, "b": 48}, "hoverlabel": {"font": {"family": FONT}}}
    layout.update(kw)
    return layout


MENU = {"bgcolor": "#e1e0d9", "bordercolor": "#c3c2b7", "font": {"color": "#0b0b0b"}}


def _axis(**kw) -> dict:
    a = {"gridcolor": LIGHT["grid"], "linecolor": LIGHT["axis"], "zeroline": False, "tickcolor": LIGHT["axis"],
         "ticks": "outside", "ticklen": 4}
    a.update(kw)
    return a


def _turbine_trace(farm, showlegend=False) -> go.Scatter:
    return go.Scatter(x=farm.x, y=farm.y, mode="markers+text", text=farm.labels, textposition="top right",
                      marker={"size": 9, "color": "white", "line": {"color": INK, "width": 1.5}},
                      textfont={"color": INK, "size": 11}, hovertemplate="%{text}<extra></extra>",
                      showlegend=showlegend, name="turbines")


def flow_figure(anim: np.ndarray, directions, xs, ys, farm, ws: float, start_wd: float = 240.0) -> go.Figure:
    """Animated heat map of the Gaussian model flow field, one frame per wind direction."""
    pct = np.clip(np.round(anim * 100), 0, 100).astype(np.uint8)
    k0 = int(np.argmin(np.abs(np.asarray(directions) - start_wd)))
    heat = lambda k: go.Heatmap(  # noqa: E731
        x=xs, y=ys, z=pct[k], zmin=40, zmax=100, colorscale=SPEED_SCALE,
        colorbar={"title": {"text": "% of free stream", "side": "right"}, "outlinewidth": 0, "thickness": 12},
        hovertemplate="%{x:.0f} m east, %{y:.0f} m north<br>%{z}% of free-stream speed<extra></extra>")
    fig = go.Figure([heat(k0), _turbine_trace(farm)])
    fig.frames = [go.Frame(data=[heat(k)], traces=[0], name=f"{wd:.0f}") for k, wd in enumerate(directions)]
    steps = [{"method": "animate", "label": f"{wd:.0f}",
              "args": [[f"{wd:.0f}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                                       "transition": {"duration": 0}}]} for wd in directions]
    fig.update_layout(**_base_layout(
        height=600, margin={"l": 56, "r": 16, "t": 16, "b": 120},
        xaxis=_axis(title="east (m)", range=[xs[0], xs[-1]], constrain="domain"),
        yaxis=_axis(title="north (m)", scaleanchor="x", range=[ys[0], ys[-1]], constrain="domain"),
        sliders=[{"active": k0, "steps": steps, "x": 0.16, "len": 0.84, "y": -0.1, "yanchor": "top",
                  "currentvalue": {"prefix": f"wind at {ws:g} m/s from ", "suffix": " deg", "xanchor": "right"},
                  "pad": {"t": 30}}],
        updatemenus=[{**MENU, "type": "buttons", "direction": "left", "x": 0.0, "y": -0.1, "xanchor": "left",
                      "yanchor": "top", "pad": {"t": 30, "r": 10},
                      "buttons": [{"label": "Play", "method": "animate",
                                   "args": [None, {"frame": {"duration": 180, "redraw": True},
                                                   "fromcurrent": True, "transition": {"duration": 0}}]},
                                  {"label": "Pause", "method": "animate",
                                   "args": [[None], {"mode": "immediate", "frame": {"duration": 0}}]}]}],
    ))
    return fig


def rose_figure(sectors: dict, labels: dict) -> go.Figure:
    first = next(iter(sectors.values()))
    theta = first.index.to_numpy(dtype=float)
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "polar"}, {"type": "polar"}]],
                        subplot_titles=("Share of gross energy (%)", "Wake loss per 10 deg sector (%)"))
    fig.add_trace(go.Barpolar(theta=theta, r=100 * first["energy_share"], marker={"color": "#2a78d6"},
                              name="energy share", showlegend=False,
                              hovertemplate="%{theta:.0f} deg: %{r:.1f} % of energy<extra></extra>"), 1, 1)
    for name, s in sectors.items():
        fig.add_trace(go.Scatterpolar(theta=np.r_[theta, theta[:1]], r=np.r_[s["wake_loss_pct"], s["wake_loss_pct"].iloc[:1]],
                                      mode="lines", line={"color": MODEL_COLORS[name], "width": 2}, name=labels[name],
                                      hovertemplate="%{theta:.0f} deg: %{r:.1f} % loss<extra>" + labels[name] + "</extra>"),
                      1, 2)
    polar = {"angularaxis": {"rotation": 90, "direction": "clockwise", "gridcolor": LIGHT["grid"],
                             "linecolor": LIGHT["axis"]},
             "radialaxis": {"gridcolor": LIGHT["grid"], "linecolor": LIGHT["axis"]}, "bgcolor": "rgba(0,0,0,0)"}
    fig.update_layout(**_base_layout(height=430, polar=polar, polar2=polar, margin={"l": 40, "r": 40, "t": 80, "b": 30},
                                     legend={"orientation": "h", "y": -0.08}))
    fig.update_annotations(yshift=24)  # subplot titles above the "0°" label of the angular axis
    return fig


def validation_figure(measured: pd.DataFrame, curves: dict, labels: dict, farm, sigma: float = 0.0,
                      min_n: int = 30, start: int = 1) -> go.Figure:
    """Normalised power vs direction per turbine (dropdown), SCADA with its 95 % band and the three models."""
    fig = go.Figure()
    per_turbine = 3 + len(labels)
    n_t = len(farm.labels)
    for i in range(n_t):
        m = measured.loc[i]
        m = m[m["n"] >= min_n]
        vis = i == start
        fig.add_trace(go.Scatter(x=m.index, y=m["ratio"] - m["ci95"], mode="lines", line={"width": 0},
                                 showlegend=False, hoverinfo="skip", visible=vis))
        fig.add_trace(go.Scatter(x=m.index, y=m["ratio"] + m["ci95"], mode="lines", line={"width": 0},
                                 fill="tonexty", fillcolor=LIGHT["band"], name="SCADA 95 % interval",
                                 hoverinfo="skip", visible=vis))
        fig.add_trace(go.Scatter(x=m.index, y=m["ratio"], mode="markers", marker={"size": 5, "color": INK},
                                 name="SCADA", customdata=m["n"], visible=vis,
                                 hovertemplate="%{x:.0f} deg: %{y:.3f} (%{customdata} records)<extra>SCADA</extra>"))
        for name in labels:
            c = curves[(name, sigma)].loc[i]
            fig.add_trace(go.Scatter(x=c.index, y=c["ratio"], mode="lines", name=labels[name], visible=vis,
                                     line={"color": MODEL_COLORS[name], "width": 2},
                                     hovertemplate="%{x:.0f} deg: %{y:.3f}<extra>" + labels[name] + "</extra>"))
    buttons = []
    for i, label in enumerate(farm.labels):
        visible = [j // per_turbine == i for j in range(n_t * per_turbine)]
        buttons.append({"label": label, "method": "update", "args": [{"visible": visible}]})
    fig.update_layout(**_base_layout(
        height=430, hovermode="x unified",
        xaxis=_axis(title="farm wind direction (deg)", range=[0, 360], dtick=45),
        yaxis=_axis(title="power / free-stream power", range=[0.15, 1.2]),
        legend={"orientation": "h", "y": 1.12, "x": 0.12, "traceorder": "normal"},
        updatemenus=[{**MENU, "buttons": buttons, "active": start, "x": 0.0, "y": 1.18, "xanchor": "left",
                      "showactive": True}],
    ))
    return fig


def efficiency_figure(eff: pd.DataFrame, labels: dict) -> go.Figure:
    fig = go.Figure(go.Scatter(x=eff.index, y=eff["measured"], mode="markers", marker={"size": 5, "color": INK},
                               name="SCADA", hovertemplate="%{x:.0f} deg: %{y:.3f}<extra>SCADA</extra>"))
    for name in labels:
        fig.add_trace(go.Scatter(x=eff.index, y=eff[name], mode="lines", name=labels[name],
                                 line={"color": MODEL_COLORS[name], "width": 2},
                                 hovertemplate="%{x:.0f} deg: %{y:.3f}<extra>" + labels[name] + "</extra>"))
    fig.update_layout(**_base_layout(height=340, hovermode="x unified",
                                     xaxis=_axis(title="farm wind direction (deg)", range=[0, 360], dtick=45),
                                     yaxis=_axis(title="farm efficiency"),
                                     legend={"orientation": "h", "y": 1.15}))
    return fig


def _flicker_scale() -> list:
    n = len(FLICKER_COLORS) + 1
    scale = [[0.0, "rgba(0,0,0,0)"], [1 / n, "rgba(0,0,0,0)"]]
    for k, color in enumerate(FLICKER_COLORS, start=1):
        scale += [[k / n, color], [(k + 1) / n, color]]
    return scale


def shadow_figure(grid, worst: np.ndarray, expected: np.ndarray, farm, step: int = 2) -> go.Figure:
    """Worst-case / expected flicker maps (buttons), drawn in the README's classes; hover shows hours."""
    xs, ys = grid.xs[::step], grid.ys[::step]
    fig = go.Figure()
    n = len(FLICKER_COLORS) + 1
    for k, (data, name) in enumerate(((worst, "worst case"), (expected, "expected"))):
        d = data[::step, ::step]
        cls = np.where(d > 0, np.digitize(d, FLICKER_BOUNDS[1:-1]) + 1, 0).astype(np.uint8)
        fig.add_trace(go.Heatmap(
            x=xs, y=ys, z=cls, customdata=np.round(d, 1).astype(np.float32), zmin=0, zmax=n - 1,
            colorscale=_flicker_scale(), visible=k == 0, name=name,
            colorbar={"tickvals": [j * (n - 1) / n for j in range(2, n)],   # class boundaries
                      "ticktext": [f"{b:g} h" for b in FLICKER_BOUNDS[1:-1]], "outlinewidth": 0, "thickness": 12,
                      "title": {"text": "hours / year", "side": "right"}},
            hovertemplate="%{x:.0f} m east, %{y:.0f} m north<br>%{customdata} h/year<extra>" + name + "</extra>"))
    fig.add_trace(_turbine_trace(farm))
    fig.update_layout(**_base_layout(
        height=600, margin={"l": 56, "r": 16, "t": 50, "b": 48},
        xaxis=_axis(title="east (m)", range=[xs[0], xs[-1]], constrain="domain"),
        yaxis=_axis(title="north (m)", scaleanchor="x", range=[ys[0], ys[-1]], constrain="domain"),
        updatemenus=[{**MENU, "type": "buttons", "direction": "left", "x": 0.0, "y": 1.1, "xanchor": "left",
                      "buttons": [{"label": "Astronomical worst case", "method": "update",
                                   "args": [{"visible": [True, False, True]}]},
                                  {"label": "Expected", "method": "update",
                                   "args": [{"visible": [False, True, True]}]}]}],
    ))
    return fig


def calendar_figure(cal: pd.DataFrame, farm, tz: str = "Europe/London") -> go.Figure:
    fig = go.Figure()
    for k, (turbine, g) in enumerate(cal.groupby("turbine")):
        start, end = g["start"].dt.tz_convert(tz), g["end"].dt.tz_convert(tz)
        hours = lambda t: (t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600).to_numpy()  # noqa: E731
        dates = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d").to_numpy()
        x = np.column_stack([dates, dates, np.full(len(g), None)]).ravel()
        y = np.column_stack([hours(start), hours(end), np.full(len(g), np.nan)]).ravel()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line={"color": TURBINE_COLORS[k % 3], "width": 2},
                                 name=farm.labels[turbine], hovertemplate="%{x}: %{y:.2f} h<extra>%{fullData.name}</extra>"))
    local = pd.concat([cal["start"], cal["end"]]).dt.tz_convert(tz)
    tod = local.dt.hour + local.dt.minute / 60
    fig.update_layout(**_base_layout(height=340, xaxis=_axis(tickformat="%b"),
                                     yaxis=_axis(title=f"local time ({tz})",
                                                 range=[np.floor(tod.min()) - 1, np.ceil(tod.max()) + 1], dtick=1),
                                     legend={"orientation": "h", "y": 1.12, "title": {"text": "shadow from "}}))
    return fig


def table_html(df: pd.DataFrame, formats: dict | None = None) -> str:
    formats = formats or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [df.index.name or ""] + list(df.columns))
    rows = []
    for idx, row in df.iterrows():
        cells = [f"<td>{html.escape(str(idx))}</td>"]
        for c, v in row.items():
            text = formats.get(c, "{}").format(v) if not (isinstance(v, float) and np.isnan(v)) else ""
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


CSS = """
:root { color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,0.10); --accent:#256abf; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --page:#0d0d0d;
  --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --ring:rgba(255,255,255,0.10); --accent:#6da7ec; } }
:root[data-theme="dark"] { color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10); --accent:#6da7ec; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink); font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  line-height:1.55; }
main { max-width:1080px; margin:0 auto; padding:32px 16px 64px; }
h1 { font-size:1.9rem; margin:0 0 8px; letter-spacing:-0.01em; }
h2 { font-size:1.3rem; margin:48px 0 8px; }
p, li { color:var(--ink2); max-width:78ch; }
a { color:var(--accent); }
.lead { font-size:1.05rem; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:24px 0; }
.tile { background:var(--surface); border:1px solid var(--ring); border-radius:10px; padding:14px 16px; }
.tile .v { font-size:1.6rem; font-weight:600; color:var(--ink); }
.tile .l { font-size:0.85rem; color:var(--muted); }
.card { background:var(--surface); border:1px solid var(--ring); border-radius:10px; padding:12px; margin:16px 0; }
.plot { width:100%; }
table { border-collapse:collapse; font-size:0.9rem; margin:12px 0; font-variant-numeric:tabular-nums; }
th, td { padding:6px 12px; border-bottom:1px solid var(--grid); text-align:right; }
th:first-child, td:first-child { text-align:left; }
th { color:var(--ink2); font-weight:600; }
.scroll { overflow-x:auto; }
footer { margin-top:48px; font-size:0.85rem; color:var(--muted); }
"""

THEME_JS = """
const DARK = {ink:'#f0efec', ink2:'#c3c2b7', grid:'#2c2c2a', axis:'#383835', band:'rgba(195,194,183,0.22)', marker:'#f0efec'};
const LIGHT = {ink:'#0b0b0b', ink2:'#52514e', grid:'#e1e0d9', axis:'#c3c2b7', band:'rgba(137,135,129,0.25)', marker:'#0b0b0b'};
function isDark() {
  const t = document.documentElement.dataset.theme;
  if (t === 'dark') return true; if (t === 'light') return false;
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}
function applyTheme() {
  const c = isDark() ? DARK : LIGHT;
  document.querySelectorAll('.plot').forEach(div => {
    if (!div.layout) return;
    const upd = {'font.color': c.ink2};
    Object.keys(div.layout).forEach(k => {
      if (/^[xy]axis\\d*$/.test(k)) { upd[k + '.gridcolor'] = c.grid; upd[k + '.linecolor'] = c.axis; upd[k + '.tickcolor'] = c.axis; }
      if (/^polar\\d*$/.test(k)) { ['angularaxis', 'radialaxis'].forEach(a => { upd[k + '.' + a + '.gridcolor'] = c.grid; upd[k + '.' + a + '.linecolor'] = c.axis; }); }
    });
    Plotly.relayout(div, upd);
    const idx = [], colors = [], fills = [], fidx = [];
    div.data.forEach((tr, i) => {
      if (tr.name === 'SCADA') { idx.push(i); }
      if (tr.fill === 'tonexty') { fidx.push(i); }
    });
    if (idx.length) Plotly.restyle(div, {'marker.color': idx.map(() => c.marker)}, idx);
    if (fidx.length) Plotly.restyle(div, {'fillcolor': fidx.map(() => c.band)}, fidx);
  });
}
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
window.addEventListener('load', applyTheme);
// on a phone: the two roses one above the other, the legends of the line charts below the plot
function applyNarrow() {
  if (window.innerWidth >= 600) return;
  const rose = document.getElementById('rose');
  if (rose && rose.layout) Plotly.relayout(rose, {height: 760, 'polar.domain': {x: [0, 1], y: [0.56, 1]},
    'polar2.domain': {x: [0, 1], y: [0, 0.44]}, 'annotations[0].x': 0.5, 'annotations[1].x': 0.5,
    'annotations[1].y': 0.44, 'legend.y': -0.04, 'legend.yanchor': 'top'});
  ['validation', 'efficiency'].forEach(id => {
    const d = document.getElementById(id);
    if (d && d.layout) Plotly.relayout(d, {'legend.y': -0.32, 'legend.yanchor': 'top', 'legend.x': 0,
      'margin.t': 30, 'margin.b': 160});
  });
}
window.addEventListener('load', applyNarrow);
"""


def _div(fig: go.Figure, div_id: str) -> str:
    """Plot container plus the script that draws it (numpy arrays travel as base64 typed arrays)."""
    payload = pio.to_json(fig, validate=False, pretty=False)
    return (f'<div id="{div_id}" class="plot"></div>\n<script>(function(){{const s={payload};'
            f'Plotly.newPlot("{div_id}",s.data,s.layout,{{responsive:true,displaylogo:false}})'
            f'.then(g=>{{if(s.frames&&s.frames.length)Plotly.addFrames(g,s.frames);}});}})();</script>')


def build_page(ctx: dict) -> str:
    """Assemble the page from the pipeline results (see cli.cmd_run for the keys)."""
    t = ctx["tiles"]
    tiles = "".join(f'<div class="tile"><div class="v">{html.escape(v)}</div><div class="l">{html.escape(lbl)}</div></div>'
                    for v, lbl in t)
    plotly_js = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"
    s = ctx["sections"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kelmarsh wake and shadow</title>
<meta name="description" content="Wake flow maps, wake losses checked against SCADA, and shadow flicker maps for the open Kelmarsh wind farm dataset.">
<style>{CSS}</style>
<script src="{plotly_js}" charset="utf-8"></script>
</head>
<body>
<main>
<h1>Kelmarsh wind farm: wakes and shadow flicker</h1>
<p class="lead">Six Senvion MM92 turbines in Northamptonshire, UK, published as open data by Cubico Sustainable
Investments. Engineering wake models (PyWake) are run on the real layout and checked against six years of
10-minute SCADA; shadow flicker is mapped on a regular grid around the farm.</p>
<div class="tiles">{tiles}</div>

<h2>Wake flow by wind direction</h2>
<p>{s['flow']}</p>
<div class="card">{ctx['flow_div']}</div>

<h2>Where the wake losses come from</h2>
<p>{s['rose']}</p>
<div class="card">{ctx['rose_div']}</div>
<div class="scroll">{ctx['aep_table']}</div>

<h2>Models against SCADA</h2>
<p>{s['validation']}</p>
<div class="card">{ctx['validation_div']}</div>
<div class="scroll">{ctx['validation_table']}</div>
<p>{s['efficiency']}</p>
<div class="card">{ctx['efficiency_div']}</div>

<h2>Shadow flicker</h2>
<p>{s['shadow']}</p>
<div class="card">{ctx['shadow_div']}</div>
<p>{s['calendar']}</p>
<div class="card">{ctx['calendar_div']}</div>

<h2>Method and limitations</h2>
{s['method']}

<footer>{s['footer']}</footer>
</main>
<script>{THEME_JS}</script>
</body>
</html>
"""
