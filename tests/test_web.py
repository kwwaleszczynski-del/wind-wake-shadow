import numpy as np
import pandas as pd
import plotly.graph_objects as go

from wind_wake_shadow import web


def test_div_embeds_typed_arrays_and_frames():
    fig = go.Figure(go.Heatmap(z=np.arange(6, dtype=np.uint8).reshape(2, 3)))
    fig.frames = [go.Frame(data=[go.Heatmap(z=np.ones((2, 3), dtype=np.uint8))], name="a")]
    html = web._div(fig, "x")
    assert '<div id="x" class="plot"></div>' in html
    assert '"bdata"' in html and '"dtype":"u1"' in html     # numpy arrays travel as base64, not as lists
    assert "Plotly.addFrames" in html


def test_table_html_escapes_and_formats():
    df = pd.DataFrame({"loss (%)": [7.6987, np.nan]}, index=pd.Index(["A & B", "<b>"], name="model"))
    out = web.table_html(df, {"loss (%)": "{:.1f}"})
    assert "A &amp; B" in out and "&lt;b&gt;" in out
    assert "<td>7.7</td>" in out and "<td></td>" in out


def test_build_page_has_sections_and_theme():
    sections = {k: k for k in ("flow", "rose", "validation", "efficiency", "shadow", "calendar", "method", "footer")}
    ctx = {"tiles": [("1 %", "one")], "sections": sections}
    ctx |= {k: f"<div>{k}</div>" for k in ("flow_div", "rose_div", "aep_table", "validation_div", "validation_table",
                                           "efficiency_div", "shadow_div", "calendar_div")}
    page = web.build_page(ctx)
    assert page.startswith("<!doctype html>")
    assert "<title>Kelmarsh wake and shadow</title>" in page
    assert "prefers-color-scheme: dark" in page
    assert "https://cdn.plot.ly/plotly-" in page
