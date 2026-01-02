from __future__ import annotations
import argparse
import json
import os
import numpy as np
from pathlib import Path

try:
    import dash
    from dash import html, dcc
    import plotly.express as px
except Exception:
    dash = None


def build_app(outputs_dir: str):
    outputs_dir = Path(outputs_dir)
    qpath = outputs_dir / "Q.npy"
    results = outputs_dir / "results.json"

    if not dash:
        raise RuntimeError("Dash or Plotly not available. Install 'dash' and 'plotly' to run the dashboard.")

    app = dash.Dash(__name__)

    q = None
    if qpath.exists():
        q = np.load(qpath)

    res = None
    if results.exists():
        with open(results, "r", encoding="utf-8") as f:
            res = json.load(f)

    children = [html.H3("SatComRL Dashboard")]

    if q is not None:
        fig = px.imshow(q, color_continuous_scale="viridis", labels={'x':'action', 'y':'state'}, title='Q-Value Heatmap')
        children.append(dcc.Graph(figure=fig))

    if res is not None:
        # show metrics (if present)
        if "qrouting" in res and "eval" in res["qrouting"]:
            eval = res["qrouting"]["eval"]
            children.append(html.Div([html.P(f"QRouting mean latency: {eval.get('mean_ms'):.2f} ms"), html.P(f"p95: {eval.get('p95_ms'):.2f} ms")]))

    app.layout = html.Div(children=children)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=str, default="outputs/qrouting")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()

    app = build_app(args.outputs)
    app.run_server(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
