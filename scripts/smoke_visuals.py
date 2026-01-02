from __future__ import annotations
import numpy as np
from satcomrl.visualize.plots import plot_q_heatmap, plot_latency_hist


def smoke():
    q = np.random.randn(8, 8)
    fig1 = plot_q_heatmap(q, path="outputs/smoke_q_heatmap.png")
    samples = np.abs(np.random.randn(1000) * 10.0)
    fig2 = plot_latency_hist(samples, path="outputs/smoke_latency_hist.png")
    print("Wrote outputs/smoke_q_heatmap.png and outputs/smoke_latency_hist.png")

if __name__ == "__main__":
    smoke()
