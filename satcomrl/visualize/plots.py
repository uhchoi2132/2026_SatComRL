from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence


def plot_q_heatmap(Q: np.ndarray, path: str | None = None, title: str = "Q-Value Heatmap") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(Q, cmap="viridis")
    ax.set_xlabel("action (to node)")
    ax.set_ylabel("state (node)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_latency_hist(samples: Sequence[float], path: str | None = None, title: str = "Latency Histogram") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(samples, bins=50, color="tab:blue", alpha=0.8)
    ax.set_xlabel("latency (ms)")
    ax.set_ylabel("count")
    ax.set_title(title)
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig
