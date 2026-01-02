from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from satcomrl.envs.routing_env import SatelliteNetworkEnv, RoutingConfig
from satcomrl.algos.qrouting import train_qrouting, QRoutingParams, greedy_route_from_Q
from satcomrl.algos.dijkstra import dijkstra_route, dijkstra_mq_route
from satcomrl.utils.seed import seed_everything
from satcomrl.utils.io import ensure_dir, save_json

def eval_route(env: SatelliteNetworkEnv, route: list[int], trials: int = 200, return_samples: bool = False) -> dict:
    """Evaluate expected latency by simulating the route step-by-step multiple times.

    If return_samples=True, also return list of per-hop queue latencies aggregated across trials under key "queue_samples_ms".
    """
    total = []
    q_samples = []
    for _ in range(trials):
        env.reset()
        lat = 0.0
        cur = env.cfg.src
        for nxt in route[1:]:
            # force transition; sample queue delay via env.step
            env.state = cur
            _, r, done, trunc, info = env.step(nxt)
            lat += (info["l_prop"] + info["l_queue"])
            q_samples.append(float(info["l_queue"]))
            cur = nxt
            if done or trunc:
                break
        total.append(lat)
    stats = {"mean_ms": float(np.mean(total)), "p95_ms": float(np.percentile(total, 95)), "trials": trials}
    if return_samples:
        stats["queue_samples_ms"] = q_samples
    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--epsilon", type=float, default=0.3)
    ap.add_argument("--epsilon-decay", type=float, default=0.9999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-tensorboard", type=int, default=1)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="outputs/qrouting")
    args = ap.parse_args()

    seed_everything(args.seed)
    cfg = RoutingConfig(seed=args.seed)
    env = SatelliteNetworkEnv(cfg)
    params = QRoutingParams(
        episodes=args.episodes, alpha=args.alpha, gamma=args.gamma,
        epsilon=args.epsilon, epsilon_decay=args.epsilon_decay
    )

    # optional logging
    from satcomrl.visualize import TrainerLogger
    logger = TrainerLogger(logdir=(args.outdir + "/logs"), run_name=(args.run_name or f"qr_seed{args.seed}"), use_wandb=False, config=vars(args)) if args.use_tensorboard else None

    Q = train_qrouting(env, params, logger=logger)
    q_route = greedy_route_from_Q(env, Q)
    dj_route = dijkstra_route(env)
    djmq_route = dijkstra_mq_route(env, mean_queue_ms=(cfg.lmin_ms + cfg.lmax_ms)/2.0)

    # collect detailed eval samples for plotting
    qr_eval = eval_route(env, q_route, trials=1000, return_samples=True)
    dj_eval = eval_route(env, dj_route, trials=1000, return_samples=True)
    djmq_eval = eval_route(env, djmq_route, trials=1000, return_samples=True)

    results = {
        "qrouting": {"route": q_route, "eval": {k: v for k, v in qr_eval.items() if k != "queue_samples_ms"}},
        "dijkstra": {"route": dj_route, "eval": {k: v for k, v in dj_eval.items() if k != "queue_samples_ms"}},
        "dijkstra_mq": {"route": djmq_route, "eval": {k: v for k, v in djmq_eval.items() if k != "queue_samples_ms"}},
        "params": vars(args)
    }

    out = ensure_dir(args.outdir)
    np.save(out / "Q.npy", Q)
    save_json(out / "results.json", results)

    # save visual artifacts
    try:
        from satcomrl.visualize.plots import plot_q_heatmap, plot_latency_hist

        qpath = out / "Q.png"
        plot_q_heatmap(Q, path=qpath.as_posix())

        hist_q = out / "queue_hist_qrouting.png"
        plot_latency_hist(qr_eval.get("queue_samples_ms", []), path=hist_q.as_posix(), title="QRouting queue latencies (ms)")

        hist_dj = out / "queue_hist_dijkstra.png"
        plot_latency_hist(dj_eval.get("queue_samples_ms", []), path=hist_dj.as_posix(), title="Dijkstra queue latencies (ms)")

        hist_djmq = out / "queue_hist_dijkstra_mq.png"
        plot_latency_hist(djmq_eval.get("queue_samples_ms", []), path=hist_djmq.as_posix(), title="Dijkstra MQ queue latencies (ms)")

        print("Saved visualizations under:", out)
    except Exception as e:
        print("Skipping plotting (missing deps or error):", e)

    print("Saved:", out / "results.json")
    print(results["qrouting"]["eval"], results["dijkstra"]["eval"], results["dijkstra_mq"]["eval"])

    if logger:
        logger.close()

if __name__ == "__main__":
    main()
