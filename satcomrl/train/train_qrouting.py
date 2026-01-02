from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from satcomrl.envs.routing_env import SatelliteNetworkEnv, RoutingConfig
from satcomrl.algos.qrouting import train_qrouting, QRoutingParams, greedy_route_from_Q
from satcomrl.algos.dijkstra import dijkstra_route, dijkstra_mq_route
from satcomrl.utils.seed import seed_everything
from satcomrl.utils.io import ensure_dir, save_json

def eval_route(env: SatelliteNetworkEnv, route: list[int], trials: int = 200) -> dict:
    # Evaluate expected latency by simulating the route step-by-step multiple times
    total = []
    for _ in range(trials):
        env.reset()
        lat = 0.0
        cur = env.cfg.src
        for nxt in route[1:]:
            # force transition; sample queue delay via env.step
            env.state = cur
            _, r, done, trunc, info = env.step(nxt)
            lat += (info["l_prop"] + info["l_queue"])
            cur = nxt
            if done or trunc:
                break
        total.append(lat)
    return {"mean_ms": float(np.mean(total)), "p95_ms": float(np.percentile(total, 95)), "trials": trials}

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

    results = {
        "qrouting": {"route": q_route, "eval": eval_route(env, q_route)},
        "dijkstra": {"route": dj_route, "eval": eval_route(env, dj_route)},
        "dijkstra_mq": {"route": djmq_route, "eval": eval_route(env, djmq_route)},
        "params": vars(args)
    }

    out = ensure_dir(args.outdir)
    np.save(out / "Q.npy", Q)
    save_json(out / "results.json", results)
    print("Saved:", out / "results.json")
    print(results["qrouting"]["eval"], results["dijkstra"]["eval"], results["dijkstra_mq"]["eval"])

    if logger:
        logger.close()

if __name__ == "__main__":
    main()
