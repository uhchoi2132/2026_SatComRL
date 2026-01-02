from __future__ import annotations
import argparse
import os
from pathlib import Path

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from satcomrl.envs.resource_env import SatelliteResourceEnv, ResourceConfig
from satcomrl.utils.seed import seed_everything
from satcomrl.utils.io import ensure_dir, save_json

def env_creator(env_config):
    cfg = ResourceConfig(**env_config)
    return SatelliteResourceEnv(cfg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-sats", type=int, default=4)
    ap.add_argument("--num-slots", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop-iters", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--use-wandb", type=int, default=int(os.getenv("SATCOMRL_USE_WANDB", "0")))
    ap.add_argument("--outdir", type=str, default="outputs/ppo_resource")
    args = ap.parse_args()

    seed_everything(args.seed)

    register_env("satcom_resource", env_creator)

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    env_conf = {
        "num_sats": args.num_sats,
        "num_slots": args.num_slots,
        "seed": args.seed,
        "random_opps": True,
    }

    config = (
        PPOConfig()
        .environment(env="satcom_resource", env_config=env_conf)
        .framework("torch")
        .env_runners(num_env_runners=args.num_workers)
        .training(
            model={
                # RLlib action masking expects observation dict with "action_mask"
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            train_batch_size=4000,
            minibatch_size=256,
            num_epochs=10,
            gamma=0.99,
            lr=3e-4,
        )
        .resources(num_gpus=0)
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    )

    algo = config.build_algo()

    out = ensure_dir(args.outdir)

    metrics_log = []
    checkpoint_path = None

    for i in range(args.stop_iters):
        res = algo.train()
        metrics_log.append({
            "iter": i,
            "episode_reward_mean": res.get("episode_reward_mean"),
            "episode_len_mean": res.get("episode_len_mean"),
            "timesteps_total": res.get("timesteps_total"),
        })
        reward = res.get("episode_reward_mean")
        len_mean = res.get("episode_len_mean")
        r_str = f"{reward:.3f}" if reward is not None else "None"
        l_str = f"{len_mean:.2f}" if len_mean is not None else "None"
        print(f"[iter {i}] reward_mean={r_str} len_mean={l_str}")

        if (i + 1) % 5 == 0:
            checkpoint_path = algo.save(out.as_posix())
            (out / "checkpoint_latest").write_text(checkpoint_path, encoding="utf-8")
            print("Saved checkpoint:", checkpoint_path)

    save_json(out / "metrics.json", {"config": vars(args), "env_conf": env_conf, "metrics": metrics_log})
    print("Saved:", out / "metrics.json")

    ray.shutdown()

if __name__ == "__main__":
    main()
