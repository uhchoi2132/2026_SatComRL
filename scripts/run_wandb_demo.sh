#!/usr/bin/env bash
# Simple demo: run a short PPO training with W&B in offline mode
set -euo pipefail
export WANDB_MODE=offline
export SATCOMRL_USE_WANDB=1
python -m satcomrl.train.train_ppo --stop-iters 2 --use-wandb 1 --use-tensorboard 0 --run-name demo_wandb
echo "W&B demo finished (offline). Check outputs/*/logs or local wandb directory." 
