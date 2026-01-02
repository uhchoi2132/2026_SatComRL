from __future__ import annotations
import os
from typing import Optional, Dict, Any

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


class TrainerLogger:
    """Simple logger wrapper supporting TensorBoard and optional Weights & Biases.

    Usage:
        logger = TrainerLogger(logdir="outputs/logs", run_name="exp1", use_wandb=False)
        logger.log_scalar("train/reward", reward, step)
        logger.close()
    """

    def __init__(self, logdir: str = "outputs/logs", run_name: Optional[str] = None, use_wandb: bool = False, config: Optional[Dict[str, Any]] = None):
        self.logdir = os.fspath(logdir)
        self.run_name = run_name or "run"
        self.tb_writer = None
        self.wandb = None

        tb_dir = os.path.join(self.logdir, self.run_name)
        os.makedirs(tb_dir, exist_ok=True)

        if SummaryWriter is not None:
            self.tb_writer = SummaryWriter(log_dir=tb_dir)

        if use_wandb:
            try:
                import wandb

                self.wandb = wandb
                # Initialize wandb with provided config if any
                wandb.init(project="satcomrl", name=self.run_name, config=config, dir=self.logdir)
            except Exception:
                self.wandb = None

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.tb_writer:
            try:
                self.tb_writer.add_scalar(tag, value, step)
            except Exception:
                pass
        if self.wandb:
            try:
                self.wandb.log({tag: value}, step=step)
            except Exception:
                pass

    def log_scalars(self, scalar_dict: Dict[str, float], step: int) -> None:
        for k, v in scalar_dict.items():
            self.log_scalar(k, v, step)

    def log_config(self, config: Dict[str, Any]) -> None:
        if self.wandb:
            try:
                self.wandb.config.update(config)
            except Exception:
                pass

    def close(self) -> None:
        if self.tb_writer:
            try:
                self.tb_writer.close()
            except Exception:
                pass
        if self.wandb:
            try:
                self.wandb.finish()
            except Exception:
                pass
