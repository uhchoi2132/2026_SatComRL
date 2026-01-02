from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, List

import ray
from ray import serve
from ray.rllib.algorithms.algorithm import Algorithm
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="SatComRL Policy Service")

class ActRequest(BaseModel):
    obs: List[float]
    mask: List[float]

@serve.deployment(route_prefix="/")
@serve.ingress(app)
class PolicyServer:
    def __init__(self, checkpoint: str):
        self.algo = Algorithm.from_checkpoint(checkpoint)

    @app.get("/health")
    def health(self):
        return {"status": "ok"}

    @app.post("/act")
    def act(self, req: ActRequest):
        # RLlib expects dict obs with action_mask
        obs = {"obs": req.obs, "action_mask": req.mask}
        action = self.algo.compute_single_action(obs, explore=False)
        return {"action": int(action)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to RLlib checkpoint dir or file")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    ray.init(ignore_reinit_error=True)
    serve.start(detached=False, http_options={"host": args.host, "port": args.port})
    PolicyServer.bind(args.checkpoint)
    print(f"Serving on http://{args.host}:{args.port}")
    import time
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()
