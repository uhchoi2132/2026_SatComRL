# SatComRL — Web/Cloud-ready Satellite Constellation RL Sandbox

SatComRL is a **cloud-first, reproducible** research template for:
- **Use Case A (SatCom Routing):** Q-routing / Q-learning over an ISL graph with stochastic queueing delays.
- **Use Case B (EO Resource Scheduling):** PPO over a Gymnasium environment with battery/memory dynamics and AT/GS opportunities.
- **Experiment tracking:** optional Weights & Biases (W&B)
- **Serving / deployment:** Ray Serve + FastAPI-style JSON inference endpoint
- **Web/Cloud development:** GitHub Codespaces (devcontainer) + Docker

> This repo is intentionally minimal and runnable. Replace the toy scenario generators with your own orbit/access-window pipeline later (e.g., Orekit/GODOT/your simulator).

---

## Quickstart (Local)
### 1) Create venv (recommended)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Train routing (tabular Q-learning)
```bash
python -m satcomrl.train.train_qrouting --episodes 20000
```

### 3) Train resource scheduling (RLlib PPO)
```bash
python -m satcomrl.train.train_ppo --stop-iters 20 --num-sats 4 --num-slots 40
```

Artifacts are written into `./outputs/`.

---

## Quickstart (Docker)
```bash
docker build -t satcomrl:latest .
docker run --rm -it -p 8000:8000 -v "$PWD:/workspace" satcomrl:latest bash
```

Then run:
```bash
python -m satcomrl.train.train_qrouting --episodes 20000
python -m satcomrl.train.train_ppo --stop-iters 20 --num-sats 4 --num-slots 40
```

---

## GitHub Codespaces (Web IDE)
1. Push this repo to GitHub.
2. **Code → Codespaces → Create codespace**
3. Terminal inside Codespaces:
```bash
python -m satcomrl.train.train_ppo --stop-iters 10
```

---

## Experiment Tracking (Optional)
Set env var:
```bash
export WANDB_API_KEY=...
export SATCOMRL_USE_WANDB=1
```
Training scripts will log metrics automatically.

---

## Serving a Trained Policy (Ray Serve)
After PPO training, start the server:
```bash
python -m satcomrl.serve.serve_policy --checkpoint outputs/ppo_resource/checkpoint_latest
```

Then query it:
```bash
curl -X POST http://localhost:8000/act -H "Content-Type: application/json" -d '{"obs":[0.5,0.1,0,0,1, 0.6,0.2,0,0,0], "mask":[1,1,1,1, ...]}'
```

In practice you will call this from your simulator/back-end.

---

## Repo Layout
- `satcomrl/envs/` Gymnasium environments
- `satcomrl/algos/` Q-routing (tabular) + utilities
- `satcomrl/train/` training entrypoints
- `satcomrl/serve/` Ray Serve deployment
- `.devcontainer/` Codespaces / Dev Container config
- `docker-compose.yml` optional local orchestration

---

## Notes
- The resource environment uses **action masking** for feasibility (AT/GS presence). Battery/memory violations are discouraged via reward penalties.
- The routing environment samples queueing delays per link; Dijkstra baselines are included.
- This template uses **PyTorch** backend for RLlib by default for portability.

