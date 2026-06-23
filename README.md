# XQC — Online Condition-Number Research

Built on the [XQC implementation](https://github.com/danielpalenicek/xqc) (Palenicek et al., ICLR 2026).

## Background

XQC is a JAX/Flax actor-critic built on Soft Actor-Critic (SAC). Its central claim is that the *conditioning* of the critic's optimization landscape drives sample efficiency. Conditioning is quantified by the condition number κ = |λ_max| / |λ_min| of the critic's loss Hessian.

The paper shows that combining three architectural components produces κ orders of magnitude smaller than baselines:

| Component | Flag | XQC default |
|---|---|---|
| Batch normalization (pre-activation) | `agent.use_batch_norm` / `agent.pre_activation_bn` | `1` / `1` |
| Weight normalization (weights projected to unit sphere each step) | `agent.use_weight_norm` | `1` |
| Distributional cross-entropy (CE) loss (C51-style, 101 bins) | `agent.critic_loss` | `categorical` |

The paper measures κ **offline** — full Hessian eigenspectrum via Lanczos on saved checkpoints. This repo closes that loop by computing κ **online during training** as a cheap signal, eventually to steer hyperparameters. Phase 1 (this repo) is measurement only.

## What we're testing

The core question: does a cheap online κ estimator (power iteration, 2–3 HVPs, one fixed minibatch) reproduce the paper's offline finding?

**Expected result:** XQC (BN + WN + CE) → low, stable κ. SAC (all three off) → high, volatile κ.

To test this cleanly, we confirmed that XQC reduces to plain SAC via config flags alone — no new code needed. The three components are each individually toggleable, and four additional flags align the remaining hyperparameters with canonical SAC defaults.

## Setup

### 1. Clone

```bash
git clone --recurse-submodules <your-repo-url>
cd xqc
```

### 2. Install dependencies

```bash
uv sync
```

All commands below use `uv run python train_parallel.py ...`.

## Running experiments

### XQC (full)

```bash
uv run python train_parallel.py env=<env> seed=0
```

### Built-in baselines

```bash
# CrossQ (BN post-activation, MSE loss, no WN)
uv run python train_parallel.py agent=crossq env=<env> seed=0

# CrossQ + Weight Normalization
uv run python train_parallel.py agent=crossq_wn env=<env> seed=0
```

### Recovering SAC via ablation

All three XQC components off, plus flags to align remaining hyperparameters with canonical SAC:

```bash
uv run python train_parallel.py \
  agent=xqc \
  agent.use_batch_norm=0 \
  agent.use_weight_norm=0 \
  agent.critic_loss=mse \
  agent.reward_normalization=false \
  agent.policy_delay=1 \
  agent.lr_end=3e-4 \
  agent.hidden_dims_critic=[256,256] \
  agent.hidden_dims_actor=[256,256] \
  env=<env> seed=0
```

| Extra flag | Why |
|---|---|
| `agent.policy_delay=1` | XQC default is 3 (delayed actor updates); SAC updates every step |
| `agent.lr_end=3e-4` | Matches `actor_lr` to flatten the built-in linear LR decay schedule |
| `agent.hidden_dims_*=[256,256]` | XQC default is 4 layers × 512/256; canonical SAC uses 2 layers × 256 |

### Single-component ablations

```bash
# No CE loss (keep BN + WN, switch to MSE)
uv run python train_parallel.py agent=xqc agent.critic_loss=mse env=<env> seed=0

# No weight normalization
uv run python train_parallel.py agent=xqc agent.use_weight_norm=0 env=<env> seed=0

# No batch normalization
uv run python train_parallel.py agent=xqc agent.use_batch_norm=0 env=<env> seed=0
```

### Key training arguments

| Argument | Default | Description |
|---|---|---|
| `env` | `h1-walk-v0` | Environment name (`dmc`, `mujoco`, `myo`, `hb` suites) |
| `seed` | `0` | Random seed |
| `max_steps` | `1_000_000` | Total training steps |
| `num_seeds` | `10` | Parallel seeds (JAX vmap) |
| `wandb.mode` | `disabled` | Set to `online` to log to Weights & Biases |

## Online κ logging

κ logging is off by default. Enable with `kappa_logging.enabled=true`. It logs κ, λ_max, and λ_min to wandb every K env steps using power iteration (2–3 HVPs per call on a fixed minibatch). Overhead is negligible.

### Config flags

| Flag | Default | Description |
|---|---|---|
| `kappa_logging.enabled` | `false` | Enable online κ logging |
| `kappa_logging.interval` | `1000` | Log every this many env steps |
| `kappa_logging.n_iters_max` | `3` | Power-iteration steps for λ_max |
| `kappa_logging.n_iters_min` | `5` | Spectral-shift steps for λ_min |

Wandb series: `seed{i}/kappa/kappa`, `seed{i}/kappa/lambda_max`, `seed{i}/kappa/lambda_min`. Use log-scale y-axes.

### Test 1 — verify logging doesn't affect training

κ computation is purely functional (JAX `jvp`/`grad`, no agent state mutation). Confirm by running the same seed with logging on and off and checking that the reward traces are numerically identical.

```bash
# Reference
uv run python train_parallel.py \
  env=dog-trot seed=0 max_steps=10000 num_seeds=1 \
  kappa_logging.enabled=false

# With logging
uv run python train_parallel.py \
  env=dog-trot seed=0 max_steps=10000 num_seeds=1 \
  kappa_logging.enabled=true kappa_logging.interval=1000
```

`seed0/r` must be numerically identical across both runs. Any divergence means something is leaking into the gradient path.

### Test 2 — κ contrast (XQC vs SAC)

Canonical validation environment: `dog-trot` (DMC).

```bash
# XQC — expect low, stable κ
uv run python train_parallel.py \
  agent=xqc env=dog-trot seed=0 num_seeds=1 \
  kappa_logging.enabled=true kappa_logging.interval=1000 \
  wandb.mode=online

# SAC ablation — expect high, volatile κ
uv run python train_parallel.py \
  agent=xqc env=dog-trot seed=0 num_seeds=1 \
  agent.use_batch_norm=0 \
  agent.use_weight_norm=0 \
  agent.critic_loss=mse \
  agent.reward_normalization=false \
  agent.policy_delay=1 \
  agent.lr_end=3e-4 \
  agent.hidden_dims_critic=[256,256] \
  agent.hidden_dims_actor=[256,256] \
  kappa_logging.enabled=true kappa_logging.interval=1000 \
  wandb.mode=online
```

Start with `num_seeds=1` for a quick check; use `num_seeds=10` for the final figure.

### Test 3 — single-component ablations with κ

```bash
# XQC minus CE loss only
uv run python train_parallel.py \
  agent=xqc env=dog-trot seed=0 \
  agent.critic_loss=mse \
  kappa_logging.enabled=true kappa_logging.interval=1000 wandb.mode=online

# XQC minus WN only
uv run python train_parallel.py \
  agent=xqc env=dog-trot seed=0 \
  agent.use_weight_norm=0 \
  kappa_logging.enabled=true kappa_logging.interval=1000 wandb.mode=online

# XQC minus BN only
uv run python train_parallel.py \
  agent=xqc env=dog-trot seed=0 \
  agent.use_batch_norm=0 \
  kappa_logging.enabled=true kappa_logging.interval=1000 wandb.mode=online
```
