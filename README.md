# Condition-Number-Controlled Training

Built on [XQC](https://github.com/danielpalenicek/xqc) (Palenicek et al., ICLR 2026).

## Background

XQC is a JAX/Flax actor-critic that argues the condition number κ = |λ_max| / |λ_min| of the critic's loss Hessian drives sample efficiency. Three components together produce κ orders of magnitude smaller than SAC: batch normalization (BN), weight normalization (WN), and a distributional cross-entropy (CE) loss.

The paper measures κ offline via full Lanczos on saved checkpoints. This repo adds a cheap **online** κ estimator (power iteration, 2–3 HVPs per call, one fixed minibatch) and uses it to adaptively control the UTD (updates-to-data) ratio: reduce UTD when κ is high (ill-conditioned, risky to over-update), increase it when κ is low (well-conditioned, safe to extract more value per step).

**UTD context:** The default UTD is 2 — standard for SAC-family methods (the original SAC paper used 1). Model-free off-policy methods are bottlenecked by replay buffer diversity, so high UTD risks overfitting to stale data. Model-based methods (Dreamer, MBPO) can use 16–64 because a world model generates cheap synthetic data. CrossQ pushes UTD to 16+ using a special joined batch norm forward pass that stabilizes high-UTD training. The hypothesis here is that low κ directly licenses higher UTD: if the critic's optimization landscape is well-conditioned, more gradient steps per env step are safe. That's what Experiments 3 and 4 test.

## Setup

```bash
git clone --recurse-submodules https://github.com/paul2pam/Condition-Number-Controlled-Training.git
cd Condition-Number-Controlled-Training
uv sync
```

## Experiments

### Experiment 1 — Verify κ logging doesn't affect training

Run the same seed with logging on and off. The reward traces must be numerically identical — this confirms κ computation is purely observational.

```bash
# Reference (logging off)
uv run python train_parallel.py \
  env=dog-trot seed=0 max_steps=10000 num_seeds=1

# With κ logging
uv run python train_parallel.py \
  env=dog-trot seed=0 max_steps=10000 num_seeds=1 \
  kappa_logging.enabled=true kappa_logging.interval=1000
```

**Pass criterion:** `seed0/r` is identical across both runs. Any divergence means κ computation is leaking into the gradient path.

### Experiment 2 — κ contrast: XQC vs SAC

Expected: XQC produces low, stable κ. SAC produces high, volatile κ. Run both on `dog-trot` and compare `seed0/kappa/kappa` in wandb (log-scale y-axis).

```bash
# XQC
uv run python train_parallel.py \
  agent=xqc env=dog-trot seed=0 num_seeds=1 \
  kappa_logging.enabled=true kappa_logging.interval=1000 \
  wandb.mode=online

# SAC
uv run python train_parallel.py \
  agent=xqc env=dog-trot seed=0 num_seeds=1 \
  agent.use_batch_norm=0 agent.use_weight_norm=0 agent.critic_loss=mse \
  agent.reward_normalization=false agent.policy_delay=1 agent.lr_end=3e-4 \
  agent.hidden_dims_critic=[256,256] agent.hidden_dims_actor=[256,256] \
  kappa_logging.enabled=true kappa_logging.interval=1000 \
  wandb.mode=online
```

Start with `num_seeds=1` for a quick check; scale to `num_seeds=10` for the final figure.

### Experiment 3 — κ-controlled UTD on XQC

**Do this after Experiment 2.** Look at the `seed0/kappa/kappa` range for XQC on dog-trot, then set `kappa_low` just above it and `kappa_high` well above SAC's range. Expected result: UTD stays near the starting value (XQC's κ is low and stable, so control rarely fires).

```bash
uv run python train_parallel.py \
  agent=xqc env=dog-trot seed=0 num_seeds=1 \
  kappa_logging.enabled=true kappa_logging.interval=1000 \
  kappa_control.enabled=true \
  kappa_control.kappa_high=<from Exp 2> \
  kappa_control.kappa_low=<from Exp 2> \
  wandb.mode=online
```

Watch `seed0/utd_ratio` alongside `seed0/kappa/kappa`.

### Experiment 4 — κ-controlled UTD on SAC

Same thresholds as Experiment 3. Expected result: UTD drops when SAC's κ spikes above `kappa_high`, then stays low or oscillates.

```bash
uv run python train_parallel.py \
  agent=xqc env=dog-trot seed=0 num_seeds=1 \
  agent.use_batch_norm=0 agent.use_weight_norm=0 agent.critic_loss=mse \
  agent.reward_normalization=false agent.policy_delay=1 agent.lr_end=3e-4 \
  agent.hidden_dims_critic=[256,256] agent.hidden_dims_actor=[256,256] \
  kappa_logging.enabled=true kappa_logging.interval=1000 \
  kappa_control.enabled=true \
  kappa_control.kappa_high=<from Exp 2> \
  kappa_control.kappa_low=<from Exp 2> \
  wandb.mode=online
```

## Reference

### κ logging flags

| Flag | Default | Description |
|---|---|---|
| `kappa_logging.enabled` | `false` | Enable online κ logging |
| `kappa_logging.interval` | `1000` | Steps between κ estimates |
| `kappa_logging.n_iters_max` | `3` | Power-iteration steps for λ_max |
| `kappa_logging.n_iters_min` | `5` | Spectral-shift steps for λ_min |

### SAC ablation flags

The SAC command above sets these on top of `agent=xqc`:

| Flag | Value | Why |
|---|---|---|
| `agent.use_batch_norm` | `0` | Remove BN |
| `agent.use_weight_norm` | `0` | Remove WN |
| `agent.critic_loss` | `mse` | MSE Bellman loss instead of CE |
| `agent.reward_normalization` | `false` | XQC-specific, not in SAC |
| `agent.policy_delay` | `1` | SAC updates actor every step; XQC default is 3 |
| `agent.lr_end` | `3e-4` | Flattens the built-in LR decay schedule |
| `agent.hidden_dims_*` | `[256,256]` | Canonical SAC width; XQC default is 4×512 |

### Other agents

```bash
uv run python train_parallel.py agent=crossq env=dog-trot seed=0
uv run python train_parallel.py agent=crossq_wn env=dog-trot seed=0
```

### κ control flags

| Flag | Default | Description |
|---|---|---|
| `kappa_control.enabled` | `false` | Enable UTD adjustment. Requires `kappa_logging.enabled=true` |
| `kappa_control.kappa_high` | `1000.0` | **PLACEHOLDER** — κ above this reduces UTD. Set from Experiment 2 |
| `kappa_control.kappa_low` | `10.0` | **PLACEHOLDER** — κ below this increases UTD. Set from Experiment 2 |
| `kappa_control.utd_min` | `1` | Floor on UTD ratio |
| `kappa_control.utd_max` | `8` | Ceiling on UTD ratio |
| `kappa_control.utd_step` | `1` | How much UTD changes per adjustment |

To verify control is firing before using calibrated thresholds, use `kappa_control.kappa_high=0` — this forces UTD to drop to `utd_min` on the first κ estimate and `seed0/utd_ratio` should show the change in wandb.

### Single-component ablations with κ

```bash
uv run python train_parallel.py agent=xqc env=dog-trot seed=0 \
  agent.critic_loss=mse kappa_logging.enabled=true wandb.mode=online        # no CE

uv run python train_parallel.py agent=xqc env=dog-trot seed=0 \
  agent.use_weight_norm=0 kappa_logging.enabled=true wandb.mode=online      # no WN

uv run python train_parallel.py agent=xqc env=dog-trot seed=0 \
  agent.use_batch_norm=0 kappa_logging.enabled=true wandb.mode=online       # no BN
```
