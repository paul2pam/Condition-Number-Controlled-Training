# Implementation Guide: Online Condition-Number (κ) Logging + κ-Controlled UTD

This is a **portable, drop-in spec** for adding two features to a JAX/Flax actor-critic codebase:

1. **Online κ logging** — cheaply estimate the condition number κ = |λ_max| / |λ_min| of the critic's loss Hessian *during* training (not offline on checkpoints).
2. **κ-controlled UTD** — use that κ signal to adaptively raise/lower the updates-to-data (UTD) ratio.

An agent implementing this in a new codebase should read this whole file first, then adapt the integration points to the host code. The estimator itself (Part A) is nearly copy-paste; the integration (Parts B–D) requires mapping to the host's training loop and config system.

---

## Conceptual background

The condition number κ of the critic loss Hessian measures how well-conditioned the optimization landscape is. Low κ → stable gradients, safe to take more update steps per env step (high UTD). High κ → ill-conditioned, more updates risk divergence.

The strategy:
- Estimate λ_max (largest eigenvalue) and λ_min (smallest) of the Hessian via **matrix-free power iteration** — never materialize the full Hessian.
- The Hessian-vector product (HVP) is computed with **forward-over-reverse autodiff** (`jvp` of `grad`), costing ~1–2× a single gradient. A handful of HVPs every K steps is negligible overhead.
- Feed κ into a simple bang-bang controller that nudges UTD up or down.

**Critical correctness requirement:** the Hessian must be taken over the *same loss the critic is actually trained on*, w.r.t. the *critic parameters only*. If the codebase supports multiple losses (e.g. MSE vs. distributional cross-entropy), the estimator must read the active loss from config — taking an MSE Hessian on a CE-trained critic silently produces a meaningless κ.

---

## Prerequisites / assumptions about the host codebase

This spec assumes the host codebase has:

| Requirement | Why | How to check |
|---|---|---|
| An autodiff framework (JAX **or** PyTorch) | HVP relies on second-order autodiff | `import jax` or `import torch` |
| A critic loss function `loss(params, ...) -> (scalar, aux)` | Hessian is taken over this | Find the function passed to the optimizer |
| Critic params accessible as a pytree | `ravel_pytree` flattens them | Usually `model.params` or `state.params` |
| A training loop with an explicit UTD / "updates per step" variable | Control mutates it | Look for `for _ in range(updates_per_step)` |
| A way to sample a fixed minibatch | κ must use the same batch each time to be comparable | Replay buffer `.sample()` |
| A metric logger (wandb, tensorboard, etc.) | Log κ, λ_max, λ_min, UTD | `wandb.log` or similar |

**Framework:** Part A gives the JAX estimator; **Part A-PT** gives the complete PyTorch equivalent. Parts B (config), C (loop integration), and D (extensions) are framework-agnostic — they wrap whichever estimator you picked.

---

## Part A — The κ estimator (core reusable module)

Drop this into a new file, e.g. `kappa.py`. The only host-specific assumption is the **loss function signature**: here it is `loss_fn(critic_params=..., <other bound args>) -> (scalar_loss, aux_dict)`. Adapt the `partial(...)` binding in `compute_kappa_metrics` to match the host's loss signature.

```python
from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, jvp
from jax.flatten_util import ravel_pytree


def _model_for_seed(seed, model):
    """If the codebase vmaps over seeds, slice out one seed's params/batch_stats.
    If there is no seed vmap, delete this and pass the model directly."""
    return model.replace(
        params=jax.tree.map(lambda p: p[seed], model.params),
        batch_stats=(
            jax.tree.map(lambda p: p[seed], model.batch_stats)
            if model.batch_stats else None
        ),
    )


def _make_hvp_fn(loss_fn_fixed, flat_params, unravel):
    """Returns jit-compiled (flat_params, flat_v) -> flat Hv via forward-over-reverse.
    H = Hessian of loss w.r.t. critic params. Never materialized."""
    def flat_loss(flat_p):
        return loss_fn_fixed(critic_params=unravel(flat_p))[0]  # [0] = scalar loss

    @jit
    def hvp_fn(flat_p, flat_v):
        return jvp(grad(flat_loss), [flat_p], [flat_v])[1]

    return hvp_fn


def _power_iter_lambda_max(hvp_fn, flat_params, n_iters):
    """Power iteration for the dominant eigenvalue. 2–3 iters is enough online."""
    n = flat_params.shape[0]
    v = jax.random.normal(jax.random.PRNGKey(42), (n,))   # fixed key → comparable across time
    v = v / jnp.linalg.norm(v)
    lam = jnp.array(0.0)
    for _ in range(n_iters):
        Hv = hvp_fn(flat_params, v)
        lam = jnp.dot(v, Hv)                               # Rayleigh quotient
        v = Hv / (jnp.linalg.norm(Hv) + 1e-8)
    return lam


def _spectral_shift_lambda_min(hvp_fn, flat_params, lambda_max, n_iters):
    """Power-iterate on (λ_max·I − H). Its dominant eigenvalue is (λ_max − λ_min),
    so λ_min = λ_max − dominant_eigenvalue_of_shifted. This is the finicky part."""
    n = flat_params.shape[0]
    v = jax.random.normal(jax.random.PRNGKey(99), (n,))
    v = v / jnp.linalg.norm(v)
    lam_shifted = jnp.array(0.0)
    for _ in range(n_iters):
        Hv = hvp_fn(flat_params, v)
        shifted_Hv = lambda_max * v - Hv
        lam_shifted = jnp.dot(v, shifted_Hv)
        v = shifted_Hv / (jnp.linalg.norm(shifted_Hv) + 1e-8)
    return lambda_max - lam_shifted


def compute_kappa_metrics(agent, fixed_batch, num_seeds, n_iters_max=3, n_iters_min=5):
    """Online κ = |λ_max| / |λ_min| of the critic loss Hessian.

    - Uses the agent's ACTIVE critic loss (so it follows whatever loss config is set).
    - Hessian is over critic.params only, with batch_stats held fixed.
    - fixed_batch is sampled ONCE and reused every call → κ comparable across checkpoints.
    - Purely functional: does NOT mutate agent state. Safe to call inside the loop.
    Returns dict of per-seed arrays for the logger.
    """
    kappas, lambdas_max, lambdas_min = [], [], []

    for j in range(num_seeds):
        critic_j = _model_for_seed(j, agent.critic)

        # Bind every loss arg EXCEPT critic_params. Adapt these kwargs to the host loss.
        loss_fn_fixed = partial(
            agent.critic.loss_fn,                          # <-- active loss (follows config)
            critic_batch_stats=critic_j.batch_stats,
            key=jax.random.PRNGKey(0),
            actor=_model_for_seed(j, agent.actor),
            critic=critic_j,
            target_critic=_model_for_seed(j, agent.target_critic),
            temperature=_model_for_seed(j, agent.temperature),
            batch=jax.tree.map(lambda x: x[j, 0], fixed_batch),  # (seeds, 1, B, ...) -> (B, ...)
        )

        flat_params, unravel = ravel_pytree(critic_j.params)
        assert flat_params.shape[0] > 0, f"[kappa] seed {j}: critic has no params"

        hvp_fn = _make_hvp_fn(loss_fn_fixed, flat_params, unravel)

        lam_max = _power_iter_lambda_max(hvp_fn, flat_params, n_iters_max)
        lam_min = _spectral_shift_lambda_min(hvp_fn, flat_params, lam_max, n_iters_min)
        kappa = jnp.abs(lam_max) / (jnp.abs(lam_min) + 1e-8)

        assert jnp.isfinite(lam_max), f"[kappa] seed {j}: lambda_max not finite"
        assert jnp.isfinite(lam_min), f"[kappa] seed {j}: lambda_min not finite (raise n_iters_min)"

        kappas.append(float(kappa))
        lambdas_max.append(float(lam_max))
        lambdas_min.append(float(lam_min))

    return {
        "kappa/kappa":      jnp.array(kappas),
        "kappa/lambda_max": jnp.array(lambdas_max),
        "kappa/lambda_min": jnp.array(lambdas_min),
    }
```

### Adapting Part A to a host codebase

1. **Loss signature.** Find the critic loss. Rewrite the `partial(...)` in `compute_kappa_metrics` so that `critic_params` is the only free argument and everything else (batch, target net, temperature, etc.) is bound. The HVP is taken w.r.t. that free argument.
2. **Where params live.** Replace `critic_j.params` / `agent.critic` with the host's param container.
3. **Seed vmap.** If the host does *not* vmap over seeds, delete `_model_for_seed`, set `num_seeds=1`, and bind the model directly.
4. **Loss reduction.** Power iteration is scale-invariant for κ (a constant factor cancels in λ_max/λ_min), so it does not matter whether the loss is a sum or a mean — but be consistent with whatever the trainer uses.
5. **PyTorch.** See Part A-PT below for a complete equivalent.

---

## Part A-PT — The κ estimator (PyTorch variant)

Functionally identical to Part A. The only real difference is the HVP: JAX uses forward-over-reverse (`jvp(grad(...))`); PyTorch uses **double-backward** (`autograd.grad` twice with `create_graph=True`). The power iteration and controller are otherwise the same.

Drop this into `kappa.py`:

```python
import torch


def _flatten(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def _make_hvp_fn(loss_closure, params):
    """Returns v -> Hv, where H is the Hessian of loss w.r.t. `params` (a list of
    leaf tensors). `loss_closure()` recomputes the scalar critic loss each call.

    Uses reverse-over-reverse: grad once with create_graph=True to keep the graph,
    then grad of (grad . v) to get Hv. ~2x a single backward pass.
    """
    def hvp_fn(flat_v):
        # First backward: build a differentiable gradient.
        loss = loss_closure()
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = _flatten(grads)
        # Second backward: d/dparams (grad . v) = H v.
        dot = (flat_grad * flat_v).sum()
        hv = torch.autograd.grad(dot, params, retain_graph=False)
        return _flatten(hv).detach()

    return hvp_fn


def _power_iter_lambda_max(hvp_fn, n_params, n_iters, device, generator):
    v = torch.randn(n_params, generator=generator, device=device)
    v = v / v.norm()
    lam = torch.tensor(0.0, device=device)
    for _ in range(n_iters):
        Hv = hvp_fn(v)
        lam = torch.dot(v, Hv)                      # Rayleigh quotient
        v = Hv / (Hv.norm() + 1e-8)
    return lam


def _spectral_shift_lambda_min(hvp_fn, n_params, lambda_max, n_iters, device, generator):
    v = torch.randn(n_params, generator=generator, device=device)
    v = v / v.norm()
    lam_shifted = torch.tensor(0.0, device=device)
    for _ in range(n_iters):
        Hv = hvp_fn(v)
        shifted_Hv = lambda_max * v - Hv            # power-iterate on (λ_max·I − H)
        lam_shifted = torch.dot(v, shifted_Hv)
        v = shifted_Hv / (shifted_Hv.norm() + 1e-8)
    return lambda_max - lam_shifted


def compute_kappa_metrics(agent, fixed_batch, n_iters_max=3, n_iters_min=5):
    """Online κ = |λ_max| / |λ_min| of the critic loss Hessian (PyTorch).

    - Uses the agent's ACTIVE critic loss (follows loss config).
    - Hessian over critic params only.
    - fixed_batch reused every call → κ comparable across checkpoints.
    - Does NOT step the optimizer or mutate weights.
    """
    critic = agent.critic
    params = [p for p in critic.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    assert n_params > 0, "[kappa] critic has no trainable params"

    device = params[0].device
    # Fixed generators → comparable estimates across time and runs.
    g_max = torch.Generator(device=device).manual_seed(42)
    g_min = torch.Generator(device=device).manual_seed(99)

    # Bind everything except the params being differentiated. Adapt to host loss.
    # IMPORTANT: critic must be in eval() mode so BatchNorm/Dropout don't change
    # running stats or inject noise during the HVP (otherwise κ is not reproducible
    # and you risk leaking state into training).
    was_training = critic.training
    critic.eval()
    try:
        def loss_closure():
            return agent.critic_loss(critic, fixed_batch)   # <-- active loss; returns scalar

        hvp_fn = _make_hvp_fn(loss_closure, params)

        lam_max = _power_iter_lambda_max(hvp_fn, n_params, n_iters_max, device, g_max)
        lam_min = _spectral_shift_lambda_min(hvp_fn, n_params, lam_max, n_iters_min, device, g_min)
        kappa = lam_max.abs() / (lam_min.abs() + 1e-8)
    finally:
        critic.train(was_training)                          # restore mode; never leave it changed

    assert torch.isfinite(lam_max), "[kappa] lambda_max not finite"
    assert torch.isfinite(lam_min), "[kappa] lambda_min not finite (raise n_iters_min)"

    return {
        "kappa/kappa":      float(kappa),
        "kappa/lambda_max": float(lam_max),
        "kappa/lambda_min": float(lam_min),
    }
```

### PyTorch-specific gotchas

- **`create_graph=True` on the first `grad` is mandatory** — without it the second backward has nothing to differentiate and Hv is zero/errors.
- **Put the critic in `eval()` during estimation**, then restore. BatchNorm in `train()` mode updates running statistics on every forward pass — that both corrupts κ reproducibility and leaks state into training (breaking the no-leakage guarantee). The `try/finally` above restores the original mode even if an assertion fires.
- **No optimizer step.** The function only calls `autograd.grad` (which does not touch `.grad` buffers or the optimizer). Never call `loss.backward()` here — that would accumulate into `.grad` and pollute the next real update.
- **Multi-seed.** The JAX version vmaps over seeds in one process; PyTorch typically runs one seed per process, so there is no seed loop. If you ensemble critics in one process, loop over them and average κ as in Part C3.
- **Device.** Keep the power-iteration vector on the same device as the params; the code above does this via `device=params[0].device`.

---

## Part B — Config flags

Add two config groups. **Both default to disabled** so existing runs are byte-for-byte unchanged.

```yaml
kappa_logging:
  enabled: false
  interval: 1000      # compute κ every this many env steps
  n_iters_max: 3      # power-iteration steps for λ_max
  n_iters_min: 5      # spectral-shift steps for λ_min (the finicky one — raise if NaN)

kappa_control:
  enabled: false
  # PLACEHOLDER thresholds — calibrate from a logging-only run first.
  kappa_high: 1000.0  # κ above this → reduce UTD by utd_step (ill-conditioned)
  kappa_low:  10.0    # κ below this → increase UTD by utd_step (well-conditioned)
  utd_min: 1          # floor on UTD ratio
  utd_max: 8          # ceiling on UTD ratio
  utd_step: 1         # increment/decrement per adjustment
```

`kappa_control.enabled` requires `kappa_logging.enabled=true` (control reads the κ computed by the logging block).

---

## Part C — Training-loop integration

Three edits to the training loop. The snippets below use the JAX estimator's signature (`compute_kappa_metrics(agent, batch, num_seeds, ...)`). For the PyTorch variant, drop the `num_seeds` argument and the `[j, 0]` batch indexing — the metric values are plain floats, and the C3 `np.mean(...)` reduces a single value harmlessly.

### C1. Before the loop: introduce a mutable UTD variable

The host loop almost certainly reads a constant like `cfg.updates_per_step`. Replace every *read* of it inside the loop with a mutable local that starts at the config value:

```python
current_utd = cfg.updates_per_step      # was a constant; now adjustable
kappa_fixed_batch = None                # sampled lazily on first κ estimate
```

Then change the update block to use `current_utd`:

```python
# before:  batches = buffer.sample(batch_size, cfg.updates_per_step)
#          infos   = agent.update(batches, num_updates=cfg.updates_per_step)
batches = buffer.sample(batch_size, current_utd)
infos   = agent.update(batches, num_updates=current_utd)
update_count += current_utd
```

### C2. κ estimation + logging (cheap, every K steps)

Insert after the agent update, gated on `kappa_logging.enabled`:

```python
if cfg.kappa_logging.enabled and i > cfg.start_training:
    kappa_interval = max(1, cfg.kappa_logging.interval // cfg.env.action_repeat)
    if i % kappa_interval == 0:
        if kappa_fixed_batch is None:                     # capture ONE fixed batch, reuse forever
            kappa_fixed_batch = buffer.sample(batch_size, 1)
        kappa_metrics = compute_kappa_metrics(
            agent, kappa_fixed_batch, cfg.num_seeds,
            n_iters_max=cfg.kappa_logging.n_iters_max,
            n_iters_min=cfg.kappa_logging.n_iters_min,
        )
        logger.log(step, kappa_metrics)

        # --- C3 control goes here ---
```

### C3. The controller (bang-bang on UTD)

Append inside the `if i % kappa_interval == 0:` block:

```python
if cfg.kappa_control.enabled:
    mean_kappa = float(np.mean(np.array(kappa_metrics["kappa/kappa"])))
    if mean_kappa > cfg.kappa_control.kappa_high:
        current_utd = max(cfg.kappa_control.utd_min,
                          current_utd - cfg.kappa_control.utd_step)
    elif mean_kappa < cfg.kappa_control.kappa_low:
        current_utd = min(cfg.kappa_control.utd_max,
                          current_utd + cfg.kappa_control.utd_step)
    logger.log(step, {"utd_ratio": np.array([current_utd] * cfg.num_seeds)})
```

Control fires at most once per `interval`, so UTD changes gradually — not every gradient step.

---

## Part D — Extending to other hyperparameters (future work)

UTD is the first knob because it is a plain Python int read once per loop iteration — trivial to mutate. The same controller pattern extends to:

- **Critic learning rate** — harder: most JAX optimizers bake the LR/schedule into the optax transform at init. Either build the optimizer with `optax.inject_hyperparams(optax.adam)(learning_rate=...)` so the LR lives in the opt-state and can be overwritten each step, or rebuild the transform on change.
- **Target update rate (τ / polyak)** — easy if τ is read from a variable each soft-update; same mutable-variable pattern as UTD.
- **Batch size** — usually fixed by buffer/jit shapes; changing it triggers recompilation, so prefer discrete preset sizes.

General recipe for each new knob: (1) make it a mutable variable initialized from config, (2) replace in-loop reads with the variable, (3) add a branch in the C3 controller mapping κ → the knob, (4) log it.

---

## Verification checklist

Run these in order. Do **not** trust κ numbers until tests 1–2 pass.

1. **Backward compatibility.** Run with everything default (both groups disabled). Behavior must be identical to before the changes (same seed → same returns).

2. **Observation-only (no leakage).** Run the same seed twice: once `kappa_logging.enabled=false`, once `kappa_logging.enabled=true` (control still off). The reward/loss traces must be **numerically identical**. If they diverge, the κ computation is leaking into the gradient path (e.g. you accidentally mutated agent state or shared a PRNG key) — fix before proceeding. This is guaranteed by construction if `compute_kappa_metrics` is purely functional.

3. **κ sanity / contrast.** On a well-conditioned config vs. an ill-conditioned one (e.g. full architecture vs. a stripped baseline), the well-conditioned run should show lower, flatter κ. This validates the estimator against expectation.

4. **Controller fires.** Set `kappa_control.enabled=true kappa_control.kappa_high=0`. Since any κ > 0 exceeds the threshold, UTD must drop to `utd_min` on the first estimate. Confirm the `utd_ratio` log shows the change. Then set `kappa_low=1e12` to force the opposite (UTD climbs to `utd_max`).

5. **Calibrate thresholds.** From the logging-only run (test 3), read the κ range for each regime. Set `kappa_low` just above the well-conditioned range and `kappa_high` just below where the ill-conditioned regime sits. Only then enable control for real experiments.

---

## Defensive-coding notes

- Guard every entry point with the `enabled` flags so the feature is fully inert by default.
- Assert finite λ_max / λ_min and non-empty params with **loud, specific** messages — eigen-estimation on GPU clusters fails in opaque ways otherwise.
- Use **fixed PRNG keys** for the power-iteration init vectors (here 42 and 99) and a fixed minibatch, so κ is comparable across time and across runs.
- λ_min via spectral shift is the least-stable piece. If it returns NaN or noise, raise `n_iters_min` first; if still unstable, consider shift-and-invert or a small damping term.
