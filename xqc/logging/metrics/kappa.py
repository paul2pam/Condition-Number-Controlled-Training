from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, jvp
from jax.flatten_util import ravel_pytree


def _model_for_seed(seed, model):
    return model.replace(
        params=jax.tree.map(lambda p: p[seed], model.params),
        batch_stats=(
            jax.tree.map(lambda p: p[seed], model.batch_stats)
            if model.batch_stats else None
        ),
    )


def _make_hvp_fn(loss_fn_fixed, flat_params, unravel):
    """Returns a jit-compiled (flat_params, flat_v) -> flat Hv using forward-over-reverse."""
    def flat_loss(flat_p):
        return loss_fn_fixed(critic_params=unravel(flat_p))[0]

    @jit
    def hvp_fn(flat_p, flat_v):
        return jvp(grad(flat_loss), [flat_p], [flat_v])[1]

    return hvp_fn


def _power_iter_lambda_max(hvp_fn, flat_params, n_iters):
    """Power iteration for λ_max, 2–3 iters is sufficient for a rough online estimate."""
    n = flat_params.shape[0]
    v = jax.random.normal(jax.random.PRNGKey(42), (n,))
    v = v / jnp.linalg.norm(v)
    lam = jnp.array(0.0)
    for _ in range(n_iters):
        Hv = hvp_fn(flat_params, v)
        lam = jnp.dot(v, Hv)
        v = Hv / (jnp.linalg.norm(Hv) + 1e-8)
    return lam


def _spectral_shift_lambda_min(hvp_fn, flat_params, lambda_max, n_iters):
    """Spectral-shifted power iteration for λ_min.

    Power-iterates on (λ_max·I − H). Its dominant eigenvalue is (λ_max − λ_min),
    so λ_min = λ_max − dominant_eigenvalue.
    """
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


def compute_kappa_metrics(agent, fixed_batch, num_seeds, n_iters_max=3, n_iters_min=5,
                          lambda_min_floor=1e-3):
    """Online κ = |λ_max| / max(|λ_min|, lambda_min_floor) of the critic loss Hessian.

    Uses the active loss from agent.critic.loss_fn (follows critic_loss config —
    categorical CE when critic_loss=categorical, MSE when critic_loss=mse).
    Hessian is taken w.r.t. critic.params only, with batch_stats held fixed.

    fixed_batch: sampled once and reused each call so κ is comparable across
    checkpoints. Expected shape (num_seeds, 1, batch_size, ...).

    lambda_min_floor: absolute damping floor on |λ_min|. Neural-net Hessians are
    indefinite with a near-zero smallest eigenvalue, so |λ_min| straddles 0 and the
    raw ratio explodes. Clamping |λ_min| up to this floor means that when λ_min is in
    the noise (the common case), κ ≈ |λ_max| / floor and therefore tracks λ_max —
    which is where the XQC-vs-SAC conditioning signal actually lives. When |λ_min|
    rises above the floor, κ is the true ratio. Use an ABSOLUTE floor (not relative
    to λ_max), or κ saturates and the λ_max signal is lost. Tune to just above the
    observed |λ_min| noise scale.

    Does NOT mutate agent state (no agent.rng touch, purely functional JAX).
    Returns dict with per-seed jnp arrays of shape (num_seeds,).
    """
    kappas, lambdas_max, lambdas_min = [], [], []

    for j in range(num_seeds):
        critic_j = _model_for_seed(j, agent.critic)

        loss_fn_fixed = partial(
            agent.critic.loss_fn,
            critic_batch_stats=critic_j.batch_stats,
            key=jax.random.PRNGKey(0),
            actor=_model_for_seed(j, agent.actor),
            critic=critic_j,
            target_critic=_model_for_seed(j, agent.target_critic),
            temperature=_model_for_seed(j, agent.temperature),
            batch=jax.tree.map(lambda x: x[j, 0], fixed_batch),
        )

        flat_params, unravel = ravel_pytree(critic_j.params)
        n_params = flat_params.shape[0]
        assert n_params > 0, f"[kappa] seed {j}: critic has no parameters (n_params=0)"

        hvp_fn = _make_hvp_fn(loss_fn_fixed, flat_params, unravel)

        lam_max = _power_iter_lambda_max(hvp_fn, flat_params, n_iters_max)
        lam_min = _spectral_shift_lambda_min(hvp_fn, flat_params, lam_max, n_iters_min)
        kappa = jnp.abs(lam_max) / jnp.maximum(jnp.abs(lam_min), lambda_min_floor)

        assert jnp.isfinite(lam_max), (
            f"[kappa] seed {j}: lambda_max={float(lam_max):.6g} is not finite — "
            "check that the loss is well-defined on the fixed batch"
        )
        assert jnp.isfinite(lam_min), (
            f"[kappa] seed {j}: lambda_min={float(lam_min):.6g} is not finite — "
            "spectral shift may have collapsed; try increasing n_iters_min"
        )

        kappas.append(float(kappa))
        lambdas_max.append(float(lam_max))
        lambdas_min.append(float(lam_min))

    return {
        "kappa/kappa":      jnp.array(kappas),
        "kappa/lambda_max": jnp.array(lambdas_max),
        "kappa/lambda_min": jnp.array(lambdas_min),
    }
