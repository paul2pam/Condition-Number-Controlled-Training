from typing import Tuple

import jax.numpy as jnp

from xqc.replay_buffer.parallel_replay_buffer import Batch
from xqc.networks.common import InfoDict, Model, Params, PRNGKey, norm_network


def update_actor(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    temperature: Model,
    batch: Batch,
    use_weight_norm: bool,
) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(
        actor_params: Params, actor_batch_stats: Params = None
    ) -> Tuple[jnp.ndarray, InfoDict]:

        dist, actor_state_updates = actor.apply(
            actor_params,
            batch.observations,
            batch_stats=actor_batch_stats,
            mutable=["batch_stats", "activations"],
            training=True,
        )

        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)

        (q_values, _), _ = critic(
            batch.observations,
            actions,
            training=False,
            return_normalized=False,
            mutable="batch_stats",
        )

        q_values = jnp.min(q_values, axis=0)
        actor_loss = log_probs * temperature() - q_values
        actor_loss = actor_loss.mean()

        return actor_loss, {
            "actor_loss": actor_loss,
            "actor_activations": actor_state_updates.get("activations"),
            "actor_batch_stats": actor_state_updates.get("batch_stats"),
            "entropy": -log_probs.mean(),
            "actor q": q_values.mean(),
        }

    new_actor, grads, info = actor.apply_gradient(actor_loss_fn)
    info["actor_grads"] = grads

    if use_weight_norm:
        new_actor = norm_network(model=new_actor)
    new_actor = new_actor.replace(batch_stats=info.pop("actor_batch_stats"))

    return new_actor, info
