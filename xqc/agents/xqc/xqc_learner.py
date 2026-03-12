"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from xqc.logging import print_total_param_count
from xqc.agents.xqc.temperature import Temperature, update as update_temperature
from xqc.agents.xqc.actor import update_actor
from xqc.agents.xqc.critic import update_critic, categorical_critic_loss_fn, mse_critic_loss_fn
from xqc.agents.xqc.utils import target_update
from xqc.replay_buffer import Batch
from xqc.networks import critic_net, policies
from xqc.networks.common import InfoDict, Model, PRNGKey, norm_network


@functools.partial(
    jax.vmap,
    in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None, None),
)
def _update(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temperature: Model,
    batch: Batch,
    tau: float,
    target_net_update_freq: int,
    step: int,
    target_entropy: float,
    use_weight_norm: bool,
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:
    rng, critic_key, actor_key = jax.random.split(rng, 3)

    new_critic, critic_info = update_critic(
        key=critic_key,
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        temperature=temperature,
        batch=batch,
        use_weight_norm=use_weight_norm,
    )

    tau = jnp.where(critic.step % target_net_update_freq == 0, tau, 0.0)
    new_target_critic = target_update(new_critic, target_critic, tau)

    new_actor, actor_info = update_actor(
        key=actor_key,
        actor=actor,
        critic=new_critic,
        temperature=temperature,
        batch=batch,
        use_weight_norm=use_weight_norm,
    )
    new_temperature, alpha_info = update_temperature(
        temperature=temperature, entropy=actor_info["entropy"], target_entropy=target_entropy
    )

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic,
        new_temperature,
        {**critic_info, **actor_info, **alpha_info},
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "tau",
        "target_net_update_freq",
        "target_entropy",
        "num_updates",
        "use_weight_norm",
    ),
)
def _do_multiple_updates(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temperature: Model,
    batches: Batch,
    tau: float,
    target_net_update_freq: int,
    target_entropy: float,
    step: int,
    num_updates: int,
    use_weight_norm: bool,
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:
    def one_step(i, state):
        step, rng, actor, critic, target_critic, temperature, info = state
        new_rng, new_actor, new_critic, new_target_critic, new_temperature, info = _update(
            rng,
            actor,
            critic,
            target_critic,
            temperature,
            jax.tree.map(lambda x: jnp.take(x, i, axis=1), batches),
            tau,
            target_net_update_freq,
            step,
            target_entropy,
            use_weight_norm,
        )
        return (
            step + 1,
            new_rng,
            new_actor,
            new_critic,
            new_target_critic,
            new_temperature,
            info,
        )

    return jax.lax.fori_loop(
        1,
        num_updates,
        one_step,
        one_step(0, (step, rng, actor, critic, target_critic, temperature, {})),
    )


class XQCLearner(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float,
        critic_lr: float,
        temp_lr: float,
        hidden_dims_critic: Sequence[int],
        policy_delay: int,
        hidden_dims_actor: Sequence[int],
        tau: float,
        target_entropy: Optional[float],
        init_temperature: float,
        num_seeds: int,
        updates_per_step: int,
        max_v: float,
        min_v: float,
        critic_loss: str,
        target_net_update_freq: int,
        n_critics: int,
        num_interactions: int,
        lr_end: float,
        pre_activation_bn: bool,
        weight_decay: float,
        normalize_last_layer: bool,
        use_layer_norm: bool,
        use_batch_norm: bool,
        decay_bn: bool,
        use_weight_norm: bool,
        skip_connections: bool,
        **kwargs,
    ) -> None:
        self.action_dim = actions.shape[-1]
        self.tau = tau
        self.target_net_update_freq = target_net_update_freq
        self.use_weight_norm = use_weight_norm
        self.critic_loss = critic_loss
        self.target_entropy = -self.action_dim / 2 if target_entropy == "auto" else float(target_entropy)
        self.n_outputs = {"categorical": 101, "mse": 1}[critic_loss]

        # Learning rate schedules
        lr_schedule_fn = functools.partial(
            optax.linear_schedule,
            end_value=lr_end,
            transition_steps=num_interactions * updates_per_step,
        )

        # Model initialization
        def init_models(seed):
            # Random keys
            rng, actor_key, critic_key, temperature_key, encoder_key, latent_model_key = (
                jax.random.split(jax.random.PRNGKey(seed), 6)
            )

            ######################
            # Optimizer definition
            ######################

            def weight_decay_mask(params):
                # For CrossQ+WN we want to have weight decay on the last layer and on the batch norm layers
                def should_decay(path, _):
                    key = path[-2].key
                    if decay_bn and 'BatchNorm' in key:
                        return True
                    elif not normalize_last_layer and key in ["value", "log_std", "mean"]:
                        # If we are already normalizing the last layer to the unit sphere,
                        # we don't want to apply weight decay on it
                        return True
                    else:
                        return False

                return jax.tree_util.tree_map_with_path(should_decay, params)

            actor_tx = optax.conditionally_mask(
                optax.adamw(
                    learning_rate=lr_schedule_fn(actor_lr),
                    weight_decay=weight_decay,
                    mask=weight_decay_mask,
                ),
                lambda step: jnp.mod(step, policy_delay) == 0,
            )

            critic_tx = optax.adamw(
                learning_rate=lr_schedule_fn(critic_lr),
                weight_decay=weight_decay,
                mask=weight_decay_mask,
            )

            temperature_tx = optax.conditionally_mask(
                optax.adam(learning_rate=lr_schedule_fn(actor_lr)),
                lambda step: jnp.mod(step, policy_delay) == 0,
            )

            ######################
            # Model initialization
            ######################
            training = False

            actor = Model.create(
                policies.NormalTanhPolicy(
                    hidden_dims=tuple(hidden_dims_actor),
                    action_dim=self.action_dim,
                    pre_activation_bn=pre_activation_bn,
                    use_layer_norm=use_layer_norm,
                    use_batch_norm=use_batch_norm,
                    skip_connections=skip_connections,
                ), 
                inputs=[actor_key, observations, training], 
                tx=actor_tx
            )

            critic_def = critic_net.VMapCritic(
                hidden_dims=tuple(hidden_dims_critic),
                n_outputs=self.n_outputs,
                max_v=max_v,
                min_v=min_v,
                n_critics=n_critics,
                pre_activation_bn=pre_activation_bn,
                use_layer_norm=use_layer_norm,
                use_batch_norm=use_batch_norm,
                skip_connections=skip_connections,
            )

            critic = Model.create(
                critic_def, 
                inputs=[critic_key, observations, actions, training],
                tx=critic_tx, 
                loss_fn={
                    "categorical": categorical_critic_loss_fn,
                    "mse": mse_critic_loss_fn
                }[critic_loss]
            )
            target_critic = Model.create(critic_def, inputs=[critic_key, observations, actions, training])
            
            temperature = Model.create(Temperature(init_temperature), inputs=[temperature_key], tx=temperature_tx)

            if self.use_weight_norm:
                # Norm Networks in the beginning.
                actor = norm_network(actor)
                critic = norm_network(critic)

            target_critic = target_critic.replace(params=critic.params)

            ######################
            # Finishing up
            ######################

            print_total_param_count([actor, critic])

            return actor, critic, target_critic, temperature, rng

        self.seeds = jnp.arange(seed, seed + num_seeds)

        self.init_models = jax.vmap(init_models)
        self.actor, self.critic, self.target_critic, self.temperature, self.rng = self.init_models(self.seeds)
        self.step = 1

    def sample_actions(
        self, observations: np.ndarray, temperature: float = 1.0
    ) -> jnp.ndarray:
        return self.sample_actions_with_log_probs(observations, temperature)[0]

    def sample_actions_with_log_probs(
        self, observations: np.ndarray, temperature: float = 1.0
    ) -> jnp.ndarray:
        self.rng, actions, log_probs = policies.sample_actions_with_log_probs(
            self.rng,
            self.actor.apply_fn,
            self.actor.params,
            self.actor.batch_stats,
            observations,
            temperature,
        )
        return np.clip(np.asarray(actions), -1, 1), log_probs

    def reset(self):
        self.actor, self.critic, self.target_critic, self.temperature, self.rng = self.init_models(self.seeds)

    def update(
        self,
        batch: Batch,
        num_updates: int = 1,
        time_to_intervene: bool = False,
    ) -> InfoDict:
        self.step, self.rng, self.actor, self.critic, self.target_critic, self.temperature, info = _do_multiple_updates(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.temperature,
            batch,
            self.tau,
            self.target_net_update_freq,
            self.target_entropy,
            self.step,
            num_updates,
            self.use_weight_norm,
        )
        return info
