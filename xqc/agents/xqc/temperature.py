import jax.numpy as jnp
from flax import linen as nn


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)


def update(temperature, entropy: float, target_entropy: float):
    def temperature_loss_fn(params, *args, **kwargs):
        temp_value = temperature.apply(params)
        temp_loss = temp_value * (entropy - target_entropy).mean()
        return temp_loss, {"temperature": temp_value, "temp_loss": temp_loss}

    new_temperature, grad, info = temperature.apply_gradient(temperature_loss_fn)

    return new_temperature, info
