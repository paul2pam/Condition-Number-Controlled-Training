import os
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore

from xqc.envs.envs import resolve_env_benchmark
from xqc.envs.dmc import DMC_ALL
from xqc.envs.humanoid_bench import HB_ALL
from xqc.envs.mujoco import MUJOCO_ALL
from xqc.envs.myosuite import MYOSUITE_TASKS


# Register Resolvers
def call_resolve_env_benchmark(env_name: str) -> str:
    return resolve_env_benchmark(env_name)


def resolve_env_default(name, default):
    return os.environ.get(name, default)


OmegaConf.register_new_resolver("resolve_benchmark", call_resolve_env_benchmark)
OmegaConf.register_new_resolver("env_default", resolve_env_default)


# Register Configs
cs = ConfigStore.instance()

action_repeats = {
    "mujoco": 1,
    "dmc": 2,
    "myosuite": 2,
    "humanoid_bench": 2,
}

# Register Aliases for defaults
aliases = {
    "mujoco": "HalfCheetah-v4",
    "dmc": "cheetah-run",
    "myo": "myo-reach",
    "humanoid_bench": "h1-walk-v0",
}

for alias, target in aliases.items():
    cs.store(
        group="env",
        name=alias,
        node={
            "name": target,
            "benchmark": resolve_env_benchmark(target),
            "action_repeat": action_repeats[resolve_env_benchmark(target)],
        },
    )

for env_name in (
    MUJOCO_ALL + list(set(DMC_ALL)) + list(MYOSUITE_TASKS) + list(set(HB_ALL))
):
    benchmark = resolve_env_benchmark(env_name)
    cs.store(
        group="env",
        name=env_name,
        node={
            "name": env_name,
            "benchmark": benchmark,
            "action_repeat": action_repeats[benchmark],
        },
    )
