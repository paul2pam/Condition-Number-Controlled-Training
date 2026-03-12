import random

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RescaleAction, TimeLimit

from xqc.envs.dmc import DMC_ALL, make_dmc_env
from xqc.envs.humanoid_bench import HB_ALL, make_humanoid_env

# Benchmark wrappers adapted from https://github.com/DAVIAN-Robotics/SimbaV2/tree/master/scale_rl/envs
from xqc.envs.mujoco import MUJOCO_ALL, make_mujoco_env
from xqc.envs.myosuite import MYOSUITE_TASKS, make_myosuite_env
from xqc.envs.wrapper import RepeatAction, SinglePrecision


def resolve_env_benchmark(env_name: str) -> str:
    if env_name in MUJOCO_ALL:
        return "mujoco"
    elif env_name in DMC_ALL:
        return "dmc"
    elif env_name in MYOSUITE_TASKS:
        return "myosuite"
    elif env_name in HB_ALL:
        return "humanoid_bench"
    else:
        raise ValueError(f"Unknown env: {env_name}")


def make_env(env_name: str, seed: int = 0, action_repeat=1) -> gym.Env:
    benchmark = resolve_env_benchmark(env_name)

    if benchmark == "mujoco":
        env = make_mujoco_env(env_name, seed)
        env = RescaleAction(env, -1.0, 1.0)

    elif benchmark == "dmc":
        env = make_dmc_env(env_name, seed)
        env = RescaleAction(env, -1.0, 1.0)
        env = SinglePrecision(env)
        env = TimeLimit(env, max_episode_steps=1000)

    elif benchmark == "myosuite":
        env = make_myosuite_env(env_name, seed)
        env.env._max_episode_steps = 100

    elif benchmark == "humanoid_bench":
        env = make_humanoid_env(env_name, seed)

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    if action_repeat > 1:
        env = RepeatAction(env, action_repeat)

    return env


class ParallelEnv:
    """
    Parallel env adapted from https://github.com/naumix/BiggerRegularizedCategorical/blob/main/jaxrl/envs.py
    """

    def __init__(self, env_names: list, seed: int = 0, action_repeat=1):
        np.random.seed(seed)
        random.seed(seed)

        envs = []
        obs_dims = np.zeros(len(env_names), dtype=np.int32)
        act_dims = np.zeros(len(env_names), dtype=np.int32)
        for i, env_name in enumerate(env_names):
            envs.append(make_env(env_name, seed=seed, action_repeat=action_repeat))
            obs_dims[i] = envs[-1].observation_space.shape[0]
            act_dims[i] = envs[-1].action_space.shape[0]

        max_state_dim = int(np.max(obs_dims))
        max_action_dim = int(np.max(act_dims))
        state_dim_differences = max_state_dim - obs_dims

        dtype = envs[-1].observation_space.dtype
        observation_space = gym.spaces.Box(
            low=(np.ones(max_state_dim, dtype=dtype)[None, :] - np.inf).repeat(
                len(envs), axis=0
            ),
            high=(np.ones(max_state_dim, dtype=dtype)[None, :] + np.inf).repeat(
                len(envs), axis=0
            ),
            shape=(len(envs), max_state_dim),
            dtype=dtype,
        )

        action_dtype = envs[-1].action_space.dtype
        action_space = gym.spaces.Box(
            low=(np.ones(max_action_dim, dtype=action_dtype)[None, :] * -1).repeat(
                len(envs), axis=0
            ),
            high=(np.ones(max_action_dim, dtype=action_dtype)[None, :]).repeat(
                len(envs), axis=0
            ),
            shape=(len(envs), max_action_dim),
            dtype=action_dtype,
        )

        self.envs = envs
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.state_dim_differences = state_dim_differences
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_tasks = len(envs)

    @property
    def max_episode_steps(self):
        if self.envs is None:
            return None

        env = self.envs[0]
        while hasattr(env, "env"):
            if isinstance(env, TimeLimit):
                return env._max_episode_steps
            env = env.env
        return None

    def _reset_idx(self, idx: int):
        seed = np.random.randint(0, 1e8)
        state, _ = self.envs[idx].reset(seed=seed)
        state = np.concatenate(
            (state, np.zeros(self.state_dim_differences[idx], dtype=np.float32)), axis=0
        )
        return state

    def generate_masks(self, terminals: np.ndarray, truncates: np.ndarray):
        masks = 1 - (terminals * (1 - truncates))
        return masks

    def reset_where_done(
        self, states: np.ndarray, terminals: np.ndarray, truncates: np.ndarray
    ):
        resets = np.zeros((terminals.shape))
        for j, (terminal, truncate) in enumerate(zip(terminals, truncates)):
            if terminal or truncate:
                states[j], terminals[j], truncates[j] = self._reset_idx(j), False, False
                resets[j] = 1
        return states, terminals, truncates, resets

    def reset(self):
        states = []
        for i, env in enumerate(self.envs):
            states.append(self._reset_idx(i))
        return np.stack(states)

    def _get_goal(self, info: dict):
        if "success" in info:
            goal = info["success"]
        elif "is_success" in info:
            goal = info["is_success"]
        elif "solved" in info:
            goal = info["solved"]
        else:
            goal = 0
        return goal

    def step(self, actions: np.ndarray):
        states, rewards, terminals, truncates, goals = [], [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            state, reward, terminal, truncate, info = env.step(
                action[: self.act_dims[i]]
            )
            state = np.concatenate(
                (state, np.zeros(self.state_dim_differences[i], dtype=np.float32)),
                axis=0,
            )
            states.append(state)
            rewards.append(reward)
            terminals.append(terminal)
            truncates.append(truncate)
            goals.append(self._get_goal(info))
        return (
            np.stack(states),
            np.stack(rewards),
            np.stack(terminals),
            np.stack(truncates),
            np.stack(goals),
        )

    def evaluate(
        self,
        agent,
        num_episodes,
        temperature=0.0,
        render=False,
        max_render_steps=1000,
        render_frameskip=4,
        render_num_envs=1,
    ):
        n_rollouts = np.zeros(self.num_tasks)
        returns = np.zeros(self.num_tasks)
        goals = np.zeros(self.num_tasks)
        mask = np.ones(self.num_tasks)
        mask_goals = np.ones(self.num_tasks)
        observations = self.reset()
        if render:
            render_num_envs = min(render_num_envs, self.num_tasks)
            max_frames = max_render_steps // max(render_frameskip, 1)
            frame_shape = self.render(num_envs=render_num_envs).shape
            renders = np.empty((max_frames, *frame_shape), dtype=np.uint8)
            render_idx = 0
        i = 0
        while True:
            if render:
                render_active = n_rollouts[:render_num_envs].min() < 1
                if render_active and i % render_frameskip == 0 and render_idx < max_frames:
                    renders[render_idx] = self.render(num_envs=render_num_envs)
                    render_idx += 1
            actions = agent.sample_actions(observations, temperature=temperature)
            next_observations, rewards, terms, truncs, success = self.step(actions)
            returns += rewards * mask
            goals += success * mask_goals
            mask_goals = np.where(success, 0, mask_goals)
            mask_goals = np.where(np.logical_or(terms, truncs), 1, mask_goals)
            observations = next_observations
            n_rollouts += np.logical_or(terms, truncs)
            observations, terms, truncs, _ = self.reset_where_done(
                observations, terms, truncs
            )
            mask = np.where(n_rollouts >= num_episodes, 0, 1)
            i += 1
            if n_rollouts.min() == num_episodes:
                break
        if render:
            renders = renders[:render_idx]
            renders = np.transpose(renders, (1, 0, 4, 2, 3))
            return {
                "goal": goals / num_episodes,
                "return": returns / num_episodes,
                "renders": renders,
            }
        else:
            return {"goal": goals / num_episodes, "return": returns / num_episodes}

    def render(self, num_envs=None):
        if num_envs is None:
            num_envs = self.num_tasks
        renders = []
        for i in range(num_envs):
            render = self.envs[i].render()
            renders.append(render)
        renders = np.stack(renders)
        return renders

    def __str__(self):
        s = "ParallelEnv<\n"
        for env in self.envs:
            s += f"  {env}"
            e = env
            while hasattr(e, "env"):
                if isinstance(e, TimeLimit):
                    s += f"  max_t={e._max_episode_steps}"
                    break
                e = e.env
            s += "\n"
        return s + ">"
