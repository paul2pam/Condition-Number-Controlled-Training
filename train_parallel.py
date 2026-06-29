import random
from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf

import conf.register_envs  # noqa: F401, This is need to register the environments


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    import wandb
    import xqc.utils

    xqc.utils.check_hydra_config()

    with wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project_name,
        name=f"{cfg.agent.name}_{cfg.env.name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        settings=wandb.Settings(start_method="thread"),
        mode=cfg.wandb.mode,
    ) as wandb_run:
        try:
            # Delayed imports to allow Hydra's env_set to take effect
            import numpy as np
            import tqdm

            from xqc.agents import XQCLearner
            from xqc.replay_buffer import ParallelReplayBuffer
            from xqc.envs import ParallelEnv
            from xqc.normalization import RewardNormalizer
            import xqc.logging

            ################################################################################
            # Setup
            ################################################################################

            xqc.utils.log_slurm_info(wandb_run)
            xqc.logging.print_config(cfg)

            np.random.seed(cfg.seed)
            random.seed(cfg.seed)

            # Make Envs
            make_envs = partial(
                ParallelEnv,
                env_names=[cfg.env.name] * cfg.num_seeds,
                action_repeat=cfg.env.action_repeat,
            )
            env = make_envs(seed=cfg.seed)
            eval_env = make_envs(seed=cfg.seed + 42)

            agent_kwargs = OmegaConf.to_container(cfg.agent, resolve=True)
            agent_kwargs.update({
                "seed": cfg.seed,
                "num_seeds": cfg.num_seeds,
                "updates_per_step": cfg.updates_per_step,
                "num_interactions": int(cfg.max_steps / cfg.env.action_repeat),
                "observations": env.observation_space.sample()[0, None],
                "actions": env.action_space.sample()[0, None],
            })
            agent = XQCLearner(**agent_kwargs)

            # Heuristic discount
            discount = xqc.utils.get_discount(
                env.max_episode_steps, cfg.env.action_repeat
            )

            replay_buffer = ParallelReplayBuffer(
                env.observation_space,
                env.action_space,
                capacity=cfg.replay_buffer_size,
                num_seeds=cfg.num_seeds,
                n_steps=cfg.n_steps,
                gamma=discount,
            )

            if cfg.agent.reward_normalization:
                reward_normalizer = RewardNormalizer(
                    n_seeds=cfg.num_seeds,
                    gamma=discount,
                    max_v=cfg.agent.max_v,
                )


            ################################################################################
            # Main XQC training loop
            ################################################################################

            start_step = 1
            update_count = 0
            lambda_max_fixed_batch = None
            current_utd = cfg.updates_per_step

            observations = env.reset()
            infos = {}

            for i in tqdm.tqdm(
                range(start_step, cfg.max_steps // cfg.env.action_repeat + 1),
                smoothing=0.1,
                disable=xqc.utils.is_slurm_job(),
            ):
                # Sample actions
                if i < cfg.start_training:
                    actions = env.action_space.sample()
                else:
                    actions, _ = agent.sample_actions_with_log_probs(observations)

                # Step env
                next_observations, rewards, dones, truncs, _ = env.step(actions)
                if cfg.agent.reward_normalization:
                    reward_normalizer.update(rewards, np.logical_or(dones, truncs))

                # Save to buffer
                masks = env.generate_masks(dones, truncs)
                replay_buffer.insert(observations, actions, rewards, masks, truncs, next_observations)
                observations = next_observations

                # Reset env
                observations, terms, truncs, _ = env.reset_where_done(observations, dones, truncs)

                # Update agent
                if i > cfg.start_training:
                    batches = replay_buffer.sample_parallel_multibatch(cfg.batch_size, current_utd)

                    if cfg.agent.reward_normalization:
                        normalized_rewards = reward_normalizer.normalize(batches.rewards)
                        batches = batches._replace(rewards=normalized_rewards)

                    infos = agent.update(batches, num_updates=current_utd)
                    update_count += current_utd

                ################################################################################
                # Evaluation and Logging
                ################################################################################

                # Policy Evaluation
                if i == 1 or i % cfg.eval_interval == 0:
                    eval_stats = eval_env.evaluate(
                        agent,
                        num_episodes=cfg.eval_episodes,
                        temperature=0.0,
                        render=cfg.eval_video,
                        render_frameskip=cfg.get("eval_video_frameskip", 1),
                        render_num_envs=cfg.get("eval_video_num_envs", 1),
                    )
                    xqc.logging.log_multiple_seeds_to_wandb(
                        i * cfg.env.action_repeat, 
                        eval_stats,
                        fps=cfg.get("eval_video_fps", 30)
                    )
                
                # Logging
                if i > cfg.start_training and i % cfg.log_interval == 0:
                    xqc.logging.log_multiple_seeds_to_wandb(
                        i * cfg.env.action_repeat,
                        xqc.logging.metrics.compute_logging_metrics(agent, infos)
                    )

                # Online λ_max logging (cheap: power iteration, fixed batch)
                if cfg.lambda_max_logging.enabled and i > cfg.start_training:
                    lambda_max_interval = max(1, cfg.lambda_max_logging.interval // cfg.env.action_repeat)
                    if i % lambda_max_interval == 0:
                        if lambda_max_fixed_batch is None:
                            # Sample the fixed λ_max batch without disturbing the global
                            # numpy RNG stream that drives training-batch sampling
                            # (replay_buffer uses np.random.randint). Snapshot/restore
                            # so logging-on and logging-off runs stay byte-identical.
                            _rng_state = np.random.get_state()
                            lambda_max_fixed_batch = replay_buffer.sample_parallel_multibatch(cfg.batch_size, 1)
                            np.random.set_state(_rng_state)
                        metrics = xqc.logging.metrics.compute_lambda_max_metrics(
                            agent,
                            lambda_max_fixed_batch,
                            cfg.num_seeds,
                            n_iters_max=cfg.lambda_max_logging.n_iters_max,
                        )
                        xqc.logging.log_multiple_seeds_to_wandb(
                            i * cfg.env.action_repeat, metrics
                        )

                        if cfg.lambda_max_control.enabled:
                            mean_lambda_max = float(np.mean(np.array(metrics["lambda_max"])))
                            if mean_lambda_max > cfg.lambda_max_control.lambda_max_high:
                                current_utd = max(cfg.lambda_max_control.utd_min,
                                                  current_utd - cfg.lambda_max_control.utd_step)
                            elif mean_lambda_max < cfg.lambda_max_control.lambda_max_low:
                                current_utd = min(cfg.lambda_max_control.utd_max,
                                                  current_utd + cfg.lambda_max_control.utd_step)
                            xqc.logging.log_multiple_seeds_to_wandb(
                                i * cfg.env.action_repeat,
                                {"utd_ratio": np.array([current_utd] * cfg.num_seeds)},
                            )

                # Condition number logging (expensive)
                if cfg.log_interval_condition_number and \
                    (i == cfg.start_training or i % cfg.log_interval_condition_number == 0):
                    xqc.logging.log_multiple_seeds_to_wandb(
                        i * cfg.env.action_repeat,
                        xqc.logging.metrics.compute_hessian_eigenspectrum(agent, replay_buffer, cfg.batch_size, cfg.num_seeds)
                    )
                

                # Reset agent
                if cfg.agent.reset_freq and update_count >= cfg.agent.reset_freq:
                    agent.reset()
                    update_count = 0

        finally:
            xqc.utils.save_slurm_outputs(wandb_run)


if __name__ == "__main__":
    main()
