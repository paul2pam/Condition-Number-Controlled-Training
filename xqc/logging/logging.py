import wandb
from omegaconf import OmegaConf, DictConfig
from termcolor import colored
import flax.traverse_util
import jax


def print_config(cfg: DictConfig):
    """Prints a structured table of hyperparameters."""
    print("\n" + "=" * 70)
    algo_name = cfg.agent.get("name", "AGENT")
    benchmark = cfg.env.get("benchmark", "UNKNOWN")
    env_name = cfg.env.get("name", "UNKNOWN")
    header = f" HYPERPARAMETERS - {algo_name} - {benchmark}/{env_name} "
    print(colored(header.center(70, "="), "cyan", attrs=["bold"]))
    print("=" * 70)

    def print_dict(content, indent=0):
        for k, v in content.items():
            if isinstance(v, (dict, DictConfig)):
                print("  " * indent + colored(f"{k}:", "yellow", attrs=["bold"]))
                print_dict(v, indent + 1)
            else:
                key_str = "  " * indent + colored(f"{k}:", "green")
                # Handle lists or complex types
                if isinstance(v, (list, tuple)):
                    val_str = str(list(v))
                else:
                    val_str = str(v)
                print(f"{key_str:<40} {val_str}")

    # We resolve the config to handle interpolation
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    print_dict(resolved_cfg)
    print("=" * 70 + "\n")


def log_multiple_seeds_to_wandb(step, infos, fps=30):
    dict_to_log = {}
    for info_key in infos:
        for seed, value in enumerate(infos[info_key]):
            if info_key == "renders":
                dict_to_log[f"seed{seed}/video"] = wandb.Video(value, fps=fps, format="mp4")
            else:
                dict_to_log[f"seed{seed}/{info_key}"] = value
                if info_key == "return" or info_key == "r" or info_key == "goal":
                    print(info_key, value)
    if "renders" in infos:
        print("Logged video of size", infos["renders"].shape)
    wandb.log(dict_to_log, step=step)


def print_total_param_count(models: list):
    total_params = 0

    for model in models:
        params_flat = flax.traverse_util.flatten_dict(model.params)

        for _, value in params_flat.items():
            param_count = 1
            for dim in value.shape:
                param_count *= dim
            total_params += param_count
