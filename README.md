<h1><img src="assets/xqc_logo.svg" alt="XQC" height="94"></h1>

Official implementation of 

**XQC: Well-conditioned Optimization Accelerates Deep Reinforcement Learning**\
[Daniel Palenicek](https://danielpalenicek.github.io/), 
Florian Vogt, 
[Joe Watson](https://joemwatson.github.io/), 
[Ingmar Posner](https://eng.ox.ac.uk/people/ingmar-posner) and 
[Jan Peters](https://www.ias.informatik.tu-darmstadt.de/Team/JanPeters)\
International Conference on Learning Representations (ICLR) 2026\
[[Paper]](https://arxiv.org/abs/2509.25174) [[Website]](https://danielpalenicek.github.io/projects/xqc)

> **TL;DR:** We introduce XQC; A well-conditioned critic architecture that achieves state-of-the-art sample efficiency on 70 continuous control tasks with 4.5× fewer parameters than SimbaV2.

<img src="assets/hero.png" alt="XQC Overview" style="border-radius: 8px;" />

## Abstract
Sample efficiency is a central property of effective deep reinforcement learning algorithms. Recent work has improved this through added complexity, such as larger models, exotic network architectures, and more complex algorithms, which are typically motivated purely by empirical performance. We take a more principled approach by focusing on the optimization landscape of the critic network. Using the eigenspectrum and condition number of the critic's Hessian, we systematically investigate the impact of common architectural design decisions on training dynamics. Our analysis reveals that a novel combination of batch normalization (BN), weight normalization (WN), and a distributional cross-entropy (CE) loss produces condition numbers orders of magnitude smaller than baselines. This combination also naturally bounds gradient norms, a property critical for maintaining a stable effective learning rate under non-stationary targets and bootstrapping. Based on these insights, we introduce XQC: a well-motivated, sample-efficient deep actor-critic algorithm built upon soft actor-critic that embodies these optimization-aware principles. We achieve state-of-the-art sample efficiency across 55 proprioception and 15 vision-based continuous control tasks, all while using significantly fewer parameters than competing methods.

## Setup

### 1. Clone the Repository recursively to include the necessary submodules
```bash
git clone --recurse-submodules https://github.com/danielpalenicek/xqc.git
cd xqc
```

### 2. Environment Setup

Using `uv` to sync the dependencies:
```bash
uv sync
# To run commands:
uv run python train_parallel.py ...
```



# Usage

The main entry point for training is `train_parallel.py`.

#### Basic Example
Train XQC on the `h1-walk-v0` environment:
```bash
uv run python train_parallel.py env=h1-walk-v0 seed=1
```

#### Key Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `env` | `h1-walk-v0` | Name of the environment to train on from available Benchmark suites (`dmc`, `mujoco`, `myo`, `hb`). |
| `seed` | `0` | Random seed for reproducibility. |
| `max_steps` | `1_000_000` | Total training steps. |
| `num_seeds` | `10` | Number of parallel seeds to run simultaneously (JAX vmap). |
| `wandb.mode` | `disabled` | Set to `online` to log results to Weights & Biases. |

## Acknowledgments
This codebase builds upon and adapts code from several open-source repositories. We thank the authors for their contributions:
- [jaxrl](https://github.com/ikostrikov/jaxrl) which served as the original foundation for this codebase.
- [SimbaV2](https://github.com/dojeon-ai/SimbaV2) for metrics computation tools.
- [BiggerRegularizedCategorical](https://github.com/naumix/BiggerRegularizedCategorical) for parallel environment wrappers.
- [deep-rl-plasticity](https://github.com/awjuliani/deep-rl-plasticity) for network dormancy and plasticity metric tracking.
- [spectral-density](https://github.com/google/spectral-density) for our Hessian eigenspectrum analyses.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{palenicek2026xqc,
  title={XQC: Well-Conditioned Optimization Accelerates Deep Reinforcement Learning},
  author={Palenicek, Daniel and Vogt, Florian and Watson, Joe and Posner, Ingmar and Peters, Jan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```