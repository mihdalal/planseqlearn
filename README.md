# Plan-Seq-Learn: Language Model Guided RL for Solving Long Horizon Robotics Tasks
<div style="text-align: center;">

This repository is the official implementation of Plan-Seq-Learn: Language Model Guided RL for Solving Long Horizon Robotics Tasks.

[Murtaza Dalal](https://mihdalal.github.io/)$^1$, [Tarun Chiruvolu](https://www.linkedin.com/in/tarun-chiruvolu/)$^1$,  [Devendra Chaplot](https://devendrachaplot.github.io/)$^2$, [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/)$^1$

$^1$ CMU, $^2$ Mistral AI

[Project Page](https://mihdalal.github.io/planseqlearn/)


https://github.com/mihdalal/planseqlearn/assets/21963649/19cb1e11-5844-4713-9206-658ec376d35e



<!-- <img src="TODO" width="100%" title="main gif"> -->
<div style="margin:10px; text-align: justify;">
Plan-Seq-Learn (PSL) is a method for enabling RL agents to learn to solve long-horizon robotics by leveraging LLM guidance via motion planning. We release all training code, environments and configuration files for replicating our results.

If you find this codebase useful in your research, please cite:
```bibtex
@inproceedings{dalal2024psl,
    title={Plan-Seq-Learn: Language Model Guided RL for Solving Long Horizon Robotics Tasks},
    author={Dalal, Murtaza and Chiruvolu, Tarun and Chaplot, Devendra and Salakhutdinov, Ruslan},
    journal={International Conference on Learning Representations},
    year={2024}
}
```

# Table of Contents

- [Installation](#installation)
- [Model Training](#model-training)
- [Troubleshooting and Known Issues](#troubleshooting-and-known-issues)
- [Citation](#citation)

# Installation
To install dependencies, please run the following commands:
```
git clone --recurse-submodules git@github.com:mihdalal/planseqlearn.git
cd planseqlearn
git lfs install
git lfs track "*.plugin"
./install.sh
```

Please add the following to your bashrc/zshrc:
```
export MUJOCO_GL='egl'
WANDB_API_KEY=...
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.3/
```

If you would like to use a container instead to avoid dependency installation:
Our pre-built docker image is at `mihdalal/planseqlearn` and we include a singularity container defintion file in our repo at `containers/planseqlearn.def`. The singularity image can be built using the command `sudo singularity build planseqlearn.sif planseqlearn.def`.

Two quick notes:
1. The dockerfile will always re-compile mujoco_py everytime it opens. We haven't been able to avoid this unfortunately - if anyone knows the solution please feel free to make a pull request with the fix!
2. The singularity container only contains dependencies! Within this container you still need to perform all the necessary python installations, copying of OMPL paths, etc.

# Model Training
To train the policies from a particular environment suite, run the following command:
```
python planseqlearn/launch_scripts/psl_[suite].py
```

Some notes:
For efficiency we train policies with motion planning and vision-based pose estimation disabled. If you want to train with those enabled, set `use_vision_pose_estimation=True` and `use_mp=True`. This will significantly slow down training.
We do not use the LLM in the loop to generate the plan (for our scenes the plan needs to just be generated once per task - this will save you a lot of money!). We have already generated the plans using GPT-4V and saved them out in env_text_plans.py.

# Troubleshooting and Known Issues

- If your training seems to be proceeding slowly (especially for image-based agents), it might be a problem with robomimic and more modern versions of PyTorch. We recommend PyTorch 1.12.1 (on Ubuntu, we used `pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`). It is also a good idea to verify that the GPU is being utilized during training.
- If you run into trouble with installing [egl_probe](https://github.com/StanfordVL/egl_probe) during robomimic installation (e.g. `ERROR: Failed building wheel for egl_probe`) you may need to make sure `cmake` is installed. A simple `pip install cmake` should work.
- If you run into other strange installation issues, one potential fix is to launch a new terminal, activate your conda environment, and try the install commands that are failing once again. One clue that the current terminal state is corrupt and this fix will help is if you see installations going into a different conda environment than the one you have active.

If you run into an error not documented above, please search through the [GitHub issues](https://github.com/mihdalal/planseqlearn/issues), and create a new one if you cannot find a fix.

## Citation

Please cite [the PSL paper](https://mihdalal.github.io/planseqlearn/resources/paper.pdf) if you use this code in your work:

```bibtex
@inproceedings{dalal2024psl,
    title={Plan-Seq-Learn: Language Model Guided RL for Solving Long Horizon Robotics Tasks},
    author={Dalal, Murtaza and Chiruvolu, Tarun and Chaplot, Devendra and Salakhutdinov, Ruslan},
    journal={International Conference on Learning Representations},
    year={2024}
}
```
