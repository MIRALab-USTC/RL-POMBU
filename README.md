# POMBU
Code to reproduce Deep 'Model-Based Reinforcement Learning via Estimated Uncertainty and Conservative Policy Optimization'.

## Installation
1. Install MuJoCo at ~/.mujoco and copy the license key to ~/.mujoco/mjkey.txt.
2. Clone this RL-POMBU.
3. Create a conda environment by running
```
conda env create -f environmen.yml
```

## Run

```
cd shells
./run_cheetah.sh
```

## Remarks
1. There is some obsolete code. We will refactor the code in winter vacation.
2. We provide appendix in [arxiv](<https://arxiv.org/pdf/1911.12574.pdf>).