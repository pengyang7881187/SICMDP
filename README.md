# SICMDP

Semi-infinitely Constrained Markov Decision Processes

## Requirements

* [Gurobipy](https://www.gurobi.com/)
* [OpenAI Gym](https://gym.openai.com/): ```pip install gym gym[atari]```

## Getting started

The customized environments *Toy SICMDP* and *Discharge of Sewage* are implemented in `toymdp_env.py` and `pollution_env.py` respectively.

Some utility functions and the algorithm *SI-CRL* are included in `SICMDP.py`.

The python files with prefix *experiment* are for the experiments in the paper.
