
# ADAPT: Auction-Based Dynamic Prioritization for Multi-Agent Coordination


## Overview

This repository contains the official implementation of ADAPT (Auction-based Dynamic Action Priority Technique), a dynamic coordination framework for multi-agent reinforcement learning (MARL) designed to enhance coordination under partial observability and synchronous execution. ADAPT combines:

- **Transformer-based observation encoding** to extract compact and informative features from high-dimensional observations.
- **Message generation with mutual information objectives** to produce meaningful and informative messages for communication.
- **Dynamic priority scheduling via distributed auctions** to assign execution priorities based on real-time inter-agent dependencies.
- **Autoregressive action inference** to model a learned causal order for policy inference.
- **Observation reconstruction** to recover global observation embeddings from compact messages.

Experiments on StarCraft Multi-Agent Challenge v2 (SMACv2) and Google Research Football (GRF) show that ADAPT achieves significantly higher win rates and reduced communication overhead compared to state-of-the-art baselines. For example, on SMACv2's \texttt{Terran\_10\_vs\_11} map, ADAPT attains a win rate of 53.70%, surpassing CommFormer (31.47%) and SeqComm (30.41%). ADAPT also demonstrates strong generalization on GRF tasks. In terms of communication efficiency, ADAPT reduces message bytes per timestep by up to 51.54% compared to SeqComm and 3.77% compared to CommFormer.


## Instructions

This code is implemented based on https://github.com/marlbenchmark/on-policy, and the running instructions are similar to that in the original project.

### Dependences
``` Bash
pip install -r requirements.txt
```

### StarCraft II & SMAC
Run the script
``` Bash
bash install_sc2.sh
```
Or you could install them manually to other path you like, just follow here: https://github.com/oxwhirl/smac.

### Google Research Football
Please following the instructios in https://github.com/google-research/football. 


### Quick Start

When your environment is ready, you could run shells in the "scripts" folder with algo="ADAPT". For example:

``` Bash
bash ./train_smacv2.sh  # run ADAPT on SMACv2
```
If you would like to change the configs of experiments, you could modify sh files or look for config.py for more details.


## Citation

If you find this project helpful, please consider to cite the following paper:

```
@inproceedings{Xie2025ADAPT,
  author    = {Xie, Zaipeng and Qiao, Chentai and Yang, Nuo and Zhao, Yiming},
  title     = {ADAPT: Auction-Based Dynamic Prioritization for Multi-Agent Coordination},
  booktitle = {Proceedings of the 28th European Conference on Artificial Intelligence (ECAI 2025)},
  year      = {2025},
  address   = {Bologna, Italy},
  month     = {October 25--30},
  note      = {in press},
}
```

