
# ADAPT: Auction-Based Dynamic Prioritization for Multi-Agent Coordination


## Overview

Effective coordination in multi-agent systems remains challenging in dynamic and partially observable environments, where agents must reason over evolving interdependencies and limited communication bandwidth. We propose \textbf{ADAPT} (\textit{Auction-based Dynamic Action Priority Technique}), a unified framework for multi-agent coordination that integrates message compression, dependency estimation, and a novel auction-based dynamic prioritization mechanism. 

In ADAPT, agents exchange compact messages and compute dependency scores to determine how much their behavior depends on others. A distributed auction protocol then assigns priority positions, guiding autoregressive decision-making in a manner aligned with inter-agent influence. This enables flexible, influence-aware coordination without centralized control or extensive communication rounds. Experiments on SMACv2 and Google Research Football (GRF) show that ADAPT achieves higher win rates, faster convergence, and lower communication cost compared to state-of-the-art baselines. Further analyses confirm its scalability to large teams, compatibility with value decomposition, and runtime efficiency. These results highlight ADAPT's potential for scalable and responsive multi-agent coordination under real-world constraints.


## Installation

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


## Quick Start

When your environment is ready, you could run shells in the "scripts" folder with algo="ADAPT". For example:

``` Bash
bash ./train_smacv2.sh  # run ADAPT on SMACv2
```
If you would like to change the configs of experiments, you could modify sh files or look for config.py for more details.




