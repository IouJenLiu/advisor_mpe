## Bridging the Imitation Gap by Adaptive Insubordination #


The repository contains Pytorch implementation of ADVISOR and baselines on the Coordinated Navigation (n=3) Task.


#### Platform and Dependencies:
* Ubuntu 16.04
* Python 3.7
* Pytorch 1.1.0
* OpenAI gym 0.10.9 (https://github.com/openai/gym)
* matplotlib
* numba 0.43.1
* llvmlite 0.32.1


### Install the improved MPE:
    cd multiagent-particle-envs
    pip install -e .
Please ensure that `multiagent-particle-envs` has been added to your `PYTHONPATH`.


### Evaluation
    cd maddpg/script
    sh eval_advisor.sh
    sh eval_il_rl_static.sh
    sh eval_il.sh
    sh eval_rl.sh
The above scripts evaluate the agents trained by ADVISOR, RL + IL with static weighting, IL, and RL on Cooperative Navigation. 


### Training
    cd maddpg/script
    sh train_advisor.sh
    sh train_il_rl_static.sh
    sh train_il.sh
    sh train_rl.sh
The above scripts train the agents  using ADVISOR, RL + IL with static weighting, IL, and RL on Cooperative Navigation


### Reference
If you use this work, please cite:

```text
@inproceedings{advisor,
  title={Bridging the Imitation Gap by Adaptive Insubordination},
  author={Weihs, Luca and Jain, Unnat and Liu, Iou-Jen and Salvador, Jordi and Lazebnik, Svetlana and Kembhavi, Aniruddha and Schwing, Alexander},
  booktitle={NeurIPS},
  year={2021},
  note = {the first two authors contributed equally},
}
```
