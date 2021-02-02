# Continuous Control with Deep Reinforcement Learning (DDPG)

This implements a reinforcement learning algorithm [DDPG](https://arxiv.org/abs/1509.02971). Using the methodology employed for DQN, DDPG can resolve continous action space environments.

Following shows that DDPG can solve one of the continuous action space environment in `OpenAI Gym`.

|Episode: 0|Episode: 500|Episode: 900|
|---|---|---|
|![eps-0](assets/eps-0.gif)|![eps-500](assets/eps-500.gif)|![eps-900](assets/eps-900.gif)|

## Score Graph for `LunarLanderContinuous-v2`
<img src="assets/score_fig.png" width="300" height="300">

## Environments

* Pytorch 1.7
* Python 3.8

Please refer `requirements.txt` for python packages for this repo.
