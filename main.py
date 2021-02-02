import gym
import numpy as np

from ddpg.agent import Agent

env = gym.make('LunarLanderContinuous-v2')

agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, batch_size=64,
              layer1_size=400, layer2_size=300, n_actions=2)

score_history = []

for i in range(1000):
    done = False
    score = 0
    obs = env.reset()

    while not done:
        act = agent.choose_action(obs)
        next_state, reward, done, info = env.step(act)
        agent.memory.push(obs, act, reward, next_state, int(done))
        agent.learn()
        score += reward
        obs = next_state

    score_history.append(score)

    print("==============================")
    print('Episode: ', i)
    print('Score: ', score)
    print('Last 100 avg: ', np.mean(score_history[-100:]))

    if i % 25 == 0:
        agent.save_models()
