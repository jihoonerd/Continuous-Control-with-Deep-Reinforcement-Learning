import gym
import numpy as np
import pyvirtualdisplay
from ddpg.agent import Agent
import imageio
import pathlib
import os
import matplotlib.pyplot as plt

display = pyvirtualdisplay.Display(visible=False)
display.start()

env = gym.make('LunarLanderContinuous-v2')

agent = Agent(lr_actor=0.000025, lr_critic=0.00025, input_dims=[8], tau=0.001, batch_size=64,
              layer1_size=400, layer2_size=300, n_actions=2)

score_history = []
record = False
img_path = 'images'
if not os.path.exists(img_path):
    pathlib.Path(img_path).mkdir(parents=True, exist_ok=True)

for i in range(1000):
    done = False
    score = 0
    obs = env.reset()
    frame_set = []

    if i % 100 == 0:
        record = True
    
    while not done:

        act = agent.choose_action(obs)
        next_state, reward, done, info = env.step(act)
        agent.memory.push(obs, act, reward, next_state, int(done))
        agent.learn()
        score += reward

        if record:
            frame_set.append(env.render(mode='rgb_array'))
        obs = next_state
    
    if record:
        imageio.mimsave(os.path.join(img_path, f'eps-{i}.gif'), frame_set, fps=30)
        record = False   
    score_history.append(score)

    print("==============================")
    print('Episode: ', i)
    print('Score: ', score)
    print('Last 100 avg: ', np.mean(score_history[-100:]))

    if i % 50 == 0:
        agent.save_models()

plt.plot(score_history)
plt.xlabel('episodes')
plt.ylabel('score')
plt.grid()
plt.savefig(os.path.join(img_path, "score_fig.png"))