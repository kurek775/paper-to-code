import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

policy = {0: 1, 1: 2, 2: 1, 3: 0, 4: 1, 6: 1, 8: 2, 9: 1, 10: 1, 13: 2, 14: 2}

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
n_games = 1000
win_pct = []
scores = []

for i in range(n_games):
    done = False
    obs = env.reset()[0]
    score = 0
    while not done:
        action = policy[obs]
        obs = env.step(action)[0]
        reward = env.step(action)[1]
        done = env.step(action)[2]
        info = env.step(action)[3]

        score += reward

    scores.append(score)
    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)
plt.plot(win_pct)
plt.show()

