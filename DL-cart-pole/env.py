import numpy as np
import gymnasium as gym
from ag import Agent
from util import plot_learning_curve

if __name__=='__main__':
    env = gym.make('CartPole-v1')
    n_games = 10000
    scores = []
    eps_history = []

    agent = Agent(lr=0.0001, input_dims=env.observation_space.shape, n_actions=env.action_space.n)

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()[0]

        while not done:
            action = agent.choose_action(observation)
          
            observation_, reward, done,_,info = env.step(action)
            print(observation_)
            score += reward
            agent.learn(observation, action, reward, observation_)
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print('episode ', i, 'score %.1f average score %.1f epsilon %.2f' % (score, avg_score, agent.epsilon))

    filename = 'cartpole_naive_dqn.png'
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)
