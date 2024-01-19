# kodowanie: utf-8

# Python. Uczenie maszynowe. Wydanie III
# Sebastian Raschka (https://sebastianraschka.com) i Vahid Mirjalili](http://vahidmirjalili.com)
# Packt Publishing Ltd. 2019
#
# Repozytorium kodu: https://github.com/rasbt/python-machine-learning-book-3rd-edition
#
# Licencja: MIT License (https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/LICENSE.txt)

###############################################################################################################
# Rozdział 18.  Uczenie przez wzmacnianie jako mechanizm podejmowania decyzji w skomplikowanych środowiskach
###############################################################################################################

# Skrypt: quczenie.py

from gridworld_env import GridWorldEnv
from agent import Agent
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done'))


def run_qlearning(agent, env, num_episodes=50):
    history = []
    for episode in range(num_episodes):
        state = env.reset()
        env.render(mode='human')
        final_reward, n_moves = 0.0, 0
        while True:
            action = agent.choose_action(state)
            next_s, reward, done, _ = env.step(action)
            agent._learn(Transition(state, action, reward,
                                    next_s, done))
            env.render(mode='human', done=done)
            state = next_s
            n_moves += 1
            if done:
                break
            final_reward = reward
        history.append((n_moves, final_reward))
        print('Epizod %d.: Nagroda: %.1f Liczba ruchów %d'
              % (episode, final_reward, n_moves))

    return history


def plot_learning_history(history):
    fig = plt.figure(1, figsize=(14, 10))
    ax = fig.add_subplot(2, 1, 1)
    episodes = np.arange(len(history))
    moves = np.array([h[0] for h in history])
    plt.plot(episodes, moves, lw=4,
             marker="o", markersize=10)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Epizody', size=20)
    plt.ylabel('Liczba ruchó', size=20)

    ax = fig.add_subplot(2, 1, 2)
    rewards = np.array([h[1] for h in history])
    plt.step(episodes, rewards, lw=4)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Epizody', size=20)
    plt.ylabel('Nagrody końcowe', size=20)
    plt.savefig('q-uczenie-historia.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    env = GridWorldEnv(num_rows=5, num_cols=6)
    agent = Agent(env)
    history = run_qlearning(agent, env)
    env.close()

    plot_learning_history(history)
