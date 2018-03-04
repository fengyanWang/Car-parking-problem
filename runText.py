from mazeEnv import Maze
from RLQLearing import QLearningTable

import numpy as np
import matplotlib.pyplot as plt

count = 0


def update():
    global count
    totleStepCount = []
    successStepCount = []
    errorStepCount = []
    for episode in range(1000):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            count += 1
            # break while loop when end of this episode
            if done:
                totleStepCount.append(count)
                if reward == 1:
                    successStepCount.append(count)
                elif reward == -1:
                    errorStepCount.append(count)
                count = 0
                break


    # end of game
    print('game over')
    env.destroy()

    plt.figure(1)
    plt.plot(np.arange(len(totleStepCount)), totleStepCount)
    plt.title('totleStepCount')

    plt.figure(2)
    plt.plot(np.arange(len(successStepCount)), successStepCount)
    plt.title('successStepCount')

    plt.figure(3)
    plt.plot(np.arange(len(errorStepCount)), errorStepCount)
    plt.title('errorStepCount')

    plt.show()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
