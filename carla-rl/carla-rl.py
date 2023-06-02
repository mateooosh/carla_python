import os

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import float32, cast

from carla_python.gym_carla.envs import CarlaEnv
from agent import DQNAgent, ActorCritic

actions = [
    [0.6, -0.1],  # left
    [0.6, 0.1],  # right
    [0.6, -0.5],  # big left
    [0.6, 0.5]  # big right
]


def main():
    config = {
        'display_size': 128,
        'dt': 0.1,
        'actions': actions,
        'vehicle': 'vehicle.tesla.model3',
        'port': 2000,
        'town': 'Town07_Opt',
        'max_time_episode': 200,
        'max_waypoints': 24,
        'out_lane_distance': 1.5
    }

    episodes = 100
    batch_size = 50
    keep_learning = True
    algorithm = 'ac'
    rewards = np.loadtxt(algorithm + '/txt/rewards.txt') if (keep_learning is True) else np.asarray([])
    avg_rewards = np.loadtxt(algorithm + '/txt/avg_rewards.txt') if (keep_learning is True) else np.asarray([])
    steps_per_episode = np.loadtxt(algorithm + '/txt/steps_per_episode.txt') if (keep_learning is True) else np.asarray([])
    distances = np.loadtxt(algorithm + '/txt/distances.txt') if (keep_learning is True) else np.asarray([])
    avg_speed = np.loadtxt(algorithm + '/txt/avg_speed.txt') if (keep_learning is True) else np.asarray([])

    # ep = 2375
    # rewards = rewards[:ep]
    # avg_rewards = avg_rewards[:ep]
    # steps_per_episode = steps_per_episode[:ep]
    # distances = distances[:ep]
    # avg_speed = avg_speed[:ep]

    first_episode = 1 + len(rewards)

    # Set gym-carla environment
    if algorithm == 'dqn':
        agent = DQNAgent(len(actions))
    elif algorithm == 'ac':
        agent = ActorCritic(len(actions))

    env = CarlaEnv(config)

    if keep_learning:
        if algorithm == 'dqn':
            agent.load('dqn/model_output/weights_2300.hdf5')
        elif algorithm == 'ac':
            agent.load('ac/model_output/actor_weights_2500.hdf5', 'ac/model_output/critic_weights_2500.hdf5')

    for e in range(first_episode, first_episode + episodes):
        state = env.reset()
        state['camera'] = cast(state['camera'], float32) / 255.0

        state = [np.expand_dims(state['camera'], axis=0), np.expand_dims(state['state'], axis=0)]

        done = False
        total_reward = 0
        total_distance = 0
        steps = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state['camera'] = cast(next_state['camera'], float32) / 255.0

            print('Episode: ', e, ' | Action: ', action, ' | Reward: ', reward)

            next_state = [np.expand_dims(next_state['camera'], axis=0), np.expand_dims(next_state['state'], axis=0)]

            total_reward += reward
            total_distance += _['distance']
            steps += 1
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print("Episode: {}/{}, ocena: {}, e: {:.2}".format(e, first_episode + episodes - 1, total_reward,
                                                                   agent.epsilon))
                rewards = np.append(rewards, total_reward)
                avg_rewards = np.append(avg_rewards, total_reward / steps)
                steps_per_episode = np.append(steps_per_episode, steps)
                distances = np.append(distances, total_distance)
                avg_speed = np.append(avg_speed, total_distance / steps / config['dt'])

        if len(agent.memory) > batch_size:
            agent.train(batch_size)

        if e % 25 == 0:
            if algorithm == 'dqn':
                agent.save("./dqn/model_output/weights_" + '{:04d}'.format(e) + ".hdf5")
            elif algorithm == 'ac':
                agent.save("./ac/model_output/actor_weights_" + '{:04d}'.format(e) + ".hdf5",
                           "./ac/model_output/critic_weights_" + '{:04d}'.format(e) + ".hdf5")

            np.savetxt(algorithm + '/txt/steps_per_episode.txt', steps_per_episode, fmt='%.2f')
            np.savetxt(algorithm + '/txt/avg_rewards.txt', avg_rewards, fmt='%.2f')
            np.savetxt(algorithm + '/txt/rewards.txt', rewards, fmt='%.2f')
            np.savetxt(algorithm + '/txt/distances.txt', distances, fmt='%.2f')
            np.savetxt(algorithm + '/txt/avg_speed.txt', avg_speed, fmt='%.2f')

        if e % 5 == 0:
            plot(steps_per_episode, 'Steps per episode', algorithm + '/plots/steps_per_episode.png')
            plot(avg_rewards, 'Average Reward', algorithm + '/plots/avg_rewards.png')
            plot(distances, 'Distance', algorithm + '/plots/distances.png')
            plot(avg_speed, 'Average Speed', algorithm + '/plots/avg_speed.png')
            plot(rewards, 'Reward', algorithm + '/plots/rewards.png')


def plot(array, ylabel, path):
    plt.clf()
    xaxis = np.array(range(1, len(array) + 1))
    yaxis = array
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    if os.path.exists(path):
        os.remove(path)

    plt.plot(xaxis, yaxis)
    plt.savefig(path)
    plt.show()


if __name__ == '__main__':
    main()
