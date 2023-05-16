import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import float32, cast

from carla_python.gym_carla.envs import CarlaEnv
from agent import DQNAgent, ActorCritic
import tensorflow as tf

LABELS_MAP = {
    (0, 0, 0): 0,  # "Unlabeled",
    (70, 70, 70): 1,  # "Building",
    (100, 40, 40): 2,  # "Fence",
    (55, 90, 80): 3,  # "Other",
    (220, 20, 60): 4,  # "Pedestrian",
    (153, 153, 153): 5,  # "Pole",
    (157, 234, 50): 6,  # "RoadLine",
    (128, 64, 128): 7,  # "Road",
    (244, 35, 232): 8,  # "SideWalk",
    (107, 142, 35): 9,  # "Vegetation",
    (0, 0, 142): 10,  # "Vehicles",
    (102, 102, 156): 11,  # "Wall",
    (220, 220, 0): 12,  # "TrafficSign",
    (70, 130, 180): 13,  # "Sky",
    (81, 0, 81): 14,  # "Ground",
    (150, 100, 100): 15,  # "Bridge",
    (230, 150, 140): 16,  # "RailTrack",
    (180, 165, 180): 17,  # "GuardRail",
    (250, 170, 30): 18,  # "TrafficLight",
    (110, 190, 160): 19,  # "Static",
    (170, 120, 50): 20,  # "Dynamic",
    (45, 60, 150): 21,  # "Water",
    (145, 170, 100): 22,  # "Terrain",
}

actions = [
    [-1.0, 0.0], # brake
    [0.7, -0.1], # left
    [0.7, 0.1], # right
    [0.7, -0.5], # big left
    [0.7, 0.5] # big right
]


def main():
    # parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 0,  # 100
        'display_size': 128,  # screen size of bird-eye render
        'dt': 0.1,  # time interval between two frames
        'actions': actions,
        'discrete': True,  # whether to use discrete control space
        'ego_vehicle_filter': 'vehicle.tesla.model3',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'town': 'Town07_Opt',  # which town to simulate
        'max_time_episode': 150,  # maximum timesteps per episode
        'max_waypt': 24,  # maximum number of waypoints
        'out_lane_thres': 0.7,  # threshold for out of lane
        'desired_speed': 4,  # desired speed (m/s)
        'max_ego_spawn_times': 200  # maximum times to spawn ego vehicle
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

    # rewards = rewards[:600]
    # avg_rewards = avg_rewards[:600]
    # steps_per_episode = steps_per_episode[:600]
    # distances = distances[:600]
    # avg_speed = avg_speed[:600]

    first_episode = 1 + len(rewards)

    # Set gym-carla environment
    if algorithm == 'dqn':
        agent = DQNAgent(len(actions))
    elif algorithm == 'ac':
        agent = ActorCritic(len(actions))

    env = CarlaEnv(params)

    if keep_learning:
        if algorithm == 'dqn':
            agent.load('dqn/model_output/weights_1250.hdf5')
        elif algorithm == 'ac':
            agent.load('ac/model_output/actor_weights_1000.hdf5', 'ac/model_output/critic_weights_1000.hdf5')

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
                avg_speed = np.append(avg_speed, total_distance / steps / params['dt'])

        if len(agent.memory) > batch_size:
            agent.train(batch_size)

        if e % 25 == 0:
            if algorithm == 'dqn':
                agent.save("./dqn/model_output/weights_" + '{:04d}'.format(e) + ".hdf5")
            elif algorithm == 'ac':
                agent.save("./ac/model_output/actor_weights_" + '{:04d}'.format(e) + ".hdf5", "./ac/model_output/critic_weights_" + '{:04d}'.format(e) + ".hdf5")

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
    print("Num GPUs Available: ", tf.config.list_physical_devices())
    start = datetime.datetime.now()
    main()

    stop = datetime.datetime.now()
    print(start)
    print(stop)

