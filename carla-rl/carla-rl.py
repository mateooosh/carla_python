import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import float32, cast

from carla_python.gym_carla.envs import CarlaEnv
from agent import DQNAgent
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


def get_labels(image):
    labels = np.zeros((image.shape[0], image.shape[1]), dtype=np.float)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_color = tuple(image[i, j])
            labels[i, j] = LABELS_MAP.get(pixel_color, 0) / len(LABELS_MAP.keys())
    return labels


def main():
    # parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 40,  # 100
        'display_size': 128,  # screen size of bird-eye render
        'dt': 0.1,  # time interval between two frames
        'discrete': True,  # whether to use discrete control space
        'discrete_acc': [-1.0, 2.0],  # discrete value of accelerations
        'discrete_steer': [-0.3, 0.3],  # discrete value of steering angles
        'ego_vehicle_filter': 'vehicle.tesla.model3',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'town': 'Town07_Opt',  # which town to simulate
        'max_time_episode': 150,  # maximum timesteps per episode
        'max_waypt': 24,  # maximum number of waypoints
        'out_lane_thres': 0.7,  # threshold for out of lane
        'desired_speed': 4,  # desired speed (m/s)
        'max_ego_spawn_times': 200  # maximum times to spawn ego vehicle
    }

    episodes = 50
    batch_size = 50
    keep_learning = True
    rewards = np.loadtxt('txt/rewards.txt') if (keep_learning is True) else np.asarray([])
    avg_rewards = np.loadtxt('txt/avg_rewards.txt') if (keep_learning is True) else np.asarray([])
    steps_per_episode = np.loadtxt('txt/steps_per_episode.txt') if (keep_learning is True) else np.asarray([])
    distances = np.loadtxt('txt/distances.txt') if (keep_learning is True) else np.asarray([])
    avg_speed = np.loadtxt('txt/avg_speed.txt') if (keep_learning is True) else np.asarray([])

    # rewards = rewards[:800]
    # avg_rewards = avg_rewards[:800]
    # steps_per_episode = steps_per_episode[:800]
    # distances = distances[:800]
    # avg_speed = avg_speed[:800]

    first_episode = 1 + len(rewards)
    neural_network = 'model_3'

    # Set gym-carla environment
    agent = DQNAgent(4, neural_network)
    env = CarlaEnv(params)

    if keep_learning:
        # 1625 gdzie jest najlepej
        agent.load('model_output/weights_0850.hdf5')

    for e in range(first_episode, first_episode + episodes):
        state = env.reset()
        state['camera'] = cast(state['camera'], float32) / 255.0

        # segmentation_image = state['camera']
        # labels = get_labels(segmentation_image)
        # state['camera'] = np.resize(labels, (segmentation_image.shape[0], segmentation_image.shape[1], 1))

        state = [np.expand_dims(state['camera'], axis=0), np.expand_dims(state['state'], axis=0)]

        done = False
        total_reward = 0
        total_distance = 0
        steps = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state['camera'] = cast(next_state['camera'], float32) / 255.0

            # segmentation_image = next_state['camera']
            # labels = get_labels(segmentation_image)
            # next_state['camera'] = np.resize(labels, (segmentation_image.shape[0], segmentation_image.shape[1], 1))
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
            agent.save("model_output/weights_" + '{:04d}'.format(e) + ".hdf5")

            np.savetxt('./txt/steps_per_episode.txt', steps_per_episode, fmt='%.2f')
            np.savetxt('./txt/avg_rewards.txt', avg_rewards, fmt='%.2f')
            np.savetxt('./txt/rewards.txt', rewards, fmt='%.2f')
            np.savetxt('./txt/distances.txt', distances, fmt='%.2f')
            np.savetxt('./txt/avg_speed.txt', avg_speed, fmt='%.2f')

        if e % 5 == 0:
            plot(steps_per_episode, 'Steps per episode', './plots/steps_per_episode.png')
            plot(avg_rewards, 'Average Reward', './plots/avg_rewards.png')
            plot(distances, 'Distance', './plots/distances.png')
            plot(avg_speed, 'Average Speed', './plots/avg_speed.png')
            plot(rewards, 'Reward', './plots/rewards.png')


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

