import datetime
import os

import numpy as np
import matplotlib.pyplot as plt

from carla_python.gym_carla.envs import CarlaEnv
from agent import DQNAgent
import tensorflow as tf


def main():
    # parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 40,  # 100
        'number_of_walkers': 0,
        'display_size': 256,  # screen size of bird-eye render
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.1,  # time interval between two frames
        'discrete': True,  # whether to use discrete control space
        'discrete_acc': [-1.0, 3.0],  # discrete value of accelerations
        'discrete_steer': [-0.3, 0.0, 0.3],  # discrete value of steering angles
        'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
        'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.mercedes.coupe_2020',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'town': 'Town07_Opt',  # which town to simulate
        'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
        'max_time_episode': 100,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'obs_range': 32,  # observation range (meter)
        'lidar_bin': 0.125,  # bin size of lidar sensor (meter)   0.125
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 1.0,  # threshold for out of lane
        'desired_speed': 5,  # desired speed (m/s)
        # 'desired_speed': 8,  # desired speed (m/s)
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
        'display_route': True  # whether to render the desired route
    }

    episodes = 1
    batch_size = 32
    keep_learning = False
    rewards = np.loadtxt('txt/rewards.txt') if (keep_learning == True) else np.asarray([])
    avg_rewards = np.loadtxt('txt/avg_rewards.txt') if (keep_learning == True) else np.asarray([])
    steps_per_episode = np.loadtxt('txt/steps_per_episode.txt') if (keep_learning == True) else np.asarray([])
    first_episode = 1 + len(rewards)
    neural_network = 'LeNet'

    # Set gym-carla environment
    env = CarlaEnv(params)
    agent = DQNAgent(6, neural_network)
    if keep_learning:
        agent.load('model_output/weights_0900.hdf5')

    for e in range(first_episode, first_episode + episodes):
        state = env.reset()
        state = np.concatenate((state['birdeye'], state['camera']), axis=1)
        state = np.expand_dims(state, axis=0)

        done = False
        total_reward = 0
        steps = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            print('Episode: ', e, ' | Action: ', action, ' | Reward: ', reward)
            next_state = np.concatenate((next_state['birdeye'], next_state['camera']), axis=1)
            next_state = np.expand_dims(next_state, axis=0)

            total_reward += reward
            steps += 1
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print("Episode: {}/{}, ocena: {}, e: {:.2}".format(e, first_episode + episodes - 1, total_reward,
                                                                   agent.epsilon))
                rewards = np.append(rewards, total_reward)
                avg_rewards = np.append(avg_rewards, total_reward / steps)
                steps_per_episode = np.append(steps_per_episode, steps)

        if len(agent.memory) > batch_size:
            agent.train(batch_size)

        if e % 20 == 0:
            agent.save("model_output/weights_" + '{:04d}'.format(e) + ".hdf5")

            # plot steps per epoch
            plot(steps_per_episode, 'Steps per episode', './plots/steps_per_episode.png')
            np.savetxt('./txt/steps_per_episode.txt', steps_per_episode, fmt='%.2f')

            # plot average rewards
            plot(avg_rewards, 'Average Reward', './plots/avg_rewards.png')
            np.savetxt('./txt/avg_rewards.txt', avg_rewards, fmt='%.2f')

            # plot rewards
            plot(rewards, 'Reward', './plots/rewards.png')
            np.savetxt('./txt/rewards.txt', rewards, fmt='%.2f')

    print(rewards)


def plot(array, ylabel, path):
    plt.clf()
    xaxis = np.array(range(len(array)))
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

# CarlaUE4 -ResX=640 -ResY=480 -quality-level=Low
# CarlaUE4 -quality-level=Low -RenderOffScreen
