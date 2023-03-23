import datetime

import numpy as np

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
        'discrete_acc': [0.0, 3.0],  # discrete value of accelerations
        'discrete_steer': [-0.3, 0.0, 0.3],  # discrete value of steering angles
        'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
        'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.lincoln.*',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'town': 'Town02',  # which town to simulate
        'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
        'max_time_episode': 100,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'obs_range': 32,  # observation range (meter)
        'lidar_bin': 0.125,  # bin size of lidar sensor (meter)   0.125
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 8,  # desired speed (m/s)
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
        'display_route': True  # whether to render the desired route
    }

    episodes = 21
    batch_size = 100
    rewards = []

    # Set gym-carla environment
    env = CarlaEnv(params)
    agent = DQNAgent(6)

    for e in range(episodes):
        state = env.reset()
        state = state['birdeye']
        images_list = []
        images_list.append(np.array(state))
        state = np.asarray(images_list)

        # import numpy as np
        # transformedImage = np.expand_dims(transformedImage, axis=0)
        # transformedImage.shape

        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state['birdeye']
            images_list = []
            images_list.append(np.array(next_state))
            next_state = np.asarray(images_list)

            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print("Episode: {}/{}, ocena: {}, e: {:.2}".format(e + 1, episodes, total_reward, agent.epsilon))
                rewards.append(total_reward)

        if len(agent.memory) > batch_size:
            agent.train(batch_size)

        if e % 20 == 0:
            agent.save("model_output/weights_" + '{:04d}'.format(e) + ".hdf5")

    print(rewards)


if __name__ == '__main__':
    print("Num GPUs Available: ", tf.config.list_physical_devices())
    start = datetime.datetime.now()
    main()
    stop = datetime.datetime.now()
    print(start)
    print(stop)

# CarlaUE4 -ResX=640 -ResY=480
