import gym
from carla import LidarDetection, Location

import python.gym_carla
import carla

from python.gym_carla.envs import CarlaEnv


def main():
    # parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 20, #100
        'number_of_walkers': 0,
        'display_size': 256,  # screen size of bird-eye render
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.1,  # time interval between two frames
        'discrete': False,  # whether to use discrete control space
        'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
        'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
        'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
        'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.mercedes.coupe_2020',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'town': 'Town02',  # which town to simulate
        'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
        'max_time_episode': 1000,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'obs_range': 32,  # observation range (meter)
        'lidar_bin': 0.125,  # bin size of lidar sensor (meter)   0.125
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 8,  # desired speed (m/s)
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
        'display_route': True,  # whether to render the desired route
        'pixor_size': 64,  # size of the pixor labels
        'pixor': False,  # whether to output PIXOR observation
    }

    # Set gym-carla environment
    env = CarlaEnv(params)

    episodes = 10

    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = [2.0, 0.0]
            state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                print("Episode: {}/{}, ocena: {}".format(e, episodes - 1, total_reward))


if __name__ == '__main__':
    main()



# CarlaUE4 -ResX=640 -ResY=480
