from __future__ import division

import copy
import random
import time

import carla
from skimage.transform import resize

import gym
from gym import spaces
from gym.utils import seeding

from carla_python.gym_carla.envs.route_planner import RoutePlanner
from carla_python.gym_carla.envs.misc import *


class CarlaEnv(gym.Env):
    def __init__(self, params):
        self.display_size = params['display_size']
        self.dt = params['dt']
        self.max_time_episode = params['max_time_episode']
        self.out_lane_distance = params['out_lane_distance']
        self.max_waypoints = params['max_waypoints']
        self.obs_size = self.display_size
        self.waypoints = []
        self.actions = params['actions']
        self.action_space = spaces.Discrete(len(self.actions))

        print('Connecting...')
        client = carla.Client('localhost', params['port'])
        client.set_timeout(15.0)
        self.world = client.load_world(params['town'])
        self.map = self.world.get_map()
        print('Connected')

        self.world.set_weather(carla.WeatherParameters.CloudyNoon)
        self.vehicle_spawn_points = list(self.map.get_spawn_points())
        self.ego_bp = self._create_vehicle_blueprint(params['vehicle'], color='100,8,8')

        # Collisions
        self.collision_hist = []
        self.collision_hist_l = 1
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # Camera
        self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=2.2, z=1.4), carla.Rotation(pitch=-20.0))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
        self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
        self.camera_bp.set_attribute('fov', '110')
        self.camera_bp.set_attribute('sensor_tick', '0.02')

        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        self.reset_step = 0
        self.total_step = 0

        # pygame
        pygame.init()
        self.display = pygame.display.set_mode((self.obs_size, self.obs_size), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.display.fill((0, 0, 0))
        pygame.display.flip()


    def reset(self):
        if hasattr(self, 'collision_sensor'):
            self.collision_sensor.stop()

        if hasattr(self, 'camera_sensor'):
            self.camera_sensor.stop()

        self.collision_sensor = None
        self.camera_sensor = None
        self._clear_all_actors(['sensor.*', 'vehicle.*', 'controller.ai.*'])
        self._set_synchronous_mode(False)

        random.shuffle(self.vehicle_spawn_points)

        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)

        ego_spawn_times = 0
        while True:
            if ego_spawn_times > 200:
                self.reset()

            transform = random.choice(self.vehicle_spawn_points)

            if self._try_spawn_ego_vehicle_at(transform):
                break
            else:
                ego_spawn_times += 1
                time.sleep(0.1)

        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)

        self.collision_hist = []


        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
        self.camera_sensor.listen(lambda data: get_camera_img(data))

        def get_camera_img(data):
            data.convert(carla.ColorConverter.CityScapesPalette)
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array

        self.time_step = 0
        self.reset_step += 1
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)
        self.routeplanner = RoutePlanner(self.ego, self.max_waypoints)
        self.waypoints = self.routeplanner.run_step()

        for i in range(len(self.waypoints) - 1):
            start = carla.Location(self.waypoints[i][0], self.waypoints[i][1], self.waypoints[i][3])
            stop = carla.Location(self.waypoints[i + 1][0], self.waypoints[i + 1][1], self.waypoints[i + 1][3])
            self.world.debug.draw_line(start, stop, thickness=1.5,
                                  color=carla.Color(0, 1, 0), life_time=0.3)

        return self._get_obs()

    def step(self, action):
        acc = self.actions[action][0]
        steer = self.actions[action][1]

        throttle = np.clip(acc / 3, 0, 1)

        act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer))
        self.ego.apply_control(act)
        last_position = self.ego.get_location()

        self.world.tick()

        spectator = self.world.get_spectator()
        transform = carla.Transform(self.ego.get_transform().transform(carla.Location(x=-5, z=2)),
                                    self.ego.get_transform().rotation)
        spectator.set_transform(transform)

        current_position = self.ego.get_location()

        self.waypoints = self.routeplanner.run_step()

        for i in range(len(self.waypoints) - 1):
            start = carla.Location(self.waypoints[i][0], self.waypoints[i][1], self.waypoints[i][3])
            stop = carla.Location(self.waypoints[i + 1][0], self.waypoints[i + 1][1], self.waypoints[i + 1][3])
            self.world.debug.draw_line(start, stop, thickness=1.5,
                                       color=carla.Color(0, 1, 0), life_time=0.3)

        info = {
            'waypoints': self.waypoints,
            'distance': current_position.distance(last_position)
        }

        self.time_step += 1
        self.total_step += 1

        return self._get_obs(), self._get_reward(), self._terminal(), copy.deepcopy(info)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode):
        pass

    def _create_vehicle_blueprint(self, actor_filter, color=None, number_of_wheels=[4]):
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if
                                                     int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _set_synchronous_mode(self, synchronous=True):
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)


    def _try_spawn_ego_vehicle_at(self, transform):
        vehicle = None
        overlap = False
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break

        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            self.ego = vehicle
            return True

        return False

    def _get_actor_polygons(self, filt):
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _get_obs(self):
        camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
        camera_surface = rgb_to_display_surface(camera, self.display_size)
        self.display.blit(camera_surface, (0, 0))

        pygame.display.flip()

        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        delta_yaw = np.arcsin(np.cross(w, np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x ** 2 + v.y ** 2)
        state = np.array([lateral_dis, - delta_yaw, speed])

        obs = {
            'camera': camera.astype(np.uint8),
            'state': state,
        }

        return obs

    def _get_reward(self):
        # reward for speed
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x ** 2 + v.y ** 2)

        # reward for collision
        r_collision = 0
        if len(self.collision_hist) > 0:
            r_collision = -1

        # reward for out of lane
        ego_x, ego_y = get_pos(self.ego)
        distance, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = 0
        if abs(distance) > self.out_lane_distance:
            r_out = -1

        r_distance = -abs(distance)

        return speed + 5 * r_out + 3 * r_distance + 10 * r_collision

    def _terminal(self):
        ego_x, ego_y = get_pos(self.ego)

        if len(self.collision_hist) > 0:
            return True

        if self.time_step > self.max_time_episode:
            return True

        dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
        if abs(dis) > self.out_lane_distance:
            return True

        return False

    def _clear_all_actors(self, actor_filters):
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    actor.destroy()
