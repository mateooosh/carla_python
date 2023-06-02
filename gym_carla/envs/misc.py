import math
import numpy as np
import pygame
import skimage


def get_pos(vehicle):
    trans = vehicle.get_transform()
    x = trans.location.x
    y = trans.location.y
    return x, y


def get_lane_dis(waypoints, x, y):
    dis_min = 1000
    waypt = waypoints[0]
    for pt in waypoints:
        d = np.sqrt((x - pt[0]) ** 2 + (y - pt[1]) ** 2)
        if d < dis_min:
            dis_min = d
            waypt = pt
    vec = np.array([x - waypt[0], y - waypt[1]])
    lv = np.linalg.norm(np.array(vec))
    w = np.array([np.cos(waypt[2] / 180 * np.pi), np.sin(waypt[2] / 180 * np.pi)])
    cross = np.cross(w, vec / lv)
    dis = - lv * cross
    return dis, w


def get_preview_lane_dis(waypoints, x, y, idx=2):
    waypt = waypoints[idx]
    vec = np.array([x - waypt[0], y - waypt[1]])
    lv = np.linalg.norm(np.array(vec))
    w = np.array([np.cos(waypt[2] / 180 * np.pi), np.sin(waypt[2] / 180 * np.pi)])
    cross = np.cross(w, vec / lv)
    dis = - lv * cross
    return dis, w


def distance_vehicle(waypoint, vehicle_transform):
    loc = vehicle_transform.location
    dx = waypoint.transform.location.x - loc.x
    dy = waypoint.transform.location.y - loc.y

    return math.sqrt(dx * dx + dy * dy)


def rgb_to_display_surface(rgb, display_size):
    surface = pygame.Surface((display_size, display_size)).convert()
    display = skimage.transform.resize(rgb, (display_size, display_size))
    display = np.flip(display, axis=1)
    display = np.rot90(display, 1)
    pygame.surfarray.blit_array(surface, display)
    return surface
