def _get_reward(self):
    """Calculate the step reward."""
    # reward for speed tracking
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x ** 2 + v.y ** 2)
    r_speed = -abs(speed - self.desired_speed)

    # reward for collision
    r_collision = 0
    if len(self.collision_hist) > 0:
        r_collision = -1

    # reward for steering:
    r_steer = -self.ego.get_control().steer ** 2

    # reward for out of lane
    ego_x, ego_y = get_pos(self.ego)
    dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
    r_out = 0
    if abs(dis) > self.out_lane_thres:
        r_out = -1

    # longitudinal speed
    lspeed = np.array([v.x, v.y])
    lspeed_lon = np.dot(lspeed, w)
    accel = self.ego.get_control().throttle
    if lspeed_lon < 0.3:
        lspeed_lon = -1
        if accel > 0:
            lspeed_lon = 1

    # cost for too fast
    r_fast = 0
    if lspeed_lon > self.desired_speed:
        r_fast = -1
        # r_fast = -1

    # cost for lateral acceleration
    r_lat = - abs(self.ego.get_control().steer) * lspeed_lon ** 2

    # default
    # return 200 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat - 0.1

    # # model w miare
    # return 50 * r_collision + 1 * lspeed_lon + 10 * r_fast + 20 * r_out + r_steer * 5 + 0.2 * r_lat - 0.1

    r_dis = 0
    if abs(dis) > 0.2:
        r_dis = -abs(dis)  # [-0.7, -0.2]

    ego_trans = self.ego.get_transform()
    ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
    delta_yaw = np.arcsin(np.cross(w, np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))

    r_angle = 0
    if delta_yaw > 0.25 or delta_yaw < -0.25:
        r_angle = - abs(delta_yaw)  # [-0,5, -0.2]

    steer = self.ego.get_control().steer
    r_steer = 0
    if (delta_yaw > 0.3 and steer >= 0) or (delta_yaw < -0.3 and steer <= 0):
        r_steer = -1

    # # model w miare2
    # return 50 * r_collision + lspeed_lon + 5 * r_fast + 10 * r_out - r_dis * 7 + 0.2 * r_lat + r_angle
    # return 50 * r_collision + lspeed_lon + 5 * r_fast + 2 * r_out + r_dis + r_angle + r_steer - 0.1
    return 50 * r_collision + lspeed_lon + 5 * r_fast + 5 * r_out + 2 * r_angle + 2 * r_steer - 0.1