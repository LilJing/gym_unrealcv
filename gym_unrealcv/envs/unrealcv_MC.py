import math
import os
import time
import gym
import numpy as np
from gym import spaces
from gym_unrealcv.envs.tracking import reward
from gym_unrealcv.envs.tracking.visualization import show_info
from gym_unrealcv.envs.utils import env_unreal
from gym_unrealcv.envs.tracking.interaction import Tracking
import gym_unrealcv
import cv2
import random
import matplotlib.pyplot as plt
import time
# from pylab import *

''' 
It is an env for active object tracking.

State : raw color image and depth
Action:  (linear velocity ,angle velocity) 
Done : the relative distance or angle to target is larger than the threshold.
Task: Learn to follow the target object(moving person) in the scene
'''
# np.random.seed(123)

class UnrealCvMC(gym.Env):
    def __init__(self,
                 setting_file,
                 reset_type,
                 action_type='discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 reward_type='distance',  # distance
                 docker=False,
                 resolution= (320, 240),
                 nav='Random',  # Random, Goal, Internal
                 ):

        self.docker = docker
        self.reset_type = reset_type
        self.roll = 0
        self.nav = nav
        setting = self.load_env_setting(setting_file)
        self.env_name = setting['env_name']
        self.cam_id = setting['cam_id']
        self.target_list = setting['targets']
        self.discrete_actions = setting['discrete_actions']
        self.discrete_actions_player = setting['discrete_actions_player']
        self.continous_actions = setting['continous_actions']
        self.continous_actions_player = setting['continous_actions_player']
        self.max_steps = setting['max_steps']
        self.max_distance = setting['max_distance']
        self.min_distance = setting['min_distance']
        self.max_direction = setting['max_direction']
        self.max_obstacles = setting['max_obstacles']
        self.height = setting['height']
        self.pitch = setting['pitch']
        self.objects_env = setting['objects_list']
        if setting.get('reset_area'):
            self.reset_area = setting['reset_area']
        if setting.get('cam_area'):
            self.cam_area = setting['cam_area']

        self.test = False if 'MCMTRoom' in self.env_name else True

        if setting.get('goal_list'):
            self.goal_list = setting['goal_list']
        if setting.get('camera_loc'):
            self.camera_loc = setting['camera_loc']

        self.target_move = setting['target_move']
        self.camera_move = setting['camera_move']
        self.scale_rate = setting['scale_rate']
        self.pose_rate = setting['pose_rate']

        self.background_list = setting['backgrounds']
        self.light_list = setting['lights']
        self.target_num = setting['target_num']
        self.exp_distance = setting['exp_distance']
        texture_dir = setting['imgs_dir']
        gym_path = os.path.dirname(gym_unrealcv.__file__)
        texture_dir = os.path.join(gym_path, 'envs', 'UnrealEnv', texture_dir)
        self.textures_list = os.listdir(texture_dir)
        self.safe_start = setting['safe_start']
        self.start_area = self.get_start_area(self.safe_start[0], 100)

        self.num_target = len(self.target_list)
        self.resolution = resolution
        self.num_cam = len(self.cam_id)

        self.cam_height = [setting['height'] for i in range(self.num_cam)]

        for i in range(len(self.textures_list)):
            if self.docker:
                self.textures_list[i] = os.path.join('/unreal', setting['imgs_dir'], self.textures_list[i])
            else:
                self.textures_list[i] = os.path.join(texture_dir, self.textures_list[i])

        # start unreal env
        self.unreal = env_unreal.RunUnreal(ENV_BIN=setting['env_bin'])
        env_ip, env_port = self.unreal.start(docker, resolution)

        # connect UnrealCV
        self.unrealcv = Tracking(cam_id=self.cam_id[0], port=env_port, ip=env_ip,
                                 env=self.unreal.path2env, resolution=resolution)
        self.unrealcv.color_dict = self.unrealcv.build_color_dic(self.target_list)
        # define action
        self.action_type = action_type
        assert self.action_type == 'Discrete' or self.action_type == 'Continuous'
        if self.action_type == 'Discrete':
            self.action_space = [spaces.Discrete(len(self.discrete_actions)) for i in range(self.num_cam)]
            player_action_space = [spaces.Discrete(len(self.discrete_actions_player)) for i in range(1)]
        elif self.action_type == 'Continuous':
            self.action_space = [spaces.Box(low=np.array(self.continous_actions['low']),
                                            high=np.array(self.continous_actions['high'])) for i in range(self.num_cam)]
            player_action_space = spaces.Discrete(len(self.continous_actions_player))

        self.count_steps = 0

        # define observation space,
        # color, depth, rgbd, ...
        self.observation_type = observation_type
        assert self.observation_type in ['Color', 'Depth', 'Rgbd', 'Gray']
        self.observation_space = [self.unrealcv.define_observation(self.cam_id[i], self.observation_type, 'fast')
                                  for i in range(self.num_cam)]

        self.unrealcv.pitch = self.pitch
        # define reward type
        # distance, bbox, bbox_distance
        self.reward_type = reward_type
        self.reward_function = reward.Reward(setting)

        if not self.test:
            if self.reset_type >= 0:
                self.unrealcv.init_objects(self.objects_env)

        self.rendering = False

        self.count_close = 0

        self.person_id = 0
        self.unrealcv.set_location(0, [self.safe_start[0][0], self.safe_start[0][1], self.safe_start[0][2]+600])
        self.unrealcv.set_rotation(0, [0, -180, -90])
        self.unrealcv.set_obj_location("TargetBP", [-3000, -3000, 220])
        if 'Random' in self.nav:
            self.random_agents = [RandomAgent(self.continous_actions_player) for i in range(self.num_target)]
        if 'Goal' in self.nav:
            if not self.test:
                self.random_agents = [GoalNavAgent(self.continous_actions_player, self.reset_area) for i in range(self.num_target)]
            else:
                self.random_agents = [GoalNavAgentTest(self.continous_actions_player, goal_list=self.goal_list)
                                      for i in range(self.num_target)]
        if 'Internal' in self.nav:
            self.unrealcv.set_random(self.target_list[0])
            self.unrealcv.set_maxdis2goal(target=self.target_list[0], dis=500)
        if 'Interval' in self.nav:
            self.unrealcv.set_interval(30)

        self.cam_angles = np.array([0 for i in range(self.num_cam)])

        self.scale = 1000.0
        self.gate_ids = np.ones(self.num_cam)
        self.record_eps = 0
        self.max_mask_area = np.ones(self.num_cam) * 0.001 * self.resolution[0] * self.resolution[1]  # 76.8

    def step(self, actions):
        info = dict(
            Done=False,
            Reward=[0 for i in range(self.num_cam)],
            Target_Pose=[],
            Cam_Pose=[],
            Steps=self.count_steps,
        )

        self.current_cam_pos = self.cam_pose.copy()
        self.current_target_pos = self.target_pos.copy()
        self.current_states = self.states

        self.gate_rewards = []
        self.gt_masks = []
        self.mask_bboxs = []
        self.target_masks = []
        self.zoom_masks = []
        areas = []
        bboxs = []

        for i in range(self.num_cam):
            if self.test:
                object_mask = self.unrealcv.read_image(self.cam_id[i], 'object_mask', 'fast')

                zoom_mask = object_mask[int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2): (
                        self.resolution[1] - int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2)),
                            int(self.resolution[0] * (1 - self.zoom_scales[i]) / 2): (self.resolution[0] - int(
                                self.resolution[0] * (1 - self.zoom_scales[i]) / 2)), :]
                zoom_mask = cv2.resize(zoom_mask, self.resolution)
                bbox, target_mask = self.unrealcv.get_bboxes(zoom_mask, self.target_list)
                bboxs.append(bbox)

                self.mask_bboxs.append(bbox)
                w = self.resolution[0] * (bbox[0][1][0] - bbox[0][0][0])
                h = self.resolution[1] * (bbox[0][1][1] - bbox[0][0][1])
                area = w * h
                areas.append(area)
                self.target_masks.append(object_mask)

                sparse_reward = 0 if area <= self.max_mask_area[i] else 1
            else:
                sparse_reward = 0
            self.gate_rewards.append(sparse_reward)

        actions = np.squeeze(actions)
        actions2cam = []
        for i in range(self.num_cam):
            if self.action_type == 'Discrete':
                actions2cam.append(self.discrete_actions[actions[i]])  # delta_yaw, delta_pitch
            else:
                actions2cam.append(actions[i])  # delta_yaw, delta_pitch

        actions2target = []
        for i in range(len(self.target_list)):
            if 'Random' in self.nav:
                if self.action_type == 'Discrete':
                    # actions2target.append(self.discrete_actions_player[self.random_agents[i].act()])
                    actions2target.append(self.random_agents[i].act())
                else:
                    actions2target.append(self.random_agents[i].act())
            if 'Goal' in self.nav:
                    actions2target.append(self.random_agents[i].act(self.target_pos[i]))

        for i, target in enumerate(self.target_list):
            self.unrealcv.set_move(target, actions2target[i][1], actions2target[i][0])

        states = []
        zoom_masks = []
        self.mask_gt_ids = []
        areas = []
        zoom_bboxs = []
        self.gt_actions = []

        self.gate_gt_ids = np.ones(len(self.cam_id), int)

        for i, cam in enumerate(self.cam_id):
            cam_rot = self.unrealcv.get_rotation(cam, 'hard')
            cam_loc = self.cam_pose[i][:3]

            if len(actions2cam[i]) == 2:
                cam_rot[1] += actions2cam[i][0] * self.zoom_scales[i]
                cam_rot[2] += actions2cam[i][1] * self.zoom_scales[i]

                cam_rot[2] = cam_rot[2] if cam_rot[2] < 80.0 else 80.0
                cam_rot[2] = cam_rot[2] if cam_rot[2] > - 80.0 else -80.0

                self.unrealcv.set_rotation(cam, cam_rot)
                self.cam_pose[i][-3:] = cam_rot
                raw_state = self.unrealcv.get_observation(cam, self.observation_type, 'fast')

                zoom_state = raw_state[int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2): (
                            self.resolution[1] - int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2)),
                             int(self.resolution[0] * (1 - self.zoom_scales[i]) / 2): (self.resolution[0] - int(self.resolution[0] * (1 - self.zoom_scales[i]) / 2)),:]
                state = cv2.resize(zoom_state, self.resolution)

            else:  # zoom action

                raw_state = self.unrealcv.get_observation(cam, self.observation_type, 'fast')
                if actions2cam[i][0] == 1:  # zoom in
                    # self.limit_scales[i] = 1 if self.zoom_scales[i] <= 0.4 else 0
                    self.zoom_scales[i] = self.zoom_in_scale[i] * self.zoom_scales[i] if self.zoom_in_scale[i] * self.zoom_scales[i] >= self.min_scale else self.zoom_scales[i]
                    zoom_state = raw_state[int(self.resolution[1]*(1-self.zoom_scales[i])/2): (self.resolution[1] -
                            int(self.resolution[1]*(1-self.zoom_scales[i])/2)), int(self.resolution[0]*(1-self.zoom_scales[i])/2): (self.resolution[0] -
                                                                                    int(self.resolution[0]*(1-self.zoom_scales[i])/2)), :]

                    state = cv2.resize(zoom_state, self.resolution)
                elif actions2cam[i][0] == -1: # zoom out
                    # self.limit_scales[i] = 1 if self.zoom_scales[i] >= 1 else 0
                    self.zoom_scales[i] = self.zoom_out_scale[i] * self.zoom_scales[i] if self.zoom_out_scale[i] * self.zoom_scales[i] <= 1 else self.zoom_scales[i]
                    zoom_state = raw_state[int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2): (self.resolution[1] - int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2)),
                                 int(self.resolution[0] * (1 - self.zoom_scales[i]) / 2): (self.resolution[0] -int(self.resolution[0] * (1 - self.zoom_scales[i]) / 2)),:]
                    state = cv2.resize(zoom_state, self.resolution)
                else:
                    zoom_state = raw_state[int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2): ( self.resolution[1] - int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2)),
                                 int(self.resolution[0] * (1 - self.zoom_scales[i]) / 2): (self.resolution[0] - int(self.resolution[0] * (1 - self.zoom_scales[i]) / 2)),:]
                    state = cv2.resize(zoom_state, self.resolution)

            if  not self.test:
                object_mask = self.unrealcv.read_image(self.cam_id[i], 'object_mask', 'fast')
                zoom_mask = object_mask[int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2): (
                        self.resolution[1] - int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2)),
                            int(self.resolution[0] * (1 - self.zoom_scales[i]) / 2): (self.resolution[0] - int(
                                self.resolution[0] * (1 - self.zoom_scales[i]) / 2)), :]
                zoom_mask = cv2.resize(zoom_mask, self.resolution)
                bbox, _ = self.unrealcv.get_bboxes(zoom_mask, self.target_list)
                zoom_bboxs.append(bbox[0])
                zoom_masks.append(zoom_mask)
                w = self.resolution[0] * (bbox[0][1][0] - bbox[0][0][0])
                h = self.resolution[1] * (bbox[0][1][1] - bbox[0][0][1])
                area = w * h
                areas.append(area)

                if area <= self.max_mask_area[i]:
                    self.mask_gt_ids.append(0)
                else:
                    self.mask_gt_ids.append(1)

            states.append(state)
            self.unrealcv.set_rotation(cam, cam_rot)

        self.states = states


        imgs = np.hstack([self.states[0], self.states[1], self.states[2], self.states[3]])
        cv2.imshow("Pose-assisted-multi-camera-collaboration", imgs)
        cv2.waitKey(10)

        self.count_steps += 1

        obj_masks = []
        cam_ws = []
        cam_hs = []

        directions = []
        rewards = []
        gt_locations = []

        zoom_rewards = []
        pose_rewards = []
        expected_scales = []
        hori_rewards = []
        verti_rewards = []
        distances = []
        verti_directions = []
        cal_target_observed = np.zeros(len(self.cam_id))
        self.target_observed = np.zeros(len(self.cam_id))

        # get bbox and reward
        for i in range(len(self.cam_id)):
            # get bbox
            if self.reset_type >= 6:
                object_mask = self.unrealcv.read_image(self.cam_id[i], 'object_mask', 'fast')
                bbox, _ = self.unrealcv.get_bboxes(object_mask, self.target_list)
                # get bbox size
                bbox_shape = np.array(bbox[0][1]) - np.array(bbox[0][0])
                if bbox_shape[0] * bbox_shape[1] < 0.01:
                    self.target_observed[i] = 1
                obj_masks.append(object_mask)

            # get relative location and reward
            direction = self.get_direction(self.cam_pose[i], self.target_pos[0])
            hori_reward = 1 - 2*abs(direction) / 45.0

            verti_direction = self.get_verti_direction(self.cam_pose[i], self.target_pos[0])
            verti_reward = 1 - 2*abs(verti_direction) / 30.0

            hori_rewards.append(hori_reward)
            verti_rewards.append(verti_reward)

            d = self.unrealcv.get_distance(self.cam_pose[i], self.target_pos[0], 3)

            gt_locations.append([direction, verti_direction, d])

            expected_scale = self.scale_function(d)
            expected_scale = expected_scale if expected_scale >= self.min_scale else self.min_scale
            expected_scale = expected_scale if expected_scale <= 1 else 1
            zoom_error = abs(self.zoom_scales[i] - expected_scale) / (1 - self.min_scale)
            zoom_reward = 1 - zoom_error
            expected_scales.append(expected_scale)
            pose_reward = max(2 - abs(direction) / 45.0 - abs(verti_direction) / 30.0, -2) / 2

            if abs(direction) <= 45.0 * self.zoom_scales[i] and abs(verti_direction) <= 30.0 * self.zoom_scales[i]:
                cal_target_observed[i] = 1
                if not self.test:
                    sparse_reward = pose_reward + zoom_reward if self.mask_gt_ids[i] != 0 else 0
            else:
                sparse_reward = -1

            reward = sparse_reward if not self.test else pose_reward
            pose_rewards.append(pose_reward)
            rewards.append(reward)
            zoom_rewards.append(zoom_reward)
            # zoom_exist_rewards.append(zoom_exist_reward)
            hori_rewards.append(hori_reward)
            verti_rewards.append(verti_reward)
            directions.append(direction)
            verti_directions.append(verti_direction)
            distances.append(d)

        info['zoom masks'] = zoom_masks
        info['zoom bboxs'] = zoom_bboxs
        info['states'] = self.states
        info['mask gt ids'] = self.mask_gt_ids

        info['Distance'] = distances
        info['zoom reward'] = zoom_rewards
        info['hori reward'] = hori_rewards
        info['verti reward'] = verti_rewards

        info['areas'] = areas
        info['w'] = cam_ws
        info['h'] = cam_hs
        info['expected scale'] = expected_scales
        info['zoom scale'] = self.zoom_scales

        info['hori reward'] = hori_rewards
        info['verti reward'] = verti_rewards

        info['target poses'] = self.current_target_pos
        info['camera poses'] = self.current_cam_pos
        info['tracking reward'] = pose_rewards
        info['zoom reward'] = zoom_rewards
        info['bboxs'] = bboxs

        info['zoom scale'] = self.zoom_scales
        # info['gate_gt_ids'] = self.gate_gt_ids
        info['gate rewards'] = self.gate_rewards

        if self.count_steps > self.max_steps:
            info['Done'] = True

        if not self.test:
            if sum(cal_target_observed) < 2:
                self.count_close += 1
            else:
                self.count_close = 0

            if self.count_close > 10:
                info['Done'] = True

        if info['Done']:
            self.record_eps += 1

        info['Reward'] = rewards
        info['Pose Reward'] = pose_rewards
        info['gt actions'] = self.gt_actions
        info['gt locations'] = gt_locations
        info['Success rate'] = sum(cal_target_observed) / self.num_cam
        info['Success ids'] = cal_target_observed

        for i, target in enumerate(self.target_list):
            self.target_pos[i] = self.unrealcv.get_obj_pose(target)

        return self.states, info['Reward'], info['Done'], info

    def reset(self, ):

        self.zoom_scales = np.ones(self.num_cam)

        self.zoom_in_scale = np.ones(self.num_cam) * 0.9
        self.zoom_out_scale = np.ones(self.num_cam) * 1.1
        self.stand_d = 500
        self.min_scale = 0.3
        self.limit_scales = np.zeros(self.num_cam)

        self.C_reward = 0
        self.count_close = 0
        # stop

        self.target_pos = np.array([np.random.randint(self.start_area[0], self.start_area[1]),
                                    np.random.randint(self.start_area[2], self.start_area[3]),
                                    self.safe_start[0][-1]])

        for i, target in enumerate(self.target_list):
            self.unrealcv.set_move(target, self.safe_start[i][0], self.safe_start[i][1])

        if self.reset_type >= 1:
            if self.env_name == 'MCMTRoom':
                map_id = [2, 3, 6, 7, 9]
                spline = False
                object_app = np.random.choice(map_id)
            else:
                map_id = [6, 7, 9, 3]
                spline = True
                object_app = map_id[self.person_id % len(map_id)]
                self.person_id += 1

            self.unrealcv.set_appearance(self.target_list[0], object_app, spline)
            self.obstacles_num = self.max_obstacles
            self.obstacle_scales = [[1, 1.2] if np.random.binomial(1, 0.5) == 0 else [1.5, 2] for k in range(self.obstacles_num)]
            self.unrealcv.clean_obstacles()
            if not self.test:
                self.unrealcv.random_obstacles(self.objects_env, self.textures_list,
                                           self.obstacles_num, self.reset_area, self.start_area, self.obstacle_scales)
        # light
        if self.reset_type >= 2:
            for lit in self.light_list:
                if 'sky' in lit:
                    self.unrealcv.set_skylight(lit, [1, 1, 1], np.random.uniform(0.5, 2))
                else:
                    lit_direction = np.random.uniform(-1, 1, 3)
                    if 'directional' in lit:
                        lit_direction[0] = lit_direction[0] * 60
                        lit_direction[1] = lit_direction[1] * 80
                        lit_direction[2] = lit_direction[2] * 60
                    else:
                        lit_direction *= 180
                    self.unrealcv.set_light(lit, lit_direction, np.random.uniform(1, 4), np.random.uniform(0.1,1,3))

        # target appearance
        if self.reset_type >= 3:
            self.unrealcv.random_player_texture(self.target_list[0], self.textures_list, 3)
            self.unrealcv.random_texture(self.background_list, self.textures_list, 5)

        # texture
        if self.reset_type >= 4 and not self.test:
            self.obstacle_scales = [[1, 1.5] if np.random.binomial(1, 0.5) == 0 else [2, 2.5] for k in  # 2.5-3.5 before
                                    range(self.obstacles_num)]
            self.obstacles_num = self.max_obstacles
            self.unrealcv.clean_obstacles()
            if not self.test:
                self.unrealcv.random_obstacles(self.objects_env, self.textures_list,
                                           self.obstacles_num, self.reset_area, self.start_area, self.obstacle_scales)

        self.target_pos = []
        for i, target in enumerate(self.target_list):
            self.unrealcv.set_obj_location(target, self.safe_start[i])
            self.target_pos.append(self.unrealcv.get_obj_pose(target))


        states = []
        self.cam_pose = []
        self.fixed_cam = True if self.test  else False
        self.gt_actions = []
        self.gate_rewards = []
        self.mask_gt_ids = []

        if self.reset_type >= 5:
            for i, cam in enumerate(self.cam_id):
                if self.fixed_cam:
                    cam_loc = self.camera_loc[i]
                else:
                    cam_loc = [np.random.randint(self.cam_area[i][0], self.cam_area[i][1]),
                               np.random.randint(self.cam_area[i][2], self.cam_area[i][3]),
                               np.random.randint(self.cam_area[i][4], self.cam_area[i][5])]

                self.unrealcv.set_location(cam, cam_loc)
                cam_rot = self.unrealcv.get_rotation(cam, 'hard')
                angle_h = self.get_direction(cam_loc+cam_rot, self.target_pos[0])
                angle_v = self.get_verti_direction(cam_loc+cam_rot, self.target_pos[0])
                cam_rot[1] += angle_h
                cam_rot[2] -= angle_v


                self.unrealcv.set_rotation(cam, cam_rot)
                self.cam_pose.append(cam_loc + cam_rot)

                raw_state = self.unrealcv.get_observation(cam, self.observation_type, 'fast')

                object_mask = self.unrealcv.read_image(self.cam_id[i], 'object_mask', 'fast')
                zoom_mask = object_mask[int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2): (
                        self.resolution[1] - int(self.resolution[1] * (1 - self.zoom_scales[i]) / 2)),
                            int(self.resolution[0] * (1 - self.zoom_scales[i]) / 2): (self.resolution[0] - int(
                                self.resolution[0] * (1 - self.zoom_scales[i]) / 2)), :]
                zoom_mask = cv2.resize(zoom_mask, self.resolution)
                bbox, _ = self.unrealcv.get_bboxes(zoom_mask, self.target_list)
                w = self.resolution[0] * (bbox[0][1][0] - bbox[0][0][0])
                h = self.resolution[1] * (bbox[0][1][1] - bbox[0][0][1])
                area = w * h

                if area <= self.max_mask_area[i] * self.zoom_scales[i]:
                    self.mask_gt_ids.append(0)
                else:
                    self.mask_gt_ids.append(1)

                states.append(raw_state)
                sparse_reward = 0 if area <= self.max_mask_area[i] else 1
                self.gate_rewards.append(sparse_reward)

        else:

            for i, cam in enumerate(self.cam_id):
                if self.test:
                    cam_loc = self.camera_loc[i]
                    self.unrealcv.set_location(cam, cam_loc)
                    cam_rot = self.unrealcv.get_rotation(cam, 'hard')
                else:
                    cam_loc = [np.random.randint(self.cam_area[i][0], self.cam_area[i][1]),
                               np.random.randint(self.cam_area[i][2], self.cam_area[i][3]),
                               np.random.randint(self.cam_area[i][4], self.cam_area[i][5])]
                    cam_rot = self.unrealcv.get_rotation(cam, 'hard')

                angle_h = self.get_direction(cam_loc + cam_rot, self.target_pos[0])
                angle_v = self.get_verti_direction(cam_loc + cam_rot, self.target_pos[0])
                cam_rot[1] += angle_h
                cam_rot[2] -= angle_v

                self.unrealcv.set_location(cam, cam_loc)
                self.unrealcv.set_rotation(cam, cam_rot)
                self.cam_pose.append(cam_loc+cam_rot)

                raw_state = self.unrealcv.get_observation(cam, self.observation_type, 'fast')
                states.append(raw_state)

                object_mask = self.unrealcv.read_image(self.cam_id[i], 'object_mask', 'fast')
                bbox, _ = self.unrealcv.get_bboxes(object_mask, self.target_list)
                w = self.resolution[0] * (bbox[0][1][0] - bbox[0][0][0])
                h = self.resolution[1] * (bbox[0][1][1] - bbox[0][0][1])
                area = w * h

                if area <= self.max_mask_area[i] * self.zoom_scales[i]:
                    self.mask_gt_ids.append(0)
                else:
                    self.mask_gt_ids.append(1)

                sparse_reward = 0 if area <= self.max_mask_area[i] else 1
                self.gate_rewards.append(sparse_reward)


        self.count_steps = 0
        if 'Random' in self.nav or 'Goal' in self.nav:
            for i in range(len(self.random_agents)):
                self.random_agents[i].reset()
        if 'Internal' in self.nav:
            self.unrealcv.set_speed(self.target_list[0], np.random.randint(30, 200))
        self.states = states
        self.current_states = self.states
        self.current_cam_pos = self.cam_pose.copy()
        self.current_target_pos = self.target_pos.copy()

        return self.states

    def close(self):
        self.unreal.close()

    def render(self, mode='rgb_array', close=False):
        if close==True:
            self.unreal.close()
        for i in range(self.num_cam):
            cv2.imshow('tracker_{}'.format(str(i)), self.states[i])
        cv2.waitKey(10)
        return self.unrealcv.img_color

    def to_render(self, choose_ids):
        map_render(self.cam_pose, self.target_pos[0],  choose_ids, self.target_move, self.camera_move, self.scale_rate, self.pose_rate)

    def seed(self, seed=None):
        self.person_id = seed

    def get_action_size(self):
        return len(self.action)

    def get_direction(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        angle_now = np.arctan2(y_delt, x_delt)/np.pi*180-current_pose[4]
        if angle_now > 180:
            angle_now -= 360
        if angle_now < -180:
            angle_now += 360
        return angle_now

    def get_angle(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        angle_now = np.arctan2(y_delt, x_delt) / np.pi * 180
        return angle_now

    def get_2d_distance(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        d = np.sqrt(y_delt * y_delt + x_delt * x_delt)
        return d

    def get_distance(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        z_delt = target_pose[2] - current_pose[2]
        d = np.sqrt(y_delt * y_delt + x_delt * x_delt + z_delt * z_delt)
        return d

    def get_verti_direction(self, current_pose, target_pose):
        # person_mid_height = target_pose[2] / 2
        person_height = target_pose[2]
        plane_distance = self.get_2d_distance(current_pose, target_pose)
        height = current_pose[2] - person_height
        angle = np.arctan2(height, plane_distance) / np.pi * 180
        angle_now = angle + current_pose[-1]
        return angle_now


    def load_env_setting(self, filename):
        gym_path = os.path.dirname(gym_unrealcv.__file__)
        setting_path = os.path.join(gym_path, 'envs', 'setting', filename)

        f = open(setting_path)
        f_type = os.path.splitext(filename)[1]
        if f_type == '.json':
            import json
            setting = json.load(f)
        else:
            print('unknown type')

        return setting

    def get_settingpath(self, filename):
        import gym_unrealcv
        gympath = os.path.dirname(gym_unrealcv.__file__)
        return os.path.join(gympath, 'envs/setting', filename)

    def get_start_area(self, safe_start, safe_range):
        start_area = [safe_start[0] - safe_range, safe_start[0] + safe_range,
                      safe_start[1] - safe_range, safe_start[1] + safe_range]
        return start_area

    def scale_function(self, d):
        scale = 449 / d
        return scale

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.step_counter = 0
        self.keep_steps = 0
        self.action_space = action_space
        self.velocity_high = action_space['high'][0]
        self.velocity_low = action_space['low'][0]
        self.angle_high = action_space['high'][1]
        self.angle_low = action_space['low'][1]

    def act(self):

        velocity = np.random.randint(self.velocity_low, self.velocity_high)
        angle = np.random.randint(self.angle_low, self.angle_high)

        return (velocity, angle)

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0

class GoalNavAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space, goal_area, nav='New'):
        self.step_counter = 0
        self.keep_steps = 0
        self.velocity_high = action_space['high'][0]
        self.velocity_low = action_space['low'][0]
        self.angle_high = action_space['high'][1]
        self.angle_low = action_space['low'][1]
        self.goal_area = goal_area
        self.goal = self.generate_goal(self.goal_area)
        if 'Base' in nav:
            self.discrete = True
        else:
            self.discrete = False
        if 'Old' in nav:
            self.max_len = 30
        else:
            self.max_len = 100

    def act(self, pose):
        self.step_counter += 1
        if self.pose_last == None:
            self.pose_last = pose
            d_moved = 100
        else:
            d_moved = np.linalg.norm(np.array(self.pose_last) - np.array(pose))
            self.pose_last = pose
        if self.check_reach(self.goal, pose) or d_moved < 10 or self.step_counter > self.max_len:
            self.goal = self.generate_goal(self.goal_area)
            if self.discrete:
                self.velocity = (self.velocity_high + self.velocity_low)/2
            else:
                self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
            # self.velocity = 70
            self.step_counter = 0

        delt_yaw = self.get_direction(pose, self.goal)
        if self.discrete:
            if abs(delt_yaw) > self.angle_high:
                velocity = 0
            else:
                velocity = self.velocity
            if delt_yaw > 3:
                self.angle = self.angle_high / 2
            elif delt_yaw <-3:
                self.angle = self.angle_low / 2
        else:
            self.angle = np.clip(delt_yaw, self.angle_low, self.angle_high)
            velocity = self.velocity * (1 + 0.2*np.random.random())
        return (velocity, self.angle)

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0
        self.goal = self.generate_goal(self.goal_area)
        self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
        self.pose_last = None

    def generate_goal(self, goal_area):
        x = np.random.randint(goal_area[0], goal_area[1])
        y = np.random.randint(goal_area[2], goal_area[3])
        goal = np.array([x, y])
        return goal

    def check_reach(self, goal, now):
        error = np.array(now[:2]) - np.array(goal[:2])
        distance = np.linalg.norm(error)
        return distance < 50

    def get_direction(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        angle_now = np.arctan2(y_delt, x_delt) / np.pi * 180 - current_pose[4]
        if angle_now > 180:
            angle_now -= 360
        if angle_now < -180:
            angle_now += 360
        return angle_now


def map_render(camera_pos, target_pos, choose_ids, target_move, camera_move, scale_rate, pose_rate):

    length = 600
    coordinate_delta = np.mean(np.array(camera_pos)[:, :2], axis=0)
    img = np.zeros((length + 1, length + 1, 3)) + 255
    num_cam = len(camera_pos)

    camera_position_origin = np.array([camera_pos[i][:2] for i in range(num_cam)])
    target_position_origin = np.array(target_pos[:2])

    lengths = []
    for i in range(num_cam):
        length = np.sqrt(sum(np.array(camera_position_origin[i] - coordinate_delta)) ** 2)
        lengths.append(length)
    pose_scale = max(lengths)

    pose_scale = pose_scale * pose_rate
    target_position = length * (np.array([scale_rate + (target_position_origin[0] - coordinate_delta[0]) / pose_scale, scale_rate +
                                          (target_position_origin[1] - coordinate_delta[0]) / pose_scale])) / 2 + np.array(target_move)

    camera_position = []
    for i in range(num_cam):
        position_transfer = length * (np.array([scale_rate + (camera_position_origin[i][0] - coordinate_delta[0]) / pose_scale,
                                                scale_rate + (camera_position_origin[i][1] - coordinate_delta[1]) / pose_scale])) / 2 + np.array(camera_move)
        camera_position.append(position_transfer)

    abs_angles = [camera_pos[i][4] for i in range(num_cam)]

    color_dict = {'red': [255, 0, 0], 'black': [0, 0, 0], 'blue': [0, 0, 255], 'green': [0, 255, 0],
                  'darkred': [128, 0, 0], 'yellow': [255, 255, 0], 'deeppink': [255, 20, 147]}

    # plot camera
    for i in range(num_cam):
        img[int(camera_position[i][1])][int(camera_position[i][0])][0] = color_dict["black"][0]
        img[int(camera_position[i][1])][int(camera_position[i][0])][1] = color_dict["black"][1]
        img[int(camera_position[i][1])][int(camera_position[i][0])][2] = color_dict["black"][2]

    # plot target
    img[int(target_position[1])][int(target_position[0])][0] = color_dict['blue'][0]
    img[int(target_position[1])][int(target_position[0])][1] = color_dict['blue'][1]
    img[int(target_position[1])][int(target_position[0])][2] = color_dict['blue'][2]

    plt.cla()
    plt.imshow(img.astype(np.uint8))

    # get camera's view space positions

    visua_len = 60
    for i in range(num_cam):
        theta = abs_angles[i] + 90.0
        # theta = -180
        dx = visua_len * math.sin(theta * math.pi / 180)
        dy = - visua_len * math.cos(theta * math.pi / 180)
        plt.arrow(camera_position[i][0], camera_position[i][1], dx, dy, width=0.1, head_width=8, head_length = 8, length_includes_head=True)

        plt.annotate(str(i), xy=(camera_position[i][0], camera_position[i][1]),
                     xytext=(camera_position[i][0], camera_position[i][1]), fontsize=10, color='blue')

        # top-left
        if int(choose_ids[i]) == 0:
            plt.annotate('cam {0} use pose'.format(i), xy=(camera_position[i][0], camera_position[i][1]),  xytext=(350, (1 + i) * 50 + 250), fontsize=10, color='red')
        else:
            plt.annotate('cam {0} use vision'.format(i), xy=(camera_position[i][0], camera_position[i][1]), xytext=(350, (1 + i) * 50 + 250), fontsize=10,
                         color='blue')

    plt.plot(target_position[0], target_position[1], 'ro')
    plt.title("Top-view")
    plt.xticks([])
    plt.yticks([])
    plt.pause(0.01)


class GoalNavAgentTest(object):
    """The world's simplest agent!"""

    def __init__(self, action_space, nav='New', goal_list=None):
        self.step_counter = 0
        self.keep_steps = 0
        self.goal_id = 0
        self.velocity_high = action_space['high'][0]
        self.velocity_low = action_space['low'][0]
        self.angle_high = action_space['high'][1]
        self.angle_low = action_space['low'][1]
        self.goal_list = goal_list

        self.goal = self.generate_goal()

        if 'Base' in nav:
            self.discrete = True
        else:
            self.discrete = False
        if 'Old' in nav:
            self.max_len = 30
        else:
            self.max_len = 1000

    def act(self, pose):

        self.step_counter += 1
        if self.pose_last == None:
            self.pose_last = pose
            d_moved = 100
        else:
            d_moved = np.linalg.norm(np.array(self.pose_last) - np.array(pose))
            self.pose_last = pose
        if self.check_reach(self.goal, pose) or d_moved < 3 or self.step_counter > self.max_len:
            self.goal = self.generate_goal()
            if self.discrete:
                self.velocity = (self.velocity_high + self.velocity_low) / 2
            else:
                self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
            self.step_counter = 0

        delt_yaw = self.get_direction(pose, self.goal)
        if self.discrete:
            if abs(delt_yaw) > self.angle_high:
                velocity = 0
            else:
                velocity = self.velocity
            if delt_yaw > 3:
                self.angle = self.angle_high / 2
            elif delt_yaw < -3:
                self.angle = self.angle_low / 2
        else:
            self.angle = np.clip(delt_yaw, self.angle_low, self.angle_high)
            velocity = self.velocity * (1 + 0.2 * np.random.random())

        return (velocity, self.angle)

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0
        self.goal_id = 0
        self.goal = self.generate_goal()
        self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
        self.pose_last = None

    def generate_goal(self):
        index = self.goal_id % len(self.goal_list)
        goal = np.array(self.goal_list[index])

        self.goal_id += 1
        return goal

    def check_reach(self, goal, now):
        error = np.array(now[:2]) - np.array(goal[:2])
        distance = np.linalg.norm(error)
        return distance < 50

    def get_direction(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        angle_now = np.arctan2(y_delt, x_delt) / np.pi * 180 - current_pose[4]
        if angle_now > 180:
            angle_now -= 360
        if angle_now < -180:
            angle_now += 360
        return angle_now
