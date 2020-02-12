from gym.envs.registration import register
import logging
logger = logging.getLogger(__name__)
use_docker = False  # True: use nvidia docker   False: do not use nvidia-docker


def load_env_setting(filename):
    import os
    import gym_unrealcv
    gympath = os.path.dirname(gym_unrealcv.__file__)
    gympath = os.path.join(gympath, 'envs/setting', filename)
    f = open(gympath)
    filetype = os.path.splitext(filename)[1]
    if filetype == '.json':
        import json
        setting = json.load(f)
    else:
        print ('unknown type')

    return setting


# Multi-Camera Collaboration env 

for env in ['MCRoomLarge', 'Garden', 'UrbanTreeOBST']:
    for i in range(7):  # reset type
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd', 'Gray']:  # observation type
                for nav in ['Random', 'Goal', 'Internal', 'None',
                            'RandomInterval', 'GoalInterval', 'InternalInterval', 'NoneInterval']:

                    name = 'Unreal{env}-{action}{obs}{nav}-v{reset}'.format(env=env, action=action, obs=obs, nav=nav, reset=i)
                    if 'Interval' in nav:
                        setting_file = 'MCMT/{env}_interval.json'.format(env=env)
                    else:
                        setting_file = 'MCMT/{env}.json'.format(env=env)
                    register(
                        id=name,
                        entry_point='gym_unrealcv.envs:UnrealCvMC',
                        kwargs={'setting_file': setting_file,
                                'reset_type': i,
                                'action_type': action,
                                'observation_type': obs,
                                'reward_type': 'distance',
                                'docker': use_docker,
                                'nav': nav
                                },
                        max_episode_steps=500
                    )



