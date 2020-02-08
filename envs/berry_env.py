import gym
from gym.spaces import Discrete, Tuple, Box

import numpy as np


class BerryEnv(gym.Env):
    def __init__(self):
        self.stupid_mapping = {0: -1, 1: 0, 2: 1}

    @property
    def observation_space(self):
        return Box(low=-np.infty, high=np.infty, shape=(1,))

    @property
    def action_space(self):
        return Tuple((Discrete(3), Discrete(3), Discrete(3)))

    def step(self, actions):
        commands = []
        for action in actions:
            commands.append(self.stupid_mapping[action])
        #TODO(ajose)
        self.send_robot_action(commands)

        # Return state, reward, is_terminated, dict you can ignore
        return np.zeros(1), 0, False, {}

    def reset(self):
        return np.zeros(1)

    def send_robot_action(self, commands):
        pass