import gym
from gym.spaces import Discrete, Tuple, Box

import numpy as np

# for communicating with grbl on the arduino
import serial
import time

class BerryEnv(gym.Env):
    def __init__(self):
        self.stupid_mapping = {0: -1, 1: 0, 2: 1}
        self.init_grbl()
        self.step_num = 0

    def init_grbl(self):
        self.serial_cxn = serial.Serial('/dev/ttyUSB0', 115200)
        # Wake up grbl
        self.serial_cxn.write("\r\n\r\n".encode())
        time.sleep(2)                 # Wait for grbl to initialize
        self.serial_cxn.flushInput()  # Flush startup text in serial input
        
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
        self.send_robot_action(commands)

        self.step_num += 1
        print('step num is {}'.format(self.step_num))

        done = False
        if self.step_num > 100:
            done = True

        # Return state, reward, is_terminated, dict you can ignore
        return np.zeros(1), 0, done, {}

    def reset(self):
        return np.zeros(1)

    def send_robot_action(self, commands):
        grbl_cmds = []
        magnitude = 1
        
        # commands is [X, Y, Z], each is [-1, 0, 1]
        if commands[0] != 0:
            grbl_cmds.append("G91 G0  X" + str(commands[0] * magnitude))
        if commands[1] != 0:
            grbl_cmds.append("G91 G0  Y" + str(commands[1] * magnitude))
        if commands[2] != 0:
            grbl_cmds.append("G91 G0  Z" + str(commands[2] * magnitude))

        for c in grbl_cmds:
            # print('Sending: ' + c)
            self.serial_cxn.write((c + '\n').encode())        # Send g-code block to grbl
            grbl_out = self.serial_cxn.readline()  # Wait for grbl response with carriage return
            # print(' : ' + grbl_out.strip())
