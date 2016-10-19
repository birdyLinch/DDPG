"""
OpenAI Gym environment for the humanoid skeleton simulation running in Bullet physics engine
"""

import socket
import sys
import struct

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='HumanEnv-v0',
    entry_point='HumanEnv:HumanEnv',
    timestep_limit=1000,
    reward_threshold=400.0,
)

class HumanEnv(gym.Env):
    metadata = {
        'render.modes': [],
        'video.frames_per_second' : 60
    }

    def __init__(self):

        # Connect to the simulation in Bullet
        HOST, PORT = 'localhost', 47138
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((HOST, PORT))
        self.s.send(b'Hello!')

        # Height at which to fail the episode
        self.y_threshold = 0.5

        self.action_space = spaces.Box(-1.5,1.5,(16,))
        self.observation_space = spaces.Box(-1e6,1e6,(306,))

        self._seed()
        self.reset()
        self.viewer = None

        self.steps_beyond_done = None

        # Just need to initialize the relevant attributes
        self._configure()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        
        # Apply the action
        c = np.asarray(action)
        buff = struct.pack('%sf' % len(c), *c)
        self.s.send(buff)

        # Receive the state
        stateSize = self.s.recv(4);
        stateSize = struct.unpack('i',stateSize)[0]
        state = self.s.recv(1000)
        while len(state) < stateSize:
            state += self.s.recv(1000)
        state = np.asarray([struct.unpack('f',state[i:i+4])[0] for i in range(0,len(state),4)])

        # Get the y position of the root joint
        y = state[3]
        done = y < self.y_threshold

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # skeleton just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _reset(self):

        # Reset the simulation
        self.s.send(b'RESET')
        zeros = np.zeros(20).astype(int)
        buff = struct.pack('%si' % len(zeros), *zeros)
        self.s.send(buff)

        # Get the state
        stateSize = self.s.recv(4);
        stateSize = struct.unpack('i',stateSize)[0]
        state = self.s.recv(1000)
        while len(state) < stateSize:
            state += self.s.recv(1000)
        state = np.asarray([struct.unpack('f',state[i:i+4])[0] for i in range(0,len(state),4)])

        # Apply a random control to just have a new initialization every time
        c = np.random.uniform(-0.5,0.5,16)
        buff = struct.pack('%sf' % len(c), *c)
        self.s.send(buff)

        # Get the state
        stateSize = self.s.recv(4);
        stateSize = struct.unpack('i',stateSize)[0]
        state = self.s.recv(1000)
        while len(state) < stateSize:
            state += self.s.recv(1000)
        self.state = np.asarray([struct.unpack('f',state[i:i+4])[0] for i in range(0,len(state),4)])

        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        pass
