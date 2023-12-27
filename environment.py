
from gym.core import Env
import numpy as np
from gym import spaces
class Environment(Env):
    metadata = {'render.modes': ['human'] }
    def __init__(self, data):
        self.initial_request = 0.1
        self.request = 0.1
        self.min_action = -2.0
        self.max_action = 2.0
        self.min_position = 0.0
        self.max_position = 1.5
        self.max_ratio = 1.0
        self.upper_bound = 0.8
        self.lower_bound = 0.2
        self.data = data
        self.steps = 0
        self._max_episode_steps = len(self.data)
        self.low_state = np.array(
            [self.min_position, -self.max_position, self.min_position], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_position, self.max_position], dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )


    def reset(self):
        self.steps = 0
        resource = self.getResource(0, 2)
        request = self.initial_request
        # resource = resource[0]
        state = np.array([request, resource[0], resource[0]/request], dtype=np.float32).reshape(3)
        # state = resource/request
        return state

    def seed(self, seed=None):
        pass

    def getResource(self, step, n):
        step = step % len(self.data)
        block = self.data[step:step + n] if step < len(self.data) - n - 1 else self.data[len(self.data) - n - 1:len(self.data)]

        resource = []
        for i in range(n - 1):
            resource.append(block[i])
        return np.array(resource)


    def step(self, action, evaluate=False):
        step = self.steps
        done = False
        info = {}
        resource = self.getResource(step, 2)
        # if (self.request + action) <= self.min_position:
        #     reward = 0
        #     done = True
        # else:
        if ((resource / (self.request + action)) < self.upper_bound) & ((resource / (self.request + action)) > self.lower_bound):
            self.request = self.request + action
            self.request = self.request[0]
            reward = 1
        elif (resource / (self.request + action) <= self.lower_bound) & (resource / (self.request + action) > 0.0):
            reward = -0.5
        elif (resource / (self.request + action) >= self.upper_bound) & (resource / (self.request + action) < 1.0):
            reward = -0.5
        else:
            reward = -1
        resource = resource[0]
        usage = resource / self.request
        self.steps += 1
        if self.steps == len(self.data):
            self.steps = 0
            done = True
        state = np.array([self.request, resource, usage], dtype=np.float32).reshape(3)

        if evaluate == True:
            print('action:{} | state :{} |  reward :{} | request :{} | ratio :{}'
                  .format(action, resource, reward, self.request, usage))
        return state, reward, done, info