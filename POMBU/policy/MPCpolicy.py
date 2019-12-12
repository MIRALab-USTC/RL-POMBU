from . import Policy
from POMBU.utils import dict_update, batch_gen

import random
import numpy as np


class MPCPolicy(Policy):
    def __init__(self, env, kwargs={}):
        Policy.__init__(self, env)
        default_kwargs = {
            "deepth": 100,
            "width": 1000,
        }
        kwargs = dict_update(default_kwargs, kwargs)
        self.deepth = kwargs["deepth"]
        self.width = kwargs["width"]

    def set_model(self, model):
        self.model = model
    
    def step(self, state_input, stochastic=True):
        state = np.array([state_input] * self.width)
        model = self.model
        r = 0
        d = False
        first_action = action = np.array([self.action_space.sample() for _ in range(self.width)])
        for _ in range(self.deepth):
            neglogp = np.array([None] * self.width)
            _, _, state, done, _, _, _, reward_mean, _ = model.step(state, feed_dict={self.action: action, self.neglogp:neglogp})
            d = np.logical_or(d, done)
            r = r + reward_mean * d
            action = np.array([self.action_space.sample() for _ in range(self.width)])
        return first_action[np.argmax(r)], None
    