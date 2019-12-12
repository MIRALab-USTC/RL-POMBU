if __name__ == "__main__":
    import sys
    sys.path.append("/home/qizhou/nips2019/rl_temp")
    sys.path.insert(0,"/home/qizhou/libs/gym")
    
from . import DynamicsModel
from POMBU.utils import dict_update
from POMBU.tf_utils.nn import mlp

import tensorflow.contrib.layers as layers
from tensorflow import nn
import tensorflow as tf

from baselines.a2c.utils import fc

import random
import numpy as np
import os


class NoisyEnsemble(DynamicsModel):
    def __init__(self, env, policy, dataset, V_estimator, save_dir, kwargs, rollout_kwargs, alpha_UBE=1, name="DE"):

        self.save_dir = save_dir
        self.kwargs = kwargs
        self.noise_std = kwargs["noise_std"]
        self.k = self.kwargs["network_kwargs"]["k_model"]
        DynamicsModel.__init__(self, env, policy, dataset, V_estimator, save_dir, rollout_kwargs, alpha_UBE, name) 

    def state_prediction(self):
        k = self.k
        cliped_action = tf.clip_by_value(self.action_output, -0.1, 0.1)
        _state = self.state_input + cliped_action
        noises = [self.noise_std * tf.random_normal(tf.shape(_state)) for i in range(k)]
        self.next_states_output = [_state + noises[i] for i in range(k)]
    
    def save(self, indecies=None, name="backup"):
        pass
    
    def load(self, indecies=None, name="backup"):
        pass

    def evaluate_policy(self, width=200):
        pass

    def build_optimizer(self):
        pass
    
    def train(self):
        pass
    


