from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import tensorflow as tf


class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, frame_skip=4):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mujoco_env.MujocoEnv.__init__(
            self, '%s/assets/swimmer.xml' % dir_path, frame_skip=frame_skip
        )
        utils.EzPickle.__init__(self)

    def _step(self, action):
        old_ob = self._get_obs()
        self.do_simulation(action, self.frame_skip)

        if getattr(self, 'action_space', None):
            action = np.clip(action, self.action_space.low,
                             self.action_space.high)
        ob = self._get_obs()

        reward_ctrl = -0.0001 * np.square(action).sum()
        reward_run = old_ob[3]
        reward = reward_run + reward_ctrl

        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            # (self.model.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
            # self.get_body_comvel("torso")[:1],
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat,
        ])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        self.prev_qpos = np.copy(self.model.data.qpos.flat)
        return self._get_obs()

    def cost_np_vec(self, obs, acts, next_obs):
        reward_ctrl = -0.0001 * np.sum(np.square(acts), axis=1)
        reward_run = obs[:, 3]
        reward = reward_run + reward_ctrl
        return -reward
        
    def get_tf_reward(self):
        def tf_reward(old_obs, action, new_obs):
            with tf.name_scope('reward'):
                reward_run = old_obs[:,3]
                ctrl_cost = 0.0001 * tf.reduce_sum(tf.square(action), axis = 1)
                reward = reward_run - ctrl_cost
                return reward
        return tf_reward

    def get_tf_done(self):
        def tf_done(old_obs, action, new_obs):
            temp = tf.reduce_sum(old_obs, axis = 1)
            return  tf.not_equal(temp, temp)
        return tf_done
