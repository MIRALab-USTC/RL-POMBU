import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)

class PointEnv(gym.Env):

    def __init__(self, init_sampling_boundaries=(-1,1), goal=(0, 0, 0), **kwargs):
        self.init_sampling_boundaries=init_sampling_boundaries
        self.goal = np.array(goal)
        self.action_space = spaces.Box(low=np.array([-0.1, -0.1, -0.1]), high=np.array([0.1, 0.1, 0.1]))
        self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf]), high=np.array([np.inf, np.inf, np.inf]))
        self._seed()
        self._state = None
        object.__init__(self)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        prev_state = self._state
        self._state = prev_state + np.clip(action, -0.1, 0.1)
        reward = self.reward(prev_state, action, self._state)
        done = False
        next_observation = np.copy(self._state)
        return next_observation, reward, done, {}


    def _reset(self):
        self._state = self.np_random.uniform(self.init_sampling_boundaries[0], self.init_sampling_boundaries[1], size=(3,))
        observation = np.copy(self._state)
        return observation

    def _render(self, close=False):
        print('current_state:', self._state)

    def done(self, obs):
        if obs.ndim == 1:
            return np.array(False)
        elif obs.ndim == 2:
            temp = np.sum(obs, axis = 1)
            return  np.not_equal(temp, temp) 

    def reward(self, obs, act, obs_next):
        if obs_next.ndim == 1:
            return 0 - np.linalg.norm(obs_next - self.goal)
        elif obs_next.ndim == 2:
            return 0 - np.linalg.norm(obs_next - self.goal[None, :], axis=1)

    def get_tf_reward(self):
        def tf_reward(old_obs, action, new_obs):
            with tf.name_scope('reward'):
                reward = 0 - tf.sqrt(tf.reduce_sum(tf.square(new_obs - self.goal), axis=1))
                return reward
        return tf_reward

    def get_tf_done(self):
        def tf_done(old_obs, action, new_obs):
            temp = tf.reduce_sum(old_obs, axis = 1)
            return  tf.not_equal(temp, temp)
        return tf_done

        
    def get_tf_transition(self):
        def tf_transition(old_obs, action):
            _state = old_obs + tf.clip_by_value(action, -0.1, 0.1)
            return  _state
        return tf_transition

    def estimate_V(self, state, pi, num_traj=20, horizon=30, gamma=0.99):
        V = 0
        state = np.array([state] * num_traj).reshape(-1,3)
        a = 1
        for _ in range(horizon):
            action = pi.step(state)
            _state = state + np.clip(action, -0.1, 0.1)
            r = self.reward(state, action, _state)
            V = V + a * r 
            a *= gamma
            state = _state
        V = V.reshape(num_traj, -1)
        V = np.mean(V, axis=0)
        return V

    def estimate_Q(self, state, action, pi, num_traj=20, horizon=30, gamma=0.99):
        Q = 0
        state = np.array([state] * num_traj).reshape(-1,3)
        action = np.array([action] * num_traj).reshape(-1,3)
        a = 1
        for _ in range(horizon):
            _state = state + np.clip(action, -0.1, 0.1)
            r = self.reward(state, action, _state)
            Q = Q + a * r 
            a *= gamma
            state = _state
            action = pi.step(state)
        Q = Q.reshape(num_traj, -1)
        Q = np.mean(Q, axis=0)
        return Q

    def estimate_optimal_return(self, horizon=30):
        V = 0
        state = self.np_random.uniform(self.init_sampling_boundaries[0], self.init_sampling_boundaries[1], size=(500, 2))
        for _ in range(horizon):
            action = 0 - state
            _state = state + np.clip(action, -0.1, 0.1)
            r = self.reward(state, action, _state)
            V = V + r 
            state = _state
        V = np.mean(V)
        print("optimal return: %f"%V)


