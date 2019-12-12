import tensorflow as tf

class Policy:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.state_space = env.observation_space

    def step(self):
        raise NotImplementedError

    def tf_step(self, state):
        self.action_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.action_space.shape[0]))
        self.neglogp_placeholder = tf.placeholder(dtype=tf.float32, shape=(None))
        return None, None, self.action_placeholder, self.neglogp_placeholder


class RandomPolicy(Policy):
    def __init__(self, env):
        Policy.__init__(self, env)

    def step(self, state):
        return self.action_space.sample(), None
