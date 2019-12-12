import tensorflow as tf
import warnings
class VEstimator:
    def __init__(self, env):
        self.env = env
        self.state_space = self.env.observation_space

    def tf_value(self, state_input):
        warnings.warn("Please rewrite the tf_value funciton")
        return None, tf.placeholder(dtype=tf.float32, shape=(None,))
    
    def tf_uncertainty(self, state_input, Vs, V):
        warnings.warn("Please rewrite the tf_uncertainty funciton")
        return None, tf.placeholder(dtype=tf.float32, shape=(None,))
