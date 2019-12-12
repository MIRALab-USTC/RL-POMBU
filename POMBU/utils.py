import tensorflow as tf
from tensorflow import nn
import numpy as np
import random
import json
import os
from baselines import logger
from baselines.common.tf_util import get_session

import POMBU
from POMBU.envs.utils import get_inner_env

def dump_params_to_file(params, save_dir):
    target_file = os.path.join(save_dir, "params.json")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(target_file, "w") as f:
        json.dump(params, f)

def load_params_from_file(exp_name, subexp_name=""):
    _file = POMBU.__file__
    if subexp_name == "" or subexp_name == None:
        params_dir = os.path.join(_file[:_file.rfind("/")], "params", exp_name + ".json")
    else:
        params_dir = os.path.join(_file[:_file.rfind("/")], "params", exp_name, subexp_name + ".json")
    with open(params_dir, "r") as f:
        return json.load(f)

def set_seed(env, seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.set_random_seed(seed)
    except Exception as e:
        print(e)
    
    while hasattr(env, "wrapped_env") or hasattr(env, "env") or hasattr(env, "seed"):
        if hasattr(env, "seed"):
            temp_seed = env.seed(seed)
            if temp_seed != None and temp_seed != []:
                logger.info("Seed: %d. Set seed successfully"%env.seed(seed)[0])
                return
        if hasattr(env, "wrapped_env"): 
            env = env.wrapped_env
        else:
            env = env.env
    logger.error("env should have the attribution seed()")

def get_tf_done(env):
    while hasattr(env, "wrapped_env") or hasattr(env, "env") or hasattr(env, "get_tf_done"):
        if hasattr(env, "get_tf_done"):
            return env.get_tf_done()
        elif hasattr(env, "wrapped_env"): 
            env = env.wrapped_env
        else:
            env = env.env
    logger.error("env should have the attribution get_tf_done()")

def get_tf_reward(env):
    while hasattr(env, "wrapped_env") or hasattr(env, "env") or hasattr(env, "get_tf_reward"):
        if hasattr(env, "get_tf_reward"):
            return env.get_tf_reward()
        elif hasattr(env, "wrapped_env"): 
            env = env.wrapped_env
        else:
            env = env.env
    logger.error("env should have the attribution get_tf_reward()")


def eval_dict(kwargs_dict):
    for key, val in kwargs_dict.items():
        if type(val) == dict:
            kwargs_dict[key] = eval_dict(val)
        elif type(val) == str and ("nn." in val or "lambda" in val or "np." in val or val == "None"):
            kwargs_dict[key] = eval(val)
    return kwargs_dict


def dict_update(a ,b):
    assert type(a) == dict and type(b) == dict
    for key in b:
        if type(b[key]) == dict:
            if key in a and type(a[key]) == dict:
                dict_update(a[key], b[key])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a

def batch_gen(batch_size, *data):
    length = len(data[0])
    index = np.arange(length)
    np.random.shuffle(index)
    i = 0
    while True:
        data_batch = []
        for item in data:
            data_batch.append(item[index[i:i+batch_size]])
        real_size = len(data_batch[0])
        yield data_batch if real_size > 0 else None
        i += batch_size

def data_filter(condition, *data):
    assert callable(condition)
    index = condition(*data)
    print("remain ratio:%f"%np.mean(index))
    new_data = []
    for item in data:
        new_data.append(item[index])
    return new_data


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class TfRunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    '''
    TensorFlow variables-based implmentation of computing running mean and std
    Benefit of this implementation is that it can be saved / loaded together with the tensorflow model
    '''
    def __init__(self, epsilon=1e-4, shape=(), scope=''):
        sess = get_session()

        self._new_mean = tf.placeholder(shape=shape, dtype=tf.float64)
        self._new_var = tf.placeholder(shape=shape, dtype=tf.float64)
        self._new_count = tf.placeholder(shape=(), dtype=tf.float64)


        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self._mean  = tf.get_variable('mean',  initializer=np.zeros(shape, 'float64'),      dtype=tf.float64)
            self._var   = tf.get_variable('std',   initializer=np.ones(shape, 'float64'),       dtype=tf.float64)
            self._count = tf.get_variable('count', initializer=np.full((), epsilon, 'float64'), dtype=tf.float64)
            self.f32_mean = tf.cast(self._mean, tf.float32)
            self.f32_var = tf.cast(self._var, tf.float32)

        self.update_ops = tf.group([
            self._var.assign(self._new_var),
            self._mean.assign(self._new_mean),
            self._count.assign(self._new_count)
        ])

        sess.run(tf.variables_initializer([self._mean, self._var, self._count]))
        self.sess = sess
        self._set_mean_var_count()

    def _set_mean_var_count(self):
        self.mean, self.var, self.count = self.sess.run([self._mean, self._var, self._count])

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        new_mean, new_var, new_count = update_mean_var_count_from_moments(self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

        self.sess.run(self.update_ops, feed_dict={
            self._new_mean: new_mean,
            self._new_var: new_var,
            self._new_count: new_count
        })

        self._set_mean_var_count()

if __name__ == "__main__":
    print("test data_filter")
    a = np.array([1,2,3,4,5,6])
    b = np.array([15,72,3,48,50,16])
    c = np.array([12,23,34,45,56,67])
    condition = lambda a,b,c: a + b > c
    print(data_filter(condition, a, b, c))



