import gym
import numpy as np
import tensorflow as tf

from envs.gym import env_name_to_gym_registry
from envs.proxy_env import ProxyEnv

def turn_on_video_recording():
    import builtins
    builtins.visualize = True


def turn_off_video_recording():
    import builtins
    builtins.visualize = False


def get_video_recording_status():
    import builtins
    return getattr(builtins, "visualize", False)

def get_inner_env(env):
    # TODO (Jenny) too hacky... refactor
    return env.wrapped_env


def get_env(env_name, video_dir=None):

    unnormalized_env = gym.make(env_name_to_gym_registry[env_name])

    turn_off_video_recording()

    def video_callable(_):
        return get_video_recording_status()

    if video_dir:
        unnormalized_env = gym.wrappers.Monitor(unnormalized_env, video_dir,
                                                video_callable=video_callable, force=True)

    return ProxyEnv(unnormalized_env)
