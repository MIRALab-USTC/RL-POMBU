import tensorflow as tf
import numpy as np
import random
import warnings
from tqdm import tqdm
import gym
import json
import os
import matplotlib.mlab as mlab

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

SMALL_SIZE = 15
MEDIUM_SIZE = 17
BIGGER_SIZE = 19
LINEWIDTH = 3

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

from baselines.common.tf_util import get_session
from POMBU.utils import TfRunningMeanStd, get_tf_done, get_tf_reward

#tf_.....
#policy
#V


class DynamicsModel:
    def __init__(self, env, policy, dataset, V_estimator, save_dir, rollout_kwargs, alpha_UBE, name, epsilon=1e-8):
    #alpha_UBE should be a hyperparameter
    #summary should be defined in subclass
    #runtime/ rollout
        self.env = env
        self.inner_env = self._get_inner_env()
        self.horizon = self.env.horizon
        self.policy = policy
        self.session = get_session()
        self.dataset = dataset
        self.name = name
        self.V_estimator = V_estimator
        self.tf_reward = get_tf_reward(env)
        self.tf_done = get_tf_done(env)
        self.state_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.alpha_UBE = alpha_UBE
        self.epsilon = epsilon
        self.rollout_norm_state = rollout_kwargs["norm_state"]
        self.rollout_norm_action = rollout_kwargs["norm_action"]
        self.rollout_norm_rew = rollout_kwargs["norm_reward"]
        self.rollout_clip_rew = rollout_kwargs["clip_reward"]
        self.method = rollout_kwargs["method"]
        self.lam = rollout_kwargs["lambda"]
        self.gamma = rollout_kwargs["gamma"]
        self.add_index = rollout_kwargs["add_index"]
        self.save_dir = save_dir
            
        with tf.name_scope(name):
            self.build_graph() 
            self.build_optimizer()    

        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        self.init_all = tf.variables_initializer(all_vars)

        self.writer = tf.summary.FileWriter('data/' if save_dir==None else save_dir+'/data/', self.session.graph)
        self.session.run([self.init_all])



    def state_prediction(self):
        k = self.state_input.shape.as_list()[0]
        size = self.state_input.shape.as_list()[2]
        self.next_states_output = tf.reshape(self.state_input, shape=(k,-1,size))
        warnings.warn("Please rewrite the sate_perdiction funciton")

    def tf_modify_state(self, state, step):
        if self.rollout_norm_state:
            state = self.dataset.tf_normalize_state(state)
        return tf.concat([0.0 * state[:,0:1] + (step * 2)/(self.horizon-1) -1, state], axis=1) if self.add_index else state

    def tf_modify_reward(self, reward):
        if self.rollout_clip_rew > 0:
            reward = tf.clip_by_value(reward, -self.rollout_clip_rew, self.rollout_clip_rew)
        if self.rollout_norm_rew:
            reward = self.dataset.tf_normalize_reward(reward)
        return reward

    def modify_state(self, state, step):
        if self.rollout_norm_state:
            state = self.dataset.normalize_state(state)
        if len(state.shape) == 2:
            return np.concatenate([0.0 * state[:,0:1] + (step * 2)/(self.horizon-1) -1, state], axis=1) if self.add_index else state
        if len(state.shape) == 1:
            return np.concatenate([[(step * 2) / (self.horizon - 1) -1], state], axis=0) if self.add_index else state
    
    def modify_action(self, action):
        if self.rollout_norm_action:
            action = self.dataset.normalize_action(action)
        return action

    def modify_reward(self, reward):
        if self.rollout_clip_rew > 0:
            reward = np.clip(reward, -self.rollout_clip_rew, self.rollout_clip_rew)
        if self.rollout_norm_rew:
            reward = self.dataset.normalize_reward(reward)
        return reward

    def tf_recover_action(self, action):
        if self.rollout_norm_action:
            action = self.dataset.tf_unnormalize_action(action)
        return action

    def recover_action(self, action):
        if self.rollout_norm_action:
            action = self.dataset.unnormalize_action(action)
        return action

    def build_graph(self):

        state_space = self.state_space

        self.state_input = tf.placeholder(shape=(None, state_space.shape[0]), dtype=state_space.dtype, name='state_input')
        self.index_input = tf.placeholder(shape=(), dtype=state_space.dtype, name='index_input')

        self.state = state = self.tf_modify_state(self.state_input, self.index_input)
        _, _, self.action, self.neglogp, _ = self.policy.tf_step(state)

        self.action_output = self.tf_recover_action(self.action)

        self.V = self.V_estimator.tf_value(state)
        _, self.U = self.V_estimator.tf_uncertainty(state)

        self.state_prediction() 

        k  = len(self.next_states_output)

        self.rewards_output = [self.tf_reward(self.state_input, self.action_output, self.next_states_output[i]) for i in range(k)]
        rewards = tf.reshape(tf.concat(self.rewards_output,axis=0), shape=(k,-1))
        self.dones = [self.tf_done(self.state_input, self.action_output, self.next_states_output[i]) for i in range(k)]


        next_states = [self.tf_modify_state(self.next_states_output[i], self.index_input+1) for i in range(k)]
        next_Vs = [self.V_estimator.tf_value(next_states[i]) for i in range(k)]
        next_Vs = tf.reshape(tf.concat(next_Vs,axis=0), shape=(k,-1))

        rewards = self.tf_modify_reward(rewards)
        self.reward_mean = reward_mean = tf.reduce_mean(rewards, axis=0, name="reward_mean")
        self.reward_var = reward_var = tf.reduce_mean(tf.square(rewards-reward_mean), axis=0, name="reward_var")
        next_Vs_mean = tf.reduce_mean(next_Vs, axis=0, name="next_Vs_mean")
        self.next_Vs_var = tf.reduce_mean(tf.square(next_Vs-next_Vs_mean), axis=0, name="next_Vs_var")
            
        Qs = rewards + self.gamma * next_Vs
        Q_mean = tf.reduce_mean(Qs, axis=0, name="Q_mean")
        with tf.name_scope("uncertainty"):
            self.u_MBUI = tf.reduce_mean(tf.square(Qs-Q_mean), axis=0, name="u_MBUI")
            self.u_UBE = tf.add(reward_var, self.alpha_UBE * Q_mean * Q_mean, name="u_UBE")
            next_state_mean = tf.reduce_mean(self.next_states_output, axis=0)
            next_state_var = tf.sqrt(tf.reduce_mean(tf.square(self.next_states_output-next_state_mean), axis=0))
            self.next_state_std = tf.reduce_sum(next_state_var, axis=1)
            if hasattr(self.inner_env, "get_tf_transition"):
                tf_transition = self.inner_env.get_tf_transition()
                next_state_real = tf_transition(self.state_input, self.action_output)
                next_state_error = tf.reduce_sum(tf.abs(self.next_states_output - next_state_real), axis=-1)
                self.next_state_error = tf.reduce_mean(next_state_error, axis=0, name="next_state_error")


    def build_optimizer(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def step(self, state, step, index_p): 
        run_list = [self.U, self.V, self.state, self.action, self.neglogp, self.next_states_output, self.dones, self.reward_mean, self.reward_var, self.next_Vs_var]
        
        if "MBUI" == self.method:
            run_list.append(self.u_MBUI)
        if "UBE" == self.method:
            run_list.append(self.u_UBE)

        state = np.clip(state, *self.state_space.bounds)
        state = np.clip(state, -1e5, 1e5)
        
        U, V, state, action, neglogp, next_states, dones, r, r_var, next_V_var, u = self.session.run(run_list, feed_dict={self.state_input: state, self.index_input: step})

        k = len(next_states)
        next_state = np.empty(shape = next_states[0].shape)
        done = np.empty(shape = dones[0].shape, dtype=bool)

        index_set = np.arange(k)
        if type(index_p) == type(None):
            index_p = np.full(k, 1/k)

        for i in range(next_states[0].shape[0]):
            index = np.random.choice(index_set, p=index_p)
            next_state[i] = next_states[index][i]
            done[i] = dones[index][i]

        return U, V, u, r, done, state, action, neglogp, r_var, next_V_var, next_state

    def _get_inner_env(self):
        env = self.env
        while hasattr(env, "wrapped_env") or hasattr(env, "env"):
            if hasattr(env, "get_tf_done"):
                break
            elif hasattr(env, "wrapped_env"): 
                env = env.wrapped_env
            else:
                env = env.env
        if hasattr(env, "get_tf_done"):
            return env
        else:
            return None

    def plot_img(self, n_epoch, resolution=20, mod="both", plot_interval=1):
        states = np.random.randn(4000,3) * 1.6 - 0.8
        self.policy.set_deterministic()
        trs, _ = self.rollout(5000, print_inf=False)
        for i in range(n_epoch):
          
            if i % plot_interval == 0:
                if mod == "V" or mod == "both":
                    uncertainties, values = self.estimate_UV(states, 20)
                    self._plot_img(i, states, values, uncertainties)
                if mod == "Q" or mod == "both":
                    print(states.shape)
                    actions, uncertainties, values = self.estimate_UQ(states, 20)
                    print(states.shape)
                    self._plot_img(i, states, values, uncertainties, mod="Q", actions=actions)

            _, _ = self.V_estimator.train(trs["state"], trs["newU"], trs["newV"], 5e-5)

    def _plot_img(self, polt_id, states, values, uncertainties, mod="V", actions=None):
        point_2d_env = self.inner_env
        print(states.shape)
        if mod=="V":
            scatter_filename = os.path.join(self.save_dir, 'scatter_V%d.png' % (polt_id))
            hist_filename_1 = os.path.join(self.save_dir, 'hist_ratio_V%d.png' % (polt_id))
            hist_filename_2 = os.path.join(self.save_dir, 'hist_error_V%d.png' % (polt_id))
            real_V = point_2d_env.estimate_V(states, self.policy, 20, self.horizon, self.gamma)
            if self.rollout_norm_rew:
                real_V = self.dataset.normalize_reward(real_V)
            errors = values - real_V
        elif mod=="Q":
            scatter_filename = os.path.join(self.save_dir, 'scatter_Q%d.png' % (polt_id))
            hist_filename_1 = os.path.join(self.save_dir, 'hist_ratio_Q%d.png' % (polt_id))
            hist_filename_2 = os.path.join(self.save_dir, 'hist_error_Q%d.png' % (polt_id))
            real_Q = point_2d_env.estimate_Q(states, actions, self.policy, 20, self.horizon, self.gamma)
            if self.rollout_norm_rew:
                real_Q = self.dataset.normalize_reward(real_Q)
            errors = values - real_Q 
        size = len(uncertainties)
        index = np.random.choice(np.arange(size), 333, replace=False)
        plt.scatter(uncertainties[index], errors[index], alpha=0.35,edgecolors='white', color="red")

        xmin = 0
        xmax = 0.03

        plt.xlim(xmin, xmax)
        plt.ylim(-0.04, 0.04)
        line_x = np.linspace(0, xmax, 3)
        plt.plot(line_x, line_x, linewidth=0.8, linestyle='--', label="x=|y|",color="orange")
        plt.plot(line_x, 0-line_x, linewidth=0.8, linestyle='--', label="", color="orange")
        plt.xlabel('uncertainty')
        plt.ylabel('error')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(scatter_filename)
        plt.close()
        print("the file %s has been saved."%scatter_filename)
        norm = lambda x: np.exp(0-x*x/2)/np.sqrt(2*np.pi)
        _, bins, _ = plt.hist(errors/uncertainties, bins=30, range=(-2.5, 2.5), density=True, facecolor = 'orange', alpha = 0.65)
        y_norm = norm(bins)
        plt.plot(bins, y_norm, 'r--', label="N(0,1)")
        plt.legend()
        plt.savefig(hist_filename_1)
        plt.close()
        print("the file %s has been saved."%hist_filename_1)
        plt.hist(errors, bins=30, range=(-0.04, 0.04), density=True, histtype='stepfilled')
        plt.savefig(hist_filename_2)
        plt.close()
        print("the file %s has been saved."%hist_filename_2)

    def estimate_UV(self, state, num_traj=10):
        state = np.array([state] * num_traj).reshape(-1,3)
        size = len(state)
        trs, _ = self.rollout(init_states=state)
        U = trs["newU"][:size]
        V = trs["newV"][:size]
        U = U.reshape(num_traj, -1)
        U = np.mean(U, axis=0)
        uncertainties = np.sqrt(U+0.0000001)
        V = V.reshape(num_traj, -1)
        values = np.mean(V, axis=0)
        return uncertainties, values

    def estimate_UQ(self, state, num_traj=10):
        state = np.array([state] * num_traj).reshape(-1,3)
        size = len(state)
        trs, _ = self.rollout(init_states=state)
        U = trs["newU"][:size]
        Q = trs["newV"][:size]
        U = U.reshape(num_traj, -1)
        U = np.mean(U, axis=0)
        uncertainties = np.sqrt(U+0.0000001)
        Q = Q.reshape(num_traj, -1)
        values = np.mean(Q, axis=0)
        action = trs["action"][:len(values)]
        return action, uncertainties, values

    def rollout(self, width=200, print_inf=False, name="", log_num=0, index_p=None, init_states=None):
        if type(init_states) == type(None):
            states = []
            for _ in range(width):
                state = self.env.reset()
                states.append(state)
            states = np.array(states)
        else:
            states = np.array(init_states)
        trs = [[], [], [], [], [], [], [], [], [], [],]

        '''
        0: oldU_list=[]
        1: oldV_list=[]
        2: u_list=[]
        3: r_list=[]
        4: done_list=[]
        5: state_list = []
        6: a_list=[]
        7: neglogp_list=[]
        8: r_var = []
        9: next_V_var = []
        '''

        print("rollout......")
        for i in tqdm(range(self.horizon)):
            temp = list(self.step(states, i, index_p))
            states = temp[-1]
            for item, item_list in zip(temp[:-1], trs):
                item_list.append(item)

        return self.prepare_trs(trs, print_inf, name, log_num)
        

    def prepare_trs(self, trs, print_inf, name="", log_num=0):
        UV_list = uv_trace(*trs[:5], self.lam, self.gamma)

        if print_inf:
            for i in range(self.horizon):
                std = np.sqrt(trs[0][i] + UV_list[0][i])
                print("\nrollout_step%d:\n AV:%f \t newU:%f \t newV:%f \t std_frac:%f"\
                    %(i, np.mean(UV_list[3][i]), np.mean(UV_list[0][i]), np.mean(UV_list[1][i]), np.mean(np.abs(std / (UV_list[3][i]+ 0.0001)))))

        ret = 0
        for i in range(self.horizon):
            ret = ret + trs[3][i]
        if self.rollout_norm_rew:
            ret = self.dataset.unnormalize_reward(ret)
        mean_ret = np.mean(ret)

        if log_num != 0:
            log_trs = {}
            log_trs["newU"] = np.array(UV_list[0]).swapaxes(0,1) 
            log_trs["newV"] = np.array(UV_list[1]).swapaxes(0,1) 
            log_trs["u"] = np.array(trs[2]).swapaxes(0,1)
            log_trs["r"] = np.array(trs[3]).swapaxes(0,1) 
            if len(trs) == 10:
                log_trs["r_var"] = np.array(trs[8]).swapaxes(0,1)
                log_trs["next_V_var"] = np.array(trs[9]).swapaxes(0,1) 
            for key in log_trs:
                log_trs[key] = log_trs[key][:log_num].tolist()
            save_file = '/trs_%s.json'%name if self.save_dir==None else self.save_dir+'/trs_%s.json'%name
            with open(save_file, 'w') as f:
                json.dump(log_trs, f)

        new_trs = {}
        not_terminal = np.logical_not(np.array(trs[4])).reshape(-1)
        new_trs["state"] = data_filter(trs[5], not_terminal)
        new_trs["action"] = data_filter(trs[6], not_terminal)
        new_trs["neglogp"] = data_filter(trs[7], not_terminal)
        new_trs["oldU"] = data_filter(trs[0], not_terminal)
        new_trs["oldV"] = data_filter(trs[1], not_terminal)
        new_trs["newU"] = data_filter(UV_list[0], not_terminal)
        new_trs["newV"] = data_filter(UV_list[1], not_terminal)
        new_trs["AU"] = data_filter(UV_list[2], not_terminal)
        new_trs["AV"] = data_filter(UV_list[3], not_terminal)


        return new_trs, mean_ret

def data_filter(data, not_terminal):
    data = np.array(data)
    if len(data.shape) == 2:
        return data.flatten()[not_terminal]
    elif len(data.shape) == 3:
        return data.reshape(-1,data.shape[-1])[not_terminal]
    else:
        raise RuntimeError("dim of data should be 2 or 3")

def uv_trace(U_list, V_list, u_list, r_list, done_list, lam, gamma=1):
    T = len(U_list)
    for i in range(T-1):
        not_done = np.logical_not(done_list[i]).astype(np.float)
        U_list[i+1] *= not_done
        V_list[i+1] *= not_done
        u_list[i+1] *= not_done
        r_list[i+1] *= not_done
        done_list[i+1] = np.logical_or(done_list[i], done_list[i+1])

    done_list[1:] = done_list[:-1]
    done_list[0].fill(False)

    U_list.append(U_list[0] * 0)
    V_list.append(V_list[0] * 0)
    newU_list = []
    newV_list = []
    AU_list = [0]
    AV_list = [0]

    gamma_U = gamma * gamma 
    for i in reversed(range(T)):
        deltaU = u_list[i] + gamma_U * U_list[i+1] - U_list[i]
        AU_list.insert(0, deltaU + gamma_U * lam * AU_list[0])
        newU_list.insert(0, AU_list[0] + U_list[i])

        deltaV = r_list[i] + gamma * V_list[i+1] - V_list[i]
        AV_list.insert(0, deltaV + gamma * lam * AV_list[0])
        newV_list.insert(0, AV_list[0] + V_list[i])

    U_list.pop()
    V_list.pop()
    return newU_list, newV_list, AU_list[:-1], AV_list[:-1]


