import numpy as np
import random
import tensorflow as tf
from POMBU.utils import TfRunningMeanStd
import json
from baselines.common.tf_util import get_session

class TrainValidData:
        def __init__(self,
                     data, 
                     train_index,
                     valid_index,
                     batch_train, 
                     batch_valid, 
                     delta, 
                     n_virtual, 
                     state_var, 
                     action_var, 
                     min_step):
            self.n_train = len(train_index)
            self.n_valid = len(valid_index)
            self.train_data = {}
            self.valid_data = {}
            
            train_index = train_index.astype(np.int32)
            valid_index = valid_index.astype(np.int32)

            for key in data:
                self.train_data[key] =  data[key][train_index]
                self.valid_data[key] =  data[key][valid_index]

                if self.n_train < min_step and n_virtual > 0:
                    virtual_data = np.concatenate([self.train_data[key]] * n_virtual, axis=0)
                    if key == "states":
                        noise = np.random.normal(0, delta, virtual_data.shape) * np.sqrt(state_var)
                        virtual_data = virtual_data + noise
                    if key == "actions":
                        noise = np.random.normal(0, delta, virtual_data.shape) * np.sqrt(action_var)
                        virtual_data = virtual_data + noise
                    self.train_data[key] = np.concatenate([self.train_data[key], virtual_data], axis=0)
            
            print("train set:", self.n_train, "\tvalid set:", self.n_valid)
            
            self.batch_train = batch_train if batch_train < self.n_train else self.n_train
            self.batch_valid = batch_valid if batch_valid < self.n_valid else self.n_valid

            self.train_start = 0
            self.train_index = np.arange(self.n_train)
            self.valid_index = np.arange(self.n_valid)
            np.random.shuffle(self.train_index)

        def get_valid_batch(self):
            index = np.random.choice(self.valid_index, self.batch_valid, replace=False)
            return [self.valid_data[key][index] for key in self.valid_data]

        def get_train_batch(self):
            if self.train_start >= self.n_train: 
                np.random.shuffle(self.train_index)
                self.train_start = 0
                return None
                
            ts = self.train_start
            self.train_start = te = ts + self.batch_train
            index = self.train_index[ts:te]
            return [self.train_data[key][index] for key in self.train_data]

        def get_batch(self):
            return self.get_train_batch(), self.get_valid_batch()



class Dataset:
    # test code can be added here
    def __init__(self, env, k_model, gamma=0.99, save_dir=None, max_size=2e5, epsilon=1e-8, name="DS"):
        self.horizon = env.horizon
        self.gamma = gamma
        self.max_size = max_size
        with tf.variable_scope(name):    
            self.state_rms = TfRunningMeanStd(shape=env.observation_space.shape, scope="state_rms")
            self.change_rms = TfRunningMeanStd(shape=env.observation_space.shape, scope="change_rms")
            self.action_rms = TfRunningMeanStd(shape=env.action_space.shape, scope="action_rms")
            self.return_rms = TfRunningMeanStd(shape=(), scope="return_rms")
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        self.saver = tf.train.Saver(all_vars)
        self.k_model = k_model
        self.session = get_session()

        self.trajectory = {
            "states":[],
            "actions":[],
            "changes":[],
            "next_states":[],
        }

        self.dataset = {
            "states":[],
            "actions":[],
            "changes":[],
            "next_states":[],
        }

        self.rewards = []

        self.epsilon = epsilon
        self.save_dir = save_dir
    
        self.train_index = []
        self.valid_index = []

        for _ in range(k_model):
            self.train_index.append([])
            self.valid_index.append([])

    def save(self, step):
        dataset = {
            "states":[],
            "actions":[],
            "changes":[],
            "next_states":[],
        }
        for item in dataset:
            for i in range(len(self.dataset[item])):
                dataset[item].append(self.dataset[item][i].tolist())
        save_file = '/dataset%d.json'%step if self.save_dir==None else self.save_dir+'/dataset%d.json'%step
        with open(save_file, 'w') as f:
            json.dump(dataset, f)
        save_file = self.save_dir + "/dataset/dataset_rms%d.dkpt" % (step) if self.save_dir != None else "dataset/dataset_rms%d.dkpt" % (step)
        self.saver.save(self.session, save_file)

        

    def load(self,step):
        save_file = '/dataset%d.json'%step if self.save_dir==None else self.save_dir+'/dataset%d.json'%step
        with open(save_file, 'r') as f:
            dataset = json.load(f)
        for item in self.dataset:
            for i in range(len(self.dataset[item])):
                self.dataset[item][i] = np.array(dataset[item][i])
        save_file = self.save_dir + "/dataset/dataset_rms%d.dkpt" % (step) if self.save_dir != None else "dataset/dataset_rms%d.dkpt" % (step)
        self.saver.restore(self.session, save_file)
        

    def flush(self):

        self.trajectory["states"].pop()

        for i in reversed(range(len(self.rewards) - 1)):
            self.rewards[i] += self.gamma * self.rewards[i+1]
        self.return_rms.update(np.array(self.rewards))


        for key in self.trajectory:
            self.trajectory[key] = np.array(self.trajectory[key])
            if self.dataset[key] == []:
                self.dataset[key] = self.trajectory[key]
            else:
                self.dataset[key] = np.concatenate([self.dataset[key], self.trajectory[key]], axis=0)
        size = self.get_total_steps()
        if size > self.max_size:
            for i in range(self.k_model):
                self.train_index[i] -= (size - self.max_size)
                self.valid_index[i] -= (size - self.max_size)
            for key in self.dataset: 
                self.dataset[key] = self.dataset[key][-self.max_size:]

        self.change_rms.update(self.trajectory["changes"])
        
        self.trajectory = {
            "states":[],
            "actions":[],
            "next_states":[],
            "changes":[],
        }

        self.rewards = []

    def add(self, action, next_state, reward):
        self.trajectory["actions"].append(action)
        change = next_state - self.trajectory["states"][-1]
        self.trajectory["states"].append(next_state)
        self.trajectory["next_states"].append(next_state)
        self.trajectory["changes"].append(change)
        self.state_rms.update(np.array([next_state]))
        self.action_rms.update(np.array([action])) 
        self.rewards.append(reward)

    def add_first_state(self, state):
        self.trajectory["states"].append(state)
        self.state_rms.update(np.array([state]))



    def tf_normalize_change(self, change):
        return (change -self.change_rms.f32_mean) / tf.sqrt(self.change_rms.f32_var + self.epsilon)

    def tf_normalize_state(self, state):
        return (state - self.state_rms.f32_mean) / tf.sqrt(self.state_rms.f32_var + self.epsilon) 

    def tf_normalize_action(self, action):
        return (action - self.action_rms.f32_mean) / tf.sqrt(self.action_rms.f32_var + self.epsilon) 

    def tf_normalize_reward(self, reward):
        return reward / tf.sqrt(self.return_rms.f32_var + self.epsilon) 

    def tf_unnormalize_change(self, change):
        return change * tf.sqrt(self.change_rms.f32_var + self.epsilon) + self.change_rms.f32_mean

    def normalize_state(self, state):
        return (state -  self.state_rms.mean) / np.sqrt((self.state_rms.var + self.epsilon))

    def normalize_action(self, action):
        return (action -  self.action_rms.mean) / np.sqrt((self.action_rms.var + self.epsilon))

    def normalize_reward(self, reward):
        return reward / np.sqrt((self.return_rms.var + self.epsilon))

    def unnormalize_reward(self, reward):
        return reward * np.sqrt((self.return_rms.var + self.epsilon))
    
    def unnormalize_action(self, action):
        return action * np.sqrt((self.action_rms.var + self.epsilon)) + self.action_rms.mean
    
    def tf_unnormalize_action(self, action):
        return action * tf.sqrt(self.action_rms.f32_var + self.epsilon) + self.action_rms.f32_mean


    def model_train_data(
            self, 
            train_data,
            train_valid,
            valid_ratio, 
            batch_train, 
            batch_valid, 
            delta, 
            n_virtual, 
            min_step, ):

        generators = []

        states = np.array(self.dataset["states"])
        actions = np.array(self.dataset["actions"])
        next_states = np.array(self.dataset["next_states"])
        changes = np.array(self.dataset["changes"])


        dataset = {
            "states":states,
            "actions":actions,
            "changes":changes,
            "next_states":next_states,
        }

        horizon = self.horizon
        n_total = self.get_total_steps()
        n_valid_pt = int(horizon * valid_ratio)
        
        for i in range(self.k_model):
            if train_data == "incremental":
                n_cur = len(self.train_index[i]) + len(self.valid_index[i])
                if train_valid == "all":
                    n_valid = int((n_total - n_cur) * valid_ratio)
                    index = np.arange(n_cur, n_total)
                    np.random.shuffle(index)
                    self.train_index[i] = np.concatenate([self.train_index[i], index[n_valid:]])
                    self.valid_index[i] = np.concatenate([self.valid_index[i], index[:n_valid]])
                elif train_valid == "trajectory":
                    index = np.arange(n_cur, n_cur + horizon)
                    for _ in range(0, n_total - n_cur, horizon):
                        np.random.shuffle(index)
                        self.train_index[i] = np.concatenate([self.train_index[i], index[n_valid_pt:]])
                        self.valid_index[i] = np.concatenate([self.valid_index[i], index[:n_valid_pt]])
                        index += horizon

            else:
                if train_valid == "all":
                    n_valid = int(n_total * valid_ratio)
                    index = np.arange(n_total)
                    np.random.shuffle(index)
                    self.train_index[i] = index[n_valid:]
                    self.valid_index[i] = index[:n_valid]
                elif train_valid == "trajectory":
                    index = np.arange(horizon)
                    np.random.shuffle(index)
                    self.train_index[i] = index[n_valid_pt:].copy()
                    self.valid_index[i] = index[:n_valid_pt].copy()
                    for _ in range(horizon, n_total, horizon):
                        index += horizon
                        np.random.shuffle(index)
                        self.train_index[i] = np.concatenate([self.train_index[i], index[n_valid_pt:]])
                        self.valid_index[i] = np.concatenate([self.valid_index[i], index[:n_valid_pt]])

            generator = TrainValidData(dataset, self.train_index[i], self.valid_index[i], batch_train, batch_valid, delta, n_virtual, self.state_rms.var, self.action_rms.var, min_step)
            generators.append(generator)

        return generators
    
    def get_total_steps(self):
        return len(self.dataset["states"])

    def print_inf(self):
        print("number of trajectories: ", self.get_total_steps())
        print("state's mean and var: ", self.state_rms.mean, self.state_rms.var)
        print("change's mean and var: ", self.change_rms.mean, self.change_rms.var)
        print("action's mean and var: ", self.action_rms.mean, self.action_rms.var)
        print("return's mean and var: ", self.return_rms.mean, self.return_rms.var)
            

