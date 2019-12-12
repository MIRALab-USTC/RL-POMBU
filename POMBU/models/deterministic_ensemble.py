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


class DeterministicEnsemble(DynamicsModel):
    def __init__(self, env, policy, dataset, V_estimator, save_dir, kwargs, rollout_kwargs, alpha_UBE=1, name="DE"):

        self.save_dir = save_dir
        self.norm_state = kwargs["norm_state"]
        self.norm_action = kwargs["norm_action"]
        self.norm_change = kwargs["norm_change"]
        self.kwargs = kwargs
        self.network_kwargs = kwargs["network_kwargs"]
        self.optimization_kwargs = kwargs["optimization_kwargs"]
        self.train_kwargs = kwargs["train_kwargs"]
        self.k = self.network_kwargs["k_model"]
        self.global_counters = [0] * self.k
        self.iteration_counter = 0
        self.training_network_fn, self.prediction_network_fn = mlp(env.observation_space.shape[0], 
                                                                   self.network_kwargs["hidden_layers"], 
                                                                   self.network_kwargs["nonlinearities"], 
                                                                   self.network_kwargs["decays"], 
                                                                   self.network_kwargs["norm"])    
        DynamicsModel.__init__(self, env, policy, dataset, V_estimator, save_dir, rollout_kwargs, alpha_UBE, name) 
        self.session.run([self.init_network, self.init_optimizer])

    def state_prediction(self):
        #### take care next_states and train_next_states

        state_space = self.state_space
        action_space = self.action_space

        self.train_state_input = tf.placeholder(shape=(None,) + state_space.shape, dtype=state_space.dtype, name='state_input')
        self.train_action_input = tf.placeholder(shape=(None,) + action_space.shape, dtype=action_space.dtype, name='action_input')
        self.train_change_input = tf.placeholder(shape=(None,) + state_space.shape, dtype=state_space.dtype, name='change_input')
        self.train_next_state_input = tf.placeholder(shape=(None,) + state_space.shape, dtype=state_space.dtype, name='next_state_input')
        self.train_input_list = [self.train_state_input, self.train_action_input, self.train_change_input, self.train_next_state_input]

        if self.norm_state:
            state = self.dataset.tf_normalize_state(self.state_input)
            train_state = self.dataset.tf_normalize_state(self.train_state_input)
        else:
            state = self.state_input
            train_state = self.train_state_input
        if self.norm_action:
            action = self.dataset.tf_normalize_action(self.action_output)
            train_action = self.dataset.tf_normalize_action(self.train_action_input)
        else:
            action = self.action_output
            train_action = self.train_action_input

        x = tf.concat([state, action], axis=1)
        train_x = tf.concat([train_state, train_action], axis=1)

        #y: change
        self.train_ys = []
        self.valid_ys = []
        ys = []
        network_name = self.network_name = self.name + "_network"
        k = self.k
        with tf.variable_scope(network_name, reuse=tf.AUTO_REUSE):
            for i in range(k):
                with tf.variable_scope("model%d" % i):
                    self.train_ys.append(self.training_network_fn(train_x))
                    self.valid_ys.append(self.prediction_network_fn(train_x))
                    ys.append(self.prediction_network_fn(x))

        
        if self.norm_change:
            changes = [self.dataset.tf_unnormalize_change(ys[i]) for i in range(k)]
            valid_changes = [self.dataset.tf_unnormalize_change(self.valid_ys[i]) for i in range(k)]
            self.train_y = self.dataset.tf_normalize_change(self.train_change_input)
        else:
            changes = ys
            valid_changes = self.valid_ys
            self.train_y = self.train_change_input

        self.next_states_output = [self.state_input + changes[i] for i in range(k)]
        self.valid_next_states = [self.train_state_input + valid_changes[i] for i in range(k)]

        self.network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=network_name)
        self.init_network = tf.variables_initializer(self.network_vars)
        self.network_var_list = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=network_name + "/model%d"%i) for i in range(k)]
        self.savers = [tf.train.Saver(self.network_var_list[i]) for i in range(k)]
    
    def save(self, indecies=None, name="backup"):
        if indecies == None:
            indecies = [i for i in self.k]
        elif type(indecies) == int:
            indecies = [indecies]
        for index in indecies:
            file_name = "dkpt/model%d_%s.dkpt" % (index, name)
            file_name = file_name if self.save_dir == None else os.path.join(self.save_dir, file_name)
            self.savers[index].save(self.session, file_name)
    
    def load(self, indecies=None, name="backup"):
        if indecies == None:
            indecies = [i for i in self.k]
        elif type(indecies) == int:
            indecies = [indecies]
        for index in indecies:
            file_name = "dkpt/model%d_%s.dkpt" % (index, name)
            file_name = file_name if self.save_dir == None else os.path.join(self.save_dir, file_name)
            self.savers[index].restore(self.session, file_name)

    def evaluate_policy(self, width=200):
        print("evaluate_policy......")
        for i in  range(self.k):
            states = []
            for _ in range(width):
                state = self.env.reset()
                states.append(state)
            states = np.array(states)
            rets = 0
            for _ in range(self.horizon):
                states, rewards = self.session.run([self.next_states_output[i], self.rewards_output[i]], feed_dict={self.state_input:states, self.index_input:i})                
                states = np.clip(states, *self.state_space.bounds)
                states = np.clip(states, -1e5, 1e5)
                rewards = np.clip(rewards, -100, 100)
                rets = rewards + rets
            mean_return = np.mean(rets)
            print(mean_return)
        print("\n\n")

    def build_optimizer(self):
        k = self.k
        self.lr = tf.placeholder(tf.float32, [])
        self.reg_coef = tf.placeholder(tf.float32, [])
        self.losses = [tf.reduce_mean(tf.square(self.train_ys[i] - self.train_y)) for i in range(k)]
        self.valid_losses = [tf.reduce_mean(tf.square(self.valid_ys[i] - self.train_y)) for i in range(k)]
        self.diff_next_state = [tf.reduce_mean(tf.square(self.valid_next_states[i] - self.train_next_state_input)) for i in range(k)]

        reg_losses = []
        with tf.variable_scope(self.network_name, reuse=tf.AUTO_REUSE):
            for i in range(k):
                reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.name + "/" + self.network_name + "/model%d"%i)
                if reg_loss != []:
                    reg_losses.append(tf.add_n(reg_loss))
                else:
                    reg_losses.append(0.0)
        
        self.regularized_losses = [tf.add(self.losses[i], reg_losses[i], name="regularized_loss%d"%i) for i in range(k)]
        max_grad = self.optimization_kwargs["max_grad"]
        self.train_ops = []
        self.train_summaries = []
        self.valid_summaries = []
        self.model_writer = tf.summary.FileWriter('model_data' if self.save_dir==None else os.path.join(self.save_dir, "model_data"), self.session.graph)
        self.optimizer_name = optimizer_name = self.name + "_optimizer"
        with tf.variable_scope(optimizer_name):
            for i in range(k):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + "/" + self.network_name + "/model%d"%i)
                with tf.control_dependencies(update_ops):
                    trainer = tf.train.AdamOptimizer(learning_rate=self.lr)
                    variables = tf.trainable_variables(self.network_name+"/model%d"%i)
                    grads_and_var = trainer.compute_gradients(self.regularized_losses[i], variables)
                    grads, var = zip(*grads_and_var)
                    if max_grad is not None:
                        grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad)
                    grads_and_var = list(zip(grads, var))
                    train_op = trainer.apply_gradients(grads_and_var)
                self.train_ops.append(train_op)

                train_summary_list = []
                train_summary_list.append(tf.summary.scalar('train_loss%d'%i, self.losses[i]))
                train_summary_list.append(tf.summary.scalar('train_regularized_loss%d'%i, self.regularized_losses[i]))
                valid_summary_list = []
                valid_summary_list.append(tf.summary.scalar('valid_loss%d'%i, self.valid_losses[i]))
                valid_summary_list.append(tf.summary.scalar('diff_next_state%d'%i, self.diff_next_state[i]))

                self.train_summaries.append(tf.summary.merge(train_summary_list))
                self.valid_summaries.append(tf.summary.merge(valid_summary_list))
        self.optimizer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=optimizer_name)
        self.init_optimizer = tf.variables_initializer(self.optimizer_vars)
        

    
    def train(self):
        reinitialize_every = self.train_kwargs["reinitialize_every"]
        if reinitialize_every > 0 and self.iteration_counter % reinitialize_every == 0:
            print("reinitialize model and optimizer...")
            self.session.run([self.init_network, self.init_optimizer])
            validate_every = self.train_kwargs["re_validate_every"]
            threshold = self.train_kwargs["re_threshold"]
            train_data = "reinitialize"
            lr = self.optimization_kwargs["re_learning_rate"]
        else:
            print("reinitialize optimizer...")
            self.session.run(self.init_optimizer)
            validate_every = self.train_kwargs["validate_every"]
            threshold = self.train_kwargs["threshold"]
            train_data = "incremental"
            lr = self.optimization_kwargs["learning_rate"]
        load_backup = self.train_kwargs["load_backup"]

        max_iterations = self.train_kwargs["max_iterations"]
        train_valid = self.train_kwargs["train_valid"]
        valid_ratio = self.train_kwargs["valid_ratio"]
        batch_train = self.train_kwargs["batch_train"]
        batch_valid = self.train_kwargs["batch_valid"]
        delta = self.train_kwargs["delta"]
        n_virtual = self.train_kwargs["n_virtual"]
        min_step = self.train_kwargs["min_step"]
        max_iterations = self.train_kwargs["max_iterations"]
        log_loss = self.train_kwargs["log_loss"]
        total_step = self.dataset.get_total_steps()

        generators = self.dataset.model_train_data(train_data, train_valid, valid_ratio, batch_train, batch_valid, delta(total_step), n_virtual, min_step)

        for i in range(self.k):
            print("\n\ntraining the model%d......\n"%i)
            min_loss = None
            tv_data = generators[i]
            train_counter = 0
            not_update = 0
            while True:
                if train_counter >= max_iterations or not_update > threshold:
                    break

                while True:
                    train_batch = tv_data.get_train_batch()
                    if train_batch == None:
                        if train_counter % validate_every == 0:
                            valid_batch = tv_data.get_valid_batch()

                            if log_loss:
                                valid_log, valid_loss = self.session.run([self.valid_summaries[i], self.valid_losses[i]], feed_dict=dict(zip(self.train_input_list, valid_batch)))
                                train_log = self.session.run(self.train_summaries[i], feed_dict=train_dict)
                                self.model_writer.add_summary(valid_log, self.global_counters[i])
                                self.model_writer.add_summary(train_log, self.global_counters[i])
                            else:
                                valid_loss = self.session.run(self.valid_losses[i], feed_dict=dict(zip(self.train_input_list, valid_batch)))

                            if min_loss == None or min_loss >= valid_loss:
                                min_loss = valid_loss
                                not_update = 0
                                if load_backup:
                                    self.save(i)
                            else:
                                not_update += validate_every

                            print("step%d: valid_loss:%f\n"%(train_counter, valid_loss))
                        train_counter += 1
                        self.global_counters[i] += 1
                        break
                    else:
                        train_dict = dict(zip(self.train_input_list, train_batch))
                        train_dict[self.lr] = lr(train_counter)
                        self.session.run(self.train_ops[i], feed_dict=train_dict)
            if load_backup:
                self.load(i)
        self.iteration_counter += 1
    


