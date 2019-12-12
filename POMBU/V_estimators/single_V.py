from . import VEstimator
from POMBU.utils import dict_update, batch_gen, data_filter
from POMBU.tf_utils.nn import mlp

import tensorflow as tf
from tensorflow import nn

from baselines.a2c.utils import fc
from baselines.common.tf_util import get_session

import random
import numpy as np
import os


class SingleV(VEstimator):
    def __init__(self, env, extra_n, save_dir, kwargs, name="SV"):
        VEstimator.__init__(self, env)
        self.extra_n = extra_n
        self.save_dir = save_dir
        self.name = name 
        self.session = get_session()
        self.graph = self.session.graph
        self.kwargs = kwargs
        self.network_kwargs = self.kwargs["network_kwargs"]
        self.optimization_kwargs = self.kwargs["optimization_kwargs"]
        self.train_kwargs = self.kwargs["train_kwargs"]

        if self.network_kwargs["u_network"] == "share":
            self.training_network_fn, self.prediction_network_fn = mlp(2, 
                                                                    self.network_kwargs["hidden_layers"], 
                                                                    self.network_kwargs["nonlinearities"], 
                                                                    self.network_kwargs["decays"], 
                                                                    self.network_kwargs["norm"])   
        else:
            self.training_network_fn, self.prediction_network_fn = mlp(1, 
                                                                    self.network_kwargs["hidden_layers"], 
                                                                    self.network_kwargs["nonlinearities"], 
                                                                    self.network_kwargs["decays"], 
                                                                    self.network_kwargs["norm"])                                                                       
        self.epoch_counter = self.iter_counter = 0
        with tf.name_scope(self.name):
            self.build_graph()
            self.build_optimizer()
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        self.init_all = tf.variables_initializer(all_vars)
        self.session.run([self.init_all, self.init_network, self.init_optimizer])

    def value_for_MF(self, state_input):
        if len(state_input.shape) == len(self.state_space.shape):
            state_input = state_input.reshape(1,-1)
            not_vec = True
        else:
            not_vec = False
        assert len(state_input.shape) == len(self.state_space.shape) + 1
        V = self.session.run(self.V, feed_dict={self.state_input:state_input})
        if not_vec:
            V = V[0]
        return V

    def tf_value(self, state_input, is_training=False):
        network_name = self.network_name
        with tf.variable_scope(network_name, reuse=tf.AUTO_REUSE):
            if self.network_kwargs["u_network"] == "share":
                if is_training:
                    V = self.training_network_fn(state_input)[:,0]
                else:
                    V = self.prediction_network_fn(state_input)[:,0]

            else:
                with tf.variable_scope("value", reuse=tf.AUTO_REUSE):
                    if is_training:
                        V = self.training_network_fn(state_input)[:,0]
                    else:
                        V = self.prediction_network_fn(state_input)[:,0]
        return V

    def tf_uncertainty(self, state_input, is_training=False):
        network_name = self.network_name
        with tf.variable_scope(network_name, reuse=tf.AUTO_REUSE):
            if self.network_kwargs["u_network"] == "share":
                if is_training:
                    U = self.training_network_fn(state_input)[:,1]
                else:
                    U = self.prediction_network_fn(state_input)[:,1]

            else:
                with tf.variable_scope("uncertainty", reuse=tf.AUTO_REUSE):
                    if is_training:
                        U = self.training_network_fn(state_input)[:,0]
                    else:
                        U = self.prediction_network_fn(state_input)[:,0]
            Upred = tf.maximum(U, 0)
        return U, Upred

    def build_graph(self):       
        self.state_input = tf.placeholder(shape=(None, self.extra_n + self.state_space.shape[0]), dtype=tf.float32, name="state_input")
        self.U_input = tf.placeholder(shape=(None,), dtype=tf.float32, name="U_input")
        self.V_input = tf.placeholder(shape=(None,), dtype=tf.float32, name="V_input")
        self.input_list = [self.state_input, self.U_input, self.V_input]

        self.network_name = self.name + "_network"
        self.V = self.tf_value(self.state_input, True)
        self.U, _ = self.tf_uncertainty(self.state_input, True)
        self.network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.network_name)
        self.init_network = tf.variables_initializer(self.network_vars)
        self.saver = tf.train.Saver(self.network_vars)

    def save(self, name="backup"):
        file_name = "dkpt/UV_%s.dkpt" % (name)
        file_name = file_name if self.save_dir == None else os.path.join(self.save_dir, file_name)
        self.saver.save(self.session, file_name)
    
    def load(self, name="backup"):
        file_name = "dkpt/UV_%s.dkpt" % (name)
        file_name = file_name if self.save_dir == None else os.path.join(self.save_dir, file_name)
        self.saver.restore(self.session, file_name)

    def build_optimizer(self):
        self.lr = tf.placeholder(tf.float32, [])

        self.U_loss = U_loss = tf.reduce_mean(tf.square(self.U - self.U_input))
        self.V_loss = V_loss = tf.reduce_mean(tf.square(self.V - self.V_input))
        U_coefficient = self.optimization_kwargs["U_coefficient"]

        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.name + "/" + self.network_name)
        if reg_loss != []:
            reg_loss = tf.add_n(reg_loss)
        else:
            reg_loss = 0.0
        loss = tf.add_n([V_loss, U_coefficient * U_loss, reg_loss], name="UV_loss")

    
        max_grad = self.optimization_kwargs["max_grad"]
        self.uv_writer = tf.summary.FileWriter('UV_data/' if self.save_dir==None else os.path.join(self.save_dir, "UV_data"), self.graph)
        self.optimizer_name = optimizer_name = self.name + "_optimizer"
        with tf.variable_scope(optimizer_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + "/" + self.network_name)
            with tf.control_dependencies(update_ops):
                trainer = tf.train.AdamOptimizer(learning_rate=self.lr)
                variables = tf.trainable_variables(self.network_name)
                grads_and_var = trainer.compute_gradients(loss, variables)
                grads, var = zip(*grads_and_var)
                if max_grad is not None:
                    grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad)
                grads_and_var = list(zip(grads, var))
                self.train_op = trainer.apply_gradients(grads_and_var)

            train_summary_list = []
            train_summary_list.append(tf.summary.scalar('V_loss', V_loss))
            train_summary_list.append(tf.summary.scalar('U_loss', U_loss))
            self.train_summaries= tf.summary.merge(train_summary_list)

        optimizer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=optimizer_name)
        self.init_optimizer = tf.variables_initializer(optimizer_vars)
    
    def train(self, state, U, V, learning_rate):
        minibatch_size = self.train_kwargs["minibatch_size"]
        num_epochs = self.train_kwargs["num_epochs"]
        log_every = self.train_kwargs["log_every"]
        input_list = [state, U, V]

        print("train V...")
        for i in range(num_epochs):
            batch_generator = batch_gen(minibatch_size, *input_list)
            for items in batch_generator:
                if items == None:
                    break
                train_dict = dict(zip(self.input_list, items))
                train_dict[self.lr] = learning_rate
                self.session.run(self.train_op, feed_dict=train_dict)
            if log_every > 0 and i % log_every == 0:
                train_log = self.session.run(self.train_summaries, feed_dict=dict(zip(self.input_list, input_list)))
                self.uv_writer.add_summary(train_log, self.epoch_counter)
            self.epoch_counter += 1
            
        if log_every == 0:
            train_log = self.session.run(self.train_summaries, feed_dict=dict(zip(self.input_list, input_list)))
            self.uv_writer.add_summary(train_log, self.iter_counter)
        self.iter_counter += 1

        newU, newV, Uloss, Vloss = self.session.run([self.U, self.V, self.U_loss, self.V_loss], feed_dict={self.state_input:state, self.U_input:U, self.V_input:V})
        print("U_loss:", Uloss, "\t\tV_loss:", Vloss)
        return newU,newV
        
