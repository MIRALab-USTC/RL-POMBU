from . import Policy
from POMBU.utils import dict_update, batch_gen, data_filter

import tensorflow.contrib.layers as layers
from tensorflow import nn
import tensorflow as tf

from baselines.common.models import mlp
from baselines.a2c.utils import fc
from baselines.common.tf_util import get_session
from baselines.common.distributions import PdType, Pd, _matching_fc

import random
import numpy as np
import os

class DiagGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size
    def pdclass(self):
        return DiagGaussianPd

    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0):
        mean = _matching_fc(latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        logstd = tf.get_variable(name='pi/logstd', shape=[1, self.size], initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        return self.pdfromflat(pdparam), mean, logstd

    def param_shape(self):
        return [2*self.size]
    def sample_shape(self):
        return [self.size]
    def sample_dtype(self):
        return tf.float32
        

class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape)-1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
    def flatparam(self):
        return self.flat
    def mode(self):
        return self.mean
    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tf.reduce_sum(self.logstd, axis=-1)
    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=-1)
    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class StochasticPolicy(Policy):
    def __init__(self, env, extra_n, save_dir, kwargs, name="Pi"):
        Policy.__init__(self, env)
        self.extra_n = extra_n
        self.save_dir = save_dir
        self.name = name 
        self.session = get_session()
        self.kwargs = kwargs
        self.network_kwargs = self.kwargs["network_kwargs"]
        self.optimization_kwargs = self.kwargs["optimization_kwargs"]
        self.train_kwargs = self.kwargs["train_kwargs"]
        self.network_fn = mlp(self.network_kwargs["num_layers"], self.network_kwargs["num_hidden"], self.network_kwargs["activation"])        
        self.epoch_counter = self.iter_counter = 0
        with tf.name_scope(self.name):
            self.build_graph()
            self.build_optimizer()
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        self.init_all = tf.variables_initializer(all_vars)
        self.session.run([self.init_all, self.init_network, self.init_optimizer])
        
    def print_parameters(self):
        for var in self.network_vars:
            print(var.name, ":")
            print(self.session.run(var))

    def step(self, state_input, noise=1):
        if len(state_input.shape) == len(self.state_space.shape):
            state_input = state_input.reshape(1,-1)
            not_vec = True
        else:
            not_vec = False
        assert len(state_input.shape) == len(self.state_space.shape) + 1
        action, std = self.session.run([self.mean, self.noise], feed_dict={self.state_input:state_input})
        action = action + noise * std
        if not_vec:
            action = action[0]
        return action

    def step_for_MF(self, state_input):
        if len(state_input.shape) == len(self.state_space.shape):
            state_input = state_input.reshape(1,-1)
            not_vec = True
        else:
            not_vec = False
        assert len(state_input.shape) == len(self.state_space.shape) + 1
        action, neglogp = self.session.run([self.action, self.neglogp], feed_dict={self.state_input:state_input})
        if not_vec:
            action = action[0]
            neglogp = neglogp[0]
        return action, neglogp
    
    def tf_step(self, state_input):
        network_name = self.network_name
        with tf.variable_scope(network_name, reuse=tf.AUTO_REUSE):
            hidden_layer = self.network_fn(state_input)
            pd, mean, _logstd = self.pdtype.pdfromlatent(hidden_layer, init_scale=0.01)
            action = pd.sample()
            neglogp = pd.neglogp(action)
        return pd, mean, action, neglogp, _logstd
    
    def reinit_std(self):
        self.session.run(self.set_logstd,feed_dict={self.logstd_input:0})

    def set_deterministic(self):
        self.session.run(self.set_logstd,feed_dict={self.logstd_input:-500})

    def kl_compute_graph(self):
        other_logstd = self.other_logstd = tf.placeholder(shape=self.action_input.shape, dtype=tf.float32, name="other_logstd")
        other_mean = self.other_mean = tf.placeholder(shape=self.action_input.shape, dtype=tf.float32, name="other_mean")
        other_std = tf.exp(other_logstd)
        return tf.reduce_sum(self.logstd - other_logstd + (tf.square(other_std) + tf.square(self.mean - other_mean)) / (2.0 * tf.square(self.std)) - 0.5, axis=-1)

    def make_kl(self, state):
        logstd, mean = self.session.run([self.logstd, self.mean], feed_dict={self.state_input:state})
        def kl_compute(policy):
            return policy.session.run(policy.kl, feed_dict={policy.state_input:state, policy.other_logstd:logstd, policy.other_mean:mean})
        return kl_compute

    def build_graph(self): 
        assert len(self.action_space.shape) == 1      
        size = self.action_space.shape[0]
        self.pdtype = DiagGaussianPdType(size)   

        self.state_input = tf.placeholder(shape=(None, self.extra_n + self.state_space.shape[0]), dtype=tf.float32, name="state_input")
        self.action_input = tf.placeholder(shape=(None,) + self.action_space.shape, dtype=tf.float32, name="action_input")

        self.A_input = tf.placeholder(shape=(None,), dtype=tf.float32, name="A_input")
        self.Uv_input = tf.placeholder(shape=(None,), dtype=tf.float32, name="Uv_input")
        self.Uq_input = tf.placeholder(shape=(None,), dtype=tf.float32, name="Uq_input")
        self.oldneglogp = tf.placeholder(shape=(None,), dtype=tf.float32, name="neglogp")

        self.input_list = [self.state_input, self.action_input, self.A_input, self.Uv_input, self.Uq_input, self.oldneglogp]
        self.network_name = self.name + "_network"
        self.pd, self.mean, self.action, self.neglogp, self._logstd = self.tf_step(self.state_input)

        zeros = tf.zeros(shape=[1, size])
        self.logstd_input = tf.placeholder(shape=(), dtype=tf.float32, name="logstd_input")
        self.set_logstd = tf.assign(self._logstd, zeros+self.logstd_input)

        self.logstd, self.std = self.pd.logstd, self.pd.std
        self.noise = self.std * tf.random_normal(tf.shape(self.mean))
        self.entropy = tf.reduce_mean(self.pd.entropy())

        self.train_neglogp = self.pd.neglogp(self.action_input)
        self.kl = self.kl_compute_graph()
        self.network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.network_name)
        self.init_network = tf.variables_initializer(self.network_vars)
        self.saver = tf.train.Saver(self.network_vars)

    def save(self, name="backup"):
        file_name = "dkpt/policy_%s.dkpt" % (name)
        file_name = file_name if self.save_dir == None else os.path.join(self.save_dir, file_name)
        self.saver.save(self.session, file_name)
    
    def load(self, name="backup"):
        file_name = "dkpt/policy_%s.dkpt" % (name)
        file_name = file_name if self.save_dir == None else os.path.join(self.save_dir, file_name)
        self.saver.restore(self.session, file_name)

    def log_return(self, mean_r_model, mean_r1, mean_r2, step):
        return_log = self.session.run(self.return_summary, feed_dict={self.mean_return_model:mean_r_model, self.mean_return_deterministic:mean_r1, self.mean_return_stochastic:mean_r2})
        self.policy_writer.add_summary(return_log, step)


    def build_optimizer(self):
        self.alpha = tf.placeholder(tf.float32, [], name="alpha")
        self.cliprange = tf.placeholder(tf.float32, [], name="cliprange")
        self.lr = tf.placeholder(tf.float32, [], name="lr")

        Uq_std = tf.sqrt(self.Uq_input)
        modified_A = self.A_input + self.alpha * Uq_std
        ratio_change = tf.exp(self.oldneglogp - self.train_neglogp) - 1
        cliped_ratio_change = tf.clip_by_value(ratio_change, -self.cliprange, self.cliprange)


        true_loss = tf.reduce_mean(- self.A_input * ratio_change)
        ppo_loss = tf.reduce_mean(tf.maximum(-self.A_input * ratio_change, -self.A_input * cliped_ratio_change))
        variance = tf.reduce_mean(tf.abs(ratio_change) * Uq_std)
        vc_loss = ppo_loss + self.alpha * variance
        ex_loss = tf.reduce_mean(tf.maximum(-modified_A * ratio_change, -modified_A * cliped_ratio_change))

        self.ratio_change = tf.reduce_mean(tf.abs(ratio_change))
        self.approxkl = .5 * tf.reduce_mean(tf.square(self.train_neglogp - self.oldneglogp))
        self.ratio_invalid = tf.reduce_mean(tf.to_float(tf.less(tf.abs(self.A_input), self.alpha * Uq_std)))

        self.p_improve_true = -true_loss / (variance + 1e-5)
        self.p_improve_ppo = -ppo_loss / (variance + 1e-5)

        self.mean_return_model = tf.placeholder(tf.float32, [], name="mean_return_model")
        self.mean_return_deterministic = tf.placeholder(tf.float32, [], name="mean_return_deterministic")
        self.mean_return_stochastic = tf.placeholder(tf.float32, [], name="mean_return_stochastic")


        max_grad = self.optimization_kwargs["max_grad"]
        self.policy_writer = tf.summary.FileWriter('policy_data/' if self.save_dir==None else os.path.join(self.save_dir, "policy_data"), self.session.graph)
        self.optimizer_name = optimizer_name = self.name + "_optimizer"
        with tf.variable_scope(optimizer_name):

            model_summary = tf.summary.scalar("return_model", self.mean_return_model)
            deterministic_summary = tf.summary.scalar("return_deterministic", self.mean_return_deterministic)
            stochastic_summary = tf.summary.scalar("return_stochastic", self.mean_return_stochastic)
            summay_list = [tf.summary.scalar('entropy', self.entropy),
                           tf.summary.scalar('approxkl', self.approxkl),
                           tf.summary.scalar('ratio_change', self.ratio_change),
                           tf.summary.scalar('ppo_loss', ppo_loss),
                           tf.summary.scalar('true_loss', true_loss),
                           tf.summary.scalar('variance', variance),
                           tf.summary.scalar('p_improve_ppo', self.p_improve_ppo),
                           tf.summary.scalar('p_improve_true', self.p_improve_true),]
            self.return_summary = tf.summary.merge([model_summary, deterministic_summary, stochastic_summary])
            
            self.vc_train = get_opt(vc_loss, self.lr, max_grad)
            self.ex_train = get_opt(ex_loss, self.lr, max_grad)
            self.p_ppo_train = get_opt(-self.p_improve_ppo, self.lr, max_grad)
            
            vc_summay_list = summay_list.copy()
            vc_summay_list.append(tf.summary.scalar('vc_loss', vc_loss))
            vc_summay_list.append(tf.summary.scalar('alpha', self.alpha))
            self.vc_summary = tf.summary.merge(vc_summay_list)
            self.p_ppo_summary = tf.summary.merge(summay_list)

        optimizer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=optimizer_name)
        self.init_optimizer = tf.variables_initializer(optimizer_vars)

    def train(self, state, action, AV, Uv, Uq, oldneglogp, method, learning_rate, cliprange, alpha=None, alpha_clip=[0, np.inf], log=True):
        minibatch_size = self.train_kwargs["minibatch_size"]
        num_epochs = self.train_kwargs["num_epochs"]
        log_every = self.train_kwargs["log_every"]
        norm_adv = self.train_kwargs["norm_adv"]
        clip_adv = self.train_kwargs["clip_adv"]
        clip_U = self.train_kwargs["clip_U"]
        if method == "exploration":
            op = self.ex_train
            log = False
        elif method == "p_improve_ppo":
            op = self.p_ppo_train
            summary = self.p_ppo_summary
        elif method in {"variance_constrain_kl", 
                        "variance_constrain_ratio", 
                        "p_improve_ppo2_kl",
                        "p_improve_ppo2_ratio"}:
            op = self.vc_train
            summary = self.vc_summary

        if clip_adv > 0:
            AV = np.clip(AV, -clip_adv, clip_adv)
        if norm_adv:
            AV = (AV - AV.mean()) / (AV.std() + 1e-8)
            Uv = Uv / (AV.var() + 1e-10)
            Uq = Uq / (AV.var() + 1e-10)
        if clip_U > 0:
            Uv = np.clip(Uv, -clip_U, clip_U)
            Uq = np.clip(Uq, -clip_U, clip_U)

        #attention
        Uv = np.maximum(Uv, 1e-8)
        Uq = np.maximum(Uq, 1e-8)
        input_list = [state, action, AV, Uv, Uq, oldneglogp]

        optimization_dict = {self.lr:learning_rate, self.cliprange:cliprange}
        if method in {"variance_constrain_kl", "variance_constrain_ratio", "exploration"}:
            assert alpha != None
            optimization_dict[self.alpha] = alpha

        for i in range(num_epochs):
            batch_generator = batch_gen(minibatch_size, *input_list)
            for items in batch_generator:
                if items == None:
                    break
                train_dict = dict(zip(self.input_list, items))
                if method in {"p_improve_ppo2", "p_improve_ppo2_kl", "p_improve_ppo2_ratio"}:
                    train_dict[self.cliprange] = cliprange
                    p_improve_ppo = self.session.run(self.p_improve_ppo, feed_dict=train_dict)
                    optimization_dict[self.alpha] = np.clip(p_improve_ppo, alpha_clip[0], alpha_clip[1])
                train_dict.update(optimization_dict)
                self.session.run(op, feed_dict=train_dict)
            if log:
                if log_every > 0 and i % log_every == 0:
                    valid_dict = dict(zip(self.input_list, input_list))
                    if method in {"p_improve_ppo2", "p_improve_ppo2_kl", "p_improve_ppo2_ratio"}:
                        valid_dict[self.cliprange] = cliprange
                        p_improve_ppo = self.session.run(self.p_improve_ppo, feed_dict=valid_dict)
                        optimization_dict[self.alpha] = np.clip(p_improve_ppo, alpha_clip[0], alpha_clip[1])
                    valid_dict.update(optimization_dict)
                    valid_log = self.session.run(summary, feed_dict=valid_dict)
                    self.policy_writer.add_summary(valid_log, self.epoch_counter)
                self.epoch_counter += 1
        if log:
            if log_every == 0:
                valid_dict = dict(zip(self.input_list, input_list))
                if method in {"p_improve_ppo2", "p_improve_ppo2_kl", "p_improve_ppo2_ratio"}:
                    valid_dict[self.cliprange] = cliprange
                    p_improve_ppo = self.session.run(self.p_improve_ppo, feed_dict=valid_dict)
                    optimization_dict[self.alpha] = np.clip(p_improve_ppo, alpha_clip[0], alpha_clip[1])
                valid_dict.update(optimization_dict)
                valid_log = self.session.run(summary, feed_dict=valid_dict)                    
                self.policy_writer.add_summary(valid_log, self.iter_counter)
            self.iter_counter += 1
            
        feed_dict = dict(zip(self.input_list, input_list))
        feed_dict[self.cliprange] = cliprange
        return self.session.run([self.p_improve_ppo, self.ratio_change, self.approxkl], feed_dict=feed_dict)

def get_opt(loss, lr, max_grad):
    trainer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-5)
    grads_and_var = trainer.compute_gradients(loss)
    grads, var = zip(*grads_and_var)
    if max_grad is not None:
        grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad)
    grads_and_var = list(zip(grads, var))
    return trainer.apply_gradients(grads_and_var)




        
       
