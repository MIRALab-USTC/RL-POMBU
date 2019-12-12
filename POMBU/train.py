import gym
import json
import os
import numpy as np
from tqdm import tqdm
from tensorflow import nn 

from POMBU.policy.stochastic_policy import StochasticPolicy
from POMBU.dataset import Dataset
from POMBU.sampler import Sampler
from POMBU.V_estimators.single_V import SingleV
from POMBU.models.deterministic_ensemble import DeterministicEnsemble
from POMBU.models.point_2d_noisy_model import NoisyEnsemble
from POMBU.utils import set_seed
from POMBU.envs.utils import get_env

from baselines import logger
from baselines.common import explained_variance

def train(train_kwargs):
    running_kwargs = train_kwargs["running_kwargs"]
    model_kwargs = train_kwargs["model_kwargs"]
    rollout_kwargs = train_kwargs["rollout_kwargs"]
    policy_kwargs = train_kwargs["policy_kwargs"]
    UV_kwargs = train_kwargs["UV_kwargs"]

    env_name = running_kwargs["env_name"]
    exp_name = running_kwargs["exp_name"]
    seed = running_kwargs["seed"]
    env = get_env(env_name)
    set_seed(env, seed)

    save_dir = running_kwargs["save_dir"]
    logger.configure(save_dir)
    algo = running_kwargs["algo"]

    running_kwargs.update(running_kwargs[algo])
    max_dataset_size = running_kwargs["max_dataset_size"] if "max_dataset_size" in running_kwargs else 1e6

    add_index = rollout_kwargs["add_index"]

    print_trs_inf = rollout_kwargs["print_trs_inf"]
    log_trs_every = rollout_kwargs["log_trs_every"]
    log_num = rollout_kwargs["log_num"]
    gamma = rollout_kwargs["gamma"]

    UV_lr = UV_kwargs["optimization_kwargs"]["learning_rate"]
    policy_lr = policy_kwargs["optimization_kwargs"]["learning_rate"]
    policy_cliprange = policy_kwargs["optimization_kwargs"]["cliprange"]
    
    k_model = model_kwargs["network_kwargs"]["k_model"]

    dataset = Dataset(env, k_model, gamma, save_dir, max_dataset_size) 
    extra_n = 1 if add_index else 0
    pi = StochasticPolicy(env, extra_n, kwargs=policy_kwargs, save_dir=save_dir)
    V_estimator = SingleV(env, extra_n, kwargs=UV_kwargs, save_dir=save_dir)
    if exp_name != "point_2d":
        model = DeterministicEnsemble(env, pi, dataset, V_estimator, save_dir, model_kwargs, rollout_kwargs)
    else:
        if running_kwargs["model_type"] == "nn":
            model = DeterministicEnsemble(env, pi, dataset, V_estimator, save_dir, model_kwargs, rollout_kwargs)
        else:
            model = NoisyEnsemble(env, pi, dataset, V_estimator, save_dir, model_kwargs, rollout_kwargs)
        log_ij = running_kwargs["log_ij"]
        n_epoch = running_kwargs["n_epoch"]
        resolution = running_kwargs["resolution"]
        plot_interval = running_kwargs["plot_interval"]
    
    sampler = Sampler(env, model, dataset)

    
    if algo == "PPO":
        n_iter = running_kwargs["n_iter"]
        n_sample = running_kwargs["n_sample"]
        for i in range(n_iter):
            print("\n\n\n\niteration:%d\n"%i)
            trs = sampler.sample_for_MF(pi, V_estimator, n_sample)
            print("U_explained_variance:", explained_variance(trs['newU'], trs['oldU']))
            print("V_explained_variance:", explained_variance(trs['newV'], trs['oldV']))
            U, V = V_estimator.train(trs["state"], trs["newU"], trs["newV"], UV_lr(i,0))
            print("U_explained_variance:", explained_variance(trs['newU'], U))
            print("V_explained_variance:", explained_variance(trs['newV'], V))
            print(pi.train(trs["state"], trs["action"], trs["AV"], trs["oldU"], trs["newU"], trs["neglogp"], "modified_A", policy_lr(i,0), policy_cliprange(i,0), 0))

    elif algo == "POMBU":
        save_every = running_kwargs["save_every"]
        n_sample = running_kwargs["n_sample"]
        n_update = running_kwargs["n_update"]
        n_iter = running_kwargs["n_iter"]
        ex_method = running_kwargs["ex_method"]
        ex_noise = running_kwargs["ex_noise"]
        report_return_during_iteration = running_kwargs["report_return_during_iteration"]
        if "uncertainty" in ex_method:
            ex_alpha = running_kwargs["ex_alpha"]
            ex_width = running_kwargs["ex_width"]
            ex_keep = running_kwargs["ex_keep"]
            ex_cliprange = running_kwargs["ex_cliprange"]
            ex_lr = running_kwargs["ex_lr"]
            ex_iter_per_update = running_kwargs["ex_iter_per_update"]
            if ex_method == "uncertainty_each_model":
                ex_n_update = k_model
            else:
                ex_n_update = running_kwargs["ex_n_update"]


        op_method = running_kwargs["op_method"]
        op_width = running_kwargs["op_width"]
        if "variance_constrain" in op_method:
            op_alpha = running_kwargs["op_alpha"]
            op_times = running_kwargs["op_times"]
        elif "p_improve_ppo2" in op_method:
            op_alpha_clip = running_kwargs["op_alpha_clip"]
            op_times = running_kwargs["op_times"]

        if "kl" in op_method:
            op_kl_feasible_interval = running_kwargs["op_kl_feasible_interval"]
        elif "ratio" in op_method:
            op_ratio_feasible_interval = running_kwargs["op_ratio_feasible_interval"]

        reinitialize_std_before = running_kwargs["reinitialize_std_before"]

        for i in range(n_iter):
            print("\n\n\niteration: %d\n\n\n"%i) 
            if i < reinitialize_std_before:
                pi.reinit_std()

            if ex_method == "simple" or i==0 or ex_n_update == 0:
                mean_r = sampler.sample(n_sample, pi, ex_noise)
                print("\n\nmean_return:%f\n\n"%mean_r)
            elif "uncertainty" in ex_method:
                if ex_keep:
                    n_each = n_sample // (ex_n_update + 1)
                    n_keep = n_each + n_sample % (ex_n_update + 1)
                else:
                    n_each = n_sample // ex_n_update
                    n_keep = n_sample % ex_n_update
                if n_keep > 0:
                    mean_r = sampler.sample(n_keep, pi, ex_noise)
                    print("\n\nmean_return:%f\n\n"%mean_r)
                pi.save()
                for j in range(ex_n_update):
                    print("\n\ntrain exploration policy...")
                    if ex_method == "uncertainty_random_weight":
                        index_p = np.random.random(k_model)
                        index_p /= index_p.sum()
                    elif ex_method == "uncertainty_each_model":
                        index_p = np.zeros(k_model)
                        index_p[j] = 1
                    else:
                        index_p = np.full(k_model, 1/k_model)
                    for _ in range(ex_iter_per_update):
                        trs, _ = model.rollout(ex_width, index_p=index_p)
                        print(pi.train(trs["state"], 
                            trs["action"], 
                            trs["AV"], 
                            trs["oldU"], 
                            trs["newU"], 
                            trs["neglogp"], 
                            "exploration", 
                            ex_lr(i),                               
                            ex_cliprange(i), 
                            ex_alpha, 
                            log=False))
                    if n_each > 0:
                        mean_r = sampler.sample(n_each, pi, ex_noise)
                        print("\n\nmean_return:%f\n\n"%mean_r)
                    if ex_method != "uncertainty_multiple_update":
                        pi.load()
                if ex_method == "uncertainty_multiple_update":
                    pi.load()



            if save_every > 0 and i % save_every == 0:
                dataset.save(i)
                pi.save("policy_%d"%i)

            dataset.print_inf()
            model.train()
            if report_return_during_iteration:
                mean_r1 = sampler.sample(20, pi, noise=0, store = False)
                mean_r2 = sampler.sample(20, pi, noise=1, store = False)
                print("mean_r:%f, mean_r_stochastic:%f"%(mean_r1, mean_r2))

            for j in range(n_update):
                print("\n\niteration:%d\n"%j)
                
                if env_name == "point_2d" and log_ij(i,j):
                    model.plot_img(n_epoch, resolution, "Q", plot_interval)
                    #trs, mean_r_model = model.rollout(op_width, print_inf=print_trs_inf, init_states=init_states)
                
                if log_trs_every> 0 and j % log_trs_every == 0:
                    trs, mean_r_model = model.rollout(op_width, print_inf=print_trs_inf, name="iter%d_update%d"%(i,j), log_num=log_num)
                else:
                    trs, mean_r_model = model.rollout(op_width, print_inf=print_trs_inf)

                if env_name != "point_2d":
                    U, V = V_estimator.train(trs["state"], trs["newU"], trs["newV"], UV_lr(i,j))

                    print("mean Upred:", U.mean(), "\tmean U:", trs['newU'].mean(), "\tU_explained_variance:", explained_variance(trs['newU'], U))
                    print("mean Vpred:", V.mean(), "\tmean V:", trs['newV'].mean(), "\tV_explained_variance:", explained_variance(trs['newV'], V))

                if "p_improve_ppo2" in op_method:
                    p_improve_ppo, ratio_change, approxkl = pi.train(trs["state"], 
                                                                    trs["action"], 
                                                                    trs["AV"], 
                                                                    trs["oldU"], 
                                                                    trs["newU"], 
                                                                    trs["neglogp"], 
                                                                    op_method, 
                                                                    policy_lr(i,j), 
                                                                    policy_cliprange(i,j), 
                                                                    alpha_clip=op_alpha_clip)
                    print("current alpha_clip: (%f,%f)" % (op_alpha_clip[0], op_alpha_clip[1]))
                elif op_method == "p_improve_ppo":
                    p_improve_ppo, ratio_change, approxkl = pi.train(trs["state"], 
                                                                    trs["action"], 
                                                                    trs["AV"], 
                                                                    trs["oldU"], 
                                                                    trs["newU"], 
                                                                    trs["neglogp"], 
                                                                    op_method, 
                                                                    policy_lr(i,j), 
                                                                    policy_cliprange(i,j))

                else:
                    p_improve_ppo, ratio_change, approxkl = pi.train(trs["state"], 
                                                                    trs["action"], 
                                                                    trs["AV"], 
                                                                    trs["oldU"], 
                                                                    trs["newU"], 
                                                                    trs["neglogp"], 
                                                                    op_method, 
                                                                    policy_lr(i,j), 
                                                                    policy_cliprange(i,j), 
                                                                    op_alpha)
                    print("current alpha: %f" % op_alpha)
                
                print(p_improve_ppo, ratio_change, approxkl)
                if op_method == "variance_constrain_ratio":
                    if ratio_change > op_ratio_feasible_interval[1]:
                        op_alpha *= op_times 
                    elif ratio_change < op_ratio_feasible_interval[0]:
                        op_alpha /= op_times 
                elif op_method == "variance_constrain_kl":
                    if approxkl > op_kl_feasible_interval[1]:
                        op_alpha *= op_times 
                    elif approxkl < op_kl_feasible_interval[0]:
                        op_alpha /= op_times 
                elif op_method == "p_improve_ppo2_ratio":
                    if ratio_change > op_ratio_feasible_interval[1]:
                        for k in [0,1]:
                            op_alpha_clip[k] *= op_times 
                    elif ratio_change < op_ratio_feasible_interval[0]:
                        for k in [0,1]:
                            op_alpha_clip[k] /= op_times
                elif op_method == "p_improve_ppo2_kl":
                    if approxkl > op_kl_feasible_interval[1]:
                        for k in [0,1]:
                            op_alpha_clip[k] *= op_times 
                    elif approxkl < op_kl_feasible_interval[0]:
                        for k in [0,1]:
                            op_alpha_clip[k] /= op_times

                if report_return_during_iteration:
                    model.evaluate_policy()
                    mean_r1 = sampler.sample(20, pi, noise=0, store = False)
                    mean_r2 = sampler.sample(20, pi, noise=1, store = False)
                    print("mean_r_model:%f, mean_r:%f, mean_r_stochastic:%f"%(mean_r_model, mean_r1, mean_r2))
                    pi.log_return(mean_r_model, mean_r1, mean_r2, i*n_update+j)

            if report_return_during_iteration == False:
                    mean_r1 = sampler.sample(20, pi, noise=0, store = False)
                    mean_r2 = sampler.sample(20, pi, noise=1, store = False)
                    print("mean_r_model:%f, mean_r:%f, mean_r_stochastic:%f"%(mean_r_model, mean_r1, mean_r2))
                    pi.log_return(mean_r_model, mean_r1, mean_r2, i)
                
            logger.record_tabular("Iteration", i)
            logger.record_tabular("TotalSteps", dataset.get_total_steps())
            logger.record_tabular("AverageReturn", mean_r2)
            logger.record_tabular("AveragedReturnDeterministic", mean_r1)
            logger.dump_tabular()
        
    elif algo == "train_model":
        load_step = running_kwargs["load_step"]
        dataset.load(load_step)
        dataset.print_inf()
        model.train()
