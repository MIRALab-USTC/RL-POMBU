import numpy as np
from tqdm import tqdm
import gym
class Sampler:
    def __init__(self, env, model, dataset, num_episode=10):
        self.num_episode = num_episode
        self.env = env
        self.horizon = env.horizon
        self.dataset = dataset
        self.model = model

    def sample(self, num_episode=None, pi=None, noise=1, store=True):
        num_episode = self.num_episode if num_episode==None else num_episode
        if store:
            print("sampling......")
        else:
            print("evaluating......")

        rets = []
        for _ in tqdm(range(num_episode)):
            ret = 0
            state = self.env.reset()
            if store:
                self.dataset.add_first_state(state)
            for i in range(self.horizon):
                if pi == None:
                    action = self.env.action_space.sample()
                else:
                    state = self.model.modify_state(state, i)
                    action = pi.step(state, noise=noise)
                    action = self.model.recover_action(action)
                next_state, reward, done, _ = self.env.step(action)
                ret += reward
                if store:
                    self.dataset.add(action, next_state, reward)
                if done:
                    break
                else:
                    state = next_state
            if store:  
                self.dataset.flush()
            rets.append(ret)
        mean_ret = np.mean(rets)
        return mean_ret

    def sample_for_MF(self, pi, V_estimator, num_episode=None):
        num_episode = self.num_episode if num_episode==None else num_episode

        Us = us = [[0]*self.horizon] * num_episode
        us = [[0]*self.horizon] * num_episode
        Vs, rs, dones, states, actions, neglogps = [], [], [], [], [], []
        
        for i in range(num_episode):
            Vs.append([])
            rs.append([])
            dones.append([])
            states.append([])
            actions.append([])
            neglogps.append([])

        for i in tqdm(range(num_episode)):
            state = self.env.reset()
            self.dataset.add_first_state(state)
            for j in range(self.horizon):
                state = self.model.modify_state(state, j)
                action, neglogp = pi.step_for_MF(state)
                V = V_estimator.value_for_MF(state)

                states[i].append(state)
                Vs[i].append(V)
                actions[i].append(action)
                neglogps[i].append(neglogp)

                action = self.model.recover_action(action)
                
                next_state, reward, done, _ = self.env.step(action)
                rs[i].append(reward)
                dones[i].append(done)

                self.dataset.add(action, next_state, reward)
                if done:
                    break
                else:
                    state = next_state
            self.dataset.flush()
        mean_ret = np.mean(rs) * self.horizon
        print("mean_ret:", mean_ret)
        rs = self.model.modify_reward(np.array(rs))
        trs=[Us,Vs,us,rs,dones,states,actions,neglogps]
        for i in range(len(trs)):
            trs[i] = list(np.array(trs[i]).swapaxes(0,1))

        pi.log_return(0, mean_ret)
        trs = self.model.prepare_trs(trs, print_inf=False)
        return trs
