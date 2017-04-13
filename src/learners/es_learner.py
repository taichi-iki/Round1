# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from learners.base import BaseLearner

import numpy as np

class network(object):
    def __init__(self, time_step, embed_dim, char_count=256, sigma=0.1):
        self.embed_dim = embed_dim
        self.input_dim = 3*time_step*embed_dim
        self.char_count = char_count
        self.unit_dim_list = [
                768,
                512,
                384,
                char_count,
            ]
        self.sigma = sigma
        self.base_embed = None
        self.base_W = None
        self.base_b = None
        self.embed = None
        self.W = None
        self.b = None
    
    def set_base_weight(self, data=None):    
        if data is None:
            self.base_embed = [
                    np.random.normal(size=(self.char_count+1, self.embed_dim)), # env
                    np.random.normal(size=(self.char_count+1, self.embed_dim)), # agent-action
                    np.random.normal(size=(2+1, self.embed_dim)), # reward
                ]
            self.base_W = []
            self.base_b = []
            last_dim = self.input_dim
            for d in self.unit_dim_list:
                self.base_W.append(np.random.normal(size=(d, last_dim)))
                self.base_b.append(np.random.normal(size=(d,)))
                last_dim = d
        else:
            self.base_embed = data['embed']
            self.base_W = data['W']
            self.base_b = data['b']
        
    def get_base_weight(self):
        data = {}
        data['embed'] = self.base_embed
        data['W'] = self.base_W
        data['b'] = self.base_b
        return data
    
    def sample_genotype(self, seed):
        tmp_state = np.random.get_state()
        np.random.seed(seed)
        self.embed = []
        self.W = []
        self.b = []
        for i in range(len(self.base_embed)):
            self.embed.append(self.base_embed[i] + self.sigma*np.random.normal(size=self.base_embed[i].shape))
        for i in range(len(self.base_W)):
            self.W.append(self.base_W[i] + self.sigma*np.random.normal(size=self.base_W[i].shape))
        for i in range(len(self.base_b)):
            self.b.append(self.base_b[i] + self.sigma*np.random.normal(size=self.base_b[i].shape))
        np.random.set_state(tmp_state)
        
    def move_base_weight(self, reward_seed_list, alpha=0.01, eps=1e-6):
        n = len(reward_seed_list)
        rew = np.asarray([t[0] for t in reward_seed_list])
        rew_mean = rew.mean()
        rew_std = np.maximum(rew.std(), eps)
        for reward, seed in reward_seed_list:
            factor = ((reward - rew_mean)/rew_std)*alpha/(n*self.sigma) 
            tmp_state = np.random.get_state()
            np.random.seed(seed)
            for i in range(len(self.base_embed)):
                self.base_embed[i] += factor*np.random.normal(size=self.base_embed[i].shape)
            for i in range(len(self.base_W)):
                self.base_W[i] += factor*np.random.normal(size=self.base_W[i].shape)
            for i in range(len(self.base_b)):
                self.base_b[i] += factor*np.random.normal(size=self.base_b[i].shape)
            np.random.set_state(tmp_state)
    
    def forward(self, x_env, x_action, x_reward):
        x = np.concatenate([
                self.embed[0].take(x_env, axis=0, mode='clip').flatten(),
                self.embed[1].take(x_action, axis=0, mode='clip').flatten(),
                self.embed[2].take(x_reward, axis=0, mode='clip').flatten(),
            ], axis=0)
        for i in range(len(self.unit_dim_list)-1):
            x = np.maximum(((self.W[i]*x[None, :]).sum(axis=1) + self.b[i]), 0)
        i = len(self.unit_dim_list)-1
        x = (self.W[i]*x[None, :]).sum(axis=1) + self.b[i]
        return x.argmax()

class ESLearner(BaseLearner):
    def __init__(self):
        self.time_step_max = 25
        self.histo_env = [-1]*self.time_step_max 
        self.histo_action = [-1]*self.time_step_max 
        self.histo_reward = [-1]*self.time_step_max
        self.net = network(self.time_step_max, 50)
        self.current_step = 0
        self.total_reward = 0
    
    def reward(self, reward):
        self.histo_reward.pop(0)
        self.histo_reward.append(reward)
        self.total_reward += reward
        self.current_step += 1  
    
    def next(self, input):
        self.histo_env.pop(0)
        self.histo_env.append(ord(input))
        action = self.net.forward(self.histo_env, self.histo_action, self.histo_reward)
        self.histo_action.pop(0)
        self.histo_action.append(action)
        return unichr(action)
