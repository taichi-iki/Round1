# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from learners.base import BaseLearner

import numpy as np

class network(object):
    def __init__(self, time_step, embed_dim, char_count=256, sigma=0.01):
        self.embed_dim = embed_dim
        self.input_dim = 3*time_step*embed_dim
        self.char_count = char_count
        self.unit_dim_list = [
                1024,
                512,
                256,
                char_count,
            ]
        self.sigma = sigma
        self.embed = None
        self.W = None
        self.b = None
    
    def set_weight(self, data=None):    
        if data is None:
            scale=0.1
            self.embed = [
                    np.asarray(np.random.normal(size=(self.char_count+1, self.embed_dim)), dtype='float32')*scale, # env
                    np.asarray(np.random.normal(size=(self.char_count+1, self.embed_dim)), dtype='float32')*scale, # agent-action
                    np.asarray(np.random.normal(size=(2+1, self.embed_dim)), dtype='float32')*scale, # reward
                ]
            self.W = []
            self.b = []
            last_dim = self.input_dim
            for d in self.unit_dim_list:
                self.W.append(np.asarray(np.random.normal(size=(d, last_dim)), dtype='float32')*scale)
                self.b.append(np.asarray(np.random.normal(size=(d,)), dtype='float32')*scale)
                last_dim = d
        else:
            self.embed = data['embed']
            self.W = data['W']
            self.b = data['b']
    
    def set_genotype_weight(self, base_weight, seed):
        tmp_state = np.random.get_state()
        np.random.seed(seed)
        self.embed = []
        self.W = []
        self.b = []
        b_embed = base_weight['embed']
        b_W = base_weight['W']
        b_b = base_weight['b']
        for i in range(len(b_embed)):
            self.embed.append(b_embed[i] + self.sigma*np.asarray(np.random.normal(size=b_embed[i].shape), dtype='float32'))
        for i in range(len(b_W)):
            self.W.append(b_W[i] + self.sigma*np.asarray(np.random.normal(size=b_W[i].shape), dtype='float32'))
        for i in range(len(b_b)):
            self.b.append(b_b[i] + self.sigma*np.asarray(np.random.normal(size=b_b[i].shape), dtype='float32'))
        np.random.set_state(tmp_state)
        
    def get_weight(self):
        data = {}
        data['embed'] = self.embed
        data['W'] = self.W
        data['b'] = self.b
        return data
        
    def move_base_weight(self, reward_seed_list, alpha=0.01, eps=1e-6):
        n = len(reward_seed_list)
        rew = np.asarray([t[0] for t in reward_seed_list])
        rew_mean = rew.mean()
        rew_std = np.maximum(rew.std(), eps)
        for reward, seed in reward_seed_list:
            factor = ((reward - rew_mean)/rew_std)*alpha/(n*self.sigma) 
            tmp_state = np.random.get_state()
            np.random.seed(seed)
            for i in range(len(self.embed)):
                self.embed[i] += np.asarray(np.random.normal(size=self.embed[i].shape), dtype='float32')*factor
            for i in range(len(self.W)):
                self.W[i] += np.asarray(np.random.normal(size=self.W[i].shape), dtype='float32')*factor
            for i in range(len(self.b)):
                self.b[i] += np.asarray(np.random.normal(size=self.b[i].shape), dtype='float32')*factor
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
        self.time_step_max = 50
        self.histo_env = [-1]*self.time_step_max 
        self.histo_action = [-1]*self.time_step_max 
        self.histo_reward = [-1]*self.time_step_max
        self.net = network(self.time_step_max, 10)
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
        return chr(action)
