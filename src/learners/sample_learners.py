# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import random
from core.serializer import StandardSerializer, IdentitySerializer
from learners.base import BaseLearner

import string, copy

class BruteforceLearner(BaseLearner):
    def __init__(self):
        # self.windows_charcode_ignore_on()
        code_space = list(string.printable)
        print('initialized code space length=%d'%(len(code_space)), code_space)
        self.code_space = code_space
        self.last_reward = 0
        self.policy_list = [
                PolicyCorrectRepeat(self.code_space),
                PolicyMappingSearch(self.code_space),
                PolicyRandom(self.code_space),
            ]
        self.current_policy_id = 0
    
    def windows_charcode_ignore_on(self):
        import sys, codecs
        writer = codecs.getwriter(sys.stdout.encoding)(sys.stdout, errors='ignore')
        sys.stdout = writer
    
    def reward(self, reward):
        self.last_reward = reward

    def next(self, input):
        current_policy = self.policy_list[self.current_policy_id]
        if current_policy.expired:
            self.current_policy_id += 1
            if self.current_policy_id >= len(self.policy_list):
                self.current_policy_id = 0
            current_policy = self.policy_list[self.current_policy_id]
        output = current_policy.next(self.last_reward, input)
        print('current_policy_id', self.current_policy_id, 'last_reward', self.last_reward, 'input', input, 'output', output)
        return output
        
class PolicyCorrectRepeat(object):
    def __init__(self, code_space):
        self.expired = False
        self.code_space = code_space
        self.current_pos = 0
        self.last_correct_pos = -1
    
    def next(self, reward, input):
        if reward == 1:
            self.last_correct_pos = self.current_pos 
        else:
            self.current_pos = (self.current_pos + 1) % len(self.code_space)
            if self.last_correct_pos == self.current_pos:
                self.expired = True
        
        return self.code_space[self.current_pos]
        
class PolicyMappingSearch(object):
    def __init__(self, code_space):
        self.expired = False
        self.code_space = code_space
        self.map_candidate = {}
        for x in code_space:
            self.map_candidate[x] = copy.copy(code_space)
        self.map_found = {}
        self.last_pair = None
    
    def next(self, reward, input):
        if (not self.last_pair is None) and reward == 1:
            last_input, last_output = self.last_pair
            if not last_input in self.map_found:
                self.map_found[last_input] = last_output
                self.map_candidate[last_input] = []
        if input in self.map_found:
            output = self.map_found[input]
        else:
            if len(self.map_candidate[input]) > 0:
                output = self.map_candidate[input].pop()
            else:
                output = random.choice(self.code_space)
        left_candidate = 0
        for l in self.map_candidate.values():
            left_candidate += len(l)
        if left_candidate == 0:
            self.expired = True
        self.last_pair = (input, output)
        return output

class PolicyRandom(object):
    def __init__(self, code_space, max_time=5000):
        self.expired = False
        self.code_space = code_space
        self.left_time = max_time
    
    def next(self, reward, input):
        self.left_time -= 1
        if self.left_time <= 0:
            self.expired = True
        return random.choice(self.code_space)

class SampleRepeatingLearner(BaseLearner):
    def reward(self, reward):
        # YEAH! Reward!!! Whatever...
        pass

    def next(self, input):
        # do super fancy computations
        # return our guess
        return input


class SampleSilentLearner(BaseLearner):
    def __init__(self):
        self.serializer = StandardSerializer()
        self.silence_code = self.serializer.to_binary(' ')
        self.silence_i = 0

    def reward(self, reward):
        # YEAH! Reward!!! Whatever...
        self.silence_i = 0

    def next(self, input):
        output = self.silence_code[self.silence_i]
        self.silence_i = (self.silence_i + 1) % len(self.silence_code)
        return output


class SampleMemorizingLearner(BaseLearner):
    def __init__(self):
        self.memory = ''
        self.teacher_stopped_talking = False
        # the learner has the serialization hardcoded to
        # detect spaces
        self.serializer = StandardSerializer()
        self.silence_code = self.serializer.to_binary(' ')
        self.silence_i = 0

    def reward(self, reward):
        # YEAH! Reward!!! Whatever...
        # Now this robotic teacher is going to mumble things again
        self.teacher_stopped_talking = False
        self.silence_i = 0
        self.memory = ''

    def next(self, input):
        # If we have received a silence byte
        text_input = self.serializer.to_text(self.memory)
        if text_input and text_input[-2:] == '  ':
            self.teacher_stopped_talking = True

        if self.teacher_stopped_talking:
            # send the memorized sequence
            output, self.memory = self.memory[0], self.memory[1:]
        else:
            output = self.silence_code[self.silence_i]
            self.silence_i = (self.silence_i + 1) % len(self.silence_code)
        # memorize what the teacher said
        self.memory += input
        return output
