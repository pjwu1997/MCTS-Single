"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from abc import ABC, abstractmethod
from collections import defaultdict
import math
from functools import partial
import random
import numpy as np

from scipy.stats import norm,uniform
import scipy.stats as ss
import copy
from bisect import bisect, insort

class model:
    def __init__(self, budget, window_length=2):
        self.mean = 0
        self.std = 1
        self.distribution = 'normal'
        self.window_length = window_length
        self.record = []
        self.sliding_record = []
        self.std_numer = 1
        self.std_denom = 1
        self.t = 0
        self.budget = budget
        self.time_decay = True
    
    def prediction(self, predict_length, alpha=1):
        '''
        Create a max distribution based on model and sample from it.
        '''
        # print(self.mean + self.std * (np.sqrt((self.window_length - 1))))
        if self.time_decay:
            # print(predict_length)
            # print(self.budget)
            # pre_value = np.exp(-alpha * (self.budget - predict_length) / (self.budget))
            # print(pre_value)
            
            value = self.mean + np.exp(-alpha * (self.budget - predict_length) / (self.budget)) * self.std * np.sqrt(predict_length - 1)
        else:
            value = self.mean + self.std * np.sqrt(predict_length - 1)
        try:
            if len(value) > 0:
                return value[0]
        except:
            return value
        
    def getReward(self, value):
        """
        update mean/variance according to input value.
        """
        self.record.append(value)
        self.mean = max(self.record[-self.window_length:])
        if len(self.record) > self.window_length:
            std = np.std(self.record[-self.window_length:])
            self.std_numer = self.std_numer * 0.9 + std
            self.std_denom = self.std_denom * 0.9 + 1
            self.std = self.std_numer / self.std_denom
        else:
            self.std = 1

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1, select_type='uct', budget=1000):
        self.Q = defaultdict(float)  # total reward of each node
        self.N = defaultdict(float)  # total visit count for each node
        self.record = defaultdict(list)
        self.sorted_reward = defaultdict(list)
        self.models = defaultdict(partial(model,budget=budget))
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
        self.select_type = select_type
        self.budget = budget
        self.epsilon = 0.1
        np.random.seed(1000)

    def choose(self, node, t=1, mode='uct'):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward
        
        def bandit_score(n):
            if self.N[n] == 0:
                return float("-inf")
            return self.models[n].prediction(t)
        
        def uct_normal_score(n, t):
            if self.N[n] < 2:
                return float("-inf") # uct_normal needs at least 2 observations
            mean = self.Q[n] / self.N[n]
            squared_rewards = np.array(self.record[n]) ** 2
            sv = (squared_rewards - self.N[n] * (mean ** 2)) / (self.N[n] - 1)
            c = np.sqrt(16 * sv * np.log(t - 1) / self.N[n])
            return mean + c

        if mode == 'uct':
            return max(self.children[node], key=score)
        elif mode == 'bandit':
            return max(self.children[node], key=bandit_score)
        elif mode == 'uct_normal':
            return max(self.children[node], key=uct_normal_score)

    def do_rollout(self, node, t):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node, t)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        # print(reward)
        self._backpropagate(path, reward)

    def _select(self, node, t):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            # print(unexplored)
            # print()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            if self.select_type == 'uct':
                node = self._uct_select(node)  # descend a layer deeper
            elif self.select_type == 'bandit':
                node = self._bandit_select(node, t)
            elif self.select_type == 'uct_normal':
                node = self._uct_normal_select(node, t)
            elif self.select_type == 'uct_v':
                node = self._uct_v_select(node, t)
            elif self.select_type == 'maxmedian':
                node = self._maxmedian_select(node, t)
            elif self.select_type == 'random':
                node = self._random_select(node, t)
            elif self.select_type == 'epsilon_greedy':
                node = self._epsilon_greedy_select(node, t)
            

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    # def _simulate(self, node):
    #     "Returns the reward for a random simulation (to completion) of `node`"
    #     invert_reward = True
    #     while True:
    #         if node.is_terminal():
    #             reward = node.reward()
    #             return 1 - reward if invert_reward else reward
    #         node = node.find_random_child()
    #         invert_reward = not invert_reward
    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        while True:
            if node.is_terminal():
                reward = node.reward()
                return reward
                # return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            # invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            self.record[node].append(reward)
            self.models[node].getReward(reward)
            insort(self.sorted_reward[node], reward)
            # reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)
    
    def _bandit_select(self, node, t):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        # log_N_vertex = math.log(self.N[node])

        def bandit(n):
            "Upper confidence bound for trees"
            return self.models[n].prediction(self.budget - t)


        return max(self.children[node], key=bandit)
    
    def _uct_normal_select(self, node, t):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct_normal(n):
            "Upper confidence bound for trees"
            if self.N[n] < 8 * np.log(t):
                return float("inf")
            else:
                mean = self.Q[n] / self.N[n]
                squared_rewards = sum(np.array(self.record[n]) ** 2)
                sv = (squared_rewards - self.N[n] * (mean ** 2)) / (self.N[n] - 1)
                c = np.sqrt(16 * sv * np.log(t - 1) / self.N[n])
                return mean + c
        return max(self.children[node], key=uct_normal)

    def _uct_v_select(self, node, t):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct_v(n):
            "Upper confidence bound for trees"
            if self.N[n] < 2:
                return float("inf")
            else:
                mean = self.Q[n] / self.N[n]
                squared_rewards = sum(np.array(self.record[n]) ** 2)
                sv = (squared_rewards - self.N[n] * (mean ** 2)) / (self.N[n] - 1)
                fac1 = np.sqrt(3 * sv * np.log(1.2 * t) / self.N[n])
                fac2 = 3 * np.log(1.2 * t) / self.N[n]
                return mean + fac1 + fac2
        return max(self.children[node], key=uct_v)
    
    def _maxmedian_select(self, node, t):
        assert all(n in self.children for n in self.children[node])
        m = np.inf
        for n in self.children[node]:
            if self.N[n] < m:
                m = self.N[n]

        def explo_func(t):
            return 1 / t
        
        def rd_argmax(vector):
            """
            Compute random among eligible maximum indices
            :param vector: np.array
            :return: int, random index among eligible maximum indices
            """
            m = np.amax(vector)
            indices = np.nonzero(vector == m)[0]
            return np.random.choice(indices)

        def maxmedian(n):
            if self.N[n] < 1:
                return float("inf")
            else:
                order = np.ceil(self.N[n]/m).astype(np.int32)
                idx = self.sorted_reward[n][-order]
                return idx
        
        if np.random.binomial(1, explo_func(t)) == 1:
            #print(list(self.children[node]))
            return_node = random.choice(list(self.children[node]))
            #print(return_node)
            return return_node
        else:
            return_node = max(self.children[node], key=maxmedian)
            #print(return_node)
            return return_node
    
    def _random_select(self, node, t):
        assert all(n in self.children for n in self.children[node])
        return random.choice(list(self.children[node]))
    
    def _epsilon_greedy_select(self, node, t):
        assert all(n in self.children for n in self.children[node])
        def average(n):
            return self.Q[n] / self.N[n]
        if np.random.rand() < self.epsilon:
            return random.choice(list(self.children[node]))
        else:
            return max(self.children[node], key=average)


                    
class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True