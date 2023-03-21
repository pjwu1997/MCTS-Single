# %%
"""
An example implementation of the abstract Node class for use in MCTS
If you run this file then you can play against the computer.
A tic-tac-toe board is represented as a tuple of 9 values, each either None,
True, or False, respectively meaning 'empty', 'X', and 'O'.
The board is indexed by row:
0 1 2
3 4 5
6 7 8
For example, this game board
O - X
O X -
X - -
corrresponds to this tuple:
(False, None, True, False, True, None, True, None, None)
"""
# %%
from collections import namedtuple
from random import choice, seed
from monte_carlo_tree_search import MCTS, Node
import numpy as np
# from tqdm.notebook import tqdm
import tqdm
import matplotlib.pyplot as plt
import json
from collections import defaultdict

with open('config.json', 'r') as f:
    config = json.load(f)

_TTTB = namedtuple("SimpleTree", "tup terminal")
k_ary = 2
LAYERS = 3
BUDGET = 10000


# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class SimpleTree(_TTTB, Node):
    def find_children(board):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        return {
            board.make_move(i) for i in range(k_ary)
        }

    def find_random_child(board):
        if board.terminal:
            return None  # If the game is finished then no moves can be made
        empty_spots = list(range(k_ary))
        return board.make_move(choice(empty_spots))

    def reward(board):
        return board.cal_reward()
    
    def cal_reward(board, bounded=True):
        start = 3
        accum = 2 if bounded else 0
        for selection in board.tup:
            accum += selection * start
            start *= 0.5
        if bounded:
            return np.random.beta(accum + 1, accum + 1)
        return np.random.normal(0.5, accum) ## Trap

    def is_terminal(board):
        return board.terminal

    def make_move(board, selection):
        tup = board.tup + (selection,)
        # tup = board.tup[:index] + (board.turn,) + board.tup[index + 1 :]
        ## turn = not board.turn
        ## winner = _find_winner(tup)
        # is_terminal = (winner is not None) or not any(v is None for v in tup)
        is_terminal = (len(tup) == LAYERS)
        return SimpleTree(tup, is_terminal)

## Default Binary Tree
def play_game():
    global LAYERS
    all_tree_type = ['bandit', 'uct', 'uct_normal', 'uct_v', 'maxmedian', 'random', 'epsilon_greedy', 'sp_mcts', 'qomax']
    trees = {}
    result = defaultdict(dict)
    for layer in config['layers']:
        LAYERS = layer
        for times in config['times']:
            for trial_per_time in config['trials']:
                for budget in config['budget']:
                    budget = (k_ary ** layer) * budget
                    for tree_name in all_tree_type:
                        try:
                            trees[tree_name] = MCTS(budget=budget, select_type=tree_name, k_ary = k_ary, layers=layer)
                        except BaseException as error:
                            break
                    ## Evaluation
                    for tree in trees.values():
                        result[tree.select_type][str((layer, times, trial_per_time, budget))] = [[] for _ in range(times)]
                        for time in tqdm.trange(times):
                            ## Train a brand new MCTS
                            for j in range(budget):
                                board = new_simpletree()
                                tree.do_rollout(board, t = j)
                            ## After train, run trial per times experiments
                            for i in range(trial_per_time):
                                np.random.seed(i)
                                seed(i)
                                board = new_simpletree()
                                while True:
                                    if tree.select_type == 'bandit':
                                        board = tree.choose(board, trial_per_time - i, 'bandit')
                                        # board = tree.choose(board)
                                    else:
                                        board = tree.choose(board)
                                    if board.terminal:
                                        reward = board.reward()
                                        result[tree.select_type][str((layer, times, trial_per_time, budget))][time].append(reward)
                                        break
                            ## Reset the tree after trial_per_time experiments
                            if time != times - 1:
                                tree.__init__(select_type=tree.select_type, budget=tree.budget, seed=time, k_ary = k_ary, layers=LAYERS)

    return trees, result

def new_simpletree():
    return SimpleTree(tup=(), terminal=False)

trees, result = play_game()

with open("result.json", "w") as outfile:
    json.dump(result, outfile)


## %%
# record_mean = {}
# all_tree_type = ['bandit', 'uct', 'uct_normal', 'uct_v', 'maxmedian', 'random', 'epsilon_greedy', 'sp_mcts']
# # all_tree_type = ['bandit', 'random']
# for tree_name in all_tree_type:
#     record = result[tree_name]
#     record_mean[tree_name] = (np.mean([np.max(i) for i in result[tree_name]]), np.std([np.max(i) for i in result[tree_name]]))