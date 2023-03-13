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
from random import choice
from monte_carlo_tree_search import MCTS, Node
import numpy as np
from tqdm.notebook import tqdm
_TTTB = namedtuple("SimpleTree", "tup terminal")
LAYERS = 4
BUDGET = 300

# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class SimpleTree(_TTTB, Node):
    def find_children(board):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        return {
            board.make_move(i) for i in [0,1]
        }

    def find_random_child(board):
        if board.terminal:
            return None  # If the game is finished then no moves can be made
        empty_spots = [0,1]
        return board.make_move(choice(empty_spots))

    def reward(board):
        return board.cal_reward()
    
    def cal_reward(board, bounded=True):
        start = 1
        accum = 0
        for selection in board.tup:
            accum += selection * start
            start *= 0.5
        if bounded:
            return np.random.beta(accum + 1, accum + 1)
        return np.random.normal(0.01 - 0.01 * accum , accum) ## Trap

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

    def to_pretty_string(board):
        to_char = lambda v: ("X" if v is True else ("O" if v is False else " "))
        rows = [
            [to_char(board.tup[3 * row + col]) for col in range(3)] for row in range(3)
        ]
        return (
            "\n  1 2 3\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )


def play_game():
    all_tree_type = ['bandit', 'uct', 'uct_normal', 'uct_v', 'maxmedian', 'random', 'epsilon_greedy', 'sp_mcts']
    trees = {}
    for tree_name in all_tree_type:
        trees[tree_name] = MCTS(budget=BUDGET, select_type=tree_name)
    print(trees)
    # tree_bandit = MCTS(budget=BUDGET, select_type='bandit')
    # tree_uct = MCTS(budget=BUDGET, select_type='uct')
    # tree_uct_normal(budget=BUDGET, select_type='uct_normal')
    result = {}
    times = 100
    trial_per_time = 100
    # bandit_record = [[] for _ in range(times)]
    # uct_record = [[] for _ in range(times)]
    # print(board.to_pretty_string())
    # for tree in trees.values():
    #     for i in range(BUDGET):
    #         board = new_simpletree()
    #         tree.do_rollout(board, t = i)
    ## Evaluation
    for tree in trees.values():
        result[tree.select_type] = [[] for _ in range(times)]
        for time in tqdm(range(times)):
            ## Train a brand new MCTS
            for j in range(BUDGET):
                board = new_simpletree()
                tree.do_rollout(board, t = j)
            ## After train, run trial per times experiments
            for i in range(trial_per_time):
                board = new_simpletree()
                while True:
                    if tree.select_type == 'bandit':
                        board = tree.choose(board, trial_per_time - i, 'bandit')
                        # board = tree.choose(board)
                    else:
                        board = tree.choose(board)
                    if board.terminal:
                        reward = board.reward()
                        result[tree.select_type][time].append(reward)
                        break
            ## Reset the tree after trial_per_time experiments
            tree.__init__(select_type=tree.select_type, budget=tree.budget)

    return trees, result

def new_simpletree():
    return SimpleTree(tup=(), terminal=False)

trees, result = play_game()

# %%
record_mean = {}
all_tree_type = ['bandit', 'uct', 'uct_normal', 'uct_v', 'maxmedian', 'random', 'epsilon_greedy', 'sp_mcts']
for tree_name in all_tree_type:
    record = result[tree_name]
    record_mean[tree_name] = np.mean([np.max(i) for i in result[tree_name]])
# %%
record_mean
# %%
bandit_record
# %%
bandit_record
# %%
for key, value in tree_bandit.models.items():
    print(key)
    print(value.prediction(100))
# %%
tee_uct.Q
# %%
board = new_simpletree()
board = tree_bandit._bandit_select(board,1)
board
# %%
board.cal_reward()
# %%
