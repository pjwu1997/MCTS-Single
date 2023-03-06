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

_TTTB = namedtuple("SimpleTree", "tup terminal")
LAYERS = 5
BUDGET = 10000

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
        # empty_spots = [i for i, value in enumerate(board.tup) if value is None]
        empty_spots = [0,1]
        return board.make_move(choice(empty_spots))

    def reward(board):
        return board.cal_reward()
        # if not board.terminal:
        #     raise RuntimeError(f"reward called on nonterminal board {board}")
        # if board.winner is board.turn:
        #     # It's your turn and you've already won. Should be impossible.
        #     raise RuntimeError(f"reward called on unreachable board {board}")
        # if board.turn is (not board.winner):
        #     return 0  # Your opponent has just won. Bad.
        # if board.winner is None:
        #     return 0.5  # Board is a tie
        # # The winner is neither True, False, nor None
        # raise RuntimeError(f"board has unknown winner type {board.winner}")
    
    def cal_reward(board):
        start = 1
        accum = 0
        for selection in board.tup:
            accum += selection * start
            start *= 0.5
        return np.random.normal(0,accum)

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
    tree_bandit = MCTS(budget=BUDGET, select_type='bandit')
    tree_uct = MCTS(budget=BUDGET, select_type='uct')
    times = 1000
    trial_per_time = 1000
    bandit_record = [[] for _ in range(times)]
    uct_record = [[] for _ in range(times)]
    board = new_simpletree()
    # print(board.to_pretty_string())
    for i in range(BUDGET):
        tree_bandit.do_rollout(board, t = i)
        tree_uct.do_rollout(board, t = i)
    for time in range(times):
        for i in range(trial_per_time):
            board = new_simpletree()
            while True:
                board = tree_bandit.choose(board, trial_per_time - i, 'bandit')
                # print(board.to_pretty_string())
                if board.terminal:
                    # print(board.tup)
                    reward = board.reward()
                    # print(reward)
                    bandit_record[time].append(reward)
                    break
    for time in range(times):
        for i in range(trial_per_time):
            board = new_simpletree()
            while True:
                board = tree_uct.choose(board)
                # print(board.to_pretty_string())
                if board.terminal:
                    # print(board.tup)
                    reward = board.reward()
                    # print(reward)
                    uct_record[time].append(reward)
                    break
    return bandit_record, uct_record, tree_bandit, tree_uct


# def _winning_combos():
#     for start in range(0, 9, 3):  # three in a row
#         yield (start, start + 1, start + 2)
#     for start in range(3):  # three in a column
#         yield (start, start + 3, start + 6)
#     yield (0, 4, 8)  # down-right diagonal
#     yield (2, 4, 6)  # down-left diagonal


# def _find_winner(tup):
#     "Returns None if no winner, True if X wins, False if O wins"
#     for i1, i2, i3 in _winning_combos():
#         v1, v2, v3 = tup[i1], tup[i2], tup[i3]
#         if False is v1 is v2 is v3:
#             return False
#         if True is v1 is v2 is v3:
#             return True
#     return None

def new_simpletree():
    return SimpleTree(tup=(), terminal=False)


bandit_record, uct_record, tree_bandit, tee_uct = play_game()
# %%
np.mean([np.max(i) for i in bandit_record])
# %%
np.mean([np.max(i) for i in uct_record])
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
