"""
config.py -- every tunable number of the project lives here.
=============================================================

READ THIS FILE FIRST.  The assignment brief gives "suggested starting
parameters"; they are all collected below so you can change one value in one
place and every module (environment, planners, learners, experiments) follows.

Nothing in this file *does* anything -- it is only constants.
"""

# --------------------------------------------------------------------------
# Grid geometry and actions
# --------------------------------------------------------------------------
GRID_SIZE = 8                      # informational: the real size comes from the maze layouts in gridworld.py (8 x 8)

# The four actions.  Each is a (row_delta, col_delta) pair.
# Row 0 is the TOP of the grid, so "up" means row - 1.
# The ORDER matters: the ESP32 code uses the same numbering (0=N,1=E,2=S,3=W),
# and "slipping" uses the fact that actions 1 and 3 are perpendicular to 0 and 2.
ACTIONS = [
    (-1, 0),   # 0 = UP
    (0, +1),   # 1 = RIGHT
    (+1, 0),   # 2 = DOWN
    (0, -1),   # 3 = LEFT
]
ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]
ACTION_ARROWS = ["↑", "→", "↓", "←"]   # ↑ → ↓ ←

# --------------------------------------------------------------------------
# Rewards (from the brief)
# --------------------------------------------------------------------------
STEP_REWARD = -1.0                 # cost of every movement (encourages short paths)
GOAL_REWARD = +100.0               # received on the step that reaches the goal
COLLISION_REWARD = -10.0           # received when the robot tries to enter an obstacle
                                   # (or the outside wall). The robot stays where it is.
                                   # NOTE: this REPLACES the -1 step cost for that step.

# --------------------------------------------------------------------------
# Learning / planning hyper-parameters (from the brief)
# --------------------------------------------------------------------------
GAMMA = 0.95                       # discount factor
ALPHA = 0.1                        # learning rate for Q-learning / SARSA
EPSILON = 0.1                      # default exploration rate for the "demo" runs

# Dynamic-programming stopping rule: stop when the largest change in V is below THETA
THETA = 1e-8

# --------------------------------------------------------------------------
# Stochastic ("slippery floor") movement for Experiment C
# --------------------------------------------------------------------------
# With probability 1 - SLIP_PROB the robot moves where it intended.
# With probability SLIP_PROB it slips sideways: half of that probability goes to
# each of the two directions perpendicular to the intended one.
SLIP_PROB = 0.2

# --------------------------------------------------------------------------
# Episode limits and experiment sizes
# --------------------------------------------------------------------------
MAX_STEPS_PER_EPISODE = 200        # an episode is cut off after this many steps
N_EPISODES = 2000                  # training episodes per learning run
N_SEEDS = 10                       # independent repetitions (different random seeds)
N_EVAL_EPISODES = 100              # greedy evaluation episodes after training

# Fewer repetitions for a quick smoke-test:  python main.py all --quick
QUICK_EPISODES = 600
QUICK_SEEDS = 3

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
RESULTS_DIR = "results"
