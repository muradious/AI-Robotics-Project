"""
gridworld.py -- Task 1 (environment) and Task 2 (MDP formulation)
==================================================================

This file defines the world the robot lives in.  It is deliberately written so
that the SAME object serves two purposes:

  1. As a *simulator*  : env.reset() / env.step(a)   -> used by the learners
                         (Q-Learning, SARSA, Monte Carlo, Dyna-Q).  They only
                         see what a real robot would see: the next state and a
                         reward.  They never read the transition table.

  2. As a *known model*: env.P[s][a] = [(prob, next_state, reward, done), ...]
                         -> used by the dynamic-programming planners (Value
                         Iteration / Policy Iteration), which are allowed to
                         "look inside" the MDP because the brief says the
                         model is known for planning.

This is exactly the "known model vs learning from experience" distinction the
brief asks us to explain (Task 2).

THE MDP  (S, A, P, R, gamma)
----------------------------
  S      : every free (non-obstacle) cell of the grid.  A state is encoded as a
           single integer  s = row * N + col.
  A      : {UP, RIGHT, DOWN, LEFT}  (see config.ACTIONS)
  P(s'|s,a):
           - deterministic world (slip_prob = 0): the intended move happens
             with probability 1.
           - stochastic world (slip_prob = 0.2): intended move w.p. 0.8,
             slip to each perpendicular direction w.p. 0.1.
           - moving into an obstacle or the outer wall leaves the robot where
             it is (and counts as a COLLISION).
  R(s,a,s'): -1 for a normal move, -10 for a collision, +100 for reaching the
           goal.  The goal is TERMINAL (episode ends).
  gamma  : 0.95 (see config.py).

Markov property: the next state and reward depend only on the current cell and
the chosen action, not on how the robot got there.  The robot's position is
therefore a sufficient state description.
"""

import random

import config as C

# --------------------------------------------------------------------------
# Maze layouts.  '.' free, '#' obstacle, 'S' start, 'G' goal.
# Rows are listed top to bottom, so row 0 is the top row.
# --------------------------------------------------------------------------
LAYOUTS = {
    # A simple maze: a few walls, plenty of free space.
    "simple": [
        "S.......",
        "........",
        "..##....",
        "..##..#.",
        "......#.",
        "...#....",
        "...#....",
        ".......G",
    ],
    # A harder maze: corridors, a dead-end pocket and a long alternative route,
    # so the shortest path (22 steps) is far longer than the straight-line 14
    # and exploration matters much more.
    "complex": [
        "S...#...",
        "###.#.#.",
        "....#.#.",
        ".####.#.",
        "......#.",
        ".######.",
        "...#....",
        ".#...#.G",
    ],
}


class GridWorld:
    """A grid-world robot-navigation MDP.

    Parameters
    ----------
    layout     : name in LAYOUTS (or a list of strings in the same format)
    slip_prob  : probability of slipping sideways (0 = deterministic)
    gamma      : discount factor (stored here because planners need it)
    seed       : seed of the environment's own random generator
    """

    def __init__(self, layout="simple", slip_prob=0.0, gamma=C.GAMMA, seed=0,
                 max_steps=C.MAX_STEPS_PER_EPISODE):
        rows = LAYOUTS[layout] if isinstance(layout, str) else layout
        self.layout_name = layout if isinstance(layout, str) else "custom"
        self.n_rows = len(rows)
        self.n_cols = len(rows[0])
        self.n_states = self.n_rows * self.n_cols      # includes obstacle cells (unused)
        self.n_actions = len(C.ACTIONS)
        self.slip_prob = slip_prob
        self.gamma = gamma
        self.max_steps = max_steps
        self.rng = random.Random(seed)                 # used only by step()/reset()

        # ---- parse the text layout -------------------------------------
        self.obstacles = set()                         # set of state ids
        self.start = None
        self.goal = None
        for r, line in enumerate(rows):
            for c, ch in enumerate(line):
                s = self.rc_to_s(r, c)
                if ch == "#":
                    self.obstacles.add(s)
                elif ch == "S":
                    self.start = s
                elif ch == "G":
                    self.goal = s
        assert self.start is not None and self.goal is not None, "layout needs S and G"

        # Free states the robot can stand on (obstacles removed).
        self.free_states = [s for s in range(self.n_states) if s not in self.obstacles]

        # ---- build the full transition model once ----------------------
        self.P = self._build_model()

        # ---- episode bookkeeping used by step() ------------------------
        self.state = self.start
        self.t = 0

    # ------------------------------------------------------------------
    # State <-> (row, col) helpers
    # ------------------------------------------------------------------
    def rc_to_s(self, r, c):
        return r * self.n_cols + c

    def s_to_rc(self, s):
        return divmod(s, self.n_cols)

    # ------------------------------------------------------------------
    # Model construction  (this IS the MDP: P(s'|s,a) and R(s,a,s'))
    # ------------------------------------------------------------------
    def _move(self, s, direction):
        """Where does a single, definite move in `direction` take the robot?

        Returns (next_state, collided).  Moving into a wall/obstacle keeps the
        robot in place and flags a collision.
        """
        r, c = self.s_to_rc(s)
        dr, dc = C.ACTIONS[direction]
        nr, nc = r + dr, c + dc
        outside = not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols)
        if outside or self.rc_to_s(nr, nc) in self.obstacles:
            return s, True                              # bump: stay put
        return self.rc_to_s(nr, nc), False

    def _outcomes(self, a):
        """The directions the robot may *actually* move when it *intends* `a`,
        with their probabilities.  This is where slipping is defined."""
        if self.slip_prob == 0.0:
            return [(a, 1.0)]
        left_of_a, right_of_a = (a - 1) % 4, (a + 1) % 4     # perpendicular directions
        return [(a, 1.0 - self.slip_prob),
                (left_of_a, self.slip_prob / 2),
                (right_of_a, self.slip_prob / 2)]

    def _build_model(self):
        """P[s][a] = list of (probability, next_state, reward, done, collided).

        Terminal handling: the goal state is absorbing with no outgoing rewards
        (its list is empty), so V(goal) = 0 and planners simply skip it.
        """
        P = {}
        for s in self.free_states:
            P[s] = {}
            for a in range(self.n_actions):
                if s == self.goal:
                    P[s][a] = []                         # terminal: nothing happens
                    continue
                outcomes = []
                for direction, prob in self._outcomes(a):
                    s2, collided = self._move(s, direction)
                    if collided:
                        reward, done = C.COLLISION_REWARD, False
                    elif s2 == self.goal:
                        reward, done = C.GOAL_REWARD, True
                    else:
                        reward, done = C.STEP_REWARD, False
                    outcomes.append((prob, s2, reward, done, collided))
                P[s][a] = outcomes
        return P

    # ------------------------------------------------------------------
    # Simulator interface  (what the learning robot interacts with)
    # ------------------------------------------------------------------
    def reset(self):
        """Put the robot back on the start cell and begin a new episode."""
        self.state = self.start
        self.t = 0
        return self.state

    def step(self, a):
        """Execute action `a`.

        Returns (next_state, reward, done, info).  The outcome is *sampled*
        from the model above -- the learner only sees the sample, never the
        probabilities.  `done` is True at the goal or when the step limit hits.
        info["collision"] tells us if the robot bumped (used for metrics).
        """
        outcomes = self.P[self.state][a]
        if not outcomes:                                # already at the terminal goal
            return self.state, 0.0, True, {"collision": False, "goal": True, "timeout": False}
        u, acc = self.rng.random(), 0.0
        for prob, s2, reward, done, collided in outcomes:
            acc += prob
            if u <= acc:
                break                                   # fall through with last if rounding
        self.state = s2
        self.t += 1
        reached_goal = done
        timed_out = self.t >= self.max_steps
        return s2, reward, (reached_goal or timed_out), {
            "collision": collided, "goal": reached_goal, "timeout": timed_out and not reached_goal}

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def reseed(self, seed):
        self.rng = random.Random(seed)

    def describe(self):
        return (f"GridWorld[{self.layout_name}] {self.n_rows}x{self.n_cols}, "
                f"{len(self.obstacles)} obstacles, slip={self.slip_prob}, gamma={self.gamma}")
