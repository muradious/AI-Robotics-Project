"""
dp.py -- Task 3: planning by Dynamic Programming (model-based)
===============================================================

Value Iteration and Policy Iteration.  Both USE THE KNOWN MODEL env.P, i.e.
they plan "in the robot's head" without ever moving it.

Notation
--------
  V[s]      state value: expected discounted return starting from s, following
            the (optimal / current) policy.
  Q(s,a)    action value: one-step look-ahead,
                Q(s,a) = sum over outcomes of  p * ( r + gamma * V[s'] )
            (for terminal outcomes V[s'] is treated as 0 -- the episode ends).
  policy[s] the action chosen in state s.

Both algorithms return a `history` dict {iteration_number: (V_copy, policy_copy)}
for chosen iterations, so the visualiser can show "value tables and policy
arrows at selected iterations" (a Task 3 requirement).
"""

import time

import numpy as np

import config as C


# ---------------------------------------------------------------------------
# Shared helper: one-step look-ahead
# ---------------------------------------------------------------------------
def q_from_v(env, V, s, a):
    """Q(s,a) computed from the current value estimate V (Bellman expectation)."""
    total = 0.0
    for prob, s2, reward, done, _collided in env.P[s][a]:
        future = 0.0 if done else V[s2]           # no future after reaching the goal
        total += prob * (reward + env.gamma * future)
    return total


def greedy_policy_from_v(env, V):
    """For every state pick the action with the highest one-step look-ahead value.
    Ties are broken by taking the first best action (deterministic)."""
    policy = np.zeros(env.n_states, dtype=int)
    for s in env.free_states:
        if s == env.goal:
            continue
        q = [q_from_v(env, V, s, a) for a in range(env.n_actions)]
        policy[s] = int(np.argmax(q))
    return policy


def _snapshot_iterations(max_iter, snapshots):
    """Which iteration numbers should be stored for plotting."""
    return set(snapshots) if snapshots else set()


# ---------------------------------------------------------------------------
# VALUE ITERATION
# ---------------------------------------------------------------------------
def value_iteration(env, theta=C.THETA, snapshots=(1, 2, 3, 5, 10), max_iter=10_000):
    """Repeatedly apply the Bellman OPTIMALITY backup to every state:

            V(s)  <-  max_a  sum_{s'} p * ( r + gamma * V(s') )

    Information about the +100 goal spreads one cell further each sweep, so the
    value table "grows" outward from the goal.  We stop when the largest change
    in a sweep is smaller than `theta`, then read off the greedy policy.

    Returns dict with V, policy, iterations, elapsed seconds, history.
    """
    t0 = time.perf_counter()
    V = np.zeros(env.n_states)
    keep = _snapshot_iterations(max_iter, snapshots)
    history = {}
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        delta = 0.0
        for s in env.free_states:
            if s == env.goal:
                continue                                    # terminal, V stays 0
            best = max(q_from_v(env, V, s, a) for a in range(env.n_actions))
            delta = max(delta, abs(best - V[s]))
            V[s] = best                                     # in-place (Gauss-Seidel) update
        if iteration in keep:
            history[iteration] = (V.copy(), greedy_policy_from_v(env, V))
        if delta < theta:
            break
    policy = greedy_policy_from_v(env, V)
    history[iteration] = (V.copy(), policy.copy())          # always keep the final one
    return {"V": V, "policy": policy, "iterations": iteration,
            "time": time.perf_counter() - t0, "history": history}


# ---------------------------------------------------------------------------
# POLICY ITERATION
# ---------------------------------------------------------------------------
def policy_evaluation(env, policy, V=None, theta=C.THETA, max_iter=10_000):
    """Compute V^pi for a FIXED policy by sweeping the Bellman EXPECTATION backup
    until it stops changing.  Returns (V, number_of_sweeps)."""
    V = np.zeros(env.n_states) if V is None else V.copy()
    sweeps = 0
    while sweeps < max_iter:
        sweeps += 1
        delta = 0.0
        for s in env.free_states:
            if s == env.goal:
                continue
            new = q_from_v(env, V, s, policy[s])
            delta = max(delta, abs(new - V[s]))
            V[s] = new
        if delta < theta:
            break
    return V, sweeps


def policy_iteration(env, theta=C.THETA, snapshots=(1, 2, 3, 4, 5), max_iter=1_000):
    """Alternate two steps until the policy stops changing:

        1. POLICY EVALUATION   : compute V^pi for the current policy.
        2. POLICY IMPROVEMENT  : make the policy greedy w.r.t. that V^pi.

    Starts from a poor policy (always UP) to make the improvement visible.
    Converges in very few outer iterations (each one is expensive, though).
    """
    t0 = time.perf_counter()
    policy = np.zeros(env.n_states, dtype=int)              # initial policy: always UP
    V = np.zeros(env.n_states)
    keep = _snapshot_iterations(max_iter, snapshots)
    history = {}
    total_sweeps = 0
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        V, sweeps = policy_evaluation(env, policy, V, theta)        # step 1
        total_sweeps += sweeps
        new_policy = greedy_policy_from_v(env, V)                   # step 2
        # "Stable" = no state would switch to a strictly better action.  Comparing
        # raw actions can loop forever on ties, so compare the VALUE of the
        # old action against the best one.
        stable = all(
            q_from_v(env, V, s, policy[s]) >= q_from_v(env, V, s, new_policy[s]) - 1e-9
            for s in env.free_states if s != env.goal)
        if iteration in keep:
            history[iteration] = (V.copy(), new_policy.copy())
        if stable:
            break
        policy = new_policy
    history[iteration] = (V.copy(), policy.copy())
    return {"V": V, "policy": policy, "iterations": iteration,
            "evaluation_sweeps": total_sweeps,
            "time": time.perf_counter() - t0, "history": history}
