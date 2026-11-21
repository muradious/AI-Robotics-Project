"""
agents.py -- Tasks 4 & 5: model-free learning from experience
==============================================================

Unlike dp.py, nothing here looks at env.P.  The robot only calls
env.reset() / env.step(a) and learns from the (state, action, reward,
next_state) samples it gets back -- like a real robot would.

Algorithms
----------
  q_learning   : off-policy TD control.  Learns the OPTIMAL Q even while behaving
                 epsilon-greedily:   Q(s,a) += alpha * [ r + gamma*max_a' Q(s',a') - Q(s,a) ]
  sarsa        : on-policy TD control (the "additional TD method").  Uses the action
                 it will really take next:  ... + gamma * Q(s',a') ...
  monte_carlo  : first-visit Monte-Carlo control with epsilon-soft behaviour.  Waits
                 for the END of an episode and updates Q(s,a) towards the observed
                 return G (no bootstrapping).
  dyna_q       : Q-learning + a learned model that replays `n_planning` imagined
                 transitions after every real step (integrates learning and planning).

Exploration (Task 5)
--------------------
  epsilon-greedy: with probability epsilon pick a random action, otherwise the
  greedy one.  `make_epsilon_schedule` builds either a constant epsilon or a
  decaying one (high at first, lower later).

Every learner returns  (Q, log)  where Q is an (n_states x 4) numpy array and
`log` holds one number per training episode: return, steps, collisions, success.
"""

import random

import numpy as np

import config as C


# ---------------------------------------------------------------------------
# Exploration helpers
# ---------------------------------------------------------------------------
def make_epsilon_schedule(kind="constant", eps=C.EPSILON, eps_min=0.01, decay=0.995):
    """Return a function  episode_index -> epsilon.

    kind="constant" : epsilon never changes.
    kind="decay"    : epsilon = max(eps_min, eps * decay**episode)  (exponential decay).
                      With eps=1.0 and decay=0.995 it reaches ~0.08 after 500 episodes.
    """
    if kind == "constant":
        return lambda ep: eps
    if kind == "decay":
        return lambda ep: max(eps_min, eps * (decay ** ep))
    raise ValueError(kind)


def epsilon_greedy(Q_s, eps, rng):
    """Choose an action from the row Q_s = Q[s].  Ties among the best actions are
    broken at random -- important at the start when every Q is 0, otherwise the
    robot would always pick action 0 and explore badly."""
    if rng.random() < eps:
        return rng.randrange(len(Q_s))                      # explore
    best = max(Q_s)
    return rng.choice([a for a, q in enumerate(Q_s) if q == best])   # exploit


def _new_log():
    # Per-episode training statistics, plus a periodic *greedy* evaluation
    # ("if we stopped exploring right now, how good is the policy?").
    return {"return": [], "steps": [], "collisions": [], "success": [],
            "greedy_x": [], "greedy_return": []}


EVAL_EVERY = 5          # evaluate the greedy policy every 5 training episodes


def _greedy_return(env, Q, rng, n):
    """Average undiscounted return of the policy argmax_a Q(s,a), estimated by
    simulating `n` episodes DIRECTLY from the model env.P with a private random
    generator -- so it never disturbs the training episode in progress."""
    total = 0.0
    for _ in range(n):
        s, ret = env.start, 0.0
        for _t in range(env.max_steps):
            row = Q[s]
            a = row.index(max(row))
            u, acc = rng.random(), 0.0
            for prob, s2, r, done, _c in env.P[s][a]:
                acc += prob
                if u <= acc:
                    break
            ret += r
            s = s2
            if done:
                break
        total += ret
    return total / n


def _maybe_eval(log, env, Q, rng, ep):
    """Record the greedy policy's quality every EVAL_EVERY episodes.
    (Deterministic world: 1 rollout is exact. Stochastic world: average of 20.)"""
    if ep % EVAL_EVERY == 0:
        log["greedy_x"].append(ep)
        log["greedy_return"].append(_greedy_return(env, Q, rng, 20 if env.slip_prob > 0 else 1))


def _log_episode(log, ret, steps, collisions, success):
    log["return"].append(ret)
    log["steps"].append(steps)
    log["collisions"].append(collisions)
    log["success"].append(1.0 if success else 0.0)


def _init_q(env, q_init):
    # Plain nested lists: much faster than numpy for tiny per-step updates.
    return [[q_init] * env.n_actions for _ in range(env.n_states)]


# ---------------------------------------------------------------------------
# Q-LEARNING
# ---------------------------------------------------------------------------
def q_learning(env, episodes=C.N_EPISODES, alpha=C.ALPHA, gamma=C.GAMMA,
               eps_fn=None, seed=0, q_init=0.0):
    """Tabular Q-learning.  See module docstring."""
    eps_fn = eps_fn or make_epsilon_schedule("constant")
    rng = random.Random(seed)
    eval_rng = random.Random(seed + 99)                     # private rng for greedy evaluation
    env.reseed(seed + 1)
    Q = _init_q(env, q_init)
    log = _new_log()

    for ep in range(episodes):
        eps = eps_fn(ep)
        s = env.reset()
        ret, steps, coll, done, info = 0.0, 0, 0, False, {}
        while not done:
            a = epsilon_greedy(Q[s], eps, rng)              # 1. choose (behaviour policy)
            s2, r, done, info = env.step(a)                 # 2. act in the world
            # 3. TD target.  If the goal was reached there is no future, so the
            #    target is just r.  A time-out is NOT a true terminal state, so
            #    we still bootstrap from Q[s2] in that case.
            target = r if info["goal"] else r + gamma * max(Q[s2])
            Q[s][a] += alpha * (target - Q[s][a])           # 4. move Q towards target
            s = s2
            ret += r; steps += 1; coll += info["collision"]
        _log_episode(log, ret, steps, coll, info["goal"])
        _maybe_eval(log, env, Q, eval_rng, ep)
    return np.array(Q), log


# ---------------------------------------------------------------------------
# SARSA  (on-policy TD control)
# ---------------------------------------------------------------------------
def sarsa(env, episodes=C.N_EPISODES, alpha=C.ALPHA, gamma=C.GAMMA,
          eps_fn=None, seed=0, q_init=0.0):
    """SARSA = State, Action, Reward, State', Action'.
    Differs from Q-learning in ONE line: the target uses the action a2 that the
    epsilon-greedy policy will actually take next, not the greedy max.  So it
    learns the value of the exploring policy -> it tends to keep a safer
    distance from obstacles when epsilon is large."""
    eps_fn = eps_fn or make_epsilon_schedule("constant")
    rng = random.Random(seed)
    eval_rng = random.Random(seed + 99)                     # private rng for greedy evaluation
    env.reseed(seed + 1)
    Q = _init_q(env, q_init)
    log = _new_log()

    for ep in range(episodes):
        eps = eps_fn(ep)
        s = env.reset()
        a = epsilon_greedy(Q[s], eps, rng)
        ret, steps, coll, done, info = 0.0, 0, 0, False, {}
        while not done:
            s2, r, done, info = env.step(a)
            a2 = epsilon_greedy(Q[s2], eps, rng)            # choose next action NOW
            target = r if info["goal"] else r + gamma * Q[s2][a2]
            Q[s][a] += alpha * (target - Q[s][a])
            s, a = s2, a2
            ret += r; steps += 1; coll += info["collision"]
        _log_episode(log, ret, steps, coll, info["goal"])
        _maybe_eval(log, env, Q, eval_rng, ep)
    return np.array(Q), log


# ---------------------------------------------------------------------------
# MONTE CARLO CONTROL (first-visit, epsilon-soft)
# ---------------------------------------------------------------------------
def monte_carlo(env, episodes=C.N_EPISODES, gamma=C.GAMMA,
                eps_fn=None, seed=0, q_init=0.0):
    """First-visit Monte-Carlo control.

    1. Play a WHOLE episode with the epsilon-greedy policy, storing (s, a, r).
    2. Walk backwards computing the return  G_t = r_t + gamma * G_{t+1}.
    3. For the FIRST visit of each (s,a) in the episode, update the running
       average  Q(s,a) <- Q(s,a) + (G - Q(s,a)) / N(s,a).

    No bootstrapping: it is unbiased but has high variance and cannot learn
    until an episode finishes -- that is why it is slower than TD here.
    """
    eps_fn = eps_fn or make_epsilon_schedule("constant")
    rng = random.Random(seed)
    eval_rng = random.Random(seed + 99)                     # private rng for greedy evaluation
    env.reseed(seed + 1)
    Q = _init_q(env, q_init)
    N = [[0] * env.n_actions for _ in range(env.n_states)]
    log = _new_log()

    for ep in range(episodes):
        eps = eps_fn(ep)
        s = env.reset()
        trajectory, ret, coll, done, info = [], 0.0, 0, False, {}
        while not done:                                     # --- 1. generate episode
            a = epsilon_greedy(Q[s], eps, rng)
            s2, r, done, info = env.step(a)
            trajectory.append((s, a, r))
            s = s2
            ret += r; coll += info["collision"]

        first_visit = {}
        for t, (st, at, _r) in enumerate(trajectory):
            first_visit.setdefault((st, at), t)

        G = 0.0
        for t in range(len(trajectory) - 1, -1, -1):        # --- 2. backwards returns
            st, at, r = trajectory[t]
            G = r + gamma * G
            if first_visit[(st, at)] == t:                  # --- 3. first-visit update
                N[st][at] += 1
                Q[st][at] += (G - Q[st][at]) / N[st][at]
        _log_episode(log, ret, len(trajectory), coll, info["goal"])
        _maybe_eval(log, env, Q, eval_rng, ep)
    return np.array(Q), log


# ---------------------------------------------------------------------------
# DYNA-Q  (learning + planning)  -- advanced extension, Experiment D
# ---------------------------------------------------------------------------
def dyna_q(env, episodes=C.N_EPISODES, alpha=C.ALPHA, gamma=C.GAMMA,
           eps_fn=None, seed=0, n_planning=10, q_init=0.0):
    """Dyna-Q = Q-learning + a learned model.

    After every REAL step we
      (a) do the ordinary Q-learning update,
      (b) store the observed transition in a model  model[(s,a)] = (r, s', goal?),
      (c) do `n_planning` extra Q-learning updates on transitions picked at random
          from the model ("imagined" experience, costs no robot time).

    Each real step therefore improves Q many times -> much faster learning per
    real interaction.  The model simply remembers the LAST outcome of each
    (s,a), which is exact for a deterministic world and an approximation when
    movement is stochastic.
    """
    eps_fn = eps_fn or make_epsilon_schedule("constant")
    rng = random.Random(seed)
    eval_rng = random.Random(seed + 99)                     # private rng for greedy evaluation
    env.reseed(seed + 1)
    Q = _init_q(env, q_init)
    model = {}                                              # (s,a) -> (r, s2, reached_goal)
    seen = []                                               # list of keys for fast random sampling
    log = _new_log()

    def update(s, a, r, s2, goal):
        target = r if goal else r + gamma * max(Q[s2])
        Q[s][a] += alpha * (target - Q[s][a])

    for ep in range(episodes):
        eps = eps_fn(ep)
        s = env.reset()
        ret, steps, coll, done, info = 0.0, 0, 0, False, {}
        while not done:
            a = epsilon_greedy(Q[s], eps, rng)
            s2, r, done, info = env.step(a)
            update(s, a, r, s2, info["goal"])               # (a) learn from real step
            if (s, a) not in model:
                seen.append((s, a))
            model[(s, a)] = (r, s2, info["goal"])           # (b) update the model
            for _ in range(n_planning):                     # (c) planning from the model
                ps, pa = seen[rng.randrange(len(seen))]
                pr, ps2, pgoal = model[(ps, pa)]
                update(ps, pa, pr, ps2, pgoal)
            s = s2
            ret += r; steps += 1; coll += info["collision"]
        _log_episode(log, ret, steps, coll, info["goal"])
        _maybe_eval(log, env, Q, eval_rng, ep)
    return np.array(Q), log


# ---------------------------------------------------------------------------
# Turning Q into things we can inspect / evaluate
# ---------------------------------------------------------------------------
def greedy_policy_from_q(env, Q):
    """pi(s) = argmax_a Q(s,a) for every free, non-goal state."""
    policy = np.zeros(env.n_states, dtype=int)
    for s in env.free_states:
        if s != env.goal:
            policy[s] = int(np.argmax(Q[s]))
    return policy


def value_from_q(env, Q):
    """V(s) = max_a Q(s,a) (comparable to the V from Value Iteration)."""
    V = np.zeros(env.n_states)
    for s in env.free_states:
        if s != env.goal:
            V[s] = float(np.max(Q[s]))
    return V


# ---------------------------------------------------------------------------
# Evaluation of ANY policy (from DP or from learning) -- Task 6 metrics
# ---------------------------------------------------------------------------
def evaluate_policy(env, policy, n_episodes=C.N_EVAL_EPISODES, seed=12345):
    """Run the policy greedily (no exploration, no learning) and measure:

      success_rate   : fraction of episodes that reached the goal before the step limit
      mean_steps     : average path length over the SUCCESSFUL episodes
      mean_return    : average undiscounted cumulative reward
      collision_rate : collisions per step taken (bumps / moves)
    """
    env.reseed(seed)
    successes, steps_ok, returns, collisions, total_steps = 0, [], [], 0, 0
    for _ in range(n_episodes):
        s, done, ret, steps, info = env.reset(), False, 0.0, 0, {}
        while not done:
            s, r, done, info = env.step(int(policy[s]))
            ret += r; steps += 1; collisions += info["collision"]
        total_steps += steps
        returns.append(ret)
        if info["goal"]:
            successes += 1
            steps_ok.append(steps)
    return {
        "success_rate": successes / n_episodes,
        "mean_steps": float(np.mean(steps_ok)) if steps_ok else float("nan"),
        "mean_return": float(np.mean(returns)),
        "collision_rate": collisions / max(1, total_steps),
    }
