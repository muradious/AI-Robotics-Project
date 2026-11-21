"""
experiments.py -- Tasks 5 & 6: experiments A, B, C, D from the brief
=====================================================================

Each experiment_* function:
   1. builds the environment(s),
   2. runs the planners / learners (learners are repeated over several random
      seeds, because a single run of a random process proves nothing),
   3. evaluates the resulting policies greedily,
   4. writes figures (PNG) and a table (CSV + Markdown) into results/.

Read the helpers first -- they are reused by every experiment:

    run_learner(...)      train one algorithm for several seeds, evaluate each
    summarize(...)        mean +/- std of the metrics over seeds
    episodes_to_converge  the "convergence speed" metric
    save_table(...)       write CSV + Markdown
"""

import csv
import os
import time

import numpy as np

import agents
import config as C
import dp
import visualize as viz
from gridworld import GridWorld

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
LEARNERS = {
    "Q-Learning": agents.q_learning,
    "SARSA": agents.sarsa,
    "Monte Carlo": agents.monte_carlo,
    "Dyna-Q": agents.dyna_q,
}


def out(name):
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    return os.path.join(C.RESULTS_DIR, name)


def episodes_to_converge(log, reference, frac=0.9, smooth=5):
    """CONVERGENCE SPEED (lower = faster).

    Every few episodes the learner records the return its *greedy* policy would
    get ("if exploration stopped now").  We smooth that curve with a moving
    average over `smooth` evaluations (to cancel noise, which matters on the
    slippery floor) and report the first training episode at which it reaches
    `frac` (90 %) of the reference return -- the return of the optimal planner.
    A value equal to the number of training episodes means "never converged".
    """
    xs = np.array(log["greedy_x"])
    g = np.array(log["greedy_return"], dtype=float)
    target = reference - (1 - frac) * abs(reference)
    ma = np.convolve(g, np.ones(smooth) / smooth, mode="valid")      # ma[i] covers evals i..i+smooth-1
    hits = np.where(ma >= target)[0]
    return int(xs[hits[0] + smooth - 1]) if len(hits) else len(log["return"])


def run_learner(name, make_env, seeds, episodes, reference, **kwargs):
    """Train learner `name` once per seed.  Returns a dict with raw curves and
    the evaluation metrics of every seed."""
    fn = LEARNERS[name]
    res = {"logs": [], "evals": [], "conv": [], "verr": [], "Q": []}
    for seed in seeds:
        env = make_env()
        Q, log = fn(env, episodes=episodes, seed=seed, **kwargs)
        policy = agents.greedy_policy_from_q(env, Q)
        res["logs"].append(log)
        res["evals"].append(agents.evaluate_policy(env, policy))
        res["conv"].append(episodes_to_converge(log, reference))
        res["Q"].append(Q)
    return res


def curves(res, key):
    """Stack one log field into an array (n_seeds x n_episodes)."""
    return np.array([log[key] for log in res["logs"]], dtype=float)


def summarize(res, label, extra=None):
    """Collapse per-seed numbers into 'mean +/- std' columns (one table row)."""
    def ms(vals):
        vals = np.array(vals, dtype=float)
        if np.all(np.isnan(vals)):               # e.g. no successful episode -> no path length
            return float("nan"), 0.0
        return float(np.nanmean(vals)), float(np.nanstd(vals))
    row = {"method": label}
    for key in ("success_rate", "mean_steps", "mean_return", "collision_rate"):
        row[key] = ms([e[key] for e in res["evals"]])
    row["train_collision_rate"] = ms(
        [np.sum(l["collisions"]) / max(1, np.sum(l["steps"])) for l in res["logs"]])
    row["episodes_to_converge"] = ms(res["conv"])
    row["final_100_ep_return"] = ms([np.mean(l["return"][-100:]) for l in res["logs"]])
    row.update(extra or {})
    return row


def planner_row(env, result, label):
    """Table row for a DP planner (deterministic, so no std)."""
    ev = agents.evaluate_policy(env, result["policy"])
    row = {"method": label}
    for key in ("success_rate", "mean_steps", "mean_return", "collision_rate"):
        row[key] = (ev[key], 0.0)
    row["iterations"] = (result["iterations"], 0.0)
    row["time_s"] = (result["time"], 0.0)
    return row


def save_table(rows, name, columns):
    """Write rows to results/<name>.csv and results/<name>.md.  Each cell is a
    (mean, std) pair or a plain string."""
    def fmt(v):
        if isinstance(v, tuple):
            m, s = v
            if np.isnan(m):
                return "n/a"
            return f"{m:.3f}" if s == 0 and abs(m) < 5 else (f"{m:.2f}" if s == 0 else f"{m:.2f} ± {s:.2f}")
        return str(v)
    with open(out(name + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([fmt(r.get(c, "")) for c in columns])
    with open(out(name + ".md"), "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("|" + "---|" * len(columns) + "\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r.get(c, "")) for c in columns) + " |\n")
    print(f"  saved results/{name}.csv / .md")
    print("  " + " | ".join(columns))
    for r in rows:
        print("  " + " | ".join(fmt(r.get(c, "")) for c in columns))


def settings(quick):
    return (C.QUICK_EPISODES if quick else C.N_EPISODES,
            list(range(C.QUICK_SEEDS if quick else C.N_SEEDS)))


def reference_return(env, policy):
    """Average undiscounted return of the planner's policy: the 'perfect score'
    the learners are compared against."""
    return agents.evaluate_policy(env, policy, n_episodes=200)["mean_return"]


COLS = ["method", "success_rate", "mean_steps", "mean_return", "collision_rate",
        "episodes_to_converge", "final_100_ep_return"]


# ---------------------------------------------------------------------------
# EXPERIMENT A -- planning versus learning (same maze, same rewards, same dynamics)
# ---------------------------------------------------------------------------
def experiment_A(quick=False):
    print("\n=== Experiment A: planning vs learning (simple maze, deterministic) ===")
    episodes, seeds = settings(quick)
    env = GridWorld("simple", slip_prob=0.0)

    # --- planners (they use env.P) ----------------------------------------
    vi = dp.value_iteration(env)
    pi = dp.policy_iteration(env)
    viz.plot_environment(env, save=out("A_environment.png"))
    viz.plot_dp_snapshots(env, vi["history"], "Value Iteration: V(s) and greedy policy", out("A_value_iteration_snapshots.png"))
    viz.plot_dp_snapshots(env, pi["history"], "Policy Iteration: V(s) and policy after each improvement", out("A_policy_iteration_snapshots.png"))
    viz.plot_value_policy(env, vi["V"], vi["policy"], "Optimal V*(s) and policy (Value Iteration)", out("A_optimal_policy.png"))
    viz.animate_policy(env, vi["policy"], out("A_robot_value_iteration.gif"), V=vi["V"])
    ref = reference_return(env, vi["policy"])
    print(f"  VI: {vi['iterations']} sweeps  |  PI: {pi['iterations']} improvements "
          f"({pi['evaluation_sweeps']} evaluation sweeps)  |  reference return = {ref:.1f}")
    print(f"  VI and PI value tables agree: max|V_vi - V_pi| = {np.abs(vi['V'] - pi['V']).max():.2e}")

    rows = [planner_row(env, vi, "Value Iteration"), planner_row(env, pi, "Policy Iteration")]
    for r in rows:
        r["episodes_to_converge"] = f"{int(r['iterations'][0])} sweeps" if r["method"].startswith("Value") \
            else f"{int(r['iterations'][0])} improv."

    # --- learners (they only use env.step) --------------------------------
    results = {}
    for name in ("Q-Learning", "SARSA", "Monte Carlo"):
        t0 = time.time()
        results[name] = run_learner(name, lambda: GridWorld("simple"), seeds, episodes, ref)
        v_err = np.mean([np.mean(np.abs(agents.value_from_q(env, Q) - vi["V"])[
            [s for s in env.free_states if s != env.goal]]) for Q in results[name]["Q"]])
        rows.append(summarize(results[name], name, {"V_error": (v_err, 0.0)}))
        print(f"  {name}: {time.time() - t0:.1f}s")

    save_table(rows, "A_table", COLS)
    viz.plot_qtable(env, results["Q-Learning"]["Q"][0], "Q-table learned by Q-Learning (seed 0)", out("A_qtable_qlearning.png"))
    Qbest = results["Q-Learning"]["Q"][0]
    viz.plot_value_policy(env, agents.value_from_q(env, Qbest), agents.greedy_policy_from_q(env, Qbest),
                          "Q-Learning: max_a Q(s,a) and learned policy", out("A_learned_policy_qlearning.png"))
    viz.animate_policy(env, agents.greedy_policy_from_q(env, Qbest), out("A_robot_qlearning.gif"))
    viz.plot_learning_curves({k: curves(v, "return") for k, v in results.items()},
                             "return per episode (smoothed)", "Experiment A: cumulative reward vs episode",
                             out("A_return_curves.png"), ylim=(-250, 100))
    viz.plot_learning_curves({k: curves(v, "steps") for k, v in results.items()},
                             "steps per episode (smoothed)", "Experiment A: steps to finish vs episode",
                             out("A_steps_curves.png"))
    viz.plot_learning_curves({k: curves(v, "greedy_return") for k, v in results.items()},
                             "return of the greedy policy", "Experiment A: quality of the learned policy vs episode",
                             out("A_greedy_curves.png"), window=1, x_step=agents.EVAL_EVERY, ylim=(-250, 100))


# ---------------------------------------------------------------------------
# EXPERIMENT B -- exploration strategies, learning rate, discount factor (Task 5)
# ---------------------------------------------------------------------------
def experiment_B(quick=False):
    print("\n=== Experiment B: exploration, learning rate, discount (complex maze) ===")
    episodes, seeds = settings(quick)
    mk = lambda: GridWorld("complex")
    env = mk()
    ref = reference_return(env, dp.value_iteration(env)["policy"])

    # ---- B1: epsilon-greedy variants ------------------------------------
    configs = {
        "ε=0.9 (high)": dict(eps_fn=agents.make_epsilon_schedule("constant", 0.9)),
        "ε=0.3 (moderate)": dict(eps_fn=agents.make_epsilon_schedule("constant", 0.3)),
        "ε=0.05 (low)": dict(eps_fn=agents.make_epsilon_schedule("constant", 0.05)),
        "ε decays 1.0→0.01": dict(eps_fn=agents.make_epsilon_schedule("decay", 1.0, 0.01, 0.99)),
        # No exploration at all + pessimistic initial Q: the robot trusts its first
        # (bad) guesses forever. This is the "insufficient exploration" demo.
        "ε=0 (greedy, Q₀=-50)": dict(eps_fn=agents.make_epsilon_schedule("constant", 0.0), q_init=-50.0),
    }
    res, rows = {}, []
    for label, kw in configs.items():
        res[label] = run_learner("Q-Learning", mk, seeds, episodes, ref, **kw)
        rows.append(summarize(res[label], label))
    save_table(rows, "B1_epsilon_table", COLS)
    viz.plot_learning_curves({k: curves(v, "return") for k, v in res.items()}, "return per episode (smoothed)",
                             "Experiment B: exploration strategies (Q-Learning, complex maze)",
                             out("B1_epsilon_return.png"))
    viz.plot_learning_curves({k: curves(v, "greedy_return") for k, v in res.items()}, "return of the greedy policy",
                             "Experiment B: quality of the learned policy (exploration switched off)",
                             out("B1_epsilon_greedy_quality.png"), window=1, x_step=agents.EVAL_EVERY, ylim=(-250, 100))
    viz.plot_learning_curves({k: curves(v, "success") for k, v in res.items()}, "fraction of episodes reaching goal",
                             "Experiment B: success during training", out("B1_epsilon_success.png"))
    viz.plot_bars({r["method"]: {"sr": r["success_rate"]} for r in rows}, "sr", "greedy success rate after training",
                  "Experiment B: final greedy performance", out("B1_epsilon_final_success.png"))

    # ---- B2: learning rate alpha ----------------------------------------
    rows2, res2 = [], {}
    for alpha in (0.01, 0.1, 0.5, 1.0):
        label = f"α={alpha}"
        res2[label] = run_learner("Q-Learning", mk, seeds, episodes, ref, alpha=alpha,
                                  eps_fn=agents.make_epsilon_schedule("decay", 1.0, 0.01, 0.99))
        rows2.append(summarize(res2[label], label))
    save_table(rows2, "B2_alpha_table", COLS)
    viz.plot_learning_curves({k: curves(v, "return") for k, v in res2.items()}, "return per episode (smoothed)",
                             "Experiment B: effect of learning rate α", out("B2_alpha_return.png"), ylim=(-250, 100))

    # ---- B3: discount factor gamma --------------------------------------
    rows3, res3 = [], {}
    for gamma in (0.5, 0.9, 0.95, 0.99):
        label = f"γ={gamma}"
        res3[label] = run_learner("Q-Learning", mk, seeds, episodes, ref, gamma=gamma,
                                  eps_fn=agents.make_epsilon_schedule("decay", 1.0, 0.01, 0.99))
        rows3.append(summarize(res3[label], label))
    save_table(rows3, "B3_gamma_table", COLS)
    viz.plot_learning_curves({k: curves(v, "return") for k, v in res3.items()}, "return per episode (smoothed)",
                             "Experiment B: effect of discount factor γ", out("B3_gamma_return.png"), ylim=(-250, 100))

    # Visual proof of "poor policy from too little exploration"
    bad = res["ε=0 (greedy, Q₀=-50)"]["Q"][0]
    viz.plot_value_policy(env, agents.value_from_q(env, bad), agents.greedy_policy_from_q(env, bad),
                          "ε=0 with pessimistic Q₀: policy never explores", out("B1_bad_policy_no_exploration.png"))


# ---------------------------------------------------------------------------
# EXPERIMENT C -- environment complexity and stochastic movement
# ---------------------------------------------------------------------------
def experiment_C(quick=False):
    print("\n=== Experiment C: environment complexity & slippery floor ===")
    episodes, seeds = settings(quick)
    rows = []
    eps_fn = agents.make_epsilon_schedule("decay", 1.0, 0.02, 0.995)
    for layout in ("simple", "complex"):
        for slip in (0.0, C.SLIP_PROB):
            tag = f"{layout}, slip={slip}"
            mk = lambda layout=layout, slip=slip: GridWorld(layout, slip_prob=slip)
            env = mk()
            vi, pi = dp.value_iteration(env), dp.policy_iteration(env)
            ref = reference_return(env, vi["policy"])
            vi_row = planner_row(env, vi, f"Value Iteration [{tag}]")
            pi_row = planner_row(env, pi, f"Policy Iteration [{tag}]")
            vi_row["episodes_to_converge"] = f"{vi['iterations']} sweeps"
            pi_row["episodes_to_converge"] = f"{pi['iterations']} improv."
            rows += [vi_row, pi_row]
            res = {}
            for name in ("Q-Learning", "SARSA"):
                res[name] = run_learner(name, mk, seeds, episodes, ref, eps_fn=eps_fn)
                rows.append(summarize(res[name], f"{name} [{tag}]"))
            safe = tag.replace(", ", "_").replace("=", "")
            viz.plot_value_policy(env, vi["V"], vi["policy"], f"Optimal policy [{tag}]", out(f"C_policy_{safe}.png"))
            viz.plot_learning_curves({k: curves(v, "return") for k, v in res.items()}, "return per episode (smoothed)",
                                     f"Experiment C [{tag}]", out(f"C_return_{safe}.png"), ylim=(-250, 100))
    save_table(rows, "C_table", COLS)


# ---------------------------------------------------------------------------
# EXPERIMENT D -- Dyna-Q (advanced extension)
# ---------------------------------------------------------------------------
def experiment_D(quick=False):
    print("\n=== Experiment D: Dyna-Q vs Q-Learning (complex maze) ===")
    episodes, seeds = settings(quick)
    episodes = min(episodes, 400)           # Dyna-Q converges quickly; 400 is plenty
    mk = lambda: GridWorld("complex")
    env = mk()
    ref = reference_return(env, dp.value_iteration(env)["policy"])
    eps_fn = agents.make_epsilon_schedule("constant", 0.1)
    res, rows = {}, []
    for n in (0, 5, 20, 50):
        label = "Q-Learning (n=0)" if n == 0 else f"Dyna-Q (n={n})"
        if n == 0:
            res[label] = run_learner("Q-Learning", mk, seeds, episodes, ref, eps_fn=eps_fn)
        else:
            res[label] = run_learner("Dyna-Q", mk, seeds, episodes, ref, eps_fn=eps_fn, n_planning=n)
        rows.append(summarize(res[label], label))
    save_table(rows, "D_table", COLS)
    viz.plot_learning_curves({k: curves(v, "return") for k, v in res.items()}, "return per episode (smoothed)",
                             "Experiment D: Dyna-Q learns faster per real interaction",
                             out("D_return.png"), window=10, ylim=(-250, 100))
    viz.plot_learning_curves({k: curves(v, "steps") for k, v in res.items()}, "steps per episode (smoothed)",
                             "Experiment D: steps per episode", out("D_steps.png"), window=10)
    viz.plot_learning_curves({k: curves(v, "greedy_return") for k, v in res.items()}, "return of the greedy policy",
                             "Experiment D: policy quality vs real episodes", out("D_greedy.png"),
                             window=1, x_step=agents.EVAL_EVERY, ylim=(-250, 100))
