"""
main.py -- the entry point.  Run `python main.py --help`.
=========================================================

  python main.py demo            quick tour: environment, VI, PI, Q-Learning (30 seconds)
  python main.py A | B | C | D   run one experiment
  python main.py all             run all experiments (a few minutes)
  python main.py all --quick     same, with fewer episodes/seeds (smoke test)
  python main.py play            open the optional Pygame viewer (needs `pip install pygame`)

Suggested reading order for understanding the code:
  config.py -> gridworld.py -> dp.py -> agents.py -> visualize.py -> experiments.py -> main.py
"""

import argparse

import agents
import config as C
import dp
import experiments as ex
import visualize as viz
from gridworld import GridWorld


def demo():
    """Small end-to-end tour that mirrors milestones M2 and M3 in miniature."""
    print("== DEMO ==")
    env = GridWorld("simple")
    print(env.describe())
    viz.plot_environment(env, save="results/demo_environment.png")           # Task 1

    vi = dp.value_iteration(env)                                              # Task 3 (planning)
    print(f"Value Iteration converged in {vi['iterations']} sweeps; V*(start) = {vi['V'][env.start]:.2f}")
    viz.plot_value_policy(env, vi["V"], vi["policy"], "Value Iteration: V* and policy", "results/demo_value_iteration.png")

    pi = dp.policy_iteration(env)
    print(f"Policy Iteration converged in {pi['iterations']} improvements")

    Q, log = agents.q_learning(env, episodes=500, eps_fn=agents.make_epsilon_schedule("decay", 1.0, 0.02, 0.99))
    policy = agents.greedy_policy_from_q(env, Q)                              # Task 4 (learning)
    print("Q-Learning policy evaluation:", agents.evaluate_policy(env, policy))
    viz.plot_qtable(env, Q, "Q-table after 500 episodes", "results/demo_qtable.png")
    viz.animate_policy(env, policy, "results/demo_robot.gif")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["demo", "A", "B", "C", "D", "all", "play"])
    p.add_argument("--quick", action="store_true", help="fewer episodes and seeds")
    p.add_argument("--layout", default="simple", help="layout for `play` (simple/complex)")
    args = p.parse_args()

    if args.what == "demo":
        demo()
    elif args.what == "play":
        from pygame_viewer import run_viewer
        run_viewer(args.layout)
    else:
        todo = "ABCD" if args.what == "all" else args.what
        for letter in todo:
            getattr(ex, f"experiment_{letter}")(quick=args.quick)
        print("\nDone. Figures and tables are in the results/ folder.")


if __name__ == "__main__":
    main()
