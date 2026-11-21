# AI630 Milestone 2 — Environment and Planning Prototype

**Due: Thursday 26 November 2026**

M2 asks for the simulation, Value Iteration, Policy Iteration and an initial policy visualisation.
All of it is already implemented in this folder. This file maps each requirement to code and results.

## How to run (about 30 seconds)

```bash
pip install -r requirements.txt
python main.py demo                  # environment, Value Iteration, Policy Iteration, Q-Learning check -> results/demo_*.png / .gif
python main.py A                     # Experiment A only: planning vs learning (writes results/A_*)
python hardware/tests/test_planner.py   # optional: on-board planner matches the Python planner
```

## Requirement checklist

| M2 requirement | Where | Evidence |
|---|---|---|
| Grid-world simulation (start, goal, obstacles, transitions, rewards) | `gridworld.py`, `config.py` | `results/A_environment.png` |
| Visualisation of the robot's movement | `visualize.py`, `pygame_viewer.py` | `results/A_robot_value_iteration.gif` |
| Value Iteration | `dp.value_iteration` | 15 sweeps, V*(start) = 41.60 |
| Policy Iteration | `dp.policy_iteration` | 4 improvements (628 evaluation sweeps in total) |
| V(s) and policy arrows at chosen iterations | `visualize.py` | `results/A_value_iteration_snapshots.png`, `results/A_policy_iteration_snapshots.png` |
| Resulting policy | `visualize.py` | `results/A_optimal_policy.png` |

## Model (for the write-up)

- **States:** every non-obstacle cell of the 8×8 grid. **Actions:** up, right, down, left.
- **Rewards:** −1 per move, −10 for bumping an obstacle or wall (robot stays put), +100 at the goal (terminal). **γ = 0.95.**
- **Transitions:** deterministic by default. With slip enabled, the intended move has probability 0.8 and each perpendicular direction 0.1.
- **Mazes:** `simple` (8 obstacles, optimal path 14 steps) and `complex` (23 obstacles, optimal path 22 steps).
- The planners read the known model `env.P` / `R`. The learners (later milestones) only see samples from `env.step`.

## Results to quote

- Value Iteration and Policy Iteration produce **identical value tables** (max difference 0.0).
- Both give the **14-step optimal path** on the simple maze, with return 87.0, success 100 % and zero collisions.
- V*(start) = 41.60 on the simple maze. It equals 100·0.95¹³ − Σ(0.95ᵏ) for k = 0..12, which is 51.3 − 9.7.
- The on-board C++ planner for the rover matches the Python planner (`hardware/tests`, all checks pass).

## Not part of M2 (already built, for M3 and M4)

Q-Learning, SARSA, Monte Carlo, Dyna-Q, Experiments A–D, the report outline and the ESP32 rover.

## Open items

- The rover firmware has not yet been run on real hardware.
- M3 and M4 dates are still to be confirmed against the syllabus.
- Check who on the team owns which section before submission.
