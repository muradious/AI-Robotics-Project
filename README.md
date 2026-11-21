# AI630 Project — Intelligent Autonomous Robot Navigation (Planning + Reinforcement Learning)

A grid-world robot that must get from the top-left to the bottom-right while avoiding obstacles.
We model it as an MDP, **plan** the route with Value / Policy Iteration (known model), and **learn** it from
experience with Q-Learning, SARSA and Monte Carlo. Dyna-Q combines both. An optional ESP32 rover then drives the
policy learned in simulation on a real taped floor grid.

## 1. Quick start

```bash
pip install -r requirements.txt          # numpy, matplotlib, pillow  (pygame is optional)
python main.py demo                      # 30-second tour -> results/demo_*.png / .gif
python main.py all --quick               # all experiments, few seeds (~20 s)
python main.py all                       # full experiments, 10 seeds each (~1 min)
python main.py play                      # optional interactive Pygame viewer
python hardware/tests/test_planner.py    # check the robot's on-board planner against the Python code
```

Everything the programs produce (figures, GIFs, CSV/Markdown tables) is written to `results/`.
A copy of a full run is already there, so you can look at the results before running anything.

## 2. How the code flows — read in this order

```
config.py        all numbers (rewards, gamma, alpha, epsilon, slip, #episodes, #seeds)
   │
gridworld.py     THE MDP.  Layouts + P(s'|s,a) + R(s,a,s').  Two faces:
   │               env.P[s][a]        -> the known model      (used by planners)
   │               env.reset()/step() -> a sampled simulator  (used by learners)
   │
   ├── dp.py         PLANNING (uses env.P):   value_iteration(), policy_iteration()
   │
   ├── agents.py     LEARNING (uses env.step only):
   │                   q_learning(), sarsa(), monte_carlo(), dyna_q()
   │                   + epsilon-greedy / epsilon schedules + evaluate_policy()
   │
   ├── visualize.py  every plot: maze, V(s)+arrows, Q-table, learning curves, robot GIF
   │
   ├── experiments.py  Experiments A, B, C, D (runs algorithms over seeds, writes tables + figures)
   │
   └── main.py       command line entry point

pygame_viewer.py     optional interactive viewer
hardware/            the real robot (section 5)
report/outline.md    skeleton of the technical report with the numbers to quote
```

**One episode of learning, in plain words** (`agents.q_learning`):
`reset()` puts the robot on the start → pick an action (ε-greedy: random with prob. ε, else best Q) → `step(a)` returns
the next cell and a reward → nudge `Q[s][a]` towards `r + γ·max Q[s']` → repeat until the goal (or 200 steps).

**What "planning" does differently** (`dp.value_iteration`): it never moves the robot. It sweeps over all cells using the
known transition table until the values stop changing; the +100 at the goal "flows backward" through the maze
(see `results/A_value_iteration_snapshots.png`).

### How the project maps to the brief

| Brief | Where |
|---|---|
| Task 1 – environment, visualisation | `gridworld.py`, `visualize.py`, `pygame_viewer.py` |
| Task 2 – MDP formulation | docstring of `gridworld.py`, `config.py` |
| Task 3 – Value / Policy Iteration, V(s), policy, snapshots | `dp.py`, `results/A_*snapshots.png` |
| Task 4 – Q-Learning + SARSA + Monte Carlo, Q-table, policy | `agents.py`, `results/A_qtable_qlearning.png` |
| Task 5 – ε, schedule, α, γ, "too little exploration" | Experiment B |
| Task 6 – compare, seeds, success / path / collisions | Experiments A and C, `results/*_table.md` |
| Experiments A, B, C, D | `experiments.py` (`python main.py A` … `D`) |
| Advanced extension (Dyna-Q) | `agents.dyna_q`, Experiment D |
| Physical robot (extra marks) | `hardware/` |

### Modelling choices (state these in the report)

* **States**: every non-obstacle cell, encoded `s = row·8 + col`. **Actions**: up/right/down/left.
* **Collision**: moving into an obstacle or the outer wall keeps the robot in place and gives **−10** (instead of the −1 step cost).
* **Goal** is terminal and gives **+100**. A 200-step time-out ends an episode but is *not* treated as terminal for bootstrapping.
* **Slip (stochastic floor)**: intended move with probability 0.8, each perpendicular direction with 0.1.
* **Mazes**: `simple` (optimal path 14 steps) and `complex` (corridors + a dead-end pocket, optimal path 22 steps). Edit `LAYOUTS` in `gridworld.py` to design your own.
* **Convergence speed** = first training episode where the (smoothed) return of the *greedy* policy reaches 90 % of the optimal planner's return (see `experiments.episodes_to_converge`). A value equal to the number of episodes means "never".

## 3. Results of the full run (10 seeds, mean ± std — see `results/`)

**Experiment A — planning vs learning (simple maze, deterministic)**

| Method | Success | Path length | Convergence |
|---|---|---|---|
| Value Iteration | 100 % | 14 | 15 sweeps |
| Policy Iteration | 100 % | 14 | 4 improvements (628 evaluation sweeps in total) |
| Q-Learning | 100 % | 14 | ≈142 ± 12 episodes |
| SARSA | 100 % | 14 | ≈158 ± 18 episodes |
| Monte Carlo | 100 % | 14.6 ± 0.9 | ≈381 ± 529 episodes (very high variance) |

VI and PI produce identical value tables. Learners reach the planners' optimal path, but need experience to do so.

**Experiment B — exploration (complex maze, Q-Learning)**
Greedy policy quality after training is perfect for every ε > 0, but the *online* return during training differs hugely:
ε = 0.05 → +73, ε = 0.3 → +29, ε = 0.9 → −810 (it keeps bumping into walls). A decaying ε gives the best of both (+77.7).
With **ε = 0 and pessimistic initial Q** the robot never explores and fails in 100 % of episodes
(`results/B1_bad_policy_no_exploration.png`). Learning rate: α = 0.01 needs ≈1160 episodes, α = 0.5 ≈ 47.
Discount: γ = 0.5 is slowest (≈182 episodes), γ ≥ 0.9 is similar.

**Experiment C — complexity and slippery floor**
On the slippery floor (20 % slip) the optimal path gets longer (14 → 17 steps on the simple maze, 22 → 29 on the
complex one) and collisions appear (6 % / 14 % of steps), but Q-Learning and SARSA reach the same quality as the planners.

**Experiment D — Dyna-Q (complex maze)**
Episodes to converge: Q-Learning 134 → Dyna-Q n=5: 62 → n=20: 30 → n=50: 25. Planning with a learned model makes
each real step far more valuable.

(Numbers above are copied from `results/*_table.md`; re-running with the same seeds reproduces them.)

## 4. Honest limitations (worth a paragraph in the report)

* Tabular methods only: the state space is tiny, so function approximation / policy gradients were not implemented.
* Dyna-Q's model stores the *last* observed outcome per (s, a) — exact for the deterministic world, only approximate for the slippery one.
* Planners and learners are compared in simulation; the real robot adds sensor noise, wheel slip and timing errors that the MDP ignores.
* Monte Carlo is run with ε-greedy first-visit updates; its high seed-to-seed variance is real, not a bug.

## 5. The physical robot (optional, extra marks)

See **`hardware/HARDWARE.md`** for parts (with Jordanian prices), wiring, floor grid, calibration and test steps.
Short version: `python hardware/export_policy.py` turns the policy learned in simulation into `map.h`;
the ESP32 sketch follows tape lines between crossings, executes the policy, and **re-plans on board with Value Iteration**
when its ultrasonic sensor finds an obstacle that was not on the map.
The planner code that runs on the ESP32 is tested on a PC against the Python implementation; the motor/sensor code
has **not** been run on real hardware yet.
