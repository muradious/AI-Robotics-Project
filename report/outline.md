# Technical report outline (AI630 Project)

Each section lists what to write, where the material is, and which rubric marks it earns
(MDP formulation 3 · environment 4 · code 12 · experiments 2 · report 4 · presentation 5).

## 1. Introduction  (½ page)
Problem: a mobile robot must reach a goal cell quickly, without collisions. Why planning *and* learning (known vs unknown model).

## 2. Problem formulation as an MDP  (1 page — rubric: MDP 3)
* States, actions, P(s'|s,a) (deterministic and 0.8/0.1/0.1 slip), R(s,a,s'), γ = 0.95. Source: docstring in `gridworld.py`, `config.py`.
* Explain the **Markov property** (position suffices) and the **discount factor** (γ near 1 → far-sighted, γ small → greedy; our γ lets the +100 be "seen" from the start: V*(start) = 41.6 on the simple maze).
* **Known model vs learning from experience**: planners use `env.P`; learners only see samples from `env.step`.
* Figure: `results/A_environment.png`.

## 3. Environment and simulation  (½ page — rubric: environment 4)
Grid, layouts `simple` / `complex`, collision handling, episode limits, visualisation (`visualize.py`, Pygame viewer, GIFs).

## 4. Methods  (2 pages — rubric: code 12)
* Value Iteration, Policy Iteration (equations, stopping rule θ). Figures: `A_value_iteration_snapshots.png`, `A_policy_iteration_snapshots.png`, `A_optimal_policy.png`.
* Q-Learning (off-policy TD), SARSA (on-policy TD), first-visit Monte Carlo control. Update equations. Figure: `A_qtable_qlearning.png`.
* ε-greedy and ε-decay; Dyna-Q.
* Metrics: success rate, path length, cumulative reward, collision rate, episodes-to-converge (definition in README).

## 5. Experiments and results  (2–3 pages — rubric: experiments 2)
| Experiment | Question | Key figure / table | One-line finding (verify against your own run) |
|---|---|---|---|
| A | planning vs learning | `A_table.md`, `A_return_curves.png`, `A_greedy_curves.png` | all find the 14-step optimum; VI 15 sweeps, PI 4 improvements; Q-Learning ≈ 142 episodes |
| B1 | exploration rate | `B1_epsilon_table.md`, `B1_epsilon_return.png`, `B1_epsilon_greedy_quality.png` | high ε learns a fine policy but behaves terribly while learning; ε = 0 never explores and fails |
| B2/B3 | α, γ | `B2_*`, `B3_*` | tiny α is very slow; γ = 0.5 slowest |
| C | maze complexity + slip | `C_table.md`, `C_policy_*.png` | slip lengthens paths and causes collisions; learners match planners |
| D | Dyna-Q | `D_table.md`, `D_greedy.png` | planning steps cut episodes-to-converge from 134 to 25 |
Report mean ± std over 10 seeds; say why (random processes).

## 6. Physical robot (optional)  (½–1 page)
Architecture, parts and cost (`hardware/HARDWARE.md`), policy export, on-board re-planning, what worked / what differed from the simulation. Photos + a short video for the presentation.

## 7. Discussion and limitations
Strengths/weaknesses: DP needs a model and scales badly with state count; TD needs no model but needs many episodes; MC high variance; SARSA vs Q-Learning (safer vs optimal); Dyna-Q trades computation for experience. Limitations listed in README §4.

## 8. Conclusions and future work
Function approximation for larger maps, policy gradients, continuous-state simulation (Gymnasium), real-robot noise models.

## Presentation (5 marks) — suggested 8 slides
1. Problem & demo GIF  2. MDP  3. Planning (VI snapshots)  4. Learning (Q-table, curves)  5. Exploration experiment  6. Slippery floor  7. Dyna-Q  8. Real robot demo + conclusions.
