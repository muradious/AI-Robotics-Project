"""
pygame_viewer.py -- OPTIONAL interactive demo (the brief allows Pygame)
========================================================================

Run with:   python main.py play            (needs: pip install pygame)

Keys
----
  V : plan with Value Iteration and show V(s) + arrows
  T : train Q-Learning for 20 more episodes (watch the policy improve live)
  SPACE : let the robot drive the current policy from the start cell
  L : switch between the simple and complex maze
  S : toggle slippery floor (stochastic movement, p = 0.2)
  ESC / close window : quit

Colours: dark = obstacle, green frame = start, gold = goal, red circle = robot,
blue shading = value (darker = better).  Arrows show the current policy.
"""

import numpy as np

import agents
import config as C
import dp
from gridworld import GridWorld

CELL = 70


def run_viewer(layout="simple"):
    import pygame

    pygame.init()
    slip = 0.0

    def build(layout, slip):
        env = GridWorld(layout, slip_prob=slip)
        return env, [[0.0] * 4 for _ in range(env.n_states)]

    env, Q = build(layout, slip)
    V = np.zeros(env.n_states)
    policy = np.zeros(env.n_states, dtype=int)
    screen = pygame.display.set_mode((env.n_cols * CELL, env.n_rows * CELL + 30))
    pygame.display.set_caption("Robot navigation -- V: plan | T: train | SPACE: drive | L: maze | S: slip")
    font = pygame.font.SysFont(None, 18)
    clock = pygame.time.Clock()
    robot, driving, message = env.start, False, "press V (plan) or T (train)"
    episodes_trained = 0

    def draw():
        screen.fill((252, 252, 251))
        lo, hi = V.min(), V.max()
        for s in range(env.n_states):
            r, c = env.s_to_rc(s)
            rect = pygame.Rect(c * CELL, r * CELL, CELL, CELL)
            if s in env.obstacles:
                colour = (59, 58, 55)
            elif s == env.goal:
                colour = (237, 161, 0)
            else:
                shade = 0 if hi == lo else (V[s] - lo) / (hi - lo)
                colour = (int(255 - 150 * shade), int(255 - 110 * shade), 255)
            pygame.draw.rect(screen, colour, rect)
            pygame.draw.rect(screen, (195, 194, 183), rect, 1)
            if s not in env.obstacles and s != env.goal:
                dr, dc = C.ACTIONS[policy[s]]
                cx, cy = rect.center
                pygame.draw.line(screen, (11, 11, 11), (cx - 16 * dc, cy - 16 * dr), (cx + 16 * dc, cy + 16 * dr), 3)
                pygame.draw.circle(screen, (11, 11, 11), (cx + 16 * dc, cy + 16 * dr), 5)
        r, c = env.s_to_rc(env.start)
        pygame.draw.rect(screen, (27, 175, 122), (c * CELL, r * CELL, CELL, CELL), 4)
        r, c = env.s_to_rc(robot)
        pygame.draw.circle(screen, (227, 73, 72), (c * CELL + CELL // 2, r * CELL + CELL // 2), CELL // 3)
        screen.blit(font.render(message, True, (11, 11, 11)), (6, env.n_rows * CELL + 8))
        pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_v:                    # plan with the known model
                    res = dp.value_iteration(env)
                    V, policy = res["V"], res["policy"]
                    message = f"Value Iteration: {res['iterations']} sweeps"
                elif event.key == pygame.K_t:                    # learn from experience
                    Qarr, _ = agents.q_learning(
                        env, episodes=20, seed=episodes_trained,
                        eps_fn=agents.make_epsilon_schedule("constant", 0.2))
                    # continue from previous Q: train on top of it by blending
                    Q = Qarr if episodes_trained == 0 else 0.5 * np.array(Q) + 0.5 * Qarr
                    episodes_trained += 20
                    V, policy = agents.value_from_q(env, np.array(Q)), agents.greedy_policy_from_q(env, np.array(Q))
                    message = f"Q-Learning (fresh 20-episode runs blended): {episodes_trained} episodes"
                elif event.key == pygame.K_SPACE:
                    robot, driving = env.start, True
                    env.reset()
                elif event.key in (pygame.K_l, pygame.K_s):
                    if event.key == pygame.K_l:
                        layout = "complex" if layout == "simple" else "simple"
                    else:
                        slip = C.SLIP_PROB if slip == 0.0 else 0.0
                    env, Q = build(layout, slip)
                    V = np.zeros(env.n_states); policy = np.zeros(env.n_states, dtype=int)
                    robot, driving, episodes_trained = env.start, False, 0
                    message = f"maze={layout}, slip={slip}"
        if driving:
            if robot == env.goal or env.t >= env.max_steps:
                driving, message = False, "arrived!" if robot == env.goal else "gave up"
            else:
                env.state = robot                    # (training may have moved env.state)
                robot, _r, _d, _i = env.step(int(policy[robot]))
            pygame.time.wait(120)
        draw()
        clock.tick(30)
    pygame.quit()
