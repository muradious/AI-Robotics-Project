"""
visualize.py -- Task 1 (visualise environment & robot) and Task 3/4 (show tables)
===================================================================================

Everything that draws a picture lives here, so the algorithm files stay clean.
All functions SAVE a PNG (or GIF) and return nothing; call them from main.py or
experiments.py.  Matplotlib runs headless (Agg backend) so it works anywhere.

What you can produce
--------------------
  plot_environment      : the maze, start, goal, optional path
  plot_value_policy     : heat-map of V(s) with the value number and a policy arrow per cell
  plot_dp_snapshots     : V(s) + arrows at selected iterations (Task 3 requirement)
  plot_qtable           : the four Q(s,a) values written inside every cell (Task 4 requirement)
  plot_learning_curves  : mean +/- std over random seeds, with smoothing
  plot_bars             : grouped bar chart for final metrics
  animate_policy        : a GIF of the robot driving along the policy (for the presentation)
"""

import os

import matplotlib
matplotlib.use("Agg")                       # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import config as C

# ---------------------------------------------------------------------------
# Style: a small colour-blind-safe palette, used in a FIXED order so a given
# algorithm keeps the same colour in every figure. Lines also differ in dash
# style, so identity never depends on colour alone.
# ---------------------------------------------------------------------------
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#6250d6", "#e34948", "#eda100"]
DASHES = ["-", "--", "-.", ":", "-", "--", "-."]
OBSTACLE, START_C, GOAL_C = "#3b3a37", "#1baf7a", "#eda100"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False,
})


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def _draw_cells(ax, env, V=None, cmap="Blues"):
    """Paint the grid.  Free cells are white (or coloured by V if given),
    obstacles dark, start green, goal gold.  Row 0 is drawn at the TOP."""
    ax.set_xlim(0, env.n_cols)
    ax.set_ylim(env.n_rows, 0)               # inverted y-axis: row 0 on top
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    if V is not None:
        vals = np.array([V[s] for s in env.free_states if s != env.goal])
        lo, hi = vals.min(), vals.max()
        norm = lambda v: 0.08 + 0.55 * (v - lo) / (hi - lo + 1e-9)
        colormap = plt.get_cmap(cmap)

    for s in range(env.n_states):
        r, c = env.s_to_rc(s)
        if s in env.obstacles:
            colour = OBSTACLE
        elif s == env.goal:
            colour = GOAL_C
        elif V is not None:
            colour = colormap(norm(V[s]))
        else:
            colour = "white"
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor=colour, edgecolor="#c3c2b7", lw=0.8))
    # start marker
    r, c = env.s_to_rc(env.start)
    ax.add_patch(Rectangle((c, r), 1, 1, fill=False, edgecolor=START_C, lw=3))
    ax.text(c + 0.08, r + 0.2, "S", fontsize=9, fontweight="bold", color=START_C, va="center")
    r, c = env.s_to_rc(env.goal)
    ax.text(c + 0.5, r + 0.5, "G", fontsize=14, fontweight="bold", ha="center", va="center", color=INK)


def _draw_arrows(ax, env, policy):
    """One arrow per free non-goal cell showing pi(s)."""
    for s in env.free_states:
        if s == env.goal:
            continue
        r, c = env.s_to_rc(s)
        dr, dc = C.ACTIONS[policy[s]]
        ax.annotate("", xy=(c + 0.5 + 0.3 * dc, r + 0.5 + 0.3 * dr),
                    xytext=(c + 0.5 - 0.3 * dc, r + 0.5 - 0.3 * dr),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))


def _draw_values(ax, env, V, dy=0.0):
    for s in env.free_states:
        if s == env.goal:
            continue
        r, c = env.s_to_rc(s)
        ax.text(c + 0.5, r + 0.88 + dy, f"{V[s]:.0f}", ha="center", va="center",
                fontsize=7, color=MUTED)


def _draw_path(ax, env, path, colour="#e34948"):
    pts = [(env.s_to_rc(s)[1] + 0.5, env.s_to_rc(s)[0] + 0.5) for s in path]
    ax.plot(*zip(*pts), color=colour, lw=2.4, marker="o", ms=4, zorder=5)


# ---------------------------------------------------------------------------
# Public figures
# ---------------------------------------------------------------------------
def plot_environment(env, path=None, title=None, save="results/environment.png"):
    """Task 1: show the maze (and optionally a path the robot took)."""
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    _draw_cells(ax, env)
    if path:
        _draw_path(ax, env, path)
    ax.set_title(title or env.describe(), fontsize=9)
    _save(fig, save)


def plot_value_policy(env, V, policy, title="Value function and policy", save="results/value_policy.png"):
    """Heat-map of V(s) (darker = higher value) with a policy arrow in every cell."""
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    _draw_cells(ax, env, V)
    _draw_values(ax, env, V)
    _draw_arrows(ax, env, policy)
    ax.set_title(title)
    _save(fig, save)


def plot_dp_snapshots(env, history, title, save):
    """Task 3: V(s) and policy arrows at several iterations side by side,
    so you can watch the goal's value 'flow' backwards through the maze."""
    its = sorted(history)
    cols = min(3, len(its))
    rows = int(np.ceil(len(its) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, it in zip(axes, its):
        V, pol = history[it]
        _draw_cells(ax, env, V)
        _draw_values(ax, env, V)
        _draw_arrows(ax, env, pol)
        ax.set_title(f"iteration {it}", fontsize=10)
    for ax in axes[len(its):]:
        ax.axis("off")
    fig.suptitle(title, fontweight="bold")
    _save(fig, save)


def plot_qtable(env, Q, title="Learned Q-table", save="results/qtable.png"):
    """Task 4: write Q(s,UP/RIGHT/DOWN/LEFT) at the top/right/bottom/left edge of
    each cell; the best action's number is bold and the policy arrow is drawn."""
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_cells(ax, env)
    offsets = [(0.5, 0.14), (0.86, 0.5), (0.5, 0.86), (0.14, 0.5)]   # up, right, down, left
    for s in env.free_states:
        if s == env.goal:
            continue
        r, c = env.s_to_rc(s)
        best = int(np.argmax(Q[s]))
        for a, (ox, oy) in enumerate(offsets):
            ax.text(c + ox, r + oy, f"{Q[s][a]:.0f}", ha="center", va="center", fontsize=6.5,
                    fontweight="bold" if a == best else "normal",
                    color=INK if a == best else MUTED)
        dr, dc = C.ACTIONS[best]
        ax.annotate("", xy=(c + 0.5 + 0.12 * dc, r + 0.5 + 0.12 * dr),
                    xytext=(c + 0.5 - 0.12 * dc, r + 0.5 - 0.12 * dr),
                    arrowprops=dict(arrowstyle="-|>", color="#2a78d6", lw=1.4))
    ax.set_title(title)
    _save(fig, save)


def _smooth(y, window):
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(np.pad(y, (window - 1, 0), mode="edge"), kernel, mode="valid")


def plot_learning_curves(curves, ylabel, title, save, window=25, ylim=None, x_step=1):
    """curves: {label: 2-D array (n_seeds x n_episodes)}.
    Draws the mean across seeds as a line and +/- 1 std as a faint band."""
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for i, (label, data) in enumerate(curves.items()):
        data = np.asarray(data, dtype=float)
        sm = np.array([_smooth(row, window) for row in data])
        mean, std = sm.mean(axis=0), sm.std(axis=0)
        x = np.arange(len(mean)) * x_step
        ax.plot(x, mean, color=SERIES[i % len(SERIES)], ls=DASHES[i % len(DASHES)], lw=2, label=label)
        ax.fill_between(x, mean - std, mean + std, color=SERIES[i % len(SERIES)], alpha=0.12, lw=0)
    ax.set_xlabel("training episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="lower right")
    _save(fig, save)


def plot_bars(table, metric, ylabel, title, save):
    """table: {label: {metric: (mean, std)}} -> one bar per label."""
    fig, ax = plt.subplots(figsize=(7.4, 4))
    labels = list(table)
    means = [table[l][metric][0] for l in labels]
    stds = [table[l][metric][1] for l in labels]
    ax.bar(range(len(labels)), means, yerr=stds, capsize=3, width=0.6,
           color=[SERIES[i % len(SERIES)] for i in range(len(labels))], edgecolor=SURFACE)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    _save(fig, save)


# ---------------------------------------------------------------------------
# Rolling out a policy and animating it
# ---------------------------------------------------------------------------
def rollout_path(env, policy, max_steps=100):
    """Follow `policy` from the start and return the list of visited states.
    Uses the model's MOST LIKELY outcome (so it is deterministic and reproducible)."""
    s, path = env.start, [env.start]
    for _ in range(max_steps):
        if s == env.goal:
            break
        outcomes = env.P[s][int(policy[s])]
        s = max(outcomes, key=lambda o: o[0])[1]
        path.append(s)
        if len(path) > 3 and path[-1] == path[-2] == path[-3]:
            break                                # stuck pushing into a wall
    return path


def animate_policy(env, policy, save="results/robot.gif", V=None, fps=3):
    """Save a GIF of the robot driving along the policy -- great for the demo."""
    from matplotlib.animation import PillowWriter
    path = rollout_path(env, policy)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    writer = PillowWriter(fps=fps)
    os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
    with writer.saving(fig, save, dpi=90):
        for k in range(len(path)):
            ax.clear()
            _draw_cells(ax, env, V)
            _draw_arrows(ax, env, policy)
            if k > 0:
                _draw_path(ax, env, path[:k + 1])
            r, c = env.s_to_rc(path[k])
            ax.add_patch(plt.Circle((c + 0.5, r + 0.5), 0.3, color="#e34948", zorder=6))
            ax.set_title(f"step {k}/{len(path) - 1}")
            writer.grab_frame()
    plt.close(fig)
    print(f"  saved {save}")
