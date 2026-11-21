// planner.h -- the SAME maths as dp.py (Value Iteration), written so it fits on an ESP32.
// ==========================================================================================
// It has NO Arduino-specific code, so it is also compiled and tested on a normal PC
// (see hardware/tests/test_planner.py) and compared against the Python implementation.
//
// Why have a planner on the robot at all?
//   The policy learned in simulation tells the robot what to do on the MAP IT WAS TRAINED ON.
//   If someone puts a NEW obstacle on the floor, that policy is wrong.  The robot detects the
//   obstacle with its ultrasonic sensor, marks the cell as blocked, and re-runs Value Iteration
//   on the 8x8 grid (takes a few milliseconds) to get a fresh policy.  This is the
//   "known model -> planning" half of the course, running on a real robot.
//
// Conventions (identical to config.py / gridworld.py):
//   actions: 0 = UP (row-1), 1 = RIGHT (col+1), 2 = DOWN (row+1), 3 = LEFT (col-1)
//   rewards: step -1, collision -10 (robot stays), goal +100 (terminal), gamma = 0.95

#pragma once
#include <stdint.h>

#ifndef GRID_N
#define GRID_N 8
#endif

namespace planner {

static const int8_t DR[4] = {-1, 0, 1, 0};
static const int8_t DC[4] = {0, 1, 0, -1};

static const float GAMMA = 0.95f;
static const float STEP_R = -1.0f;
static const float COLLISION_R = -10.0f;
static const float GOAL_R = 100.0f;

struct Grid {
  bool blocked[GRID_N][GRID_N];     // true = obstacle
  int goalR, goalC;
  float V[GRID_N][GRID_N];          // state values  (V* after planning)
  uint8_t policy[GRID_N][GRID_N];   // best action per cell
};

inline bool inside(int r, int c) { return r >= 0 && r < GRID_N && c >= 0 && c < GRID_N; }

// One-step look-ahead  Q(s,a) = r + gamma * V(s')   (deterministic world)
inline float qValue(const Grid &g, int r, int c, int a) {
  int nr = r + DR[a], nc = c + DC[a];
  if (!inside(nr, nc) || g.blocked[nr][nc])           // bump: stay in place, pay -10
    return COLLISION_R + GAMMA * g.V[r][c];
  if (nr == g.goalR && nc == g.goalC)                  // reaching the goal ends the episode
    return GOAL_R;
  return STEP_R + GAMMA * g.V[nr][nc];
}

// Value Iteration:  V(s) <- max_a Q(s,a)  until nothing changes, then take the argmax.
inline void valueIteration(Grid &g, float theta = 1e-4f, int maxSweeps = 2000) {
  for (int r = 0; r < GRID_N; r++)
    for (int c = 0; c < GRID_N; c++) g.V[r][c] = 0.0f;

  for (int sweep = 0; sweep < maxSweeps; sweep++) {
    float delta = 0.0f;
    for (int r = 0; r < GRID_N; r++) {
      for (int c = 0; c < GRID_N; c++) {
        if (g.blocked[r][c] || (r == g.goalR && c == g.goalC)) continue;
        float best = qValue(g, r, c, 0);
        for (int a = 1; a < 4; a++) {
          float q = qValue(g, r, c, a);
          if (q > best) best = q;
        }
        float d = best - g.V[r][c];
        if (d < 0) d = -d;
        if (d > delta) delta = d;
        g.V[r][c] = best;
      }
    }
    if (delta < theta) break;
  }

  // Greedy policy (first best action wins ties, like numpy.argmax in dp.py)
  for (int r = 0; r < GRID_N; r++) {
    for (int c = 0; c < GRID_N; c++) {
      uint8_t bestA = 0;
      float bestQ = qValue(g, r, c, 0);
      for (int a = 1; a < 4; a++) {
        float q = qValue(g, r, c, a);
        if (q > bestQ) { bestQ = q; bestA = a; }
      }
      g.policy[r][c] = bestA;
    }
  }
}

// Is there ANY free path from (r,c) to the goal?  (breadth-first search from the goal)
// Value Iteration alone cannot tell us "impossible", so we check explicitly.
inline bool reachable(const Grid &g, int r, int c) {
  bool seen[GRID_N][GRID_N] = {};
  int qr[GRID_N * GRID_N], qc[GRID_N * GRID_N];
  int head = 0, tail = 0;
  qr[tail] = g.goalR; qc[tail] = g.goalC; tail++;
  seen[g.goalR][g.goalC] = true;
  while (head < tail) {
    int cr = qr[head], cc = qc[head]; head++;
    if (cr == r && cc == c) return true;
    for (int a = 0; a < 4; a++) {
      int nr = cr + DR[a], nc = cc + DC[a];
      if (inside(nr, nc) && !g.blocked[nr][nc] && !seen[nr][nc]) {
        seen[nr][nc] = true; qr[tail] = nr; qc[tail] = nc; tail++;
      }
    }
  }
  return false;
}

// Re-plan after the map changed.  Returns false if the goal can no longer be reached.
inline bool replan(Grid &g, int curR, int curC) {
  valueIteration(g);
  return reachable(g, curR, curC);
}

}  // namespace planner
