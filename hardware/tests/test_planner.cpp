// Host-side test of planner.h (the code that will run on the ESP32).
// Build:  g++ -DMAP_HEADER='"map_x.h"' test_planner.cpp -o test_planner
// It prints machine-readable lines that test_planner.py compares with the Python implementation.
#include <stdio.h>
#include <stdint.h>
#include MAP_HEADER
#include "../esp32_rover/planner.h"

using namespace planner;

// Walk a policy table from (r,c) to the goal; returns number of steps (-1 if it never arrives).
static int walk(const Grid &g, const uint8_t pol[GRID_N][GRID_N], int r, int c, int *cellsR, int *cellsC) {
  int steps = 0;
  if (cellsR) { cellsR[0] = r; cellsC[0] = c; }
  while (!(r == g.goalR && c == g.goalC) && steps < 200) {
    int a = pol[r][c];
    int nr = r + DR[a], nc = c + DC[a];
    if (!inside(nr, nc) || g.blocked[nr][nc]) return -1;      // policy walks into an obstacle
    r = nr; c = nc; steps++;
    if (cellsR) { cellsR[steps] = r; cellsC[steps] = c; }
  }
  return (r == g.goalR && c == g.goalC) ? steps : -1;
}

int main() {
  Grid g;
  for (int r = 0; r < GRID_N; r++)
    for (int c = 0; c < GRID_N; c++) {
      g.blocked[r][c] = (MAP_ROWS[r][c] == '#');
      g.policy[r][c] = 0;
    }
  g.goalR = GOAL_R_CELL; g.goalC = GOAL_C_CELL;

  // 1. plan on the original map
  valueIteration(g);
  int cr[300], cc[300];
  int n = walk(g, g.policy, START_R, START_C, cr, cc);
  printf("VSTART %.4f\n", g.V[START_R][START_C]);
  printf("PATH_VI %d\n", n);
  printf("PATH_EXPORTED %d\n", walk(g, PLANNED_POLICY, START_R, START_C, 0, 0));

  // 2. a surprise obstacle appears on the path -> the robot (standing just before it) re-plans.
  //    Pick the first path cell whose blocking still leaves the goal reachable (in a corridor
  //    maze, blocking some cells makes the goal impossible -- that case is tested in step 3).
  int br = -1, bc = -1, fr = -1, fc = -1, m = -1;
  bool ok = false;
  for (int i = 3; i <= n - 1 && !ok; i++) {
    br = cr[i]; bc = cc[i]; fr = cr[i - 1]; fc = cc[i - 1];
    g.blocked[br][bc] = true;
    ok = replan(g, fr, fc);
    if (!ok) g.blocked[br][bc] = false;
  }
  if (ok) m = walk(g, g.policy, fr, fc, 0, 0);
  printf("BLOCK %d %d\nFROM %d %d\nREACHABLE %d\nREPLAN_STEPS %d\n", br, bc, fr, fc, ok ? 1 : 0, m);

  // 3. completely walled-in goal must be reported unreachable
  for (int a = 0; a < 4; a++) {
    int nr = g.goalR + DR[a], nc = g.goalC + DC[a];
    if (inside(nr, nc)) g.blocked[nr][nc] = true;
  }
  printf("WALLED_REACHABLE %d\n", replan(g, fr, fc) ? 1 : 0);
  return 0;
}
