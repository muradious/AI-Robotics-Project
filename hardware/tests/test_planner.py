"""
Verifies the ESP32 planner (planner.h) against the Python implementation, on a PC.

    python hardware/tests/test_planner.py

For both mazes it checks that
  * the C++ Value Iteration gives the same V*(start) as dp.value_iteration,
  * the C++ policy and the policy exported from Python both reach the goal in the shortest number of steps,
  * after a surprise obstacle is added, re-planning finds the new shortest path (BFS check),
  * a walled-in goal is reported as unreachable.
"""
import os, subprocess, sys, tempfile
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "hardware"))
import dp
from export_policy import build_header
from gridworld import GridWorld


def bfs(env, src, blocked_extra=()):
    d = {src: 0}; q = deque([src])
    while q:
        s = q.popleft()
        for a in range(4):
            s2, hit = env._move(s, a)
            if not hit and s2 not in blocked_extra and s2 not in d:
                d[s2] = d[s] + 1; q.append(s2)
    return d.get(env.goal)


failures = 0
def check(name, cond, detail=""):
    global failures
    print(("  PASS " if cond else "  FAIL ") + name + (f"  {detail}" if detail else ""))
    failures += (not cond)

with tempfile.TemporaryDirectory() as tmp:
    for layout in ("simple", "complex"):
        print(f"[{layout}]")
        hdr = os.path.join(tmp, f"map_{layout}.h")
        open(hdr, "w").write(build_header(layout, "q"))
        exe = os.path.join(tmp, f"t_{layout}")
        subprocess.run(["g++", "-std=c++11", "-Wall", f'-DMAP_HEADER="{hdr}"',
                        os.path.join(HERE, "test_planner.cpp"), "-o", exe], check=True)
        out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout
        v = {l.split()[0]: l.split()[1:] for l in out.strip().splitlines()}
        env = GridWorld(layout)
        vi = dp.value_iteration(env)
        check("V*(start) matches Python", abs(float(v["VSTART"][0]) - vi["V"][env.start]) < 0.01,
              f"C++={v['VSTART'][0]} py={vi['V'][env.start]:.4f}")
        opt = bfs(env, env.start)
        check("C++ path is shortest", int(v["PATH_VI"][0]) == opt, f"{v['PATH_VI'][0]} vs {opt}")
        check("exported Q-learning policy is shortest", int(v["PATH_EXPORTED"][0]) == opt, f"{v['PATH_EXPORTED'][0]} vs {opt}")
        br, bc = map(int, v["BLOCK"]); fr, fc = map(int, v["FROM"])
        exp = bfs(env, env.rc_to_s(fr, fc), {env.rc_to_s(br, bc)})
        check("re-plan after surprise obstacle is optimal", v["REACHABLE"][0] == "1" and int(v["REPLAN_STEPS"][0]) == exp,
              f"{v['REPLAN_STEPS'][0]} vs {exp}")
        check("walled-in goal reported unreachable", v["WALLED_REACHABLE"][0] == "0")

print("ALL PASSED" if not failures else f"{failures} FAILURE(S)")
sys.exit(1 if failures else 0)
