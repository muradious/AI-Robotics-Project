# Building the ESP32 rover

> **Status:** the on-board planner (`planner.h`) is tested on a PC against the Python code
> (`python hardware/tests/test_planner.py`). The motor, sensor and timing code in `esp32_rover.ino` has **not** been
> run on a real robot — expect to calibrate it (section 5). Treat the first run as a debugging session, and
> keep the simulation results as your safe deliverable.

## 1. What the robot does

The floor carries an 8×8 grid of black tape lines; each **crossing = one MDP state**. The robot starts on the
start crossing, and at every crossing it: reads the action from the policy learned in simulation → turns to face it →
checks with the ultrasonic sensor → follows the tape to the next crossing. If an **unexpected obstacle** (a box) blocks
the way, it marks that crossing as blocked and **re-plans with Value Iteration on the ESP32**, then carries on.
That is a nice demo story for the presentation: *learned in simulation (Q-Learning) → executed on a robot → adapts
by planning (Value Iteration)*.

## 2. Parts and prices in Jordan (JOD)

Prices were read from the shop pages on 12 Oct 2026 and may change; confirm stock before ordering.

| Part | Qty | Price | Source |
|---|---|---|---|
| [ESP32 WiFi+Bluetooth dev board](https://besomi.com/jo_en/esp32-wifibluetooth-development-board.html) | 1 | 8.00 | Besomi |
| [2WD chassis kit](https://besomi.com/jo_en/dear0003-2wd-car-kit-for-arduino-motorized-robotics-chassis-with-wheels-and-dc-motors.html) (motors, wheels, caster, battery holder) | 1 | 8.50 | Besomi |
| [HG7881 dual motor driver](https://futureelectronicsjo.com/804-hg7881-dual-motor-driver-module.html) | 1 | 1.25 | Future Electronics |
| [TCRT5000 line-tracking module](https://futureelectronicsjo.com/969-tcrt5000-ir-reflective-line-tracking-sensor-module.html) | 3 | 3.00 | Future Electronics |
| [HC-SR04 ultrasonic sensor](https://besomi.com/jo_en/stsx0003-hc-sr04-ultrasonic-distance-sensor-2cm-to-400cm-range-precise-object-detection.html) | 1 | 1.75 | Besomi |
| **Core total** | | **22.50** | |
| Optional rechargeable power: [2-way 18650 holder](https://besomi.com/jo_en/bcbh0013-2-way-18650-battery-holder-wired-connection-for-37v-batteries.html) 3.00 + [2× 18650 cell](https://besomi.com/jo_en/bcbx0439-18650-37v-600-3800mah-lithium-ion-battery.html) 4.00 + [TP4056 charger](https://besomi.com/jo_en/bcbg0066-tp4056-lithium-battery-charger-module-type-c-lithium-battery-charger-module-1a.html) 0.65 | | 7.65 | Besomi |

Not priced (small, buy locally, budget ≈ JOD 3–5): jumper wires (female-female and female-male), a small breadboard,
resistors **1 kΩ and 2 kΩ** (voltage divider for the ultrasonic echo pin), black electrical tape (19 mm),
a white board/poster/paper for the floor, 4×AA batteries if you skip the 18650 option, a few cardboard boxes as obstacles.
Besomi delivers inside Jordan for JOD 3 (free over 100) or free self-pickup; Future Electronics: +962 7 9586 4624.

## 3. Wiring

Common ground everywhere. Pins match the constants at the top of `esp32_rover.ino`.

| Module pin | ESP32 pin | Note |
|---|---|---|
| HG7881 A-IA / A-IB (left motor) | GPIO 25 / 26 | motor leads to channel A outputs |
| HG7881 B-IA / B-IB (right motor) | GPIO 27 / 33 | motor leads to channel B outputs |
| HG7881 VCC / GND | battery + / − (6–7.4 V) | motor power |
| Line sensor LEFT / MIDDLE / RIGHT (OUT) | GPIO 34 / 35 / 32 | power the modules from the ESP32 **3V3** |
| HC-SR04 TRIG | GPIO 23 | |
| HC-SR04 ECHO | GPIO 19 **through a divider** | ECHO is 5 V! ECHO → 1 kΩ → GPIO19, and GPIO19 → 2 kΩ → GND |
| HC-SR04 VCC / GND | VIN (5 V) / GND | |
| Battery + | ESP32 VIN | keep the supply ≤ 9 V |
| Button / LED | GPIO 0 (BOOT) / GPIO 2 | already on the board |

If a wheel turns the wrong way, flip `LEFT_SIGN` / `RIGHT_SIGN` in the sketch instead of re-wiring.

**Sensor bar:** mount the three TCRT5000 in a row at the very front, about 1–1.5 cm above the floor, centre one on the robot's
axis. Spacing between neighbouring sensors should be a bit larger than the tape width (≈ 2–2.5 cm for 19 mm tape), so on a
straight line only the middle one sees tape. Put the ultrasonic sensor at the front, pointing forward.
Measure the distance from the **wheel axle to the sensor bar** — that sets `NUDGE_MS` (section 5).

## 4. The floor

* Black tape, **8 lines each way**, spacing **20–25 cm** (so about 1.5 × 1.5 m). Crossings are the cells, so the robot visits 8×8 = 64 crossings.
* Light, matte floor (white poster board or paper). Glossy floors confuse the IR sensors.
* The map comes from `LAYOUTS` in `gridworld.py`: obstacles (`#`) are crossings you **block physically with a box** (or simply
  never use); `S` is the start crossing (top-left), `G` the goal (bottom-right). Row 0 = the row farthest from you if you
  put the start at your top-left; just be consistent with the heading set in the sketch (`heading = 2` means the robot
  starts facing "down" = increasing row).
* Don't have 1.5 m? Make a smaller layout in `LAYOUTS` in `gridworld.py` (e.g. 5×5 — the grid size is taken from the layout), re-run the experiments, and export that map.

## 5. Step-by-step bring-up (do them in this order)

1. **Install** Arduino IDE → Boards Manager → "esp32 by Espressif Systems" **version 3.x** (needed for `analogWrite`).
   Board: *ESP32 Dev Module*.
2. **Export the policy:** `python hardware/export_policy.py` (writes `hardware/esp32_rover/map.h`; add `--layout complex` for the harder maze).
3. **Self-test:** in `esp32_rover.ino` set `#define TEST_MODE 1`, upload, open the Serial Monitor at 115200.
   * Move the robot over tape: the L/M/R values must change. If they are inverted, flip `LINE_IS_HIGH`; adjust the little blue potentiometer on each module until black = on and white = off.
   * Wave your hand in front: the distance must change. If it reads 999 always, check the echo divider.
   * Press BOOT: the motors run (left, right, both, spin). Fix `LEFT_SIGN`/`RIGHT_SIGN` so "both fwd" drives forward.
4. **Line following:** `TEST_MODE 0`. Put the robot on a crossing facing along a line and run it one cell. Tune `BASE_PWM` (lower = steadier) and `STEER_GAIN` until it tracks the tape without oscillating.
5. **Calibrate the crossing stop:** after detecting a crossing the robot rolls forward `NUDGE_MS` so its **wheel axle** is over the crossing. Adjust until, when it stops, the axle (centre of the wheels) is above the middle of the crossing. Rotation then works because the centre sensor swings over the perpendicular line exactly 90° later.
6. **Turns:** adjust `TURN_PWM` (and `TURN_OVERSHOOT_MS`) until a 90° turn ends with the middle sensor centred on the new line. Test right, left and U-turn.
7. **Full run:** put the axle on the start crossing, facing the direction given by `heading` in the sketch (default: down). Press BOOT. Watch the Serial Monitor: it prints the cell, action and target each step.
8. **Surprise obstacle demo:** while it is driving, place a box on a crossing that is ahead on its path. It should print "obstacle … re-plan" and take a detour. If the goal becomes impossible it stops and blinks.

## 6. If it misbehaves

| Symptom | Likely cause / fix |
|---|---|
| Robot wanders off the line | lower `BASE_PWM`, raise/lower `STEER_GAIN`, check sensor height and potentiometers |
| Stops at the wrong place | calibrate `NUDGE_MS`; check `MIN_SEGMENT_MS` is shorter than the time to cross one cell |
| "turn: never found the next line" | `TURN_PWM` too low (wheels stall) or too high (skips the line); battery low |
| Spins in the wrong direction | flip `LEFT_SIGN`/`RIGHT_SIGN` |
| Detects obstacles that are not there | ultrasonic sees the floor/wall at an angle: tilt it up slightly, lower `OBSTACLE_CM` |
| ESP32 resets when motors start | battery voltage dips: use fresh batteries / add a 100–470 µF capacitor across the motor driver supply |

## 7. How this ties back to the course (for the report)

* Policy learned by **Q-Learning** in simulation is executed on the robot (sim-to-real).
* **Value Iteration** (`planner.h`) runs on the robot to re-plan when the map changes (known-model planning).
* What differs from the MDP: movement is not exactly deterministic, turns/stops are time-calibrated, the ultrasonic sensor is noisy.
  Discuss how a stochastic model (Experiment C) is the more honest description of the real robot.
