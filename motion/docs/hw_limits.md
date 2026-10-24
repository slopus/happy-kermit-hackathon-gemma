# Hardware limits: xArm6, DH PGC-140-50, dexterous hands

Gathered from the web on 2026-10-24. Nothing here was measured on our hardware. When sources disagree, all values are listed and the conservative one is marked **use**. Machine-readable copy: `hw_limits.json`.

## 1. UFactory xArm6

| Joint | Range (official docs) | Range (UFACTORY US / resellers) | Max speed | Max accel |
|---|---|---|---|---|
| J1 | ±360° | ±360° | 180°/s | 1145°/s² |
| J2 | −117° … +116° | −118° … +120° | 180°/s | 1145°/s² |
| J3 | −219° … +10° | −225° … +11° | 180°/s | 1145°/s² |
| J4 | ±360° | ±360° | 180°/s | 1145°/s² |
| J5 | −97° … +180° | −97° … +180° | 180°/s | 1145°/s² |
| J6 | ±360° | ±360° | 180°/s | 1145°/s² |

**Use:** the official-docs ranges, which are narrower for J2 and J3. Sources: https://docs.xarm.ufactory.cc/8.technical_specifications.html and https://www.ufactory.us/product/ufactory-xarm-6.

**Joint limits**
- Speed: 0–180°/s. Source: the docs URL above.
- Acceleration: 1145°/s², which is about 20 rad/s²; UFACTORY's speed guide rounds it to 1146. Sources: https://www.ufactory.cc/wp-content/uploads/2023/05/xArm-User-Manual-V2.0.0.pdf and https://docs.supportarticle.ufactory.cc/support_articles/developer/guide-to-run-robotic-arm-at-the-maximum-speed.html.
- Jerk: 28 647°/s³. Source: the V2.0.0 manual above.
- In xarm_ros, `set_max_acc_joint` goes up to 20 rad/s². Source: https://github.com/xArm-Developer/xarm_ros.

**TCP (tool tip) limits**
- Linear speed: 0–1000 mm/s. Source: the docs URL above.
- Linear acceleration: 50 000 mm/s². Source: the V2.0.0 manual.
- Rotational speed and acceleration: 180°/s and 1145°/s². Source: the V2.0.0 manual.
- Jerk: the manuals disagree. V1.6.9 says 10 000 mm/s³ (https://cdn.robotshop.com/media/u/ufa/rb-ufa-18/pdf/xarm-6-dof-robotic-arm-user-manual.pdf) and V2.0.0 says 100 000 mm/s³.

**Other specs:** 700 mm reach, 5 kg payload, ±0.1 mm repeatability, about 12.2 kg. Source: the docs URL above.

### Streaming joint commands: `set_servo_angle_j` in mode 1 (servo mode)
Sources:
- UFACTORY guide: https://help.ufactory.cc/en/articles/3973629-guide-to-use-the-interface-set_servo_cartesian-and-set_servo_anagle_j
- Motion modes: https://docs.usermanual.ufactory.cc/user_manual/ufactoryStudio/10.motion_mode_and_state.html
- SDK issue #93: https://github.com/xArm-Developer/xArm-Python-SDK/issues/93
- SDK example: https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/example/wrapper/common/7001-servo_j.py
- xarm_ros2 docs: https://github.com/xArm-Developer/xarm_ros2/blob/humble/xarm_api/ReadMe.md

**Rate**
- The maximum is **250 Hz** on firmware 1.4.0 and later. Older firmware tops out at 100 Hz.
- Commands sent faster than 250 Hz are dropped.
- Below about 30 Hz, motion becomes jerky.
- **Use 100–250 Hz at a fixed rate.** The xarm_ros2 docs recommend this, and the official example sleeps 0.01 s per loop, which is about 100 Hz.

**Caveats**
- The `speed`, `mvacc` and `mvtime` arguments are **ignored** (issue #93). Speed comes only from how far apart the targets are and how often they're sent.
- Each target must be very close to the current pose. The arm steps to it almost immediately, so a large step causes a jump or a fault. For the Cartesian version (`set_servo_cartesian`), UFACTORY says steps over 10 mm won't run.
- You have to plan the trajectory yourself and keep velocity and acceleration under the limits above.
- Setup order: enable motion, set mode 0 and move to the start pose, then call `set_mode(1)` and `set_state(0)`, then start streaming.
- The ROS docs require a risk assessment before using servo mode.
- Alternative: **mode 6** (online joint planning, firmware 1.10.0+). Each new target interrupts the current motion and is re-planned from where the arm is, so it gives smooth retargeting without your own planner.
- I didn't find a statement specific to firmware 2.x. The 250 Hz limit and the caveats above are stated for firmware 1.4.0 and later.

## 2. DH Robotics PGC-140-50 (parallel gripper)

Main sources:
- Datasheet: https://www.dh-robotics.com/wp-content/uploads/2022/01/PGC-Series.pdf
- Product pages: https://www.dobot.nu/en/product/dh-robotics-pgc-140-50/ and https://www.devonics.com/product-page/copy-of-collaborative-parallel-gripper-pgc-140-50

| Item | Value | Source / notes |
|---|---|---|
| Stroke | 50 mm total | datasheet |
| Force | 40–140 N per jaw (set as 20–100%) | datasheet; force register 0x0101 |
| Open/close time | **0.6 s** (Dobot.nu, RobotShop, Unchained) or **0.75 s** (Devonics, OpenELAB) | sources disagree, so **use 0.75 s** |
| Position command | 0–1000‰ in register 0x0103; the gripper moves as soon as it is written | https://www.manualslib.com/manual/2822592/Dh-Robotics-Pgc-50.html |
| Position resolution | about 0.05 mm per step (50 mm / 1000), **my own calculation** | DH doesn't publish this |
| Repeatability | ±0.03 mm | datasheet |
| Speed setting | 1–100% | as above |
| Bus | Modbus RTU over RS485: 115200 baud, 8N1, slave ID 1 by default; baud register 0x0303 | AG-145 manual (same register layout): https://device.report/m/6b0275b246cfbbbe281b9c703a39719c39da36696969d37f3cc6e9a4542ceaea.pdf |
| Command rate / latency | **not published.** An 8-byte frame takes under 1 ms on the wire at 115200 baud, so the gripper's internal processing will dominate. Measure it; plan for roughly 10–20 Hz polling as a working assumption (not from a source) | — |
| Weight / rating | about 1 kg, IP67, 24 V | https://www.robotshop.com/products/dh-robotics-pgc-140-50-collaborative-parallel-gripper-mainly-cobots |

## 3. Dexterous hands (about $10k)

| Item | Inspire RH56DFX | Inspire RH56DFTP (tactile) | PSYONIC Ability Hand | Shadow Dexterous Hand |
|---|---|---|---|---|
| Active DOF / joints | 6 / 12 (linear servos) | 6 / 12 | 6 (5 finger flexion + thumb rotation; brushless motors) | 24 joints (20 actuated) |
| Joint ranges | Commanded as 0–1000 per DOF; thumb rotation more than 65° | same | No official range; the UIUC demo uses 0–114° | per joint, in the spec sheet |
| Speed | Fingers about 260°/s; thumb bend about 70°/s, thumb rotation about 107°/s; full close about 0.8 s | Fingers more than 200°/s, thumb more than 130°/s; close about 0.8 s | 90° in 200 ms; distributor lists about 375°/s (thumb rotation 550°/s) | Full range about 1 Hz; F1 joint about 160°/s ("indicative") |
| Control rate | Not published. RS485 at 115200 8N1, also CAN | same, plus Ethernet; tactile data at 30 Hz or faster | Not published. One lab's setup used 500 Hz over UART and a 100 Hz ROS demo | — |
| Force | 10 N per finger, 15 N thumb | 30 N five-finger grip | 66 N power grasp, 9.3 N pinch | — |
| Weight | about 540 g (650 g with wrist) | 790 ± 10 g | 490 g (older versions 460–470 g) | — |
| Mounting / power | 24 V; RS485/CAN; left and right versions | 24 V, up to 4.5 A | 7.4 V battery or external supply; BLE/I2C/UART/RS485 | — |

Sources for the table:
- **RH56DFX:** https://www.knoxlabs.com/products/inspire-robots-rh56dfx-dexterous-hand and the Inspire RH56 manual https://en.inspire-robots.com/wp-content/uploads/2024/02/INSPIRE-ROBOTS-THE-DEXTEROUS-HAND-RH56-SERIES-USER-MANUAL.pdf. Some listings show the faster RH56BFX's numbers (0.4 s close, 570°/s) under the DFX name, so check the datasheet for the unit you have.
- **RH56DFTP:** https://en.inspire-robots.com/wp-content/uploads/2025/01/INSPIRE-ROBOTS-The-Dexterous-Hand-RH56DFTP-User-Manual-V1.0.0.pdf and https://robosavvy.co.uk/rh56dftphands.html (about £10.4k).
- **Ability Hand:**
  - PSYONIC pages: https://www.psyonic.io/ability-hand and https://www.psyonic.io/news/psyonic-releases-ability-hand-for-research-users
  - UIUC lab code: https://github.com/uiuckimlab/psyonic_hand_control
  - Distributor speeds: https://www.spsco.com/ability-hand-system.html
- **Shadow Hand:** https://www.shadowrobot.com/wp-content/uploads/2024/06/20240610-UPDATED-shadow_dexterous_hand_e_technical_specification.pdf and https://www.shadowrobot.com/wp-content/uploads/2022/03/shadow_teleop_technical_specification.pdf. Wevolver says 120°/s, which contradicts this; the official sheet is the safer figure.

**Not found:** an official joint-angle range in degrees for each Inspire DOF (the manual only uses the 0–1000 scale), and an official command rate for any of the hands. Measure these on the hardware.
