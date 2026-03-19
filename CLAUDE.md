# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAVEN II dual-arm surgical robot simulation with RRT motion planning and trajectory data collection. The Python simulation (`raven_sim.py`) is standalone and does not depend on the `raven2/` C++ ROS control package, though both implement the same robot kinematics.

## Running the Simulation

```bash
# Interactive GUI mode (default) - 3D visualization with sliders and buttons
python raven_sim.py --interactive

# Batch mode - generate N trajectory datasets to data/
python raven_sim.py --batch 100
```

Dependencies: `numpy`, `matplotlib` (no requirements.txt — install manually via pip).

## Architecture of raven_sim.py

The simulation is organized in sequential sections (~1025 lines):

1. **Forward Kinematics** (lines ~25-127): `DH_proximal()` builds 4x4 DH matrices. `raven_left_arm_frames()` and `raven_right_arm_frames()` compute FK with different DH parameters per arm (alpha, theta offsets differ). `fm02base()` transforms Frame-0 coordinates to base frame via T_0B matrices.

2. **Collision Detection** (lines ~130-180): `AABBObstacle` class with `check_collision()` using sampled points along links (not full geometry). 5mm safety margin.

3. **RRT Planner** (lines ~250-360): `rrt_plan()` with goal bias, `smooth_path()` with shortcut attempts. All planning functions take an `arm='left'|'right'` parameter.

4. **Data Recording** (lines ~360-480): `compute_trajectory_data()` and `save_trajectory_csv()` output 42-column CSVs with joint positions, EE pose, velocities, obstacle distances.

5. **Interactive Mode** (lines ~645-870): Matplotlib GUI with 14 joint sliders (7 per arm), workspace visualization, obstacle placement, RRT planning, and arm toggle button.

6. **Batch Mode** (lines ~876-940): Automated trajectory generation alternating left/right arms.

## Key Kinematic Details

- **7 DOF per arm**: J1-J2 revolute (shoulder), J3 prismatic (insertion), J4 revolute (tool roll), J5-J7 revolute (wrist/grasp)
- **DH convention**: Craig's modified/proximal convention
- **Left vs Right arm differences**: Different alpha values (e.g., alpha1: 0° vs 180°, alpha3: 128° vs 52°), different theta offsets (e.g., th1: +205° vs +25°), different T_0B transforms
- **T_0B matrices**: Transform from each arm's Frame-0 (remote motion center) to that arm's own Base Frame. The real C++ controller (`raven2/src/raven/r2_kinematics.cpp`) does NOT apply T_0B — it works entirely in Frame-0 coordinates.

## raven2/ C++ Reference Code

Separate git repo containing the real robot's ROS control software. Useful as a reference for:
- DH parameters and FK/IK: `r2_kinematics.cpp` / `r2_kinematics.h`
- Mass/COM data for gravity compensation: `grav_comp.cpp`
- Joint limits, transmission ratios, cable geometry: `defines.h`
- Control gains: `params/r2params.yaml`

## Reference Document

`Kinematic analysis of Raven.pdf` — Source of truth for DH parameters, T_0B transforms (Equations 1-2), joint offset equations, and frame assignments. Extracted text available in `page1.txt` through `page22.txt`.
