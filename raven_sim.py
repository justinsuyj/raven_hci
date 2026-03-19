"""
RAVEN Arm Simulation with RRT Motion Planning and Data Collection.

Usage:
    python raven_sim.py --interactive          # GUI with sliders, obstacles, RRT
    python raven_sim.py --batch 100            # generate 100 trajectory CSVs
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ===========================================================================
# 1. Forward kinematics (from monte_carlo_raven_arm_sim_visual.ipynb)
# ===========================================================================

def DH_proximal(alpha, a, d, theta):
    deg2rad = np.pi / 180.0
    alpha = alpha * deg2rad
    theta = theta * deg2rad
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st,  0,  a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa,  ca,  d * ca],
        [0, 0, 0, 1]
    ])


def raven_left_arm_frames(jpos): # jpos = [J1, ... J7]
    """Return transforms T00..T06 (7 frames) for all joint origins."""
    th1 = jpos[0] + 205
    th2 = jpos[1] + 180
    d3  = jpos[2]
    th4 = jpos[3]
    th5 = jpos[4] - 90
    th6 = 0.5 * jpos[6] - 0.5 * jpos[5]

    T01 = DH_proximal(0,            0,    0,      th1)
    T12 = DH_proximal(75,           0,    0,      th2)
    T23 = DH_proximal(128.0,        0,    d3,     90.0)
    T34 = DH_proximal(0,            0,   -470.0,  th4)
    T45 = DH_proximal(90.0,         0,    0,      th5)
    T56 = DH_proximal(90.0,        13,    0,      th6)

    frames = [np.eye(4)]                              # T00
    accum = T01;          frames.append(accum.copy()) # T01
    accum = accum @ T12;  frames.append(accum.copy()) # T02
    accum = accum @ T23;  frames.append(accum.copy()) # T03
    accum = accum @ T34;  frames.append(accum.copy()) # T04
    accum = accum @ T45;  frames.append(accum.copy()) # T05
    accum = accum @ T56;  frames.append(accum.copy()) # T06
    return frames

def raven_right_arm_frames(jpos):
    """Return transforms T00..T06 (7 frames) for the RIGHT arm."""
    th1 = jpos[0] + 25       # vs +205 for left
    th2 = jpos[1]            # vs +180 for left
    d3  = jpos[2]
    th4 = jpos[3]
    th5 = jpos[4] - 90
    th6 = 0.5 * jpos[6] - 0.5 * jpos[5]

    T01 = DH_proximal(180.0,         0,    0,      th1)
    T12 = DH_proximal(75,            0,    0,      th2)
    T23 = DH_proximal(52.0,          0,    d3,    -90.0)
    T34 = DH_proximal(0,             0,   -470.0,  th4)
    T45 = DH_proximal(90.0,          0,    0,      th5)
    T56 = DH_proximal(90.0,         13,    0,      th6)

    frames = [np.eye(4)]                              # T00
    accum = T01;          frames.append(accum.copy()) # T01
    accum = accum @ T12;  frames.append(accum.copy()) # T02
    accum = accum @ T23;  frames.append(accum.copy()) # T03
    accum = accum @ T34;  frames.append(accum.copy()) # T04
    accum = accum @ T45;  frames.append(accum.copy()) # T05
    accum = accum @ T56;  frames.append(accum.copy()) # T06
    return frames

# Frame-0 to base-frame transforms (Equations 1-2 of reference document).
# Each arm has its own base frame at its bolt pattern on the robot platform.
T_0B_LEFT = np.array([
    [0,  0, 1, 300.71],
    [0, -1, 0, 61],
    [1,  0, 0, -7],
    [0,  0, 0, 1]
])

T_0B_RIGHT = np.array([
    [0,  0, -1, -300.71],
    [0,  1,  0,  61],
    [1,  0,  0,  -7],
    [0,  0,  0,   1]
])

# World-frame offsets for each arm's base frame.
# On the real robot the base frames sit on opposite sides of the platform
# and the linkages extend inward so the remote motion centers (frame-0 origins)
# converge above the surgical site.  These offsets position each arm's base
# in the shared world frame so the two frame-0 origins nearly overlap.
#   Left  frame-0 in its base: [300.71, 61, -7]  → world ≈ [0, 61, -7]
#   Right frame-0 in its base: [-300.71, 61, -7] → world ≈ [0, 61, -7]
_BASE_WORLD_LEFT  = np.array([-300.71, 0.0, 0.0])
_BASE_WORLD_RIGHT = np.array([ 300.71, 0.0, 0.0])

# Keep T_0B as alias for left arm (backward compat)
T_0B = T_0B_LEFT


def fm02base(point_fm0, arm='left'):
    '''Transforms a 3D point from frame-0 coordinates to world frame.
    Applies T_0B (frame-0 → arm base) then the arm's base world offset.'''
    T = T_0B_LEFT if arm == 'left' else T_0B_RIGHT
    offset = _BASE_WORLD_LEFT if arm == 'left' else _BASE_WORLD_RIGHT
    p = np.array([point_fm0[0], point_fm0[1], point_fm0[2], 1.0])
    return (T @ p)[:3] + offset



def get_robot_points(jpos, samples_per_link=10, arm='left'):
    """Return a list of 3D points (base frame) sampling the full arm body."""
    fk = raven_left_arm_frames if arm == 'left' else raven_right_arm_frames
    frames = fk(jpos)

    origins = [fm02base(f[:3, 3], arm) for f in frames]
    points = list(origins)
    for i in range(len(origins) - 1):
        for t in np.linspace(0, 1, samples_per_link + 2)[1:-1]:
            points.append(origins[i] * (1 - t) + origins[i + 1] * t)
    return points, origins, frames


# Joint limits
JOINT_LIMITS = np.array([
    [20, 60],       # J1 deg
    [55, 110],      # J2 deg
    [330, 420],     # J3 mm
    [0, 360],       # J4 deg
    [-90, 90],      # J5 deg
    [0, 90],        # J6 deg
    [0, 90],        # J7 deg
], dtype=float)

JOINT_NAMES = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']
NUM_JOINTS = 7

DEFAULT_JPOS = np.array([(lo + hi) / 2.0 for lo, hi in JOINT_LIMITS])

# Workspace bounding box in world frame (determined empirically via Monte Carlo FK)
# EE reachable region sits within these bounds
WS_EE_MIN_L = np.array([-55.0,  40.0, -160.0])
WS_EE_MAX_L = np.array([ 70.0, 175.0,  -30.0])
WS_EE_MIN_R = np.array([-70.0,  45.0, -160.0])
WS_EE_MAX_R = np.array([ 50.0, 175.0,  -30.0])
# Combined bounds covering both arms
WS_EE_MIN = np.array([-70.0,  40.0, -160.0])
WS_EE_MAX = np.array([ 70.0, 175.0,  -30.0])


def random_jpos():
    return np.array([np.random.uniform(lo, hi) for lo, hi in JOINT_LIMITS])


def random_jpos_in_workspace(obstacles=None, max_attempts=200, arm='left'):
    """Sample a random joint config whose EE lies within the workspace and is collision-free."""
    fk = raven_left_arm_frames if arm == 'left' else raven_right_arm_frames
    ws_min = WS_EE_MIN_L if arm == 'left' else WS_EE_MIN_R
    ws_max = WS_EE_MAX_L if arm == 'left' else WS_EE_MAX_R
    for _ in range(max_attempts):
        q = random_jpos()
        frames = fk(q)
        ee = fm02base(frames[-1][:3, 3], arm)
        if np.all(ee >= ws_min) and np.all(ee <= ws_max):
            if obstacles:
                col, _ = check_collision(q, obstacles, samples_per_link=6, arm=arm)
                if col:
                    continue
            return q
    return random_jpos()  # fallback


# ===========================================================================
# 2. AABB obstacles & collision detection
# ===========================================================================

class AABBObstacle:
    def __init__(self, vmin, vmax):
        self.vmin = np.asarray(vmin, dtype=float)
        self.vmax = np.asarray(vmax, dtype=float)

    def contains(self, point):
        return np.all(point >= self.vmin) and np.all(point <= self.vmax)

    def distance_to(self, point):
        """Signed distance from point to surface (negative = inside)."""
        clamped = np.clip(point, self.vmin, self.vmax)
        return np.linalg.norm(point - clamped)

    def corners(self):
        mn, mx = self.vmin, self.vmax
        return np.array([
            [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
        ])

    def faces(self):
        c = self.corners()
        return [
            [c[0], c[1], c[2], c[3]],
            [c[4], c[5], c[6], c[7]],
            [c[0], c[1], c[5], c[4]],
            [c[1], c[2], c[6], c[5]],
            [c[2], c[3], c[7], c[6]],
            [c[3], c[0], c[4], c[7]],
        ]


# Safety margin: must be >= the larger of link half_w (12) and tool sphere radius (15)
COLLISION_MARGIN = 5.0


def check_collision(jpos, obstacles, samples_per_link=10, arm='left'):
    """
    Returns (collides: bool, min_dist: float).
    Collision is true if any robot centerline point is within COLLISION_MARGIN of
    an obstacle, accounting for link box width and tool sphere radius.
    """
    if not obstacles:
        return False, float('inf')
    points, _, _ = get_robot_points(jpos, samples_per_link, arm=arm)
    min_dist = float('inf')
    for pt in points:
        for obs in obstacles:
            d = obs.distance_to(pt)
            if d < COLLISION_MARGIN:
                return True, 0.0
            if d < min_dist:
                min_dist = d
    return False, min_dist - COLLISION_MARGIN


def min_obstacle_distance(jpos, obstacles, samples_per_link=10, arm='left'):
    """Return minimum distance from robot body to any obstacle."""
    _, d = check_collision(jpos, obstacles, samples_per_link, arm=arm)
    return d


# ===========================================================================
# 3. RRT planner
# ===========================================================================

def _jdist(a, b):
    """Weighted joint-space distance (prismatic joint scaled)."""
    diff = a - b
    # Scale J3 (mm) to be comparable to degrees
    weights = np.array([1, 1, 0.1, 1, 1, 1, 1])
    return np.linalg.norm(diff * weights)


def rrt_plan(start, goal, obstacles, max_iter=5000, step_size=8.0,
             goal_thresh=15.0, goal_bias=0.1, arm='left'):
    """
    RRT in 7-D joint space.
    Returns list of joint configurations (path) or None if planning fails.
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    nodes = [start.copy()]
    parents = [-1]

    for iteration in range(max_iter):
        # Sample random config (with goal bias)
        if random.random() < goal_bias:
            q_rand = goal.copy()
        else:
            q_rand = random_jpos()

        # Find nearest node
        dists = [_jdist(q_rand, n) for n in nodes]
        nearest_idx = int(np.argmin(dists))
        q_near = nodes[nearest_idx]

        # Steer toward q_rand
        diff = q_rand - q_near
        d = _jdist(q_rand, q_near)
        if d < 1e-6:
            continue
        if d > step_size:
            q_new = q_near + diff * (step_size / d)
        else:
            q_new = q_rand.copy()

        # Clip to joint limits
        q_new = np.clip(q_new, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

        # Collision check along edge (check a few interpolated configs)
        collision = False
        for t in np.linspace(0, 1, 5):
            q_test = q_near * (1 - t) + q_new * t
            col, _ = check_collision(q_test, obstacles, samples_per_link=6, arm=arm)
            if col:
                collision = True
                break
        if collision:
            continue

        nodes.append(q_new)
        parents.append(nearest_idx)

        # Check if goal reached
        if _jdist(q_new, goal) < goal_thresh:
            # Connect to goal
            col_goal = False
            for t in np.linspace(0, 1, 5):
                q_test = q_new * (1 - t) + goal * t
                c, _ = check_collision(q_test, obstacles, samples_per_link=6, arm=arm)
                if c:
                    col_goal = True
                    break
            if not col_goal:
                nodes.append(goal.copy())
                parents.append(len(nodes) - 2)
                # Trace back
                path = []
                idx = len(nodes) - 1
                while idx != -1:
                    path.append(nodes[idx])
                    idx = parents[idx]
                path.reverse()
                return path

    return None  # planning failed


def smooth_path(path, obstacles, attempts=50, arm='left'):
    """Shortcut-based path smoothing."""
    path = [p.copy() for p in path]
    for _ in range(attempts):
        if len(path) <= 2:
            break
        i = random.randint(0, len(path) - 3)
        j = random.randint(i + 2, len(path) - 1)
        # Check if shortcut is collision-free
        ok = True
        for t in np.linspace(0, 1, max(j - i, 5)):
            q = path[i] * (1 - t) + path[j] * t
            q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
            col, _ = check_collision(q, obstacles, samples_per_link=6, arm=arm)
            if col:
                ok = False
                break
        if ok:
            # Replace segment with direct connection
            new_seg = [path[i] * (1 - t) + path[j] * t
                       for t in np.linspace(0, 1, max(3, (j - i) // 2))]
            path = path[:i] + new_seg + path[j + 1:]
    return path


# ===========================================================================
# 4. Data recording
# ===========================================================================

def compute_trajectory_data(path, obstacles, arm='left'):
    """
    Given a path (list of jpos arrays) and obstacles, compute per-waypoint data.
    Returns a list of dicts (one per waypoint).
    """
    T_0B_arm = T_0B_LEFT if arm == 'left' else T_0B_RIGHT
    rows = []
    n = len(path)
    for step, jpos in enumerate(path):
        pts, origins, frames = get_robot_points(jpos, samples_per_link=10, arm=arm)
        ee_base = origins[-1]
        T06 = frames[-1]
        R_base = T_0B_arm[:3, :3] @ T06[:3, :3]

        # Joint velocities via finite differences
        if step == 0:
            vel = np.zeros(NUM_JOINTS)
        elif step == n - 1:
            vel = np.zeros(NUM_JOINTS)
        else:
            vel = (path[step + 1] - path[step - 1]) / 2.0

        # Min distance to obstacles
        min_d = float('inf')
        for pt in pts:
            for obs in obstacles:
                d = obs.distance_to(pt)
                if d < min_d:
                    min_d = d

        row = {'step': step}
        for j in range(NUM_JOINTS):
            row[f'j{j+1}'] = jpos[j]
        row['ee_x'] = ee_base[0]
        row['ee_y'] = ee_base[1]
        row['ee_z'] = ee_base[2]
        for ri in range(3):
            for ci in range(3):
                row[f'ee_R{ri}{ci}'] = R_base[ri, ci]
        for j in range(NUM_JOINTS):
            row[f'vj{j+1}'] = vel[j]

        # Obstacle info (pad up to max obstacles, store all)
        for oi, obs in enumerate(obstacles):
            row[f'obs{oi}_min_x'] = obs.vmin[0]
            row[f'obs{oi}_min_y'] = obs.vmin[1]
            row[f'obs{oi}_min_z'] = obs.vmin[2]
            row[f'obs{oi}_max_x'] = obs.vmax[0]
            row[f'obs{oi}_max_y'] = obs.vmax[1]
            row[f'obs{oi}_max_z'] = obs.vmax[2]

        row['min_obs_dist'] = min_d
        rows.append(row)
    return rows


def save_trajectory_csv(path, obstacles, filepath, arm='left'):
    """Save one trajectory to CSV."""
    rows = compute_trajectory_data(path, obstacles, arm=arm)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_batch_summary(summaries, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not summaries:
        return
    fieldnames = list(summaries[0].keys())
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


# ===========================================================================
# 5. Drawing helpers
# ===========================================================================

BASE_CORNERS = np.array([
    [80, 0, 0], [80, 100, 0], [-80, 100, 0], [-80, 0, 0],
    [80, 0, 80], [80, 100, 80], [-80, 100, 80], [-80, 0, 80],
])
BASE_FACES = [
    [BASE_CORNERS[i] for i in [0, 1, 2, 3]],
    [BASE_CORNERS[i] for i in [4, 5, 6, 7]],
    [BASE_CORNERS[i] for i in [0, 3, 7, 4]],
    [BASE_CORNERS[i] for i in [1, 2, 6, 5]],
    [BASE_CORNERS[i] for i in [0, 1, 5, 4]],
    [BASE_CORNERS[i] for i in [2, 3, 7, 6]],
]


def _box_between(p0, p1, half_w=12.0):
    d = p1 - p0
    length = np.linalg.norm(d)
    if length < 1e-6:
        return []
    z = d / length
    ref = np.array([1, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1, 0])
    x = np.cross(z, ref)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    dx, dy = x * half_w, y * half_w
    c = [p0 - dx - dy, p0 + dx - dy, p0 + dx + dy, p0 - dx + dy,
         p1 - dx - dy, p1 + dx - dy, p1 + dx + dy, p1 - dx + dy]
    return [
        [c[0], c[1], c[2], c[3]], [c[4], c[5], c[6], c[7]],
        [c[0], c[1], c[5], c[4]], [c[1], c[2], c[6], c[5]],
        [c[2], c[3], c[7], c[6]], [c[3], c[0], c[4], c[7]],
    ]


def _sphere_points(center, radius, n=10):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def sample_workspace_surfaces(n=15, arm='left'):
    """Generate 6 boundary surfaces of the workspace for one arm.
    Each surface fixes one of J1/J2/J3 at a limit and sweeps the other two.
    Wrist joints (J4-J7) are held at midpoints for clean surface geometry."""
    fk = raven_left_arm_frames if arm == 'left' else raven_right_arm_frames
    wrist_mid = [(lo + hi) / 2.0 for lo, hi in JOINT_LIMITS[3:]]
    j1r = np.linspace(JOINT_LIMITS[0][0], JOINT_LIMITS[0][1], n)
    j2r = np.linspace(JOINT_LIMITS[1][0], JOINT_LIMITS[1][1], n)
    j3r = np.linspace(JOINT_LIMITS[2][0], JOINT_LIMITS[2][1], n)
    surfaces = []

    def _ee(q_arr):
        return fm02base(fk(q_arr)[-1][:3, 3], arm)

    # J1 at limits → sweep J2 × J3
    for j1_lim in JOINT_LIMITS[0]:
        X, Y, Z = [np.zeros((n, n)) for _ in range(3)]
        for i, j2 in enumerate(j2r):
            for k, j3 in enumerate(j3r):
                p = _ee(np.array([j1_lim, j2, j3, *wrist_mid]))
                X[i, k], Y[i, k], Z[i, k] = p
        surfaces.append((X, Y, Z))

    # J2 at limits → sweep J1 × J3
    for j2_lim in JOINT_LIMITS[1]:
        X, Y, Z = [np.zeros((n, n)) for _ in range(3)]
        for i, j1 in enumerate(j1r):
            for k, j3 in enumerate(j3r):
                p = _ee(np.array([j1, j2_lim, j3, *wrist_mid]))
                X[i, k], Y[i, k], Z[i, k] = p
        surfaces.append((X, Y, Z))

    # J3 at limits → sweep J1 × J2
    for j3_lim in JOINT_LIMITS[2]:
        X, Y, Z = [np.zeros((n, n)) for _ in range(3)]
        for i, j1 in enumerate(j1r):
            for k, j2 in enumerate(j2r):
                p = _ee(np.array([j1, j2, j3_lim, *wrist_mid]))
                X[i, k], Y[i, k], Z[i, k] = p
        surfaces.append((X, Y, Z))

    return surfaces


# Precompute once so every draw_arm call reuses it
_WS_SURFACES_L = None
_WS_SURFACES_R = None

def _get_ws_surfaces():
    global _WS_SURFACES_L, _WS_SURFACES_R
    if _WS_SURFACES_L is None:
        _WS_SURFACES_L = sample_workspace_surfaces(15, 'left')
    if _WS_SURFACES_R is None:
        _WS_SURFACES_R = sample_workspace_surfaces(15, 'right')
    return _WS_SURFACES_L, _WS_SURFACES_R


def _draw_one_arm(ax, jpos, arm, shaft_color, sphere_color):
    """Draw a single arm (tool shaft + EE sphere)."""
    _, origins, _ = get_robot_points(jpos, arm=arm)
    faces_shaft = _box_between(origins[3], origins[4], half_w=5.0)
    if faces_shaft:
        ax.add_collection3d(Poly3DCollection(
            faces_shaft, facecolors=shaft_color,
            edgecolors='k', linewidths=0.5, alpha=0.6))
    sx, sy, sz = _sphere_points(origins[4], radius=10.0)
    ax.plot_surface(sx, sy, sz, color=sphere_color, alpha=0.8)


def draw_arm(ax, jpos_left, jpos_right=None, obstacles=None, path_ee=None, zoom=1.0):
    """Draw the full scene: both arms, obstacles, EE path.
    zoom > 1.0 zooms in, zoom < 1.0 zooms out."""
    ax.cla()

    # Left arm (gold): green shaft, pink sphere
    _draw_one_arm(ax, jpos_left, 'left', '#00CC33', '#CC4466')

    # Right arm (green): blue shaft, orange sphere
    if jpos_right is not None:
        _draw_one_arm(ax, jpos_right, 'right', '#3377DD', '#DD8844')

    # Workspace boundary surfaces — unified color for both arms
    ws_L, ws_R = _get_ws_surfaces()
    all_surfs = ws_L + ws_R
    ws_color = '#B0C4DE'   # light steel blue
    ws_border = '#4682B4'  # steel blue
    for X, Y, Z in all_surfs:
        ax.plot_surface(X, Y, Z, color=ws_color, alpha=0.06,
                        linewidth=0, edgecolor='none')
        for row in [0, -1]:
            ax.plot(X[row, :], Y[row, :], Z[row, :], color=ws_border,
                    linewidth=0.8, alpha=0.5)
        for col in [0, -1]:
            ax.plot(X[:, col], Y[:, col], Z[:, col], color=ws_border,
                    linewidth=0.8, alpha=0.5)

    # Obstacles
    if obstacles:
        for obs in obstacles:
            ax.add_collection3d(Poly3DCollection(
                obs.faces(), facecolors='red', edgecolors='darkred',
                linewidths=0.8, alpha=0.35))

    # EE path trace
    if path_ee is not None and len(path_ee) > 1:
        pe = np.array(path_ee)
        ax.plot(pe[:, 0], pe[:, 1], pe[:, 2], 'g-', linewidth=2.5, label='EE path')

    # Axis limits — fit to both workspaces + margin, scaled by zoom
    margin = 30
    all_x = np.concatenate([s[0].ravel() for s in all_surfs])
    all_y = np.concatenate([s[1].ravel() for s in all_surfs])
    all_z = np.concatenate([s[2].ravel() for s in all_surfs])
    lo = np.array([all_x.min(), all_y.min(), all_z.min()]) - margin
    hi = np.array([all_x.max(), all_y.max(), all_z.max()]) + margin
    if obstacles:
        for obs in obstacles:
            lo = np.minimum(lo, obs.vmin - 10)
            hi = np.maximum(hi, obs.vmax + 10)
    mid = (lo + hi) / 2
    half_range = max(hi - lo) / 2 / zoom
    ax.set_xlim(mid[0] - half_range, mid[0] + half_range)
    ax.set_ylim(mid[1] - half_range, mid[1] + half_range)
    ax.set_zlim(mid[2] - half_range, mid[2] + half_range)
    ax.set_xlabel('x_B')
    ax.set_ylabel('y_B')
    ax.set_zlabel('z_B')
    ax.set_title(f'RAVEN II Dual-Arm Simulation (zoom: {zoom:.1f}x)')


# ===========================================================================
# 6. Random obstacle generation (for batch mode / interactive add)
# ===========================================================================

def random_obstacle_in_workspace(arm='left'):
    """Generate a random AABB obstacle inside the EE workspace region for one arm."""
    ws_min = WS_EE_MIN_L if arm == 'left' else WS_EE_MIN_R
    ws_max = WS_EE_MAX_L if arm == 'left' else WS_EE_MAX_R
    size = np.array([
        np.random.uniform(5, 15),
        np.random.uniform(5, 15),
        np.random.uniform(5, 15),
    ])
    center = np.array([
        np.random.uniform(ws_min[0] + size[0] / 2, ws_max[0] - size[0] / 2),
        np.random.uniform(ws_min[1] + size[1] / 2, ws_max[1] - size[1] / 2),
        np.random.uniform(ws_min[2] + size[2] / 2, ws_max[2] - size[2] / 2),
    ])
    vmin = np.maximum(center - size / 2, ws_min)
    vmax = np.minimum(center + size / 2, ws_max)
    return AABBObstacle(vmin, vmax)


# ===========================================================================
# 7. Interactive mode
# ===========================================================================

def run_interactive():
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.38, top=0.95)

    state = {
        'jpos_L': DEFAULT_JPOS.copy(),
        'jpos_R': DEFAULT_JPOS.copy(),
        'active_arm': 'left',   # which arm RRT/goal/obstacles operate on
        'goal': None,
        'obstacles': [],
        'path': None,
        'path_ee': None,
        'zoom': 1.0,
    }

    draw_arm(ax, state['jpos_L'], state['jpos_R'], state['obstacles'], zoom=state['zoom'])

    # --- Left arm sliders (left column) ---
    sliders_L = []
    for i in range(NUM_JOINTS):
        ax_s = fig.add_axes([0.05, 0.30 - i * 0.030, 0.25, 0.020])
        lo, hi = JOINT_LIMITS[i]
        s = Slider(ax_s, f'L-{JOINT_NAMES[i]}', lo, hi, valinit=state['jpos_L'][i])
        sliders_L.append(s)

    # --- Right arm sliders (right column) ---
    sliders_R = []
    for i in range(NUM_JOINTS):
        ax_s = fig.add_axes([0.35, 0.30 - i * 0.030, 0.25, 0.020])
        lo, hi = JOINT_LIMITS[i]
        s = Slider(ax_s, f'R-{JOINT_NAMES[i]}', lo, hi, valinit=state['jpos_R'][i])
        sliders_R.append(s)

    def on_slider(val):
        for i, s in enumerate(sliders_L):
            state['jpos_L'][i] = s.val
        for i, s in enumerate(sliders_R):
            state['jpos_R'][i] = s.val
        draw_arm(ax, state['jpos_L'], state['jpos_R'], state['obstacles'],
                 state['path_ee'], zoom=state['zoom'])
        fig.canvas.draw_idle()

    for s in sliders_L:
        s.on_changed(on_slider)
    for s in sliders_R:
        s.on_changed(on_slider)

    # --- Buttons ---
    ax_add_obs = fig.add_axes([0.70, 0.28, 0.12, 0.04])
    btn_add_obs = Button(ax_add_obs, 'Add Obstacle')

    ax_clr_obs = fig.add_axes([0.84, 0.28, 0.12, 0.04])
    btn_clr_obs = Button(ax_clr_obs, 'Clear Obs')

    ax_set_goal = fig.add_axes([0.70, 0.22, 0.12, 0.04])
    btn_set_goal = Button(ax_set_goal, 'Set Goal')

    ax_plan = fig.add_axes([0.84, 0.22, 0.12, 0.04])
    btn_plan = Button(ax_plan, 'Plan RRT')

    ax_animate = fig.add_axes([0.70, 0.16, 0.12, 0.04])
    btn_animate = Button(ax_animate, 'Animate')

    ax_save = fig.add_axes([0.84, 0.16, 0.12, 0.04])
    btn_save = Button(ax_save, 'Save Data')

    ax_rand_start = fig.add_axes([0.70, 0.10, 0.12, 0.04])
    btn_rand_start = Button(ax_rand_start, 'Rand Start')

    ax_rand_goal = fig.add_axes([0.84, 0.10, 0.12, 0.04])
    btn_rand_goal = Button(ax_rand_goal, 'Rand Goal')

    ax_zoom_in = fig.add_axes([0.70, 0.04, 0.12, 0.04])
    btn_zoom_in = Button(ax_zoom_in, 'Zoom In +')

    ax_zoom_out = fig.add_axes([0.84, 0.04, 0.12, 0.04])
    btn_zoom_out = Button(ax_zoom_out, 'Zoom Out -')

    ax_toggle_arm = fig.add_axes([0.70, 0.34, 0.26, 0.04])
    btn_toggle_arm = Button(ax_toggle_arm, 'Active: LEFT arm')

    info_ax = fig.add_axes([0.70, 0.00, 0.26, 0.03])
    info_ax.axis('off')
    info_text = info_ax.text(0, 0.5, 'Ready.', fontsize=9, va='center')

    def set_info(msg):
        info_text.set_text(msg)
        fig.canvas.draw_idle()

    def _active_arm():
        return state['active_arm']

    def _active_fk():
        return raven_left_arm_frames if _active_arm() == 'left' else raven_right_arm_frames

    def _active_jpos_key():
        return 'jpos_L' if _active_arm() == 'left' else 'jpos_R'

    def _active_sliders():
        return sliders_L if _active_arm() == 'left' else sliders_R

    def _active_ws():
        arm = _active_arm()
        ws_min = WS_EE_MIN_L if arm == 'left' else WS_EE_MIN_R
        ws_max = WS_EE_MAX_L if arm == 'left' else WS_EE_MAX_R
        return ws_min, ws_max

    def _redraw():
        draw_arm(ax, state['jpos_L'], state['jpos_R'], state['obstacles'],
                 state['path_ee'], zoom=state['zoom'])

    def on_toggle_arm(event):
        if state['active_arm'] == 'left':
            state['active_arm'] = 'right'
        else:
            state['active_arm'] = 'left'
        label = 'LEFT' if state['active_arm'] == 'left' else 'RIGHT'
        btn_toggle_arm.label.set_text(f'Active: {label} arm')
        state['goal'] = None
        state['path'] = None
        state['path_ee'] = None
        set_info(f'Active arm: {label}')
        _redraw()
        fig.canvas.draw_idle()

    def on_add_obs(event):
        obs = random_obstacle_in_workspace(arm=_active_arm())
        state['obstacles'].append(obs)
        set_info(f'{len(state["obstacles"])} obstacle(s)')
        _redraw()
        fig.canvas.draw_idle()

    def on_clr_obs(event):
        state['obstacles'].clear()
        state['path'] = None
        state['path_ee'] = None
        set_info('Obstacles cleared.')
        _redraw()
        fig.canvas.draw_idle()

    def on_set_goal(event):
        arm = _active_arm()
        jkey = _active_jpos_key()
        fk = _active_fk()
        ws_min, ws_max = _active_ws()
        goal_frames = fk(state[jkey])
        goal_ee = fm02base(goal_frames[-1][:3, 3], arm)
        if not (np.all(goal_ee >= ws_min) and np.all(goal_ee <= ws_max)):
            set_info('Goal EE is outside workspace! Adjust sliders.')
            return
        state['goal'] = state[jkey].copy()
        _redraw()
        ax.scatter(*goal_ee, c='lime', s=120, marker='*', zorder=10, label='Goal EE')
        ax.legend(loc='upper left', fontsize=8)
        set_info(f'Goal set. EE: ({goal_ee[0]:.0f}, {goal_ee[1]:.0f}, {goal_ee[2]:.0f})')
        fig.canvas.draw_idle()

    def on_rand_start(event):
        arm = _active_arm()
        jkey = _active_jpos_key()
        sliders = _active_sliders()
        q = random_jpos_in_workspace(state['obstacles'], arm=arm)
        for i in range(NUM_JOINTS):
            sliders[i].set_val(q[i])
        state[jkey] = q.copy()
        _redraw()
        fig.canvas.draw_idle()
        label = 'L' if arm == 'left' else 'R'
        set_info(f'Random {label} start set (in workspace).')

    def on_rand_goal(event):
        arm = _active_arm()
        fk = _active_fk()
        state['goal'] = random_jpos_in_workspace(state['obstacles'], arm=arm)
        _redraw()
        goal_frames = fk(state['goal'])
        goal_ee = fm02base(goal_frames[-1][:3, 3], arm)
        ax.scatter(*goal_ee, c='lime', s=120, marker='*', zorder=10, label='Goal EE')
        ax.legend(loc='upper left', fontsize=8)
        set_info(f'Goal EE: ({goal_ee[0]:.0f}, {goal_ee[1]:.0f}, {goal_ee[2]:.0f})')
        fig.canvas.draw_idle()

    def on_zoom_in(event):
        state['zoom'] = min(state['zoom'] * 1.3, 10.0)
        _redraw()
        set_info(f'Zoom: {state["zoom"]:.1f}x')
        fig.canvas.draw_idle()

    def on_zoom_out(event):
        state['zoom'] = max(state['zoom'] / 1.3, 0.2)
        _redraw()
        set_info(f'Zoom: {state["zoom"]:.1f}x')
        fig.canvas.draw_idle()

    def on_plan(event):
        arm = _active_arm()
        jkey = _active_jpos_key()
        fk = _active_fk()
        ws_min, ws_max = _active_ws()
        if state['goal'] is None:
            set_info('Set a goal first!')
            return
        start_frames = fk(state[jkey])
        start_ee = fm02base(start_frames[-1][:3, 3], arm)
        if not (np.all(start_ee >= ws_min) and np.all(start_ee <= ws_max)):
            set_info('Start EE is outside workspace! Use Rand Start.')
            return
        set_info('Planning...')
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        path = rrt_plan(state[jkey], state['goal'], state['obstacles'],
                        max_iter=5000, step_size=8.0, goal_thresh=15.0, arm=arm)
        if path is None:
            set_info('RRT failed - no path found.')
            state['path'] = None
            state['path_ee'] = None
        else:
            path = smooth_path(path, state['obstacles'], attempts=50, arm=arm)
            state['path'] = path
            ee_pts = []
            for q in path:
                frames = fk(q)
                ee_pts.append(fm02base(frames[-1][:3, 3], arm))
            state['path_ee'] = ee_pts
            set_info(f'Path found: {len(path)} waypoints.')

        _redraw()
        fig.canvas.draw_idle()

    def on_animate(event):
        if state['path'] is None or len(state['path']) < 2:
            set_info('No path to animate. Plan first!')
            return
        arm = _active_arm()
        jkey = _active_jpos_key()
        fk = _active_fk()
        sliders = _active_sliders()
        path = state['path']
        n = len(path)
        for step_i, q in enumerate(path):
            # Update the active arm's jpos for drawing
            jpos_L = q if arm == 'left' else state['jpos_L']
            jpos_R = q if arm == 'right' else state['jpos_R']
            draw_arm(ax, jpos_L, jpos_R, state['obstacles'],
                     state['path_ee'], zoom=state['zoom'])
            if state['goal'] is not None:
                gf = fk(state['goal'])
                gee = fm02base(gf[-1][:3, 3], arm)
                ax.scatter(*gee, c='lime', s=120, marker='*', zorder=10)
            set_info(f'Animating {step_i + 1}/{n}')
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.08)
        state[jkey] = path[-1].copy()
        for i in range(NUM_JOINTS):
            sliders[i].set_val(state[jkey][i])
        set_info(f'Animation done ({n} steps).')

    def on_save(event):
        if state['path'] is None:
            set_info('No path to save. Plan first!')
            return
        arm = _active_arm()
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        idx = len([f for f in os.listdir(data_dir) if f.endswith('.csv')]) if os.path.isdir(data_dir) else 0
        fpath = os.path.join(data_dir, f'trajectory_{idx:04d}.csv')
        save_trajectory_csv(state['path'], state['obstacles'], fpath, arm=arm)
        set_info(f'Saved to {os.path.basename(fpath)}')

    btn_toggle_arm.on_clicked(on_toggle_arm)
    btn_add_obs.on_clicked(on_add_obs)
    btn_clr_obs.on_clicked(on_clr_obs)
    btn_set_goal.on_clicked(on_set_goal)
    btn_plan.on_clicked(on_plan)
    btn_animate.on_clicked(on_animate)
    btn_save.on_clicked(on_save)
    btn_rand_start.on_clicked(on_rand_start)
    btn_rand_goal.on_clicked(on_rand_goal)
    btn_zoom_in.on_clicked(on_zoom_in)
    btn_zoom_out.on_clicked(on_zoom_out)

    plt.show()


# ===========================================================================
# 8. Batch mode
# ===========================================================================

def run_batch(num_episodes):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    summaries = []

    print(f'Generating {num_episodes} RRT trajectories (both arms)...')
    success_count = 0

    for ep in range(num_episodes):
        # Alternate arms each episode
        arm = 'left' if ep % 2 == 0 else 'right'

        # Random obstacles (1-3) in the active arm's workspace
        n_obs = random.randint(1, 3)
        obstacles = [random_obstacle_in_workspace(arm=arm) for _ in range(n_obs)]

        # Random collision-free start and goal within workspace
        start = random_jpos_in_workspace(obstacles, arm=arm)
        goal = random_jpos_in_workspace(obstacles, arm=arm)

        # Verify both are truly collision-free
        c1, _ = check_collision(start, obstacles, samples_per_link=6, arm=arm)
        c2, _ = check_collision(goal, obstacles, samples_per_link=6, arm=arm)
        if c1 or c2:
            print(f'  [{ep+1}/{num_episodes}] ({arm}) Could not find collision-free start/goal, skipping.')
            summaries.append({
                'episode': ep, 'arm': arm, 'success': False, 'num_obstacles': n_obs,
                'num_waypoints': 0, 'path_length': 0,
            })
            continue

        t0 = time.time()
        path = rrt_plan(start, goal, obstacles, max_iter=5000, step_size=8.0, arm=arm)
        dt = time.time() - t0

        if path is None:
            print(f'  [{ep+1}/{num_episodes}] ({arm}) RRT failed ({dt:.1f}s)')
            summaries.append({
                'episode': ep, 'arm': arm, 'success': False, 'num_obstacles': n_obs,
                'num_waypoints': 0, 'path_length': 0,
            })
            continue

        path = smooth_path(path, obstacles, attempts=50, arm=arm)

        # Compute path length in joint space
        plen = sum(_jdist(path[i], path[i+1]) for i in range(len(path)-1))

        fpath = os.path.join(data_dir, f'trajectory_{ep:04d}.csv')
        save_trajectory_csv(path, obstacles, fpath, arm=arm)
        success_count += 1
        print(f'  [{ep+1}/{num_episodes}] ({arm}) OK  {len(path)} waypoints, '
              f'length={plen:.1f}, {dt:.1f}s -> {os.path.basename(fpath)}')

        summaries.append({
            'episode': ep, 'arm': arm, 'success': True, 'num_obstacles': n_obs,
            'num_waypoints': len(path), 'path_length': round(plen, 2),
            'start': start.tolist(), 'goal': goal.tolist(),
        })

    summary_path = os.path.join(data_dir, 'batch_summary.csv')
    save_batch_summary(summaries, summary_path)
    print(f'\nDone. {success_count}/{num_episodes} successful. '
          f'Summary: {summary_path}')


# ===========================================================================
# 9. Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='RAVEN Arm RRT Simulation')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--interactive', action='store_true', default=True,
                       help='Launch interactive GUI (default)')
    group.add_argument('--batch', type=int, metavar='N',
                       help='Generate N trajectory datasets')
    args = parser.parse_args()

    if args.batch:
        run_batch(args.batch)
    else:
        run_interactive()


if __name__ == '__main__':
    main()
