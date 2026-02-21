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


def raven_arm_frames(jpos):
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

    frames = [np.eye(4)]
    accum = T01;          frames.append(accum.copy())
    accum = accum @ T12;  frames.append(accum.copy())
    accum = accum @ T23;  frames.append(accum.copy())
    accum = accum @ T34;  frames.append(accum.copy())
    accum = accum @ T45;  frames.append(accum.copy())
    accum = accum @ T56;  frames.append(accum.copy())
    return frames


# Frame-0 to base-frame transform
T_0B = np.array([
    [0,  0, 1, 300.71],
    [0, -1, 0, 61],
    [1,  0, 0, -7],
    [0,  0, 0, 1]
])


def to_base(point_fm0):
    p = np.array([point_fm0[0], point_fm0[1], point_fm0[2], 1.0])
    return (T_0B @ p)[:3]


def get_robot_points(jpos, samples_per_link=10):
    """Return a list of 3D points (base frame) sampling the full arm body."""
    frames = raven_arm_frames(jpos)
    origins = [to_base(f[:3, 3]) for f in frames]
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

# Workspace bounding box in base frame (determined empirically via Monte Carlo FK)
# EE reachable region sits within these bounds
WS_EE_MIN = np.array([250.0,   45.0, -160.0])
WS_EE_MAX = np.array([370.0,  175.0,  -30.0])
# Full arm sweep (all joint origins) bounds
WS_ARM_MIN = np.array([120.0, -250.0, -160.0])
WS_ARM_MAX = np.array([420.0,  175.0,  415.0])


def random_jpos():
    return np.array([np.random.uniform(lo, hi) for lo, hi in JOINT_LIMITS])


def random_jpos_in_workspace(obstacles=None, max_attempts=200):
    """Sample a random joint config whose EE lies within the workspace and is collision-free."""
    for _ in range(max_attempts):
        q = random_jpos()
        frames = raven_arm_frames(q)
        ee = to_base(frames[-1][:3, 3])
        if np.all(ee >= WS_EE_MIN) and np.all(ee <= WS_EE_MAX):
            if obstacles:
                col, _ = check_collision(q, obstacles, samples_per_link=6)
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


def check_collision(jpos, obstacles, samples_per_link=10):
    """
    Returns (collides: bool, min_dist: float).
    Collision is true if any robot centerline point is within COLLISION_MARGIN of
    an obstacle, accounting for link box width and tool sphere radius.
    """
    if not obstacles:
        return False, float('inf')
    points, _, _ = get_robot_points(jpos, samples_per_link)
    min_dist = float('inf')
    for pt in points:
        for obs in obstacles:
            d = obs.distance_to(pt)
            if d < COLLISION_MARGIN:
                return True, 0.0
            if d < min_dist:
                min_dist = d
    return False, min_dist - COLLISION_MARGIN


def min_obstacle_distance(jpos, obstacles, samples_per_link=10):
    """Return minimum distance from robot body to any obstacle."""
    _, d = check_collision(jpos, obstacles, samples_per_link)
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
             goal_thresh=15.0, goal_bias=0.1):
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
            col, _ = check_collision(q_test, obstacles, samples_per_link=6)
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
                c, _ = check_collision(q_test, obstacles, samples_per_link=6)
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


def smooth_path(path, obstacles, attempts=50):
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
            col, _ = check_collision(q, obstacles, samples_per_link=6)
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

def compute_trajectory_data(path, obstacles):
    """
    Given a path (list of jpos arrays) and obstacles, compute per-waypoint data.
    Returns a list of dicts (one per waypoint).
    """
    rows = []
    n = len(path)
    for step, jpos in enumerate(path):
        pts, origins, frames = get_robot_points(jpos, samples_per_link=10)
        ee_base = origins[-1]
        T06 = frames[-1]
        R_base = T_0B[:3, :3] @ T06[:3, :3]

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


def save_trajectory_csv(path, obstacles, filepath):
    """Save one trajectory to CSV."""
    rows = compute_trajectory_data(path, obstacles)
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
    [100, 0, 0], [100, 100, 0], [-200, 100, 0], [-200, 0, 0],
    [100, 0, 100], [100, 100, 100], [-200, 100, 100], [-200, 0, 100],
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


def sample_workspace_surfaces(n=15):
    """Generate 6 boundary surfaces of the workspace.
    Each surface fixes one of J1/J2/J3 at a limit and sweeps the other two.
    Wrist joints (J4-J7) are held at midpoints for clean surface geometry."""
    wrist_mid = [(lo + hi) / 2.0 for lo, hi in JOINT_LIMITS[3:]]
    j1r = np.linspace(JOINT_LIMITS[0][0], JOINT_LIMITS[0][1], n)
    j2r = np.linspace(JOINT_LIMITS[1][0], JOINT_LIMITS[1][1], n)
    j3r = np.linspace(JOINT_LIMITS[2][0], JOINT_LIMITS[2][1], n)
    surfaces = []

    def _ee(q_arr):
        return to_base(raven_arm_frames(q_arr)[-1][:3, 3])

    # J1 at limits → sweep J2 × J3 (red)
    for j1_lim in JOINT_LIMITS[0]:
        X, Y, Z = [np.zeros((n, n)) for _ in range(3)]
        for i, j2 in enumerate(j2r):
            for k, j3 in enumerate(j3r):
                p = _ee(np.array([j1_lim, j2, j3, *wrist_mid]))
                X[i, k], Y[i, k], Z[i, k] = p
        surfaces.append((X, Y, Z, 'red'))

    # J2 at limits → sweep J1 × J3 (blue)
    for j2_lim in JOINT_LIMITS[1]:
        X, Y, Z = [np.zeros((n, n)) for _ in range(3)]
        for i, j1 in enumerate(j1r):
            for k, j3 in enumerate(j3r):
                p = _ee(np.array([j1, j2_lim, j3, *wrist_mid]))
                X[i, k], Y[i, k], Z[i, k] = p
        surfaces.append((X, Y, Z, 'blue'))

    # J3 at limits → sweep J1 × J2 (green)
    for j3_lim in JOINT_LIMITS[2]:
        X, Y, Z = [np.zeros((n, n)) for _ in range(3)]
        for i, j1 in enumerate(j1r):
            for k, j2 in enumerate(j2r):
                p = _ee(np.array([j1, j2, j3_lim, *wrist_mid]))
                X[i, k], Y[i, k], Z[i, k] = p
        surfaces.append((X, Y, Z, 'green'))

    return surfaces


# Precompute once so every draw_arm call reuses it
_WS_SURFACES = None

def _get_ws_surfaces():
    global _WS_SURFACES
    if _WS_SURFACES is None:
        _WS_SURFACES = sample_workspace_surfaces(15)
    return _WS_SURFACES


def draw_arm(ax, jpos, obstacles=None, path_ee=None, zoom=1.0):
    """Draw the full scene: arm, obstacles, EE path.
    zoom > 1.0 zooms in, zoom < 1.0 zooms out."""
    ax.cla()

    _, origins, _ = get_robot_points(jpos)

    # Link 4: tool shaft from origins[3] to origins[4] (d=-470mm along z3)
    faces_shaft = _box_between(origins[3], origins[4], half_w=5.0)
    if faces_shaft:
        ax.add_collection3d(Poly3DCollection(
            faces_shaft, facecolors='#00CC33',
            edgecolors='k', linewidths=0.5, alpha=0.6))

    # Joints 5-6-7 as a single sphere at the tool shaft tip
    sx, sy, sz = _sphere_points(origins[4], radius=10.0)
    ax.plot_surface(sx, sy, sz, color='#CC4466', alpha=0.8)

    # Workspace boundary surfaces (J1/J2/J3 fixed at limits)
    ws_surfs = _get_ws_surfaces()
    for X, Y, Z, _ in ws_surfs:
        ax.plot_surface(X, Y, Z, color='cyan', alpha=0.06,
                        linewidth=0, edgecolor='none')
        # Draw border lines of each surface panel
        for row in [0, -1]:
            ax.plot(X[row, :], Y[row, :], Z[row, :], 'b-', linewidth=0.8, alpha=0.5)
        for col in [0, -1]:
            ax.plot(X[:, col], Y[:, col], Z[:, col], 'b-', linewidth=0.8, alpha=0.5)

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

    # Axis limits — fit to workspace surfaces + margin, scaled by zoom
    margin = 30
    all_x = np.concatenate([s[0].ravel() for s in ws_surfs])
    all_y = np.concatenate([s[1].ravel() for s in ws_surfs])
    all_z = np.concatenate([s[2].ravel() for s in ws_surfs])
    lo = np.array([all_x.min(), all_y.min(), all_z.min()]) - margin
    hi = np.array([all_x.max(), all_y.max(), all_z.max()]) + margin
    # Expand to include obstacles if they extend beyond
    if obstacles:
        for obs in obstacles:
            lo = np.minimum(lo, obs.vmin - 10)
            hi = np.maximum(hi, obs.vmax + 10)
    mid = (lo + hi) / 2
    half_range = max(hi - lo) / 2 / zoom  # zoom > 1 shrinks range (zoom in)
    ax.set_xlim(mid[0] - half_range, mid[0] + half_range)
    ax.set_ylim(mid[1] - half_range, mid[1] + half_range)
    ax.set_zlim(mid[2] - half_range, mid[2] + half_range)
    ax.set_xlabel('x_B')
    ax.set_ylabel('y_B')
    ax.set_zlabel('z_B')
    ax.set_title(f'RAVEN Arm Simulation (zoom: {zoom:.1f}x)')


# ===========================================================================
# 6. Random obstacle generation (for batch mode / interactive add)
# ===========================================================================

def random_obstacle_in_workspace():
    """Generate a random AABB obstacle inside the EE workspace region (arm-sized)."""
    size = np.array([
        np.random.uniform(5, 15),
        np.random.uniform(5, 15),
        np.random.uniform(5, 15),
    ])
    # Place entirely within workspace bounds
    center = np.array([
        np.random.uniform(WS_EE_MIN[0] + size[0] / 2, WS_EE_MAX[0] - size[0] / 2),
        np.random.uniform(WS_EE_MIN[1] + size[1] / 2, WS_EE_MAX[1] - size[1] / 2),
        np.random.uniform(WS_EE_MIN[2] + size[2] / 2, WS_EE_MAX[2] - size[2] / 2),
    ])
    vmin = np.maximum(center - size / 2, WS_EE_MIN)
    vmax = np.minimum(center + size / 2, WS_EE_MAX)
    return AABBObstacle(vmin, vmax)


# ===========================================================================
# 7. Interactive mode
# ===========================================================================

def run_interactive():
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.42, top=0.95)

    state = {
        'jpos': DEFAULT_JPOS.copy(),
        'goal': None,
        'obstacles': [],
        'path': None,
        'path_ee': None,
        'zoom': 1.0,
    }

    draw_arm(ax, state['jpos'], state['obstacles'], zoom=state['zoom'])

    # --- Joint sliders ---
    sliders = []
    for i in range(NUM_JOINTS):
        ax_s = fig.add_axes([0.12, 0.32 - i * 0.033, 0.50, 0.022])
        lo, hi = JOINT_LIMITS[i]
        s = Slider(ax_s, JOINT_NAMES[i], lo, hi, valinit=state['jpos'][i])
        sliders.append(s)

    def on_slider(val):
        for i, s in enumerate(sliders):
            state['jpos'][i] = s.val
        draw_arm(ax, state['jpos'], state['obstacles'], state['path_ee'], zoom=state['zoom'])
        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(on_slider)

    # --- Buttons ---
    ax_add_obs = fig.add_axes([0.70, 0.30, 0.12, 0.04])
    btn_add_obs = Button(ax_add_obs, 'Add Obstacle')

    ax_clr_obs = fig.add_axes([0.84, 0.30, 0.12, 0.04])
    btn_clr_obs = Button(ax_clr_obs, 'Clear Obs')

    ax_set_goal = fig.add_axes([0.70, 0.24, 0.12, 0.04])
    btn_set_goal = Button(ax_set_goal, 'Set Goal')

    ax_plan = fig.add_axes([0.84, 0.24, 0.12, 0.04])
    btn_plan = Button(ax_plan, 'Plan RRT')

    ax_animate = fig.add_axes([0.70, 0.18, 0.12, 0.04])
    btn_animate = Button(ax_animate, 'Animate')

    ax_save = fig.add_axes([0.84, 0.18, 0.12, 0.04])
    btn_save = Button(ax_save, 'Save Data')

    ax_rand_start = fig.add_axes([0.70, 0.12, 0.12, 0.04])
    btn_rand_start = Button(ax_rand_start, 'Rand Start')

    ax_rand_goal = fig.add_axes([0.84, 0.12, 0.12, 0.04])
    btn_rand_goal = Button(ax_rand_goal, 'Rand Goal')

    ax_zoom_in = fig.add_axes([0.70, 0.06, 0.12, 0.04])
    btn_zoom_in = Button(ax_zoom_in, 'Zoom In +')

    ax_zoom_out = fig.add_axes([0.84, 0.06, 0.12, 0.04])
    btn_zoom_out = Button(ax_zoom_out, 'Zoom Out -')

    info_ax = fig.add_axes([0.70, 0.01, 0.26, 0.04])
    info_ax.axis('off')
    info_text = info_ax.text(0, 0.5, 'Ready.', fontsize=9, va='center')

    def set_info(msg):
        info_text.set_text(msg)
        fig.canvas.draw_idle()

    def _redraw():
        draw_arm(ax, state['jpos'], state['obstacles'], state['path_ee'], zoom=state['zoom'])

    def on_add_obs(event):
        obs = random_obstacle_in_workspace()
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
        goal_frames = raven_arm_frames(state['jpos'])
        goal_ee = to_base(goal_frames[-1][:3, 3])
        if not (np.all(goal_ee >= WS_EE_MIN) and np.all(goal_ee <= WS_EE_MAX)):
            set_info('Goal EE is outside workspace! Adjust sliders.')
            return
        state['goal'] = state['jpos'].copy()
        _redraw()
        ax.scatter(*goal_ee, c='lime', s=120, marker='*', zorder=10, label='Goal EE')
        ax.legend(loc='upper left', fontsize=8)
        set_info(f'Goal set. EE: ({goal_ee[0]:.0f}, {goal_ee[1]:.0f}, {goal_ee[2]:.0f})')
        fig.canvas.draw_idle()

    def on_rand_start(event):
        q = random_jpos_in_workspace(state['obstacles'])
        for i in range(NUM_JOINTS):
            sliders[i].set_val(q[i])
        state['jpos'] = q.copy()
        _redraw()
        fig.canvas.draw_idle()
        set_info('Random start set (in workspace).')

    def on_rand_goal(event):
        state['goal'] = random_jpos_in_workspace(state['obstacles'])
        _redraw()
        goal_frames = raven_arm_frames(state['goal'])
        goal_ee = to_base(goal_frames[-1][:3, 3])
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
        if state['goal'] is None:
            set_info('Set a goal first!')
            return
        # Verify start is in workspace
        start_frames = raven_arm_frames(state['jpos'])
        start_ee = to_base(start_frames[-1][:3, 3])
        if not (np.all(start_ee >= WS_EE_MIN) and np.all(start_ee <= WS_EE_MAX)):
            set_info('Start EE is outside workspace! Use Rand Start.')
            return
        set_info('Planning...')
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        path = rrt_plan(state['jpos'], state['goal'], state['obstacles'],
                        max_iter=5000, step_size=8.0, goal_thresh=15.0)
        if path is None:
            set_info('RRT failed - no path found.')
            state['path'] = None
            state['path_ee'] = None
        else:
            path = smooth_path(path, state['obstacles'], attempts=50)
            state['path'] = path
            # Compute EE path in base frame
            ee_pts = []
            for q in path:
                frames = raven_arm_frames(q)
                ee_pts.append(to_base(frames[-1][:3, 3]))
            state['path_ee'] = ee_pts
            set_info(f'Path found: {len(path)} waypoints.')

        _redraw()
        fig.canvas.draw_idle()

    def on_animate(event):
        if state['path'] is None or len(state['path']) < 2:
            set_info('No path to animate. Plan first!')
            return
        path = state['path']
        n = len(path)
        for step_i, q in enumerate(path):
            draw_arm(ax, q, state['obstacles'], state['path_ee'], zoom=state['zoom'])
            # Mark goal EE
            if state['goal'] is not None:
                gf = raven_arm_frames(state['goal'])
                gee = to_base(gf[-1][:3, 3])
                ax.scatter(*gee, c='lime', s=120, marker='*', zorder=10)
            set_info(f'Animating {step_i + 1}/{n}')
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.08)
        # Leave arm at goal
        state['jpos'] = path[-1].copy()
        for i in range(NUM_JOINTS):
            sliders[i].set_val(state['jpos'][i])
        set_info(f'Animation done ({n} steps).')

    def on_save(event):
        if state['path'] is None:
            set_info('No path to save. Plan first!')
            return
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        idx = len([f for f in os.listdir(data_dir) if f.endswith('.csv')]) if os.path.isdir(data_dir) else 0
        fpath = os.path.join(data_dir, f'trajectory_{idx:04d}.csv')
        save_trajectory_csv(state['path'], state['obstacles'], fpath)
        set_info(f'Saved to {os.path.basename(fpath)}')

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

    print(f'Generating {num_episodes} RRT trajectories...')
    success_count = 0

    for ep in range(num_episodes):
        # Random obstacles (1-3)
        n_obs = random.randint(1, 3)
        obstacles = [random_obstacle_in_workspace() for _ in range(n_obs)]

        # Random collision-free start and goal within workspace
        start = random_jpos_in_workspace(obstacles)
        goal = random_jpos_in_workspace(obstacles)

        # Verify both are truly collision-free
        c1, _ = check_collision(start, obstacles, samples_per_link=6)
        c2, _ = check_collision(goal, obstacles, samples_per_link=6)
        if c1 or c2:
            print(f'  [{ep+1}/{num_episodes}] Could not find collision-free start/goal, skipping.')
            summaries.append({
                'episode': ep, 'success': False, 'num_obstacles': n_obs,
                'num_waypoints': 0, 'path_length': 0,
            })
            continue

        t0 = time.time()
        path = rrt_plan(start, goal, obstacles, max_iter=5000, step_size=8.0)
        dt = time.time() - t0

        if path is None:
            print(f'  [{ep+1}/{num_episodes}] RRT failed ({dt:.1f}s)')
            summaries.append({
                'episode': ep, 'success': False, 'num_obstacles': n_obs,
                'num_waypoints': 0, 'path_length': 0,
            })
            continue

        path = smooth_path(path, obstacles, attempts=50)

        # Compute path length in joint space
        plen = sum(_jdist(path[i], path[i+1]) for i in range(len(path)-1))

        fpath = os.path.join(data_dir, f'trajectory_{ep:04d}.csv')
        save_trajectory_csv(path, obstacles, fpath)
        success_count += 1
        print(f'  [{ep+1}/{num_episodes}] OK  {len(path)} waypoints, '
              f'length={plen:.1f}, {dt:.1f}s -> {os.path.basename(fpath)}')

        summaries.append({
            'episode': ep, 'success': True, 'num_obstacles': n_obs,
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
