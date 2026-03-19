"""
Data collection for left-arm planning policy training.

The right arm randomly moves between 3 waypoints in its workspace,
acting as a dynamic obstacle. For each episode, the left arm plans
a collision-free trajectory (via RRT) that avoids the right arm's
current body, and the trajectory data is recorded.

Usage:
    python collect_planning_data.py --episodes 100
    python collect_planning_data.py --episodes 50 --out data/planning
"""

import argparse
import csv
import os
import random
import time

import numpy as np

from raven_sim import (
    raven_left_arm_frames, raven_right_arm_frames,
    fm02base, get_robot_points,
    T_0B_LEFT, T_0B_RIGHT,
    JOINT_LIMITS, NUM_JOINTS, DEFAULT_JPOS,
    AABBObstacle, check_collision, COLLISION_MARGIN,
    random_jpos, random_jpos_in_workspace,
    rrt_plan, smooth_path, _jdist,
)


# ---------------------------------------------------------------------------
# Right-arm body as sampled points (for collision checking)
# ---------------------------------------------------------------------------

# Frames 0-2 are at the shared remote motion center (not physical bodies).
# Only frames 3+ (tool insertion + instrument) have collision geometry.
_FIRST_COLLISION_FRAME = 3


def _tool_points(jpos, arm, samples_per_link=6):
    """Return sampled 3D points along the tool part of the arm (frames 3+)."""
    fk = raven_left_arm_frames if arm == 'left' else raven_right_arm_frames
    frames = fk(jpos)
    origins = [fm02base(f[:3, 3], arm) for f in frames]
    # Only keep frames from the tool onward
    tool_origins = origins[_FIRST_COLLISION_FRAME:]
    points = list(tool_origins)
    for i in range(len(tool_origins) - 1):
        for t in np.linspace(0, 1, samples_per_link + 2)[1:-1]:
            points.append(tool_origins[i] * (1 - t) + tool_origins[i + 1] * t)
    return points


def get_right_arm_points(jpos_right, samples_per_link=6):
    """Return sampled 3D points along the right arm tool body."""
    return _tool_points(jpos_right, 'right', samples_per_link)


def check_collision_with_right_arm(jpos_left, right_arm_pts, safety=15.0):
    """Check if left arm tool collides with pre-computed right arm body points.

    Returns (collides, min_dist).
    """
    left_pts = _tool_points(jpos_left, 'left', samples_per_link=6)
    min_d = float('inf')
    for lp in left_pts:
        for rp in right_arm_pts:
            d = np.linalg.norm(lp - rp)
            if d < safety:
                return True, d
            if d < min_d:
                min_d = d
    return False, min_d


def check_collision_dual(jpos_left, static_obstacles, right_arm_pts, safety=15.0):
    """Combined collision check: left arm vs static obstacles AND right arm body."""
    # Static obstacles first (fast AABB check)
    if static_obstacles:
        col, d_static = check_collision(jpos_left, static_obstacles,
                                        samples_per_link=6, arm='left')
        if col:
            return True, 0.0
    else:
        d_static = float('inf')

    # Right arm body (point-to-point)
    col, d_right = check_collision_with_right_arm(jpos_left, right_arm_pts, safety)
    if col:
        return True, d_right
    return False, min(d_static, d_right)


def rrt_plan_dual(start, goal, static_obstacles, right_arm_pts,
                  max_iter=5000, step_size=12.0,
                  goal_thresh=20.0, goal_bias=0.15):
    """RRT planner for the left arm avoiding both static obstacles and the right arm."""
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    nodes = [start.copy()]
    parents = [-1]

    for _ in range(max_iter):
        if random.random() < goal_bias:
            q_rand = goal.copy()
        else:
            q_rand = random_jpos()

        dists = [_jdist(q_rand, n) for n in nodes]
        nearest_idx = int(np.argmin(dists))
        q_near = nodes[nearest_idx]

        diff = q_rand - q_near
        d = _jdist(q_rand, q_near)
        if d < 1e-6:
            continue
        if d > step_size:
            q_new = q_near + diff * (step_size / d)
        else:
            q_new = q_rand.copy()

        q_new = np.clip(q_new, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

        collision = False
        for t in np.linspace(0, 1, 5):
            q_test = q_near * (1 - t) + q_new * t
            col, _ = check_collision_dual(q_test, static_obstacles, right_arm_pts)
            if col:
                collision = True
                break
        if collision:
            continue

        nodes.append(q_new)
        parents.append(nearest_idx)

        if _jdist(q_new, goal) < goal_thresh:
            col_goal = False
            for t in np.linspace(0, 1, 5):
                q_test = q_new * (1 - t) + goal * t
                c, _ = check_collision_dual(q_test, static_obstacles, right_arm_pts)
                if c:
                    col_goal = True
                    break
            if not col_goal:
                nodes.append(goal.copy())
                parents.append(len(nodes) - 2)
                path = []
                idx = len(nodes) - 1
                while idx != -1:
                    path.append(nodes[idx])
                    idx = parents[idx]
                path.reverse()
                return path

    return None


def smooth_path_dual(path, static_obstacles, right_arm_pts, attempts=50):
    """Shortcut smoothing for left arm avoiding static obs + right arm."""
    path = [p.copy() for p in path]
    for _ in range(attempts):
        if len(path) <= 2:
            break
        i = random.randint(0, len(path) - 3)
        j = random.randint(i + 2, len(path) - 1)
        ok = True
        for t in np.linspace(0, 1, max(j - i, 5)):
            q = path[i] * (1 - t) + path[j] * t
            q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
            col, _ = check_collision_dual(q, static_obstacles, right_arm_pts)
            if col:
                ok = False
                break
        if ok:
            new_seg = [path[i] * (1 - tt) + path[j] * tt
                       for tt in np.linspace(0, 1, max(3, (j - i) // 2))]
            path = path[:i] + new_seg + path[j + 1:]
    return path


# ---------------------------------------------------------------------------
# Sample 3 random waypoints for the right arm
# ---------------------------------------------------------------------------

def sample_right_arm_waypoints(n=3):
    """Return `n` collision-free joint configs for the right arm."""
    waypoints = []
    for _ in range(n):
        q = random_jpos_in_workspace(obstacles=None, arm='right')
        waypoints.append(q)
    return waypoints


# ---------------------------------------------------------------------------
# Pre-plan right arm trajectories between all waypoint pairs
# ---------------------------------------------------------------------------

def preplan_right_arm_paths(waypoints):
    """Plan and smooth RRT paths between every ordered pair of waypoints.

    Returns a dict:  (i, j) -> list of joint configs
    """
    paths = {}
    n = len(waypoints)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            path = rrt_plan(waypoints[i], waypoints[j], obstacles=[],
                            max_iter=3000, step_size=10.0, arm='right')
            if path is None:
                # Fallback: direct interpolation (no obstacles for right arm)
                path = [waypoints[i] * (1 - t) + waypoints[j] * t
                        for t in np.linspace(0, 1, 20)]
            else:
                path = smooth_path(path, obstacles=[], attempts=30, arm='right')
            paths[(i, j)] = path
    return paths


# ---------------------------------------------------------------------------
# Interpolate the right arm along a pre-planned path
# ---------------------------------------------------------------------------

def interpolate_path(path, num_steps):
    """Resample a path to exactly `num_steps` waypoints via linear interpolation."""
    if len(path) < 2:
        return [path[0]] * num_steps

    # Cumulative arc-length parameterisation
    seg_lens = [_jdist(path[k], path[k + 1]) for k in range(len(path) - 1)]
    total = sum(seg_lens)
    if total < 1e-9:
        return [path[0]] * num_steps
    cum = np.concatenate([[0], np.cumsum(seg_lens)])

    resampled = []
    for s in np.linspace(0, total, num_steps):
        idx = np.searchsorted(cum, s, side='right') - 1
        idx = min(idx, len(path) - 2)
        local_t = (s - cum[idx]) / max(seg_lens[idx], 1e-9)
        local_t = np.clip(local_t, 0, 1)
        q = path[idx] * (1 - local_t) + path[idx + 1] * local_t
        resampled.append(q)
    return resampled


# ---------------------------------------------------------------------------
# Per-waypoint data (extends the existing format with right-arm columns)
# ---------------------------------------------------------------------------

def compute_episode_data(left_path, right_configs, static_obstacles, arm='left'):
    """Compute per-waypoint data for a left-arm trajectory.

    `right_configs` is a list of right-arm joint configs, one per left-arm
    waypoint (same length as `left_path`).  Each right-arm config is recorded
    alongside the left-arm state.
    """
    T_0B_arm = T_0B_LEFT if arm == 'left' else T_0B_RIGHT
    rows = []
    n = len(left_path)

    for step in range(n):
        jpos_left = left_path[step]
        jpos_right = right_configs[step]

        # Left-arm FK
        pts_l, origins_l, frames_l = get_robot_points(jpos_left, 10, arm='left')
        ee_base = origins_l[-1]
        T06 = frames_l[-1]
        R_base = T_0B_arm[:3, :3] @ T06[:3, :3]

        # Right-arm FK (for recording)
        _, origins_r, _ = get_robot_points(jpos_right, 0, arm='right')
        ee_right = origins_r[-1]

        # Joint velocities (finite differences)
        if 0 < step < n - 1:
            vel = (left_path[step + 1] - left_path[step - 1]) / 2.0
        else:
            vel = np.zeros(NUM_JOINTS)

        # Distances: left tool to right arm body + static obstacles
        right_pts = get_right_arm_points(jpos_right, samples_per_link=6)
        left_tool_pts = _tool_points(jpos_left, 'left', samples_per_link=6)
        min_d = float('inf')
        # Distance to static obstacles
        for pt in pts_l:
            for obs in static_obstacles:
                d = obs.distance_to(pt)
                if d < min_d:
                    min_d = d
        # Distance to right arm body (point-to-point, tool parts only)
        for lp in left_tool_pts:
            for rp in right_pts:
                d = np.linalg.norm(lp - rp)
                if d < min_d:
                    min_d = d

        # Build row
        row = {'step': step}
        for ji in range(NUM_JOINTS):
            row[f'j{ji+1}'] = jpos_left[ji]
        row['ee_x'], row['ee_y'], row['ee_z'] = ee_base
        for ri in range(3):
            for ci in range(3):
                row[f'ee_R{ri}{ci}'] = R_base[ri, ci]
        for ji in range(NUM_JOINTS):
            row[f'vj{ji+1}'] = vel[ji]

        # Right-arm state
        for ji in range(NUM_JOINTS):
            row[f'right_j{ji+1}'] = jpos_right[ji]
        row['right_ee_x'], row['right_ee_y'], row['right_ee_z'] = ee_right

        # Static obstacles
        for oi, obs in enumerate(static_obstacles):
            for k, v in zip(['min_x', 'min_y', 'min_z'], obs.vmin):
                row[f'obs{oi}_{k}'] = v
            for k, v in zip(['max_x', 'max_y', 'max_z'], obs.vmax):
                row[f'obs{oi}_{k}'] = v

        row['min_obs_dist'] = min_d
        rows.append(row)

    return rows


def save_episode_csv(rows, filepath):
    """Write episode data to CSV."""
    if not rows:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect(num_episodes, out_dir='data/planning', num_static_obs=1):
    os.makedirs(out_dir, exist_ok=True)

    print('Sampling 3 right-arm waypoints ...')
    right_waypoints = sample_right_arm_waypoints(3)
    for i, wp in enumerate(right_waypoints):
        ee = fm02base(raven_right_arm_frames(wp)[-1][:3, 3], 'right')
        print(f'  WP{i}: EE = [{ee[0]:.1f}, {ee[1]:.1f}, {ee[2]:.1f}]')

    print('Pre-planning right-arm trajectories between waypoints ...')
    right_paths = preplan_right_arm_paths(right_waypoints)
    print(f'  {len(right_paths)} paths ready')

    # Right arm patrol schedule: cycle through random transitions
    wp_indices = list(range(len(right_waypoints)))
    cur_wp = 0

    summaries = []
    success_count = 0
    t_start_all = time.time()

    for ep in range(num_episodes):
        t0 = time.time()

        # --- Right arm: pick next waypoint and get trajectory segment ----------
        next_wp = random.choice([w for w in wp_indices if w != cur_wp])
        right_path_segment = right_paths[(cur_wp, next_wp)]

        # Pick a random configuration along the right arm's current trajectory
        # to serve as the "snapshot" for this episode
        right_snapshot_idx = random.randint(0, len(right_path_segment) - 1)
        right_jpos = right_path_segment[right_snapshot_idx]

        # Precompute right arm body points for collision checking
        right_arm_pts = get_right_arm_points(right_jpos, samples_per_link=6)

        # --- Static obstacles --------------------------------------------------
        from raven_sim import random_obstacle_in_workspace
        static_obs = [random_obstacle_in_workspace(arm='left')
                      for _ in range(num_static_obs)]

        # Try up to 3 start/goal pairs before giving up
        path = None
        for _attempt in range(3):
            for _try in range(50):
                left_start = random_jpos_in_workspace(static_obs, arm='left')
                col, _ = check_collision_with_right_arm(left_start, right_arm_pts)
                if not col:
                    break
            for _try in range(50):
                left_goal = random_jpos_in_workspace(static_obs, arm='left')
                col, _ = check_collision_with_right_arm(left_goal, right_arm_pts)
                if not col:
                    break
            path = rrt_plan_dual(left_start, left_goal, static_obs, right_arm_pts)
            if path is not None:
                break

        if path is None:
            print(f'  ep {ep:4d}  FAILED (RRT)')
            summaries.append(dict(episode=ep, success=False,
                                  right_wp_from=cur_wp, right_wp_to=next_wp))
            cur_wp = next_wp
            continue

        path = smooth_path_dual(path, static_obs, right_arm_pts, attempts=50)

        # --- Assign right-arm configs along the left-arm path ------------------
        # The right arm moves from its snapshot toward the destination during
        # the left arm's execution.  We interpolate the remaining right-arm
        # path segment so it lines up with the left-arm waypoints.
        remaining_right = right_path_segment[right_snapshot_idx:]
        right_configs = interpolate_path(remaining_right, len(path))

        # --- Record data -------------------------------------------------------
        rows = compute_episode_data(path, right_configs, static_obs, arm='left')
        fpath = os.path.join(out_dir, f'episode_{ep:04d}.csv')
        save_episode_csv(rows, fpath)

        plen = sum(_jdist(path[k], path[k + 1]) for k in range(len(path) - 1))
        dt = time.time() - t0
        success_count += 1
        print(f'  ep {ep:4d}  OK  wps={len(path):3d}  len={plen:.1f}  '
              f'right {cur_wp}->{next_wp}  {dt:.1f}s  -> {os.path.basename(fpath)}')

        summaries.append(dict(
            episode=ep, success=True,
            right_wp_from=cur_wp, right_wp_to=next_wp,
            num_waypoints=len(path), path_length=round(plen, 2),
            right_snapshot_idx=right_snapshot_idx,
        ))

        cur_wp = next_wp

    # --- Batch summary ---------------------------------------------------------
    elapsed = time.time() - t_start_all
    print(f'\nDone: {success_count}/{num_episodes} succeeded in {elapsed:.1f}s')

    summary_path = os.path.join(out_dir, 'collection_summary.csv')
    if summaries:
        keys = sorted(set().union(*(s.keys() for s in summaries)))
        with open(summary_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summaries)
        print(f'Summary -> {summary_path}')

    # Save the right-arm waypoints for reproducibility
    wp_path = os.path.join(out_dir, 'right_arm_waypoints.csv')
    with open(wp_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([f'j{i+1}' for i in range(NUM_JOINTS)])
        for wp in right_waypoints:
            w.writerow(wp.tolist())
    print(f'Right-arm waypoints -> {wp_path}')


# ---------------------------------------------------------------------------
# Visualization mode
# ---------------------------------------------------------------------------

def visualize(num_episodes=5, num_static_obs=0):
    """Run a few episodes with live 3D animation so you can inspect the setup."""
    import matplotlib.pyplot as plt
    from raven_sim import draw_arm

    print('Sampling 3 right-arm waypoints ...')
    right_waypoints = sample_right_arm_waypoints(3)
    for i, wp in enumerate(right_waypoints):
        ee = fm02base(raven_right_arm_frames(wp)[-1][:3, 3], 'right')
        print(f'  WP{i}: EE = [{ee[0]:.1f}, {ee[1]:.1f}, {ee[2]:.1f}]')

    print('Pre-planning right-arm trajectories ...')
    right_paths = preplan_right_arm_paths(right_waypoints)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()

    wp_indices = list(range(len(right_waypoints)))
    cur_wp = 0

    for ep in range(num_episodes):
        next_wp = random.choice([w for w in wp_indices if w != cur_wp])
        right_path_seg = right_paths[(cur_wp, next_wp)]
        right_snapshot_idx = random.randint(0, len(right_path_seg) - 1)
        right_jpos = right_path_seg[right_snapshot_idx]
        right_arm_pts = get_right_arm_points(right_jpos, samples_per_link=6)

        from raven_sim import random_obstacle_in_workspace
        static_obs = [random_obstacle_in_workspace(arm='left')
                      for _ in range(num_static_obs)]

        # Plan left arm
        path = None
        for _attempt in range(3):
            for _try in range(50):
                left_start = random_jpos_in_workspace(static_obs, arm='left')
                col, _ = check_collision_with_right_arm(left_start, right_arm_pts)
                if not col:
                    break
            for _try in range(50):
                left_goal = random_jpos_in_workspace(static_obs, arm='left')
                col, _ = check_collision_with_right_arm(left_goal, right_arm_pts)
                if not col:
                    break
            path = rrt_plan_dual(left_start, left_goal, static_obs, right_arm_pts)
            if path is not None:
                break

        if path is None:
            print(f'  ep {ep}: RRT failed, skipping')
            cur_wp = next_wp
            continue

        path = smooth_path_dual(path, static_obs, right_arm_pts, attempts=50)

        # Interpolate right arm motion along left arm path
        remaining_right = right_path_seg[right_snapshot_idx:]
        right_configs = interpolate_path(remaining_right, len(path))

        # Animate: step through the trajectory
        print(f'  ep {ep}: animating {len(path)} steps  (right {cur_wp}->{next_wp})')
        ee_trace = []
        for step in range(len(path)):
            left_q = path[step]
            right_q = right_configs[step]

            # Collect left EE for path trace
            frames_l = raven_left_arm_frames(left_q)
            ee = fm02base(frames_l[-1][:3, 3], 'left')
            ee_trace.append(ee)

            draw_arm(ax, left_q, jpos_right=right_q,
                     obstacles=static_obs, path_ee=ee_trace, zoom=1.0)
            ax.set_title(f'Episode {ep}  step {step}/{len(path)-1}  '
                         f'right WP{cur_wp}->WP{next_wp}')
            plt.draw()
            plt.pause(0.3)

        plt.pause(1.0)
        cur_wp = next_wp

    print('Done. Close the window to exit.')
    plt.ioff()
    plt.show()


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Collect left-arm planning data with moving right arm')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to collect')
    parser.add_argument('--out', type=str, default='data/planning',
                        help='Output directory')
    parser.add_argument('--static-obs', type=int, default=1,
                        help='Number of static obstacles per episode')
    parser.add_argument('--visualize', action='store_true',
                        help='Run visual demo instead of data collection')
    parser.add_argument('--vis-episodes', type=int, default=5,
                        help='Number of episodes to visualize')
    args = parser.parse_args()

    if args.visualize:
        visualize(num_episodes=args.vis_episodes, num_static_obs=args.static_obs)
    else:
        collect(args.episodes, out_dir=args.out, num_static_obs=args.static_obs)
