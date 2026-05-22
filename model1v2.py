import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox, CheckButtons
import pandas as pd
import math
from datetime import datetime
import random
from matplotlib.patches import Polygon, Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.colors import ListedColormap

_res_cmap = ListedColormap(['mediumpurple'])

# =======================================================================
# 🚨 MODELO 1: NAIVE SYMBOLIC ENGINE - VERSION 1.9.3 (FAIR COMPARISON) 🚨
# Goal: Equal physics, GUI, and Dynamic Room Scaling to M3.2
# =======================================================================

POLICIES = {
    "conservative": {"speed": 0.45, "inflation": 1.60, "res": 0.05, "label": "CONSERVATIVE"},
    "neutral":      {"speed": 0.85, "inflation": 0.95, "res": 0.10, "label": "NEUTRAL"}
}

ANALOGY_INPUT = "Hot coffee is dangerous, safety first."
SELECTED_POLICY = POLICIES["conservative"] if "dangerous" in ANALOGY_INPUT.lower() else POLICIES["neutral"]

META_PARAMS = {
    "speed": SELECTED_POLICY["speed"],
    "inflation": SELECTED_POLICY["inflation"],
    "resolution": SELECTED_POLICY["res"],
    "f_loc": 3, "f_glob": 15, "initial_battery": 100.0,
    "init_uncertainty": 1.5, "goal_threshold": 0.15
}

ROOM_SIZE = 10.0
START, GOAL = np.array([ROOM_SIZE - 1.0, ROOM_SIZE - 1.0]), np.array([1.0, 1.0])
STEP_SIZE = 0.22
MAX_TURN_PER_FRAME    = 0.25
SPILL_COOLDOWN_FRAMES = 50

base_poly = np.array([[3.5, 4.0], [6.0, 4.0], [6.0, 4.5], [4.5, 4.5], [4.5, 6.0], [3.5, 6.0]])
REAL_POLY = base_poly.copy()

total_runs_to_execute, current_run = 5, 1
allowed_envs = [True, True]
current_run_mode = 0
current_env_is_empty, is_playing = True, False
all_iterations_data = []
battery_level = 100.0
robot_pos_true, robot_pos_est = START.copy(), START.copy()
laser_endpoints, spill_events = [], []
total_spills, total_collisions = 0, 0
prev_v_final = np.array([0.0, 0.0])
frames_since_spill = SPILL_COOLDOWN_FRAMES
particles = np.random.normal(START, META_PARAMS["init_uncertainty"], (40, 2))
frame_counter = 0

OBSTACLE_GRID = None
GRID_X_EDGES  = None
GRID_Y_EDGES  = None

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================

def generate_random_room():
    dx, dy = random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8)
    scale = random.uniform(0.85, 1.15)
    if current_run_mode == 1:
        scale *= 2.5
    center = np.mean(base_poly, axis=0)
    new_poly = center + (base_poly - center) * scale + np.array([dx, dy])
    offset = (ROOM_SIZE / 2.0) - 5.0
    new_poly += np.array([offset, offset])
    return new_poly

def ray_poly_intersection(origin, direction, poly):
    min_t = float('inf')
    for i in range(len(poly)):
        p1, p2 = poly[i], poly[(i + 1) % len(poly)]
        v1, v2, v3 = origin - p1, p2 - p1, np.array([-direction[1], direction[0]])
        det = np.dot(v2, v3)
        if abs(det) < 1e-6: continue
        t1 = np.cross(v2, v1) / det
        if t1 >= 0 and 0 <= np.dot(v1, v3) / det <= 1: min_t = min(min_t, t1)
    return min_t if min_t != float('inf') else None

def inflate_polygon(poly, amount):
    n = len(poly)
    normals = []
    for i in range(n):
        p1, p2 = poly[i], poly[(i + 1) % n]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.sqrt(dx * dx + dy * dy)
        normals.append(np.array([dy / length, -dx / length]) if length > 1e-9 else np.zeros(2))
    inflated = []
    for i in range(n):
        n_in, n_out = normals[(i - 1) % n], normals[i]
        bisector = n_in + n_out
        b_len = np.linalg.norm(bisector)
        if b_len < 1e-6:
            inflated.append(poly[i] + amount * n_out)
        else:
            b_unit = bisector / b_len
            cos_angle = np.clip(np.dot(b_unit, n_out), 0.15, 1.0)
            inflated.append(poly[i] + b_unit * (amount / cos_angle))
    return np.array(inflated)

def is_inside_poly(p, poly):
    n = len(poly); inside = False; p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if p[1] > min(p1y, p2y) and p[1] <= max(p1y, p2y) and p[0] <= max(p1x, p2x):
            if p1x == p2x or p[0] <= (p[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def build_obstacle_grid():
    global OBSTACLE_GRID, GRID_X_EDGES, GRID_Y_EDGES
    res = META_PARAMS["resolution"]
    x_edges = np.arange(0, ROOM_SIZE + res, res)
    y_edges = np.arange(0, ROOM_SIZE + res, res)
    GRID_X_EDGES, GRID_Y_EDGES = x_edges, y_edges
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    if current_env_is_empty or nx * ny > 400000:
        OBSTACLE_GRID = np.zeros((max(1, ny), max(1, nx)), dtype=bool)
        return
    cx = x_edges[:-1] + res / 2
    cy = y_edges[:-1] + res / 2
    gx, gy = np.meshgrid(cx, cy)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    p_path = MplPath(np.vstack([REAL_POLY, REAL_POLY[0]]))
    OBSTACLE_GRID = np.array(p_path.contains_points(pts)).reshape(ny, nx)

def get_closest_grid_obstacle(pos):
    if OBSTACLE_GRID is None or GRID_X_EDGES is None or GRID_Y_EDGES is None:
        return pos.copy(), float('inf')
    if not OBSTACLE_GRID.any():
        return pos.copy(), float('inf')
    res = META_PARAMS["resolution"]
    cx = GRID_X_EDGES[:-1] + res / 2
    cy = GRID_Y_EDGES[:-1] + res / 2
    occ_y, occ_x = np.nonzero(OBSTACLE_GRID)
    centers = np.column_stack([cx[occ_x], cy[occ_y]])
    dists = np.linalg.norm(centers - pos, axis=1)
    idx = int(np.argmin(dists))
    return centers[idx], float(dists[idx])

# ==========================================
# 🧠 STEP LOGIC
# ==========================================

def step_logic():
    global robot_pos_true, robot_pos_est, frame_counter, is_playing, current_run
    global laser_endpoints, battery_level, particles, prev_v_final, total_spills, total_collisions, frames_since_spill

    inf = META_PARAMS["inflation"]

    # 1. Sense
    if frame_counter % int(META_PARAMS["f_loc"]) == 0:
        laser_endpoints = []
        for angle in np.linspace(0, 2*math.pi, 36, endpoint=False):
            d = np.array([math.cos(angle), math.sin(angle)])
            t_obj = ray_poly_intersection(robot_pos_true, d, REAL_POLY) if not current_env_is_empty else None
            laser_endpoints.append((robot_pos_true + (t_obj if t_obj else 15.0)*d, "OBSTACLE" if t_obj else "WALL"))

    # 2. Particle Filter
    if frame_counter % int(META_PARAMS["f_glob"]) == 0:
        particles = np.random.normal(robot_pos_true, 1.5, (40, 2))
        robot_pos_est = np.mean(particles, axis=0)

    # 3. Path planning — avoidance uses resolution grid
    v_to_goal = GOAL - robot_pos_est
    drive_vec = v_to_goal / np.linalg.norm(v_to_goal) if np.linalg.norm(v_to_goal) > 0 else np.zeros(2)

    dist_obs = float('inf')
    in_avoidance = 0
    if not current_env_is_empty:
        closest_p, dist_obs = get_closest_grid_obstacle(robot_pos_est)
        if dist_obs < inf:
            in_avoidance = 1
            radial = ((robot_pos_est - closest_p) / dist_obs) * 6.5 if dist_obs > 0 else np.array([0, 1])
            avoid_vec = radial + np.array([radial[1], -radial[0]]) * 5.2
            drive_vec = (drive_vec * 0.1) + avoid_vec

    final_vec = drive_vec / np.linalg.norm(drive_vec) if np.linalg.norm(drive_vec) > 0 else np.zeros(2)

    # Fix 1: kinematic turn-rate cap
    if np.linalg.norm(prev_v_final) > 0 and np.linalg.norm(final_vec) > 0:
        cross = final_vec[0]*prev_v_final[1] - final_vec[1]*prev_v_final[0]
        dot   = final_vec[0]*prev_v_final[0] + final_vec[1]*prev_v_final[1]
        angle_diff = math.atan2(cross, dot)
        if abs(angle_diff) > MAX_TURN_PER_FRAME:
            clamp = MAX_TURN_PER_FRAME * (1 if angle_diff > 0 else -1)
            c, s = math.cos(clamp), math.sin(clamp)
            px, py = prev_v_final[0], prev_v_final[1]
            final_vec = np.array([c*px - s*py, s*px + c*py])

    # 4. Physics — realism fixes applied
    cur_v = META_PARAMS["speed"]
    # Fix 4: goal approach damping
    dist_to_goal = np.linalg.norm(robot_pos_est - GOAL)
    actual_speed = cur_v * (0.2 if dist_to_goal < 1.0 else 1.0)
    # Fix 2 & 3: physical jerk + vibration baseline
    speed_modifier = 0.1 if actual_speed < 0.5 else 1.0
    jerk_mag   = actual_speed * np.linalg.norm(final_vec - prev_v_final)
    speed_risk = (actual_speed * 0.002 + actual_speed**2 * 0.001) * speed_modifier
    jerk_risk  = jerk_mag * actual_speed * 0.12 * speed_modifier
    # Fix 5: post-spill cooldown
    cooldown_factor = min(1.0, frames_since_spill / SPILL_COOLDOWN_FRAMES)
    p_spill    = (speed_risk + jerk_risk) * cooldown_factor
    frames_since_spill += 1

    spilled = False
    if random.random() < p_spill:
        spill_events.append(robot_pos_true.copy()); total_spills += 1; spilled = True
        frames_since_spill = 0
    dominant_risk = "speed" if speed_risk >= jerk_risk else "jerk"
    spill_cause = dominant_risk if spilled else "none"

    collision = 1 if (not current_env_is_empty and is_inside_poly(robot_pos_true, REAL_POLY)) else 0
    if collision: total_collisions += 1

    cpu_load = min(99.8, 12.0 + (len(particles) * 0.4) + (0.05/META_PARAMS["resolution"])*45 + random.uniform(-0.5, 0.5))

    step = final_vec * (STEP_SIZE * actual_speed)
    robot_pos_true += step + np.random.normal(0, 0.02, 2)
    robot_pos_est  += step
    particles      += step + np.random.normal(0, 0.04, (40, 2))
    prev_v_final = final_vec.copy()
    sensor_cost = 0.015 if frame_counter % int(META_PARAMS["f_loc"]) == 0 else 0.0
    battery_level = max(0.0, battery_level - (actual_speed * 0.06 + (cpu_load / 100) * 0.04 + sensor_cost))

    dist_obs_log = round(dist_obs, 3) if not math.isinf(dist_obs) else float('nan')
    all_iterations_data.append({
        "Run": current_run, "Frame": frame_counter,
        "Mode": "Domestic" if current_run_mode == 0 else "Warehouse",
        "Active_Model": f"M1-{SELECTED_POLICY['label']}",
        "Speed": round(actual_speed, 3),
        "Jerk": round(jerk_mag, 4),
        "Speed_Risk": round(speed_risk, 5),
        "Jerk_Risk": round(jerk_risk, 5),
        "p_spill": round(p_spill, 5),
        "Spill": 1 if spilled else 0,
        "Spill_Cause": spill_cause,
        "In_Avoidance": in_avoidance,
        "Dist_Obs": dist_obs_log,
        "Collision": collision,
        "Loc_Error": round(np.linalg.norm(robot_pos_true - robot_pos_est), 4),
        "Bat": round(battery_level, 1), "CPU": round(cpu_load, 1)
    })

    if np.linalg.norm(robot_pos_true - GOAL) < META_PARAMS["goal_threshold"]:
        if current_run < total_runs_to_execute:
            current_run += 1; reset_run()
        else:
            is_playing = False; save_to_excel()
    frame_counter += 1

# ==========================================
# 🛠️ UTILS & EXCEL
# ==========================================

def reset_run():
    global robot_pos_true, robot_pos_est, frame_counter, REAL_POLY, battery_level, particles
    global current_env_is_empty, current_run_mode, ROOM_SIZE, START, GOAL
    global OBSTACLE_GRID, GRID_X_EDGES, GRID_Y_EDGES, frames_since_spill

    active_indices = [i for i, val in enumerate(allowed_envs) if val]
    current_run_mode = random.choice(active_indices) if active_indices else 0
    current_env_is_empty = (random.random() >= (0.10 if current_run_mode == 0 else 0.90))
    ROOM_SIZE = 30.0 if current_run_mode == 1 else 10.0
    START = np.array([ROOM_SIZE - 1.0, ROOM_SIZE - 1.0])
    GOAL  = np.array([1.0, 1.0])
    REAL_POLY = generate_random_room()
    robot_pos_true, robot_pos_est = START.copy(), START.copy()
    frame_counter, battery_level = 0, META_PARAMS["initial_battery"]
    particles = np.random.normal(START, META_PARAMS["init_uncertainty"], (40, 2))
    frames_since_spill = SPILL_COOLDOWN_FRAMES
    build_obstacle_grid()

def save_to_excel():
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    df = pd.DataFrame(all_iterations_data)
    run_scores = []
    for r in df['Run'].unique():
        rd = df[df['Run'] == r]
        spills, cols = rd['Spill'].sum(), rd['Collision'].sum()
        ideal_f = np.linalg.norm(START - GOAL) / (STEP_SIZE * 0.85)
        time_ratio = len(rd) / ideal_f
        time_factor = max(0.2, 1.3 - 0.30 * time_ratio)
        base_safety = 100 - (spills * 15) - (cols * 100)
        if spills == 0 and cols == 0: base_safety = 100
        score = max(0, base_safety * time_factor * (rd['Bat'].iloc[-1] / 100))
        run_scores.append(score)

    report = pd.DataFrame({
        "Metric": ["Avg SE Mark", "Total Spills", "Total Collisions", "Spill Rate (per frame)", "Analogy Used"],
        "Result": [
            round(np.mean(run_scores), 2),
            int(df['Spill'].sum()),
            int(df['Collision'].sum()),
            round(df['Spill'].mean(), 5),
            ANALOGY_INPUT,
        ]
    })

    spill_cols = ['Run', 'Frame', 'Mode', 'Speed', 'Jerk', 'Speed_Risk', 'Jerk_Risk',
                  'p_spill', 'Dist_Obs', 'In_Avoidance', 'Spill_Cause', 'Loc_Error', 'Bat']
    spill_events_df = df[df['Spill'] == 1][[c for c in spill_cols if c in df.columns]].copy()

    stat_cols = ['Speed', 'Jerk', 'p_spill', 'Speed_Risk', 'Jerk_Risk', 'Dist_Obs', 'In_Avoidance', 'Loc_Error']
    stat_cols = [c for c in stat_cols if c in df.columns]
    spill_stats = df.groupby('Spill')[stat_cols].agg(['mean', 'std']).round(5)
    spill_stats.index = ['No Spill', 'Spill']

    config = pd.DataFrame({
        "Parameter": ["Model", "Policy_Selected", "Analogy_Input",
                      "Speed", "Inflation", "Resolution",
                      "Freq_Loc", "Freq_Glob", "Initial_Battery_%",
                      "Init_Uncertainty", "Goal_Threshold",
                      "Total_Runs", "Room_Size_at_End", "Step_Size"],
        "Value": ["M1-SYMBOLIC", SELECTED_POLICY["label"], ANALOGY_INPUT,
                  META_PARAMS["speed"], META_PARAMS["inflation"], META_PARAMS["resolution"],
                  META_PARAMS["f_loc"], META_PARAMS["f_glob"], META_PARAMS["initial_battery"],
                  META_PARAMS["init_uncertainty"], META_PARAMS["goal_threshold"],
                  total_runs_to_execute, ROOM_SIZE, STEP_SIZE]
    })

    with pd.ExcelWriter(f"model1_final_v193_fair_{date_str}.xlsx") as writer:
        config.to_excel(writer, sheet_name='Config', index=False)
        df.to_excel(writer, sheet_name='Telemetry', index=False)
        report.to_excel(writer, sheet_name='Performance_Report', index=False)
        spill_events_df.to_excel(writer, sheet_name='Spill_Events', index=False)
        spill_stats.to_excel(writer, sheet_name='Spill_Analysis')
    print(f"✅ EXCEL GENERATED. M1 Fair Score: {np.mean(run_scores):.2f}")

# ==========================================
# 🎨 MASTER HUD & UI
# ==========================================

fig, ax = plt.subplots(figsize=(15, 9))
plt.subplots_adjust(left=0.25, bottom=0.26, right=0.95)

def draw_graphics():
    ax.clear(); ax.set_xlim(-0.5, ROOM_SIZE + 0.5); ax.set_ylim(-0.5, ROOM_SIZE + 0.5); ax.set_aspect('equal')
    inf = META_PARAMS["inflation"]
    ax.add_patch(Rectangle((0,0), ROOM_SIZE, ROOM_SIZE, fill=False, edgecolor='black', lw=2))

    res = META_PARAMS["resolution"]
    xs = np.arange(0, ROOM_SIZE + res, res)
    ys = np.arange(0, ROOM_SIZE + res, res)
    ax.vlines(xs, 0, ROOM_SIZE, colors=['slategray'], alpha=0.12, lw=0.4, zorder=0)
    ax.hlines(ys, 0, ROOM_SIZE, colors=['slategray'], alpha=0.12, lw=0.4, zorder=0)

    borders = [((0,0), ROOM_SIZE, inf), ((0, ROOM_SIZE-inf), ROOM_SIZE, inf),
               ((0,0), inf, ROOM_SIZE), ((ROOM_SIZE-inf, 0), inf, ROOM_SIZE)]
    [ax.add_patch(Rectangle(xy, w, h, color='cyan', alpha=0.1)) for xy, w, h in borders]

    if not current_env_is_empty:
        if (OBSTACLE_GRID is not None and GRID_X_EDGES is not None
                and GRID_Y_EDGES is not None and OBSTACLE_GRID.any()):
            C = np.ma.masked_where(~OBSTACLE_GRID, np.ones_like(OBSTACLE_GRID, dtype=float))
            ax.pcolormesh(GRID_X_EDGES, GRID_Y_EDGES, C, cmap=_res_cmap, alpha=0.55, zorder=3, shading='flat')
        ax.add_patch(Polygon(REAL_POLY, color='black', alpha=0.7, zorder=10))
        ax.add_patch(Polygon(inflate_polygon(REAL_POLY, inf).tolist(), color='cyan', alpha=0.15, zorder=2))

    ax.scatter(particles[:,0], particles[:,1], color='red', s=5, alpha=0.2)
    for ep, ht in laser_endpoints:
        ax.plot([robot_pos_true[0], ep[0]], [robot_pos_true[1], ep[1]],
                color='orange' if ht=="OBSTACLE" else 'green', alpha=0.1, lw=1)
    for s in spill_events: ax.scatter(s[0], s[1], color='brown', s=30)
    ax.plot(robot_pos_true[0], robot_pos_true[1], 'bo', markersize=9)
    ax.plot(robot_pos_est[0], robot_pos_est[1], 'go', markersize=7, alpha=0.7)
    ax.plot(GOAL[0], GOAL[1], 'rx', markersize=15)
    ax.text(0.1, ROOM_SIZE + 0.6, f"AGENT: M1-{SELECTED_POLICY['label']} (STATIC)", color='purple', fontweight='bold', fontsize=12)
    ax.text(0.1, ROOM_SIZE + 0.2, f"SE Analogy: '{ANALOGY_INPUT}' → Speed: {META_PARAMS['speed']} m/s", fontsize=10)

def update(frame):
    if is_playing: step_logic(); draw_graphics()

ui_cfg = [
    (0.88, 'Speed:',     str(META_PARAMS["speed"]),            lambda v: META_PARAMS.update({"speed": float(v)})),
    (0.82, 'Inflation:', str(META_PARAMS["inflation"]),         lambda v: META_PARAMS.update({"inflation": float(v)})),
    (0.76, 'Resolution:', str(META_PARAMS["resolution"]),      lambda v: (META_PARAMS.update({"resolution": float(v)}), build_obstacle_grid())),
    (0.70, 'Freq Loc:',  str(META_PARAMS["f_loc"]),            lambda v: META_PARAMS.update({"f_loc": int(v)})),
    (0.64, 'Freq Glob:', str(META_PARAMS["f_glob"]),           lambda v: META_PARAMS.update({"f_glob": int(v)})),
    (0.58, 'Runs:',      str(total_runs_to_execute),           lambda v: globals().update(total_runs_to_execute=int(v))),
    (0.52, 'Init Bat %:', str(META_PARAMS["initial_battery"]), lambda v: META_PARAMS.update({"initial_battery": float(v)}))
]

boxes = [TextBox(plt.axes([0.12, y, 0.05, 0.035]), label, initial=init) for y, label, init, func in ui_cfg]
for i, box in enumerate(boxes):
    if ui_cfg[i][3]: box.on_submit(ui_cfg[i][3])

check = CheckButtons(plt.axes([0.02, 0.35, 0.15, 0.12]), ('Domestic', 'Warehouse'), (True, True))
check.on_clicked(lambda l: allowed_envs.__setitem__(0, not allowed_envs[0]) if l=='Domestic' else allowed_envs.__setitem__(1, not allowed_envs[1]))

btn = Button(plt.axes([0.02, 0.25, 0.15, 0.06]), '▶ Start Fair M1', color='cyan')
btn.on_clicked(lambda e: globals().update({"is_playing": True, "current_run": 1, "all_iterations_data": [], "spill_events": [], "total_spills": 0, "total_collisions": 0}) or reset_run())

ax_legend = fig.add_axes([0.25, 0.01, 0.70, 0.23])
ax_legend.axis('off')
legend_handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='blue', markersize=10,
           label='Robot True Position  –  GPS/encoder ground truth'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='green', markersize=9, alpha=0.8,
           label='Robot Estimated Position  –  particle-filter mean'),
    Line2D([0],[0], marker='x', color='red', markersize=11, markeredgewidth=2.5,
           label='Goal / Destination  –  target to reach'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='red', markersize=6, alpha=0.4,
           label='Particle Cloud  –  localization uncertainty samples'),
    Line2D([0],[0], color='orange', alpha=0.8, lw=2,
           label='Laser ray → Obstacle hit  –  LIDAR return on object'),
    Line2D([0],[0], color='green', alpha=0.8, lw=2,
           label='Laser ray → Free space  –  LIDAR max-range (no hit)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='saddlebrown', markersize=9,
           label='Spill event  –  liquid spilled (speed/decel penalty)'),
    Patch(facecolor='black', alpha=0.7,  label='Obstacle  –  physical L-shaped blocker'),
    Patch(facecolor='cyan',  alpha=0.40, label='Inflation zone  –  uniform safety buffer (walls & obstacle)'),
    Line2D([0],[0], color='slategray', alpha=0.6, lw=1.5, linestyle='--',
           label='Planning grid  –  cell size = Resolution param (finer → higher CPU load)'),
    Patch(facecolor='mediumpurple', alpha=0.55,
          label='Resolution layer  –  obstacle rasterised on grid (fewer cells = blockier / less accurate robot map)'),
]
ax_legend.legend(handles=legend_handles, loc='center', ncol=3, fontsize=7.8,
                 frameon=True, fancybox=True, shadow=True, title='VISUAL LEGEND', title_fontsize=9,
                 edgecolor='gray', labelspacing=0.55, handlelength=2.0,
                 handleheight=1.2, handletextpad=0.6, columnspacing=1.2)

draw_graphics()
ani = FuncAnimation(fig, update, interval=40, cache_frame_data=False)
plt.show()
