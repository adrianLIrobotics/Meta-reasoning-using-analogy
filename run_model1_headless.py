"""
run_model1_headless.py
Drives the M1-SYMBOLIC simulation logic headlessly (no display required).
Runs 3 environment configurations and saves 3 Excel files:
  model1_symbolic_domestic_<ts>.xlsx
  model1_symbolic_mix_<ts>.xlsx
  model1_symbolic_warehouse_<ts>.xlsx

Policy is selected via ANALOGY_INPUT (same as model1v2.py):
  "dangerous" keyword  → conservative (speed=0.45, inflation=1.60, resolution=0.05)
  otherwise            → neutral      (speed=0.85, inflation=0.95, resolution=0.10)
"""

import numpy as np
import math
import random
import pandas as pd
from datetime import datetime
from matplotlib.path import Path as MplPath

# ─── POLICY SELECTION (mirrors model1v2.py) ──────────────────────────────────
ANALOGY_INPUT = "Hot coffee is dangerous, safety first."

POLICIES = {
    "conservative": {"speed": 0.45, "inflation": 1.60, "resolution": 0.05, "label": "CONSERVATIVE"},
    "neutral":      {"speed": 0.85, "inflation": 0.95, "resolution": 0.10, "label": "NEUTRAL"},
}
SELECTED_POLICY = POLICIES["conservative"] if "dangerous" in ANALOGY_INPUT.lower() else POLICIES["neutral"]

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
STEP_SIZE             = 0.22
SPILL_COOLDOWN_FRAMES = 50

META_PARAMS = {
    "speed":       SELECTED_POLICY["speed"],
    "inflation":   SELECTED_POLICY["inflation"],
    "resolution":  SELECTED_POLICY["resolution"],
    "f_loc": 3, "f_glob": 15, "initial_battery": 100.0,
    "init_uncertainty": 1.5, "goal_threshold": 0.15
}

base_poly = np.array([[3.5,4.0],[6.0,4.0],[6.0,4.5],[4.5,4.5],[4.5,6.0],[3.5,6.0]])

# ─── MUTABLE STATE ───────────────────────────────────────────────────────────
ROOM_SIZE = 10.0
START     = np.array([9.0, 9.0])
GOAL      = np.array([1.0, 1.0])
REAL_POLY = base_poly.copy()

total_runs_to_execute = 20
current_run           = 1
allowed_envs          = [True, True]
current_run_mode      = 0
current_env_is_empty  = True
all_iterations_data   = []
is_playing            = False

robot_pos_true  = START.copy()
robot_pos_est   = START.copy()
frame_counter   = 0
run_spills      = 0
run_collisions  = 0
spill_events    = []
battery_level   = 100.0
prev_v_final    = np.zeros(2)
frames_since_spill = SPILL_COOLDOWN_FRAMES
particles       = np.random.normal(START, META_PARAMS["init_uncertainty"], (40, 2))

OBSTACLE_GRID = None
GRID_X_EDGES  = None
GRID_Y_EDGES  = None

NOVEL_ENTITY_SPAWN_RUN = 8
novel_entity_active  = False
novel_entity_pos     = np.array([0.0, 0.0])
novel_entity_radius  = 0.45
novel_entity_type    = "unknown_cluster"
cold_start_data      = []

# ─── PURE LOGIC FUNCTIONS ────────────────────────────────────────────────────

def generate_random_room():
    dx, dy = random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8)
    scale  = random.uniform(0.85, 1.15)
    if current_run_mode == 1:
        scale *= 2.5
    center   = np.mean(base_poly, axis=0)
    new_poly = center + (base_poly - center) * scale + np.array([dx, dy])
    offset   = (ROOM_SIZE / 2.0) - 5.0
    return new_poly + np.array([offset, offset])

def build_obstacle_grid():
    global OBSTACLE_GRID, GRID_X_EDGES, GRID_Y_EDGES
    res     = META_PARAMS["resolution"]
    x_edges = np.arange(0, ROOM_SIZE + res, res)
    y_edges = np.arange(0, ROOM_SIZE + res, res)
    GRID_X_EDGES, GRID_Y_EDGES = x_edges, y_edges
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    if current_env_is_empty or nx * ny > 400000:
        OBSTACLE_GRID = np.zeros((max(1, ny), max(1, nx)), dtype=bool)
        return
    cx, cy = x_edges[:-1] + res/2, y_edges[:-1] + res/2
    gx, gy = np.meshgrid(cx, cy)
    pts    = np.column_stack([gx.ravel(), gy.ravel()])
    path   = MplPath(np.vstack([REAL_POLY, REAL_POLY[0]]))
    OBSTACLE_GRID = np.array(path.contains_points(pts)).reshape(ny, nx)

def is_inside_poly(p, poly):
    n = len(poly); inside = False; p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if p[1] > min(p1y, p2y) and p[1] <= max(p1y, p2y) and p[0] <= max(p1x, p2x):
            if p1x == p2x or p[0] <= (p[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def get_closest_grid_obstacle(pos):
    if OBSTACLE_GRID is None or GRID_X_EDGES is None or GRID_Y_EDGES is None:
        return pos.copy(), float('inf')
    if not OBSTACLE_GRID.any():
        return pos.copy(), float('inf')
    res = META_PARAMS["resolution"]
    cx  = GRID_X_EDGES[:-1] + res / 2
    cy  = GRID_Y_EDGES[:-1] + res / 2
    occ_y, occ_x = np.nonzero(OBSTACLE_GRID)
    centers = np.column_stack([cx[occ_x], cy[occ_y]])
    dists   = np.linalg.norm(centers - pos, axis=1)
    idx     = int(np.argmin(dists))
    return centers[idx], float(dists[idx])

def reset_run():
    global robot_pos_true, robot_pos_est, frame_counter, spill_events, particles
    global current_run_mode, current_env_is_empty, battery_level
    global run_spills, run_collisions, REAL_POLY, frames_since_spill
    global ROOM_SIZE, START, GOAL, OBSTACLE_GRID, GRID_X_EDGES, GRID_Y_EDGES
    global novel_entity_active, novel_entity_pos, novel_entity_radius, novel_entity_type

    active_indices = [i for i, val in enumerate(allowed_envs) if val]
    current_run_mode     = random.choice(active_indices) if active_indices else 0
    current_env_is_empty = (random.random() >= (0.10 if current_run_mode == 0 else 0.90))

    ROOM_SIZE = 30.0 if current_run_mode == 1 else 10.0
    START = np.array([ROOM_SIZE - 1.0, ROOM_SIZE - 1.0])
    GOAL  = np.array([1.0, 1.0])
    REAL_POLY = generate_random_room()

    robot_pos_true, robot_pos_est = START.copy(), START.copy()
    frame_counter, battery_level  = 0, META_PARAMS["initial_battery"]
    run_spills, run_collisions     = 0, 0
    spill_events                   = []
    particles = np.random.normal(START, META_PARAMS["init_uncertainty"], (40, 2))
    frames_since_spill = SPILL_COOLDOWN_FRAMES
    if current_run >= NOVEL_ENTITY_SPAWN_RUN:
        novel_entity_active = True
        mid = (START + GOAL) / 2.0
        novel_entity_pos    = mid + np.array([random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)])
        etype               = random.choice(["unknown_cluster", "fallen_chair", "foil_lid"])
        novel_entity_type   = etype
        novel_entity_radius = {"unknown_cluster": 0.45, "fallen_chair": 0.30, "foil_lid": 0.15}[etype]
    else:
        novel_entity_active = False
    build_obstacle_grid()

def step_logic():
    global robot_pos_true, robot_pos_est, frame_counter, battery_level
    global run_spills, run_collisions, prev_v_final, particles
    global current_run, is_playing, frames_since_spill

    inf = META_PARAMS["inflation"]

    # 1. Particle Filter (localization)
    if frame_counter % int(META_PARAMS["f_glob"]) == 0:
        particles     = np.random.normal(robot_pos_true, 1.5, (40, 2))
        robot_pos_est = np.mean(particles, axis=0)

    # 2. Path planning
    v_to_goal = GOAL - robot_pos_est
    drive_vec = v_to_goal / np.linalg.norm(v_to_goal) if np.linalg.norm(v_to_goal) > 0 else np.zeros(2)
    avoid_vec = np.zeros(2)
    dist_obs  = float('inf')
    in_avoidance = 0
    if not current_env_is_empty:
        closest_p, dist_obs = get_closest_grid_obstacle(robot_pos_est)
        if dist_obs < inf:
            in_avoidance = 1
            radial    = ((robot_pos_est - closest_p) / dist_obs) * 6.5 if dist_obs > 0 else np.array([0, 1])
            avoid_vec = radial + np.array([radial[1], -radial[0]]) * 5.2
            drive_vec = drive_vec * 0.1 + avoid_vec

    final_vec = drive_vec / np.linalg.norm(drive_vec) if np.linalg.norm(drive_vec) > 0 else np.zeros(2)

    # 3. Physics
    dist_to_goal  = np.linalg.norm(robot_pos_est - GOAL)
    actual_speed  = META_PARAMS["speed"] * (0.2 if dist_to_goal < 0.3 else 1.0)
    speed_modifier = 0.1 if actual_speed < 0.5 else 1.0
    jerk_mag   = actual_speed * np.linalg.norm(final_vec - prev_v_final)
    speed_risk = (actual_speed * 0.002 + actual_speed**2 * 0.001) * speed_modifier
    jerk_risk  = jerk_mag * actual_speed * 0.12 * speed_modifier
    cooldown_factor = min(1.0, frames_since_spill / SPILL_COOLDOWN_FRAMES)
    p_spill    = (speed_risk + jerk_risk) * cooldown_factor
    frames_since_spill += 1

    spilled = False
    if random.random() < p_spill:
        spill_events.append(robot_pos_true.copy()); run_spills += 1; spilled = True
        frames_since_spill = 0
    dominant_risk = "speed" if speed_risk >= jerk_risk else "jerk"
    spill_cause   = dominant_risk if spilled else "none"

    collision = 1 if (not current_env_is_empty and is_inside_poly(robot_pos_true, REAL_POLY)) else 0
    if novel_entity_active and np.linalg.norm(robot_pos_true - novel_entity_pos) < novel_entity_radius:
        collision = 1
    if collision: run_collisions += 1

    cpu_load    = min(99.8, 12.0 + (len(particles)*0.4) + (0.05/META_PARAMS["resolution"])*45 + random.uniform(-0.5, 0.5))
    sensor_cost = 0.015 if frame_counter % int(META_PARAMS["f_loc"]) == 0 else 0.0
    battery_level = max(0.0, battery_level - (actual_speed * 0.06 + (cpu_load/100)*0.04 + sensor_cost))

    step_move = final_vec * (STEP_SIZE * actual_speed)
    robot_pos_true += step_move + np.random.normal(0, 0.02, 2)
    robot_pos_est  += step_move
    particles      += step_move + np.random.normal(0, 0.04, (40, 2))
    prev_v_final    = final_vec.copy()

    dist_obs_log = round(dist_obs, 3) if not math.isinf(dist_obs) else float('nan')
    all_iterations_data.append({
        "Run": current_run, "Frame": frame_counter,
        "Mode": "Domestic" if current_run_mode == 0 else "Warehouse",
        "Active_Model": f"M1-{SELECTED_POLICY['label']}",
        "Novel_Entity_Active": int(novel_entity_active),
        "Novel_Entity_Type": novel_entity_type if novel_entity_active else "none",
        "Speed": round(actual_speed, 3),
        "Jerk": round(jerk_mag, 4),
        "Speed_Risk": round(speed_risk, 5), "Jerk_Risk": round(jerk_risk, 5),
        "p_spill": round(p_spill, 5),
        "Spill": 1 if spilled else 0, "Spill_Cause": spill_cause,
        "In_Avoidance": in_avoidance, "Dist_Obs": dist_obs_log,
        "Collision": collision,
        "Loc_Error": round(np.linalg.norm(robot_pos_true - robot_pos_est), 4),
        "Bat": round(battery_level, 1), "CPU": round(cpu_load, 1)
    })

    if np.linalg.norm(robot_pos_true - GOAL) < META_PARAMS["goal_threshold"]:
        ideal_f    = np.linalg.norm(START - GOAL) / (STEP_SIZE * 0.85)
        time_ratio = frame_counter / ideal_f
        time_factor = max(0.50, 1.15 - 0.20 * time_ratio)
        base_score = 100 - (run_spills * 25) - (run_collisions * 100)
        if run_spills == 0 and run_collisions == 0: base_score = 100
        final_se = max(0, base_score * time_factor * (battery_level / 100.0))
        cold_start_data.append({
            "run": current_run, "se": final_se,
            "entity_active": int(novel_entity_active),
            "entity_type": novel_entity_type if novel_entity_active else "none",
            "spills": run_spills, "collisions": run_collisions,
        })
        print(f"    run {current_run} done | SE={final_se:.1f} | spills={run_spills} | entity={'ON' if novel_entity_active else 'off'}")
        if current_run < total_runs_to_execute:
            current_run += 1
            reset_run()
        else:
            is_playing = False
    frame_counter += 1

def save_to_excel(cfg_name):
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    df       = pd.DataFrame(all_iterations_data)

    run_scores = []
    for r in df['Run'].unique():
        rd = df[df['Run'] == r]
        spills, cols = int(rd['Spill'].sum()), int(rd['Collision'].sum())
        ideal_f    = np.linalg.norm(START - GOAL) / (STEP_SIZE * 0.85)
        time_ratio = len(rd) / ideal_f
        time_factor = max(0.50, 1.15 - 0.20 * time_ratio)
        base_safety = 100 - (spills * 25) - (cols * 100)
        if spills == 0 and cols == 0: base_safety = 100
        score = max(0, base_safety * time_factor * (rd['Bat'].iloc[-1] / 100))
        run_scores.append(score)

    report = pd.DataFrame({
        "Metric": ["Avg SE Mark", "Total Spills", "Total Collisions",
                   "Spill Rate (per frame)", "Analogy Used"],
        "Result": [
            round(np.mean(run_scores), 2),
            int(df['Spill'].sum()),
            int(df['Collision'].sum()),
            round(df['Spill'].mean(), 5),
            ANALOGY_INPUT,
        ]
    })

    spill_cols = ['Run','Frame','Mode','Speed','Jerk','Speed_Risk','Jerk_Risk',
                  'p_spill','Dist_Obs','In_Avoidance','Spill_Cause','Loc_Error','Bat']
    spill_events_df = df[df['Spill'] == 1][[c for c in spill_cols if c in df.columns]].copy()

    stat_cols = ['Speed','Jerk','p_spill','Speed_Risk','Jerk_Risk','Dist_Obs','In_Avoidance','Loc_Error']
    stat_cols = [c for c in stat_cols if c in df.columns]
    spill_stats = df.groupby('Spill')[stat_cols].agg(['mean','std']).round(5)
    label_map = {0: 'No Spill', 1: 'Spill'}
    spill_stats.index = [label_map.get(i, str(i)) for i in spill_stats.index]

    config = pd.DataFrame({
        "Parameter": ["Model", "Config", "Policy_Selected", "Analogy_Input",
                      "Speed", "Inflation", "Resolution",
                      "Freq_Loc", "Freq_Glob", "Initial_Battery_%",
                      "Init_Uncertainty", "Goal_Threshold",
                      "Total_Runs", "Room_Size_at_End", "Step_Size"],
        "Value": ["M1-SYMBOLIC", cfg_name.upper(), SELECTED_POLICY["label"], ANALOGY_INPUT,
                  META_PARAMS["speed"], META_PARAMS["inflation"], META_PARAMS["resolution"],
                  META_PARAMS["f_loc"], META_PARAMS["f_glob"], META_PARAMS["initial_battery"],
                  META_PARAMS["init_uncertainty"], META_PARAMS["goal_threshold"],
                  total_runs_to_execute, ROOM_SIZE, STEP_SIZE]
    })

    out = f"model1_symbolic_{cfg_name}_{date_str}.xlsx"
    with pd.ExcelWriter(out) as writer:
        config.to_excel(writer, sheet_name='Config', index=False)
        df.to_excel(writer, sheet_name='Telemetry', index=False)
        report.to_excel(writer, sheet_name='Performance_Report', index=False)
        spill_events_df.to_excel(writer, sheet_name='Spill_Events', index=False)
        spill_stats.to_excel(writer, sheet_name='Spill_Analysis')
    print(f"  ✅ Saved: {out}")

# ─── MAIN ────────────────────────────────────────────────────────────────────

CONFIGS = [
    ([True,  False], "domestic"),
    ([True,  True],  "mix"),
    ([False, True],  "warehouse"),
]

MAX_FRAMES_TOTAL = 80_000

def init_config(cfg_envs):
    global current_run, allowed_envs, all_iterations_data, is_playing, prev_v_final, frames_since_spill

    allowed_envs[:]    = cfg_envs
    current_run        = 1
    all_iterations_data = []
    is_playing         = True
    prev_v_final       = np.zeros(2)
    frames_since_spill = SPILL_COOLDOWN_FRAMES

if __name__ == "__main__":
    print(f"M1-SYMBOLIC | Policy: {SELECTED_POLICY['label']} | Analogy: \"{ANALOGY_INPUT}\"")
    for cfg_envs, cfg_name in CONFIGS:
        print(f"\n{'='*55}")
        print(f"  CONFIG: {cfg_name.upper()} | allowed_envs={cfg_envs}")
        print('='*55)
        init_config(cfg_envs)
        reset_run()
        total_frames = 0
        while is_playing and total_frames < MAX_FRAMES_TOTAL:
            step_logic()
            total_frames += 1
            if total_frames % 2000 == 0:
                mode = "Warehouse" if current_run_mode else "Domestic"
                print(f"  [{cfg_name}] run {current_run}/{total_runs_to_execute} | frame {frame_counter} | env={mode} | bat={battery_level:.1f}%")
        if is_playing:
            print(f"  ⚠️  Safety cap hit ({MAX_FRAMES_TOTAL} frames) — saving partial data")
        save_to_excel(cfg_name)

    print("\n✅ All 3 configurations complete.")
