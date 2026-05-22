"""
run_model2_headless.py
Drives the M2-DST-HYBRID simulation logic headlessly (no display required).
Runs 3 environment configurations and saves 3 Excel files:
  model2_DST_domestic_<ts>.xlsx
  model2_DST_mix_<ts>.xlsx
  model2_DST_warehouse_<ts>.xlsx
"""

import numpy as np
import math
import random
import pandas as pd
from datetime import datetime
from matplotlib.path import Path as MplPath

# ─── CONSTANTS ──────────────────────────────────────────────────────────────
STEP_SIZE             = 0.22
SMOOTH_VAL            = 0.08
MAX_TURN_PER_FRAME    = 0.25
SPILL_COOLDOWN_FRAMES = 50

POLICY_M0 = {"speed": 0.65, "inflation": 0.85, "resolution": 0.10}
POLICY_M1 = {"speed": 0.40, "inflation": 1.85, "resolution": 0.05}

META_PARAMS = {
    "speed": 0.65, "inflation": 0.85, "resolution": 0.10,
    "f_loc": 3, "f_glob": 15, "initial_battery": 100.0,
    "sensor_std": 0.03, "encoder_std": 0.02, "init_uncertainty": 1.5,
    "goal_threshold": 0.15, "laser_range": 5.5
}

base_poly = np.array([[3.5,4.0],[6.0,4.0],[6.0,4.5],[4.5,4.5],[4.5,6.0],[3.5,6.0]])

# ─── MUTABLE STATE (reset per config / per run) ──────────────────────────────
historical_SE      = []
dynamic_threshold  = 0.5
mass_H_init        = 0.5
mass_H, mass_notH, mass_Theta = 0.5, 0.0, 0.5
plausibility_H     = 1.0
eff_speed          = POLICY_M0["speed"]
eff_inflation      = POLICY_M0["inflation"]
eff_resolution     = POLICY_M0["resolution"]

ROOM_SIZE = 10.0
START     = np.array([9.0, 9.0])
GOAL      = np.array([1.0, 1.0])
REAL_POLY = base_poly.copy()

total_runs_to_execute = 5
current_run           = 1
allowed_envs          = [True, True]
current_run_mode      = 0
current_env_is_empty  = True
all_iterations_data   = []
is_playing            = False

robot_pos_true     = START.copy()
robot_pos_est      = START.copy()
active_model_label = "M0-REACTIVE (FAST)"
frame_counter      = 0
run_spills         = 0
run_collisions     = 0
spill_events       = []
laser_endpoints    = []
battery_level      = 100.0
prev_v_final       = np.zeros(2)
frames_since_spill = SPILL_COOLDOWN_FRAMES
pose_variance      = 1.5
particles          = np.random.normal(START, pose_variance, (40, 2))
hits_in_path       = 0  # carries over between sensing frames

OBSTACLE_GRID = None
GRID_X_EDGES  = None
GRID_Y_EDGES  = None

# ─── PURE LOGIC FUNCTIONS ────────────────────────────────────────────────────

def generate_random_room():
    dx, dy = random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8)
    scale  = random.uniform(0.85, 1.15)
    if current_run_mode == 1:
        scale *= 2.5
    center  = np.mean(base_poly, axis=0)
    new_poly = center + (base_poly - center) * scale + np.array([dx, dy])
    offset   = (ROOM_SIZE / 2.0) - 5.0
    return new_poly + np.array([offset, offset])

def update_belief_dst(hits, total_rays, min_dist):
    global mass_H, mass_notH, mass_Theta
    density_factor   = hits / total_rays if total_rays > 0 else 0
    proximity_factor = max(0, 1.0 - (min_dist / META_PARAMS["laser_range"]))
    obs_notH  = np.clip(density_factor * 0.6 + proximity_factor * 0.4, 0, 0.95)
    loc_err   = np.linalg.norm(robot_pos_true - robot_pos_est)
    obs_Theta = np.clip(loc_err / 1.5, 0.05, 0.6)
    obs_H     = max(0, 1.0 - obs_notH - obs_Theta)
    K = (mass_H * obs_notH) + (mass_notH * obs_H)
    if K >= 1: K = 0.99
    new_H     = (mass_H * obs_H + mass_H * obs_Theta + mass_Theta * obs_H) / (1 - K)
    new_notH  = (mass_notH * obs_notH + mass_notH * obs_Theta + mass_Theta * obs_notH) / (1 - K)
    new_Theta = (mass_Theta * obs_Theta) / (1 - K)
    tot = new_H + new_notH + new_Theta
    mass_H, mass_notH, mass_Theta = new_H/tot, new_notH/tot, new_Theta/tot
    if mass_Theta < 0.05:
        diff = 0.05 - mass_Theta
        mass_Theta = 0.05
        denom = mass_H + mass_notH + 1e-9
        mass_H    -= diff * (mass_H    / denom)
        mass_notH -= diff * (mass_notH / denom)
    pl = 1.0 - mass_notH
    return min([0, 0.25, 0.5, 0.75, 1.0], key=lambda x: abs(x - pl))

def ray_poly_intersection(origin, direction, poly):
    min_t = float('inf')
    for i in range(len(poly)):
        p1, p2 = poly[i], poly[(i + 1) % len(poly)]
        v1, v2 = origin - p1, p2 - p1
        v3  = np.array([-direction[1], direction[0]])
        det = np.dot(v2, v3)
        if abs(det) < 1e-6: continue
        t1 = np.cross(v2, v1) / det
        t2 = np.dot(v1, v3) / det
        if t1 >= 0 and 0 <= t2 <= 1:
            min_t = min(min_t, t1)
    return min_t if min_t != float('inf') else None

def build_obstacle_grid():
    global OBSTACLE_GRID, GRID_X_EDGES, GRID_Y_EDGES
    res    = META_PARAMS["resolution"]
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
    global robot_pos_true, robot_pos_est, frame_counter, mass_H, mass_notH, mass_Theta, plausibility_H
    global spill_events, laser_endpoints, particles, pose_variance
    global current_run_mode, current_env_is_empty, battery_level
    global run_spills, run_collisions, REAL_POLY, dynamic_threshold, mass_H_init
    global eff_speed, eff_inflation, eff_resolution
    global ROOM_SIZE, START, GOAL, OBSTACLE_GRID, GRID_X_EDGES, GRID_Y_EDGES, frames_since_spill
    global hits_in_path

    if historical_SE:
        avg_se = np.mean(historical_SE[-3:])
        dynamic_threshold = np.clip(0.5 + (70 - avg_se) * 0.005, 0.25, 0.75)
        mass_H_init = np.clip(0.5 + (avg_se - 50) * 0.005, 0.2, 0.8)

    active_indices = [i for i, val in enumerate(allowed_envs) if val]
    current_run_mode    = random.choice(active_indices) if active_indices else 0
    current_env_is_empty = (random.random() >= (0.10 if current_run_mode == 0 else 0.90))

    if current_run_mode == 1:
        ROOM_SIZE = 30.0; META_PARAMS["laser_range"] = 15.0
    else:
        ROOM_SIZE = 10.0; META_PARAMS["laser_range"] = 5.5

    START = np.array([ROOM_SIZE - 1.0, ROOM_SIZE - 1.0])
    GOAL  = np.array([1.0, 1.0])
    REAL_POLY = generate_random_room()

    robot_pos_true, robot_pos_est = START.copy(), START.copy()
    frame_counter, battery_level, run_spills, run_collisions = 0, META_PARAMS["initial_battery"], 0, 0
    mass_H, mass_notH, mass_Theta, plausibility_H = mass_H_init, 0.0, 1.0 - mass_H_init, 1.0
    eff_speed, eff_inflation, eff_resolution = POLICY_M0["speed"], POLICY_M0["inflation"], POLICY_M0["resolution"]
    META_PARAMS["resolution"] = POLICY_M0["resolution"]
    spill_events, laser_endpoints, pose_variance = [], [], 1.5
    particles = np.random.normal(START, pose_variance, (40, 2))
    frames_since_spill = SPILL_COOLDOWN_FRAMES
    hits_in_path = 0
    build_obstacle_grid()

def step_logic():
    global robot_pos_true, robot_pos_est, frame_counter, active_model_label, plausibility_H, laser_endpoints
    global battery_level, run_spills, run_collisions, prev_v_final, mass_H, mass_notH, mass_Theta
    global current_run, is_playing, pose_variance, particles
    global eff_speed, eff_inflation, eff_resolution, frames_since_spill, hits_in_path

    # 1. Sensing & DST
    if frame_counter % int(META_PARAMS["f_loc"]) == 0:
        laser_endpoints = []
        if np.linalg.norm(prev_v_final) > 0:
            current_heading = math.atan2(prev_v_final[1], prev_v_final[0])
        else:
            vg = GOAL - robot_pos_est
            current_heading = math.atan2(vg[1], vg[0])
        hits_in_path, min_dist, total_rays_in_cone = 0, META_PARAMS["laser_range"], 0
        for angle in np.linspace(0, 2*math.pi, 36, endpoint=False):
            d     = np.array([math.cos(angle), math.sin(angle)])
            t_obj = ray_poly_intersection(robot_pos_true, d, REAL_POLY) if not current_env_is_empty else None
            adiff = abs(math.atan2(math.sin(angle - current_heading), math.cos(current_heading - angle)))
            in_cone = adiff < math.radians(35)
            if in_cone: total_rays_in_cone += 1
            if t_obj and t_obj < META_PARAMS["laser_range"]:
                if in_cone:
                    hits_in_path += 1
                    min_dist = min(min_dist, t_obj)
                laser_endpoints.append((robot_pos_true + t_obj * d, "OBSTACLE"))
            else:
                laser_endpoints.append((robot_pos_true + META_PARAMS["laser_range"] * d, "WALL"))
        plausibility_H = update_belief_dst(hits_in_path, total_rays_in_cone, min_dist)

    # 2. Fuzzy switching & smoothing
    if plausibility_H <= dynamic_threshold:
        target_v, target_inf, target_res = POLICY_M1["speed"], POLICY_M1["inflation"], POLICY_M1["resolution"]
        active_model_label = "M1-SYMBOLIC (SAFE)"
    else:
        target_v, target_inf, target_res = POLICY_M0["speed"], POLICY_M0["inflation"], POLICY_M0["resolution"]
        active_model_label = "M0-REACTIVE (FAST)"
    eff_speed      += SMOOTH_VAL * (target_v   - eff_speed)
    eff_inflation  += SMOOTH_VAL * (target_inf - eff_inflation)
    eff_resolution += SMOOTH_VAL * (target_res - eff_resolution)

    # 3. Global planning & PF
    if frame_counter % int(META_PARAMS["f_glob"]) == 0:
        pose_variance = max(0.04, pose_variance * (0.95 + (hits_in_path / 36.0) * 0.1))
        particles     = np.random.normal(robot_pos_true, pose_variance, (40, 2))
        robot_pos_est = np.mean(particles, axis=0)
        q_res = round(eff_resolution / 0.05) * 0.05
        if abs(q_res - META_PARAMS["resolution"]) > 1e-9:
            META_PARAMS["resolution"] = q_res
            build_obstacle_grid()

    # 4. Path planning
    v_to_goal = GOAL - robot_pos_est
    drive_vec = v_to_goal / np.linalg.norm(v_to_goal) if np.linalg.norm(v_to_goal) > 0 else np.zeros(2)
    avoid_vec = np.zeros(2)
    dist_obs  = float('inf')
    if not current_env_is_empty:
        closest_p, dist_obs = get_closest_grid_obstacle(robot_pos_est)
        if dist_obs < eff_inflation:
            radial    = ((robot_pos_est - closest_p) / dist_obs) * 6.5
            avoid_vec = radial + np.array([radial[1], -radial[0]]) * 5.2
            drive_vec *= 0.1
    final_vec = drive_vec + avoid_vec
    if np.linalg.norm(final_vec) > 0: final_vec /= np.linalg.norm(final_vec)

    # 5. Physics
    dist_to_goal  = np.linalg.norm(robot_pos_est - GOAL)
    actual_speed  = eff_speed * (0.2 if dist_to_goal < 0.3 else 1.0)
    speed_modifier = 0.1 if actual_speed < 0.5 else 1.0
    jerk_mag   = actual_speed * np.linalg.norm(final_vec - prev_v_final)
    speed_risk = (actual_speed * 0.002 + actual_speed**2 * 0.001) * speed_modifier
    jerk_risk  = jerk_mag * actual_speed * 0.12 * speed_modifier
    cooldown_factor = min(1.0, frames_since_spill / SPILL_COOLDOWN_FRAMES)
    p_spill    = (speed_risk + jerk_risk) * cooldown_factor
    frames_since_spill += 1
    in_avoidance = 1 if np.linalg.norm(avoid_vec) > 0 else 0
    dist_obs_log = round(dist_obs, 3) if not math.isinf(dist_obs) else float('nan')
    spilled = False
    if random.random() < p_spill:
        spill_events.append(robot_pos_true.copy()); run_spills += 1; spilled = True
        frames_since_spill = 0
    dominant_risk = "speed" if speed_risk >= jerk_risk else "jerk"
    spill_cause   = dominant_risk if spilled else "none"
    collision = 1 if (not current_env_is_empty and is_inside_poly(robot_pos_true, REAL_POLY)) else 0
    if collision: run_collisions += 1

    cpu_load    = min(99.8, 12.0 + (len(particles)*0.4) + (0.05/META_PARAMS["resolution"])*45 + random.uniform(-0.5, 0.5))
    sensor_cost = 0.015 if frame_counter % int(META_PARAMS["f_loc"]) == 0 else 0.0
    battery_level = max(0.0, battery_level - (actual_speed * 0.06 + (cpu_load/100)*0.04 + sensor_cost))

    step_move = final_vec * (STEP_SIZE * actual_speed)
    robot_pos_true += step_move + np.random.normal(0, 0.02, 2)
    robot_pos_est  += step_move
    particles      += step_move + np.random.normal(0, 0.04, (40, 2))
    prev_v_final    = final_vec.copy()

    # 6. Telemetry
    all_iterations_data.append({
        "Run": current_run, "Frame": frame_counter,
        "Mode": "Domestic" if current_run_mode == 0 else "Warehouse",
        "Active_Model": active_model_label,
        "Fuzzy_Threshold": round(dynamic_threshold, 2), "Pl_H": plausibility_H,
        "m_H": round(mass_H, 2), "m_notH": round(mass_notH, 2),
        "Speed": round(actual_speed, 3),
        "Resolution": round(META_PARAMS["resolution"], 3),
        "Jerk": round(jerk_mag, 4),
        "Speed_Risk": round(speed_risk, 5), "Jerk_Risk": round(jerk_risk, 5),
        "p_spill": round(p_spill, 5),
        "Spill": 1 if spilled else 0, "Spill_Cause": spill_cause,
        "In_Avoidance": in_avoidance, "Dist_Obs": dist_obs_log,
        "Collision": collision,
        "Loc_Error": round(np.linalg.norm(robot_pos_true - robot_pos_est), 4),
        "Bat": round(battery_level, 1), "CPU": round(cpu_load, 1)
    })

    # 7. Goal check
    if np.linalg.norm(robot_pos_true - GOAL) < META_PARAMS["goal_threshold"]:
        ideal_f    = np.linalg.norm(START - GOAL) / (STEP_SIZE * 0.85)
        time_ratio = frame_counter / ideal_f
        time_factor = max(0.2, 1.3 - 0.30 * time_ratio)
        base_score = 100 - (run_spills * 15) - (run_collisions * 100)
        if run_spills == 0 and run_collisions == 0: base_score = 100
        final_se = max(0, base_score * time_factor * (battery_level / 100.0))
        historical_SE.append(final_se)
        print(f"    run {current_run} done | SE={final_se:.1f} | spills={run_spills} | frames={frame_counter}")
        if current_run < total_runs_to_execute:
            current_run += 1
            reset_run()
        else:
            is_playing = False
    frame_counter += 1

def save_to_excel(cfg_name):
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    df       = pd.DataFrame(all_iterations_data)

    report = pd.DataFrame({
        "Metric": ["Avg SE Mark", "Fuzzy Threshold Avg", "Total Spills", "Total Collisions", "Spill Rate (per frame)"],
        "Result": [
            round(np.mean(historical_SE), 2),
            round(df['Fuzzy_Threshold'].mean(), 3),
            int(df['Spill'].sum()),
            int(df['Collision'].sum()),
            round(df['Spill'].mean(), 5),
        ]
    })

    spill_cols = ['Run','Frame','Mode','Active_Model','Resolution','Speed','Jerk',
                  'Speed_Risk','Jerk_Risk','p_spill','Dist_Obs',
                  'In_Avoidance','Spill_Cause','Pl_H','Loc_Error','Bat']
    spill_events_df = df[df['Spill'] == 1][[c for c in spill_cols if c in df.columns]].copy()

    stat_cols = ['Resolution','Speed','Jerk','p_spill','Speed_Risk','Jerk_Risk','Dist_Obs','In_Avoidance','Loc_Error']
    stat_cols = [c for c in stat_cols if c in df.columns]
    spill_stats = df.groupby('Spill')[stat_cols].agg(['mean','std']).round(5)
    spill_stats.index = ['No Spill', 'Spill']

    config = pd.DataFrame({
        "Parameter": ["Model","Config",
                      "Policy_M0_Speed","Policy_M0_Inflation","Policy_M0_Resolution",
                      "Policy_M1_Speed","Policy_M1_Inflation","Policy_M1_Resolution",
                      "Freq_Loc","Freq_Glob","Initial_Battery_%","Laser_Range",
                      "Sensor_Std","Encoder_Std","Init_Uncertainty","Goal_Threshold",
                      "Fuzzy_Threshold_Final","Mass_H_Init_Final",
                      "Total_Runs","Room_Size_at_End","Step_Size"],
        "Value": ["M2-DST-HYBRID", cfg_name.upper(),
                  POLICY_M0["speed"], POLICY_M0["inflation"], POLICY_M0["resolution"],
                  POLICY_M1["speed"], POLICY_M1["inflation"], POLICY_M1["resolution"],
                  META_PARAMS["f_loc"], META_PARAMS["f_glob"],
                  META_PARAMS["initial_battery"], META_PARAMS["laser_range"],
                  META_PARAMS["sensor_std"], META_PARAMS["encoder_std"],
                  META_PARAMS["init_uncertainty"], META_PARAMS["goal_threshold"],
                  round(dynamic_threshold, 3), round(mass_H_init, 3),
                  total_runs_to_execute, ROOM_SIZE, STEP_SIZE]
    })

    out = f"model2_DST_{cfg_name}_{date_str}.xlsx"
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

MAX_FRAMES_TOTAL = 80_000  # safety cap per config

def init_config(cfg_envs):
    global historical_SE, dynamic_threshold, mass_H_init
    global mass_H, mass_notH, mass_Theta, plausibility_H
    global eff_speed, eff_inflation, eff_resolution
    global current_run, allowed_envs, all_iterations_data, is_playing
    global prev_v_final, frames_since_spill, hits_in_path

    historical_SE     = []
    dynamic_threshold = 0.5
    mass_H_init       = 0.5
    mass_H, mass_notH, mass_Theta, plausibility_H = 0.5, 0.0, 0.5, 1.0
    eff_speed          = POLICY_M0["speed"]
    eff_inflation      = POLICY_M0["inflation"]
    eff_resolution     = POLICY_M0["resolution"]
    META_PARAMS["resolution"] = POLICY_M0["resolution"]
    allowed_envs[:]    = cfg_envs
    current_run        = 1
    all_iterations_data = []
    is_playing         = True
    prev_v_final       = np.zeros(2)
    frames_since_spill = SPILL_COOLDOWN_FRAMES
    hits_in_path       = 0

if __name__ == "__main__":
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
