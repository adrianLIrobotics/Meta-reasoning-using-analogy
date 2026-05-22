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
# 🚨 MODELO 3: DYNAMIC HISTORICAL AGENT - VERSION 3.5 (FINAL MASTER) 🚨
# Logic: Dynamic SE, Changing Rooms, Anti-Dogmatism, Dynamic Sensors
# Visuals: Fully Restored 7-Parameter Control Panel + PF + Inflation
# =======================================================================

# --- Historical Experience (The Analogy Base) ---
historical_SE = []      
dynamic_threshold = 0.5 
mass_H_init = 0.5       

# --- DST Global Beliefs ---
mass_H = mass_H_init       
mass_notH = 0.0    
mass_Theta = 1.0 - mass_H_init   
plausibility_H = 1.0

# Base Policies
POLICY_M0 = {"speed": 0.65, "inflation": 0.85, "resolution": 0.10}
POLICY_M1 = {"speed": 0.40, "inflation": 1.85, "resolution": 0.05}

# Smoothing states
eff_speed = POLICY_M0["speed"]
eff_inflation = POLICY_M0["inflation"]
eff_resolution = POLICY_M0["resolution"]
SMOOTH_VAL = 0.08
MAX_TURN_PER_FRAME    = 0.25   # rad/frame kinematic steering cap
SPILL_COOLDOWN_FRAMES = 50     # frames for cup to re-settle after a spill

META_PARAMS = {
    "speed": 0.65, "inflation": 0.85, "resolution": 0.1,
    "f_loc": 3, "f_glob": 15, "initial_battery": 100.0,
    "sensor_std": 0.03, "encoder_std": 0.02, "init_uncertainty": 1.5,
    "goal_threshold": 0.15, "laser_range": 5.5
}

# --- Dynamic Room Scaling Globals ---
ROOM_SIZE = 10.0 
START, GOAL = np.array([ROOM_SIZE - 1.0, ROOM_SIZE - 1.0]), np.array([1.0, 1.0])
STEP_SIZE = 0.22 

# Base Obstacle Pattern
base_poly = np.array([[3.5, 4.0], [6.0, 4.0], [6.0, 4.5], [4.5, 4.5], [4.5, 6.0], [3.5, 6.0]])
REAL_POLY = base_poly.copy()

# --- System State ---
total_runs_to_execute = 5
current_run = 1
allowed_envs = [True, True] 
current_run_mode = 0 
current_env_is_empty = True
all_iterations_data = []
is_playing = False
robot_pos_true, robot_pos_est = START.copy(), START.copy()
active_model_label = "M0-REACTIVE"
frame_counter = 0

run_spills = 0
run_collisions = 0
spill_events, laser_endpoints = [], []
battery_level = 100.0
prev_v_final = np.zeros(2)
frames_since_spill = SPILL_COOLDOWN_FRAMES
pose_variance = 1.5
particles = np.random.normal(START, pose_variance, (40, 2))

OBSTACLE_GRID = None   # bool 2-D array: grid cells occupied by obstacle
GRID_X_EDGES  = None   # x-edges of those cells
GRID_Y_EDGES  = None   # y-edges of those cells

# ==========================================
# 🛠️ ENGINES (DYNAMIC ROOMS & DST)
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

def update_belief_dst(hits, total_rays, min_dist):
    global mass_H, mass_notH, mass_Theta
    
    density_factor = hits / total_rays if total_rays > 0 else 0
    proximity_factor = max(0, 1.0 - (min_dist / META_PARAMS["laser_range"]))
    
    obs_notH = np.clip(density_factor * 0.6 + proximity_factor * 0.4, 0, 0.95)
    loc_err = np.linalg.norm(robot_pos_true - robot_pos_est)
    obs_Theta = np.clip(loc_err / 1.5, 0.05, 0.6)
    obs_H = max(0, 1.0 - obs_notH - obs_Theta)
    
    K = (mass_H * obs_notH) + (mass_notH * obs_H)
    if K >= 1: K = 0.99
    
    new_H = (mass_H * obs_H + mass_H * obs_Theta + mass_Theta * obs_H) / (1 - K)
    new_notH = (mass_notH * obs_notH + mass_notH * obs_Theta + mass_Theta * obs_notH) / (1 - K)
    new_Theta = (mass_Theta * obs_Theta) / (1 - K)
    
    total = new_H + new_notH + new_Theta
    mass_H, mass_notH, mass_Theta = new_H/total, new_notH/total, new_Theta/total
    
    # 🚨 FIX: ANTI-DOGMATISM 🚨
    # Ensures the robot can always instantly react to sudden danger
    if mass_Theta < 0.05:
        diff = 0.05 - mass_Theta
        mass_Theta = 0.05
        mass_H -= diff * (mass_H / (mass_H + mass_notH + 1e-9))
        mass_notH -= diff * (mass_notH / (mass_H + mass_notH + 1e-9))
    
    pl = 1.0 - mass_notH
    return min([0, 0.25, 0.5, 0.75, 1.0], key=lambda x: abs(x - pl))

def ray_poly_intersection(origin, direction, poly):
    min_t = float('inf')
    for i in range(len(poly)):
        p1, p2 = poly[i], poly[(i + 1) % len(poly)]
        v1, v2 = origin - p1, p2 - p1
        v3 = np.array([-direction[1], direction[0]])
        det = np.dot(v2, v3)
        if abs(det) < 1e-6: continue
        t1 = np.cross(v2, v1) / det
        t2 = np.dot(v1, v3) / det
        if t1 >= 0 and 0 <= t2 <= 1: min_t = min(min_t, t1)
    return min_t if min_t != float('inf') else None

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

# ==========================================
# 🧠 STEP LOGIC (THE BRAIN)
# ==========================================

def step_logic():
    global robot_pos_true, robot_pos_est, frame_counter, active_model_label, plausibility_H, laser_endpoints
    global battery_level, run_spills, run_collisions, prev_v_final, mass_H, mass_notH, mass_Theta
    global current_run, is_playing, pose_variance, particles, eff_speed, eff_inflation, eff_resolution, frames_since_spill

    # 1. Sensing & Proportional DST
    if frame_counter % int(META_PARAMS["f_loc"]) == 0:
        laser_endpoints = []
        
        # 🚨 VELOCITY-ALIGNED VISION CONE FIX 🚨
        if np.linalg.norm(prev_v_final) > 0:
            current_heading = math.atan2(prev_v_final[1], prev_v_final[0])
        else:
            # Fallback to goal if standing perfectly still
            v_goal = GOAL - robot_pos_est
            current_heading = math.atan2(v_goal[1], v_goal[0])
            
        hits_in_path, min_dist = 0, META_PARAMS["laser_range"]
        total_rays_in_cone = 0
        
        for angle in np.linspace(0, 2*math.pi, 36, endpoint=False):
            d = np.array([math.cos(angle), math.sin(angle)])
            t_obj = ray_poly_intersection(robot_pos_true, d, REAL_POLY) if not current_env_is_empty else None
            
            # Calculate difference against the current heading, not the distant goal
            angle_diff = abs(math.atan2(math.sin(angle-current_heading), math.cos(current_heading-angle)))
            in_cone = angle_diff < math.radians(35)
            if in_cone: total_rays_in_cone += 1

            if t_obj and t_obj < META_PARAMS["laser_range"]:
                if in_cone: 
                    hits_in_path += 1
                    min_dist = min(min_dist, t_obj)
                laser_endpoints.append((robot_pos_true + t_obj * d, "OBSTACLE"))
            else:
                laser_endpoints.append((robot_pos_true + META_PARAMS["laser_range"] * d, "WALL"))
        
        plausibility_H = update_belief_dst(hits_in_path, total_rays_in_cone, min_dist)

    # 2. FUZZY SWITCHING & SMOOTHING
    if plausibility_H <= dynamic_threshold:
        target_v, target_inf, target_res = POLICY_M1["speed"], POLICY_M1["inflation"], POLICY_M1["resolution"]
        active_model_label = "M1-SYMBOLIC (SAFE)"
    else:
        target_v, target_inf, target_res = POLICY_M0["speed"], POLICY_M0["inflation"], POLICY_M0["resolution"]
        active_model_label = "M0-REACTIVE (FAST)"

    eff_speed += SMOOTH_VAL * (target_v - eff_speed)
    eff_inflation += SMOOTH_VAL * (target_inf - eff_inflation)
    eff_resolution += SMOOTH_VAL * (target_res - eff_resolution)

    # 3. Path Planning & PF Maintenance
    if frame_counter % int(META_PARAMS["f_glob"]) == 0:
        pose_variance = max(0.04, pose_variance * (0.95 + (hits_in_path/36.0)*0.1))
        particles = np.random.normal(robot_pos_true, pose_variance, (40, 2))
        robot_pos_est = np.mean(particles, axis=0)
        # Dynamic resolution: quantize smoothed value and rebuild grid if it changed
        quantized_res = round(eff_resolution / 0.05) * 0.05
        if abs(quantized_res - META_PARAMS["resolution"]) > 1e-9:
            META_PARAMS["resolution"] = quantized_res
            build_obstacle_grid()

    v_to_goal = GOAL - robot_pos_est
    drive_vec = v_to_goal / np.linalg.norm(v_to_goal) if np.linalg.norm(v_to_goal) > 0 else np.zeros(2)
    avoid_vec = np.zeros(2)
    dist_obs = float('inf')
    if not current_env_is_empty:
        closest_p, dist_obs = get_closest_grid_obstacle(robot_pos_est)
        if dist_obs < eff_inflation:
            radial = ((robot_pos_est - closest_p) / dist_obs) * 6.5
            avoid_vec = radial + np.array([radial[1], -radial[0]]) * 5.2
            drive_vec *= 0.1

    final_vec = drive_vec + avoid_vec
    if np.linalg.norm(final_vec) > 0: final_vec /= np.linalg.norm(final_vec)

    # Fix 1: kinematic turn-rate cap — prevents physically impossible reversals
    if np.linalg.norm(prev_v_final) > 0 and np.linalg.norm(final_vec) > 0:
        cross = final_vec[0]*prev_v_final[1] - final_vec[1]*prev_v_final[0]
        dot   = final_vec[0]*prev_v_final[0] + final_vec[1]*prev_v_final[1]
        angle_diff = math.atan2(cross, dot)
        if abs(angle_diff) > MAX_TURN_PER_FRAME:
            clamp = MAX_TURN_PER_FRAME * (1 if angle_diff > 0 else -1)
            c, s = math.cos(clamp), math.sin(clamp)
            px, py = prev_v_final[0], prev_v_final[1]
            final_vec = np.array([c*px - s*py, s*px + c*py])

    # 4. Movement & Physics — realism fixes applied
    # Fix 4: goal approach damping
    dist_to_goal = np.linalg.norm(robot_pos_est - GOAL)
    actual_speed = eff_speed * (0.2 if dist_to_goal < 1.0 else 1.0)
    # Fix 2 & 3: physical jerk magnitude + vibration baseline speed risk
    speed_modifier = 0.1 if actual_speed < 0.5 else 1.0
    jerk_mag   = actual_speed * np.linalg.norm(final_vec - prev_v_final)
    speed_risk = (actual_speed * 0.002 + actual_speed**2 * 0.001) * speed_modifier
    jerk_risk  = jerk_mag * actual_speed * 0.12 * speed_modifier
    # Fix 5: post-spill cooldown
    cooldown_factor = min(1.0, frames_since_spill / SPILL_COOLDOWN_FRAMES)
    p_spill    = (speed_risk + jerk_risk) * cooldown_factor
    frames_since_spill += 1
    in_avoidance  = 1 if np.linalg.norm(avoid_vec) > 0 else 0
    dist_obs_log  = round(dist_obs, 3) if not math.isinf(dist_obs) else float('nan')
    spilled = False
    if random.random() < p_spill:
        spill_events.append(robot_pos_true.copy()); run_spills += 1; spilled = True
        frames_since_spill = 0
    dominant_risk = "speed" if speed_risk >= jerk_risk else "jerk"
    spill_cause = dominant_risk if spilled else "none"

    collision = 1 if (not current_env_is_empty and is_inside_poly(robot_pos_true, REAL_POLY)) else 0
    if collision: run_collisions += 1

    cpu_load = min(99.8, 12.0 + (len(particles) * 0.4) + (0.05/META_PARAMS["resolution"])*45 + random.uniform(-0.5, 0.5))
    sensor_cost = 0.015 if frame_counter % int(META_PARAMS["f_loc"]) == 0 else 0.0
    battery_level = max(0.0, battery_level - (actual_speed * 0.06 + (cpu_load / 100) * 0.04 + sensor_cost))

    step_move = final_vec * (STEP_SIZE * actual_speed)
    robot_pos_true += step_move + np.random.normal(0, 0.02, 2)
    robot_pos_est += step_move
    particles += step_move + np.random.normal(0, 0.04, (40, 2))
    prev_v_final = final_vec.copy()
    
    # 5. Log telemetry
    all_iterations_data.append({
        "Run": current_run, "Frame": frame_counter,
        "Mode": "Domestic" if current_run_mode == 0 else "Warehouse",
        "Active_Model": active_model_label,
        "Fuzzy_Threshold": round(dynamic_threshold, 2), "Pl_H": plausibility_H,
        "m_H": round(mass_H, 2), "m_notH": round(mass_notH, 2),
        "Speed": round(actual_speed, 2),
        "Resolution": round(META_PARAMS["resolution"], 3),
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
        ideal_f = np.linalg.norm(START-GOAL) / (STEP_SIZE * 0.85)
        time_ratio = frame_counter / ideal_f
        time_factor = max(0.2, 1.3 - 0.30 * time_ratio)
        
        base_score = 100 - (run_spills * 15) - (run_collisions * 100)
        if run_spills == 0 and run_collisions == 0: 
            base_score = 100
            
        final_se = max(0, base_score * time_factor * (battery_level/100.0))
        historical_SE.append(final_se)
        
        if current_run < total_runs_to_execute: 
            current_run += 1
            reset_run()
        else: 
            is_playing = False
            save_to_excel()
    frame_counter += 1

# ==========================================
# 🛠️ UTILS & UI CONTROLS
# ==========================================

def get_closest_point_on_poly(p, poly):
    min_dist = float('inf'); closest = np.array([0.0, 0.0])
    for i in range(len(poly)):
        p1, p2 = poly[i], poly[(i + 1) % len(poly)]
        v, w = p2 - p1, p - p1
        t = np.clip(np.dot(w, v) / np.dot(v, v), 0, 1)
        proj = p1 + t * v; d = np.linalg.norm(p - proj)
        if d < min_dist: min_dist, closest = d, proj
    return closest, min_dist

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
        n_in = normals[(i - 1) % n]
        n_out = normals[i]
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

def reset_run():
    global robot_pos_true, robot_pos_est, frame_counter, mass_H, mass_notH, mass_Theta, plausibility_H
    global spill_events, particles, pose_variance, current_run_mode, current_env_is_empty, battery_level
    global run_spills, run_collisions, REAL_POLY, dynamic_threshold, mass_H_init, eff_speed, eff_inflation, eff_resolution
    global ROOM_SIZE, START, GOAL, OBSTACLE_GRID, GRID_X_EDGES, GRID_Y_EDGES, frames_since_spill
    
    if historical_SE:
        avg_se = np.mean(historical_SE[-3:]) 
        dynamic_threshold = np.clip(0.5 + (70 - avg_se) * 0.005, 0.25, 0.75)
        mass_H_init = np.clip(0.5 + (avg_se - 50) * 0.005, 0.2, 0.8)
    
    active_indices = [i for i, val in enumerate(allowed_envs) if val]
    current_run_mode = random.choice(active_indices) if active_indices else 0
    current_env_is_empty = (random.random() >= (0.10 if current_run_mode == 0 else 0.90))
    
    # 🚨 FIX: DYNAMIC SENSOR UPGRADE 🚨
    if current_run_mode == 1: # Warehouse
        ROOM_SIZE = 30.0
        META_PARAMS["laser_range"] = 15.0
    else:                     # Domestic
        ROOM_SIZE = 10.0
        META_PARAMS["laser_range"] = 5.5
        
    START = np.array([ROOM_SIZE - 1.0, ROOM_SIZE - 1.0])
    GOAL = np.array([1.0, 1.0])
    
    REAL_POLY = generate_random_room() 
    
    robot_pos_true, robot_pos_est = START.copy(), START.copy()
    frame_counter, battery_level, run_spills, run_collisions = 0, META_PARAMS["initial_battery"], 0, 0
    mass_H, mass_notH, mass_Theta, plausibility_H = mass_H_init, 0.0, 1.0 - mass_H_init, 1.0
    eff_speed, eff_inflation, eff_resolution = POLICY_M0["speed"], POLICY_M0["inflation"], POLICY_M0["resolution"]
    META_PARAMS["resolution"] = POLICY_M0["resolution"]
    spill_events, laser_endpoints, pose_variance = [], [], 1.5
    particles = np.random.normal(START, pose_variance, (40, 2))
    frames_since_spill = SPILL_COOLDOWN_FRAMES
    build_obstacle_grid()

def save_to_excel():
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    df = pd.DataFrame(all_iterations_data)

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

    spill_cols = ['Run', 'Frame', 'Mode', 'Active_Model', 'Resolution', 'Speed', 'Jerk',
                  'Speed_Risk', 'Jerk_Risk', 'p_spill', 'Dist_Obs',
                  'In_Avoidance', 'Spill_Cause', 'Pl_H', 'Loc_Error', 'Bat']
    spill_events_df = df[df['Spill'] == 1][[c for c in spill_cols if c in df.columns]].copy()

    stat_cols = ['Resolution', 'Speed', 'Jerk', 'p_spill', 'Speed_Risk', 'Jerk_Risk', 'Dist_Obs', 'In_Avoidance', 'Loc_Error']
    stat_cols = [c for c in stat_cols if c in df.columns]
    spill_stats = df.groupby('Spill')[stat_cols].agg(['mean', 'std']).round(5)
    spill_stats.index = ['No Spill', 'Spill']

    config = pd.DataFrame({
        "Parameter": ["Model",
                      "Policy_M0_Speed", "Policy_M0_Inflation", "Policy_M0_Resolution",
                      "Policy_M1_Speed", "Policy_M1_Inflation", "Policy_M1_Resolution",
                      "Freq_Loc", "Freq_Glob",
                      "Initial_Battery_%", "Laser_Range",
                      "Sensor_Std", "Encoder_Std",
                      "Init_Uncertainty", "Goal_Threshold",
                      "Fuzzy_Threshold_Final", "Mass_H_Init_Final",
                      "Total_Runs", "Room_Size_at_End", "Step_Size"],
        "Value": ["M2-DST-HYBRID",
                  POLICY_M0["speed"], POLICY_M0["inflation"], POLICY_M0["resolution"],
                  POLICY_M1["speed"], POLICY_M1["inflation"], POLICY_M1["resolution"],
                  META_PARAMS["f_loc"], META_PARAMS["f_glob"],
                  META_PARAMS["initial_battery"], META_PARAMS["laser_range"],
                  META_PARAMS["sensor_std"], META_PARAMS["encoder_std"],
                  META_PARAMS["init_uncertainty"], META_PARAMS["goal_threshold"],
                  round(dynamic_threshold, 3), round(mass_H_init, 3),
                  total_runs_to_execute, ROOM_SIZE, STEP_SIZE]
    })

    with pd.ExcelWriter(f"model3_dynamic_agent_v35_final_{date_str}.xlsx") as writer:
        config.to_excel(writer, sheet_name='Config', index=False)
        df.to_excel(writer, sheet_name='Telemetry', index=False)
        report.to_excel(writer, sheet_name='Performance_Report', index=False)
        spill_events_df.to_excel(writer, sheet_name='Spill_Events', index=False)
        spill_stats.to_excel(writer, sheet_name='Spill_Analysis')
    print(f"✅ EXCEL GENERATED.")

# --- MASTER HUD ---
fig, ax = plt.subplots(figsize=(15, 9))
plt.subplots_adjust(left=0.25, bottom=0.26, right=0.95)

def draw_graphics():
    ax.clear(); ax.set_xlim(-0.5, ROOM_SIZE + 0.5); ax.set_ylim(-0.5, ROOM_SIZE + 0.5); ax.set_aspect('equal')
    
    ax.add_patch(Rectangle((0,0), ROOM_SIZE, ROOM_SIZE, fill=False, edgecolor='black', lw=2))

    res = META_PARAMS["resolution"]
    xs = np.arange(0, ROOM_SIZE + res, res)
    ys = np.arange(0, ROOM_SIZE + res, res)
    ax.vlines(xs, 0, ROOM_SIZE, colors=['slategray'], alpha=0.12, lw=0.4, zorder=0)
    ax.hlines(ys, 0, ROOM_SIZE, colors=['slategray'], alpha=0.12, lw=0.4, zorder=0)

    borders = [
        ((0,0), ROOM_SIZE, eff_inflation), 
        ((0, ROOM_SIZE-eff_inflation), ROOM_SIZE, eff_inflation), 
        ((0,0), eff_inflation, ROOM_SIZE), 
        ((ROOM_SIZE-eff_inflation, 0), eff_inflation, ROOM_SIZE)
    ]
    [ax.add_patch(Rectangle(xy, w, h, color='cyan', alpha=0.1)) for xy, w, h in borders]
    
    if not current_env_is_empty:
        # Resolution layer: draw the cached occupancy grid the robot actually
        # uses for planning — coarser resolution = blockier shape = less accuracy.
        if (OBSTACLE_GRID is not None and GRID_X_EDGES is not None
                and GRID_Y_EDGES is not None and OBSTACLE_GRID.any()):
            C = np.ma.masked_where(~OBSTACLE_GRID,
                                   np.ones_like(OBSTACLE_GRID, dtype=float))
            ax.pcolormesh(GRID_X_EDGES, GRID_Y_EDGES, C, cmap=_res_cmap,
                          alpha=0.55, zorder=3, shading='flat')

        ax.add_patch(Polygon(REAL_POLY, color='black', alpha=0.7, zorder=10))
        inflated = inflate_polygon(REAL_POLY, eff_inflation)
        ax.add_patch(Polygon(inflated.tolist(), color='cyan', alpha=0.15, zorder=2))
    
    ax.scatter(particles[:,0], particles[:,1], color='red', s=5, alpha=0.2)
    for ep, ht in laser_endpoints:
        ax.plot([robot_pos_true[0], ep[0]], [robot_pos_true[1], ep[1]], color='orange' if ht=="OBSTACLE" else 'green', alpha=0.1, lw=1)
    
    for s in spill_events: ax.scatter(s[0], s[1], color='brown', s=30)
    ax.plot(robot_pos_true[0], robot_pos_true[1], 'bo', markersize=9)
    ax.plot(robot_pos_est[0], robot_pos_est[1], 'go', alpha=0.7)
    ax.plot(GOAL[0], GOAL[1], 'rx', markersize=15)
    
    color = 'red' if "M1" in active_model_label else 'green'
    past_se = np.mean(historical_SE[-3:]) if historical_SE else 0.0
    
    ax.text(0.1, ROOM_SIZE + 0.6, f"AGENT: {active_model_label}", color=color, fontweight='bold', fontsize=12)
    ax.text(0.1, ROOM_SIZE + 0.2, f"Prior m(H): {mass_H_init:.2f} | Dyn Threshold: {dynamic_threshold:.2f} | Past SE Avg: {past_se:.1f}", fontsize=10, color='purple')
    ax.text(ROOM_SIZE - 2.5, ROOM_SIZE + 0.2, f"Plausibility H: {plausibility_H:.2f}", fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))

def update(frame):
    if is_playing: step_logic(); draw_graphics()

# --- FULL GUI RESTORED ---
ui_cfg = [
    (0.88, 'Max Speed:', str(POLICY_M0["speed"]), lambda v: POLICY_M0.update({"speed": float(v)})),
    (0.82, 'Inflation:', str(POLICY_M0["inflation"]), lambda v: POLICY_M0.update({"inflation": float(v)})),
    (0.76, 'Resolution:', str(META_PARAMS["resolution"]), lambda v: (META_PARAMS.update({"resolution": float(v)}), build_obstacle_grid())),
    (0.70, 'Freq Loc:', str(META_PARAMS["f_loc"]), lambda v: META_PARAMS.update({"f_loc": int(v)})),
    (0.64, 'Freq Glob:', str(META_PARAMS["f_glob"]), lambda v: META_PARAMS.update({"f_glob": int(v)})),
    (0.58, 'Runs:', str(total_runs_to_execute), lambda v: globals().update(total_runs_to_execute=int(v))),
    (0.52, 'Init Bat %:', str(META_PARAMS["initial_battery"]), lambda v: META_PARAMS.update({"initial_battery": float(v)}))
]

boxes = [TextBox(plt.axes([0.12, y, 0.05, 0.035]), label, initial=init) for y, label, init, func in ui_cfg]
for i, box in enumerate(boxes): box.on_submit(ui_cfg[i][3])

check = CheckButtons(plt.axes([0.02, 0.35, 0.15, 0.12]), ('Domestic', 'Warehouse'), (True, True))
check.on_clicked(lambda l: allowed_envs.__setitem__(0, not allowed_envs[0]) if l=='Domestic' else allowed_envs.__setitem__(1, not allowed_envs[1]))

btn = Button(plt.axes([0.02, 0.25, 0.15, 0.06]), '▶ Start Historical Agent', color='cyan')
btn.on_clicked(lambda e: globals().update({"is_playing": True, "current_run": 1, "all_iterations_data": [], "historical_SE": []}) or reset_run())

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
    Patch(facecolor='black', alpha=0.7,
          label='Obstacle  –  physical L-shaped blocker'),
    Patch(facecolor='cyan', alpha=0.40,
          label='Inflation zone  –  safety buffer (walls & obstacle)'),
    Line2D([0],[0], color='slategray', alpha=0.6, lw=1.5, linestyle='--',
           label='Planning grid  –  cell size = Resolution param (finer → higher CPU load)'),
    Patch(facecolor='mediumpurple', alpha=0.55,
          label='Resolution layer  –  obstacle rasterised on grid (fewer cells = blockier / less accurate robot map)'),
    Patch(facecolor='limegreen', alpha=0.8,
          label='M0-REACTIVE  –  fast mode  (Pl_H > threshold)'),
    Patch(facecolor='red', alpha=0.8,
          label='M1-SYMBOLIC  –  safe mode  (Pl_H ≤ threshold)'),
]

ax_legend.legend(
    handles=legend_handles,
    loc='center', ncol=3,
    fontsize=7.8, frameon=True, fancybox=True, shadow=True,
    title='VISUAL LEGEND', title_fontsize=9,
    edgecolor='gray', labelspacing=0.55, handlelength=2.0,
    handleheight=1.2, handletextpad=0.6, columnspacing=1.2,
)

draw_graphics()
ani = FuncAnimation(fig, update, interval=40, cache_frame_data=False)
plt.show()