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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── CONSTANTS ──────────────────────────────────────────────────────────────
STEP_SIZE             = 0.22
SMOOTH_VAL            = 0.25   # raised from 0.08 — faster policy transitions
MAX_TURN_PER_FRAME    = 0.25
SPILL_COOLDOWN_FRAMES = 50

POLICY_M0 = {"speed": 0.65, "inflation": 0.85, "resolution": 0.10}
POLICY_M1 = {"speed": 0.40, "inflation": 1.85, "resolution": 0.05}
# Neutral fallback when Pl_analogy is low — midpoint between M0 and M1.
POLICY_NEUTRAL = {
    "speed":      (POLICY_M0["speed"]      + POLICY_M1["speed"])      / 2.0,  # 0.525
    "inflation":  (POLICY_M0["inflation"]  + POLICY_M1["inflation"])  / 2.0,  # 1.35
    "resolution": (POLICY_M0["resolution"] + POLICY_M1["resolution"]) / 2.0,  # 0.075
}

# ─── ANALOGY LIBRARY ─────────────────────────────────────────────────────────
# Each analogy carries EXPERT-CALIBRATED parameters derived from domain knowledge,
# independent of M0/M1. When the analogy is correct and the environment matches,
# these parameters outperform both baselines. DST (Pl_analogy) only adjusts when
# evidence suggests the analogy is wrong; Pl_H adjusts in real-time to local hazard.
#
#  medical_care: hospital staff serving hot drinks — precise, careful, not frozen.
#    speed=0.58  (fast enough for service delivery, safe for liquid cargo)
#    inflation=1.20  (clear berths around furniture without excessive detours)
#    resolution=0.04 (fine grid → smooth, jerk-free paths → fewer spills)
#
#  efficient_courier: timed parcel delivery in a known, controlled environment.
#    speed=0.63  (near-M0, time is primary KPI)
#    inflation=0.88  (minimal margins — courier knows the route)
#    resolution=0.09 (coarser grid → fast replanning → responsive)
ANALOGY_INPUT = "Hot coffee is dangerous, safety first."

TASK_FEATURES = {
    "safety":    ["hot", "dangerous", "danger", "safety", "safe", "caution", "careful"],
    "liquid":    ["coffee", "soup", "drink", "beverage", "water", "tea", "liquid"],
    "fragile":   ["fragile", "glass", "delicate", "breakable"],
    "urgent":    ["urgent", "fast", "quickly", "emergency", "hurry", "rush"],
    "vulnerable":["elderly", "patient", "child", "sick", "injured"],
}

ANALOGY_LIBRARY = {
    "medical_care": {
        "description": "Serving fragile/dangerous items near vulnerable people — safety paramount",
        "signature": np.array([0.9, 0.1]),   # [safety_attention, urgency_attention]
        "policy": {"speed": 0.58, "inflation": 1.20, "resolution": 0.04},
    },
    "efficient_courier": {
        "description": "Time-pressured delivery in a known controlled environment",
        "signature": np.array([0.2, 0.8]),
        "policy": {"speed": 0.63, "inflation": 0.88, "resolution": 0.09},
    },
}

# Novel entity: physically present on the path but NOT in REAL_POLY.
# M0/M1 are blind to it (their ray/grid model doesn't know it exists).
# M2 detects anomalous laser returns and reasons by analogy.
NOVEL_ENTITY_SPAWN_RUN = 8   # appears starting from this run number
NOVEL_ENTITY_TYPES = {
    "unknown_cluster": {"radius": 0.45, "structured": 0.1},
    "fallen_chair":    {"radius": 0.30, "structured": 0.8},
    "foil_lid":        {"radius": 0.15, "structured": 0.3},
}

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
_sweep_threshold   = None   # when set, overrides adaptive threshold (sweep mode)
mass_H, mass_notH, mass_Theta = 0.5, 0.0, 0.5
plausibility_H     = 1.0
eff_speed          = POLICY_M0["speed"]
eff_inflation      = POLICY_M0["inflation"]
eff_resolution     = POLICY_M0["resolution"]

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

# ─── NOVEL ENTITY STATE ───────────────────────────────────────────────────────
novel_entity_active  = False
novel_entity_pos     = np.array([0.0, 0.0])
novel_entity_radius  = 0.45
novel_entity_type    = "unknown_cluster"

# ─── ANALOGY / TASK PROFILE STATE ────────────────────────────────────────────
selected_analogy_name = "medical_care"
base_policy           = POLICY_M1
task_profile          = np.array([0.9, 0.1])

# ─── COLD-START / ADAPTATION TRACKING ────────────────────────────────────────
cold_start_data = []   # list of dicts: {run, se, entity_active, analogy}

# ─── ANALOGY BELIEF (second DST layer) ───────────────────────────────────────
# Frame of discernment: {A_correct, A_wrong, Θ}
# Updated per run from outcomes; modulates effective_sa → operating envelope.
analogy_m_correct = 0.0    # DST mass: analogy is correct
analogy_m_wrong   = 0.0    # DST mass: analogy is wrong
analogy_m_open    = 1.0    # DST mass: unknown (starts fully open)
Pl_analogy        = 0.80   # Pl(A_correct) = m_correct + m_open

anomaly_detected_in_run = False   # set True when anomaly hits > 0 in any frame
analogy_history         = []      # per-run snapshots: {run, Pl_analogy, se, ...}

# ─── ANOMALY PERSISTENCE (path planner reads between sensing cycles) ───────────
anomaly_last_hits = 0
anomaly_last_dist = float('inf')
anomaly_last_dir  = np.zeros(2)
# Entity position persists across frames within a run once detected.
# Avoidance doesn't drop out when robot momentarily faces away from entity.
entity_known_pos  = None

# ─── ENVIRONMENT RELEVANCE BELIEF (second analogy DST layer) ─────────────────
# Pl_env: how much does this environment warrant the analogy's caution?
#   High (≈1): many obstacles, entity detected → analogy is needed → trust it fully
#   Low  (≈0): clear empty space → analogy is overkill → relax toward M0 efficiency
# Updated per run from avg_plausibility_H (high Pl_H = clear environment = less relevant).
Pl_env          = 0.50   # starts neutral; experiment varies this as "room-fullness belief"
sum_pl_h_in_run = 0.0    # accumulates plausibility_H per frame → gives per-run avg
cnt_pl_h_in_run = 0      # frame counter for the above

# ─── ENVIRONMENT FAMILIARITY ──────────────────────────────────────────────────
# Counts consecutive clean runs (no spills, no collisions) within a config.
consecutive_clean_runs = 0

# Force every run in a config to have no obstacles (used by Fig 12 experiment only).
_force_empty_env = False

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
    obs_notH  = np.clip(density_factor * 0.4 + proximity_factor * 0.7, 0, 0.95)
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
    return round(float(np.clip(pl, 0.0, 1.0)), 3)  # continuous, not quantized

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

def extract_task_profile(text):
    """Score safety_attention and urgency_attention from free-text analogy input."""
    t = text.lower()
    safety  = sum(0.15 for kw in TASK_FEATURES["safety"]    if kw in t)
    safety += sum(0.12 for kw in TASK_FEATURES["liquid"]    if kw in t)
    safety += sum(0.10 for kw in TASK_FEATURES["fragile"]   if kw in t)
    safety += sum(0.08 for kw in TASK_FEATURES["vulnerable"] if kw in t)
    urgency = sum(0.20 for kw in TASK_FEATURES["urgent"]    if kw in t)
    return np.array([min(1.0, safety), min(1.0, urgency)])

def select_analogy(profile):
    """Return (name, policy, cosine_score) of the analogy most similar to profile."""
    best_name, best_score = "medical_care", -1.0
    for name, analogy in ANALOGY_LIBRARY.items():
        sig   = analogy["signature"]
        score = float(np.dot(profile, sig) / (np.linalg.norm(profile) * np.linalg.norm(sig) + 1e-9))
        if score > best_score:
            best_score, best_name = score, name
    return best_name, ANALOGY_LIBRARY[best_name]["policy"], best_score

def init_analogy_belief(init_Pl=0.80):
    """Reset analogy DST masses. init_Pl = starting Pl(A_correct), derived from cosine score."""
    global analogy_m_correct, analogy_m_wrong, analogy_m_open, Pl_analogy
    init_Pl = float(np.clip(init_Pl, 0.40, 0.95))
    analogy_m_correct = 0.0
    analogy_m_open    = init_Pl
    analogy_m_wrong   = 1.0 - init_Pl
    Pl_analogy        = float(analogy_m_correct + analogy_m_open)


def update_analogy_belief(spills, had_collision, anomaly_navigated_ok):
    """
    Dempster-Shafer update for Pl(analogy_correct) based on run outcome.

    Evidence sources:
      - Collision:            strong evidence the analogy failed to protect
      - Spill:                evidence analogy under- or over-calibrated for this env
      - Clean run:            mild evidence analogy is appropriate
      - Anomaly navigated ok: mild extra credit — analogy guided caution correctly
    """
    global analogy_m_correct, analogy_m_wrong, analogy_m_open, Pl_analogy

    if selected_analogy_name == "medical_care":
        # Promises: safe delivery (spills≈0, collisions=0)
        if had_collision:
            ev_wrong, ev_correct = 0.22, 0.0   # worst failure — conservative yet collided
        elif spills > 0:
            ev_wrong   = min(0.30, spills * 0.14)
            ev_correct = 0.0
        else:
            ev_wrong   = 0.0
            ev_correct = 0.13 if anomaly_navigated_ok else 0.08
    elif selected_analogy_name == "efficient_courier":
        # Promises: fast delivery — but task_profile sa=0.9 (liquid cargo) means any
        # spill is wrong: courier analogy is misapplied to a fragile-item task.
        if had_collision:
            ev_wrong, ev_correct = 0.22, 0.0
        elif spills >= 1:
            ev_wrong   = min(0.30, spills * 0.12)
            ev_correct = 0.0
        else:
            ev_wrong   = 0.0
            ev_correct = 0.08
    else:
        ev_wrong, ev_correct = 0.0, 0.0

    ev_open = max(0.0, 1.0 - ev_wrong - ev_correct)

    # Dempster-Shafer combination (unnormalised)
    K = analogy_m_correct * ev_wrong + analogy_m_wrong * ev_correct
    K = min(K, 0.99)
    denom = 1.0 - K
    new_c = (analogy_m_correct * ev_correct + analogy_m_correct * ev_open + analogy_m_open * ev_correct) / denom
    new_w = (analogy_m_wrong   * ev_wrong   + analogy_m_wrong   * ev_open + analogy_m_open * ev_wrong)   / denom
    new_o = (analogy_m_open    * ev_open) / denom

    total = new_c + new_w + new_o + 1e-12
    analogy_m_correct = new_c / total
    analogy_m_wrong   = new_w / total
    analogy_m_open    = new_o / total
    # Plausibility = everything not ruled out as wrong
    Pl_analogy = float(np.clip(analogy_m_correct + analogy_m_open, 0.0, 1.0))


def ray_circle_intersection(origin, direction, center, radius):
    """Ray-circle intersection; returns distance t > 0 or None."""
    oc  = origin - center
    b   = 2.0 * float(np.dot(oc, direction))
    c   = float(np.dot(oc, oc)) - radius ** 2
    disc = b * b - 4.0 * c
    if disc < 0:
        return None
    t = (-b - math.sqrt(disc)) / 2.0
    return float(t) if t > 0 else None

def reset_run():
    global robot_pos_true, robot_pos_est, frame_counter, mass_H, mass_notH, mass_Theta, plausibility_H
    global spill_events, laser_endpoints, particles, pose_variance
    global current_run_mode, current_env_is_empty, battery_level
    global run_spills, run_collisions, REAL_POLY, dynamic_threshold, mass_H_init
    global eff_speed, eff_inflation, eff_resolution
    global ROOM_SIZE, START, GOAL, OBSTACLE_GRID, GRID_X_EDGES, GRID_Y_EDGES, frames_since_spill
    global hits_in_path
    global novel_entity_active, novel_entity_pos, novel_entity_radius, novel_entity_type
    global selected_analogy_name, base_policy, task_profile
    global anomaly_detected_in_run, anomaly_last_hits, anomaly_last_dist, anomaly_last_dir
    global entity_known_pos, sum_pl_h_in_run, cnt_pl_h_in_run

    anomaly_detected_in_run = False
    anomaly_last_hits = 0
    anomaly_last_dist = float('inf')
    anomaly_last_dir  = np.zeros(2)
    entity_known_pos  = None   # reset each run — re-learn entity position
    sum_pl_h_in_run   = 0.0
    cnt_pl_h_in_run   = 0

    if historical_SE:
        avg_se = np.mean(historical_SE[-3:])
        if _sweep_threshold is None:
            dynamic_threshold = np.clip(0.5 + (70 - avg_se) * 0.005, 0.25, 0.75)
        else:
            dynamic_threshold = _sweep_threshold
        mass_H_init = np.clip(0.5 + (avg_se - 50) * 0.005, 0.2, 0.8)
    elif _sweep_threshold is not None:
        dynamic_threshold = _sweep_threshold

    active_indices = [i for i, val in enumerate(allowed_envs) if val]
    current_run_mode    = random.choice(active_indices) if active_indices else 0
    if _force_empty_env:
        current_env_is_empty = True
    else:
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
    # Task profile and analogy selection (re-derived each run from static input)
    task_profile = extract_task_profile(ANALOGY_INPUT)
    selected_analogy_name, base_policy, _ = select_analogy(task_profile)
    # Start from the base policy chosen by analogy (not always M0)
    eff_speed      = base_policy["speed"]
    eff_inflation  = base_policy["inflation"]
    eff_resolution = base_policy["resolution"]
    META_PARAMS["resolution"] = base_policy["resolution"]

    # Novel entity: spawns on path midpoint once NOVEL_ENTITY_SPAWN_RUN is reached
    if current_run >= NOVEL_ENTITY_SPAWN_RUN:
        novel_entity_active = True
        mid = (START + GOAL) / 2.0
        # Place slightly off-centre so it's not trivially avoidable
        novel_entity_pos    = mid + np.array([random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)])
        etype               = random.choice(list(NOVEL_ENTITY_TYPES.keys()))
        novel_entity_type   = etype
        novel_entity_radius = NOVEL_ENTITY_TYPES[etype]["radius"]
    else:
        novel_entity_active = False

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
    global novel_entity_active, novel_entity_pos, novel_entity_radius, novel_entity_type
    global selected_analogy_name, base_policy, task_profile, cold_start_data
    global Pl_analogy, anomaly_detected_in_run, analogy_history
    global anomaly_last_hits, anomaly_last_dist, anomaly_last_dir, entity_known_pos
    global consecutive_clean_runs, Pl_env, sum_pl_h_in_run, cnt_pl_h_in_run

    # 1. Sensing & DST
    if frame_counter % int(META_PARAMS["f_loc"]) == 0:
        laser_endpoints = []
        if np.linalg.norm(prev_v_final) > 0:
            current_heading = math.atan2(prev_v_final[1], prev_v_final[0])
        else:
            vg = GOAL - robot_pos_est
            current_heading = math.atan2(vg[1], vg[0])
        hits_in_path, min_dist, total_rays_in_cone = 0, META_PARAMS["laser_range"], 0
        anomaly_hits, anomaly_min_dist = 0, META_PARAMS["laser_range"]
        anomaly_closest_dir = np.zeros(2)
        for angle in np.linspace(0, 2*math.pi, 36, endpoint=False):
            d     = np.array([math.cos(angle), math.sin(angle)])
            t_obj = ray_poly_intersection(robot_pos_true, d, REAL_POLY) if not current_env_is_empty else None
            # Also check novel entity (M2 only — extra sensing capability)
            t_ent = ray_circle_intersection(robot_pos_true, d, novel_entity_pos, novel_entity_radius) \
                    if novel_entity_active else None
            adiff   = abs(math.atan2(math.sin(angle - current_heading), math.cos(current_heading - angle)))
            in_cone = adiff < math.radians(35)
            if in_cone: total_rays_in_cone += 1
            # Known polygon obstacle
            if t_obj and t_obj < META_PARAMS["laser_range"]:
                min_dist = min(min_dist, t_obj)
                if in_cone: hits_in_path += 1
                laser_endpoints.append((robot_pos_true + t_obj * d, "OBSTACLE"))
            else:
                laser_endpoints.append((robot_pos_true + META_PARAMS["laser_range"] * d, "WALL"))
            # Anomalous return: ray hits novel entity but NOT the known polygon
            if t_ent and t_ent < META_PARAMS["laser_range"]:
                if (t_obj is None) or (t_ent < t_obj):   # entity closer than polygon
                    anomaly_hits += 1
                    if t_ent < anomaly_min_dist:           # track direction of closest hit
                        anomaly_min_dist    = t_ent
                        anomaly_closest_dir = d.copy()
                    min_dist = min(min_dist, t_ent)        # also feed DST proximity
        plausibility_H   = update_belief_dst(hits_in_path, total_rays_in_cone, min_dist)
        # Accumulate Pl_H for per-run avg → feeds Pl_env update at end of run
        sum_pl_h_in_run += plausibility_H
        cnt_pl_h_in_run += 1
        # Persist anomaly state for path planner (used every frame, not just sensing frames)
        anomaly_last_hits = anomaly_hits
        anomaly_last_dist = anomaly_min_dist if anomaly_hits > 0 else float('inf')
        anomaly_last_dir  = anomaly_closest_dir.copy()

        # ── Analogy re-selection triggered by anomaly ─────────────────────────
        # When the entity is unmodelled, anomaly_hits > 0 signals scope expansion.
        # Boost safety_attention proportional to proximity and re-select analogy.
        if novel_entity_active and anomaly_hits > 0:
            anomaly_detected_in_run = True
            proximity_signal = max(0.0, 1.0 - anomaly_min_dist / META_PARAMS["laser_range"])
            boosted_profile  = task_profile.copy()
            boosted_profile[0] = min(1.0, task_profile[0] + proximity_signal * 0.4)
            selected_analogy_name, base_policy, _ = select_analogy(boosted_profile)

    # 2. Fuzzy switching & smoothing — analogy IS the baseline, DST adjusts it.
    #
    # Architecture:
    #   (a) Analogy provides expert-calibrated parameters (speed, inflation, resolution).
    #       When analogy is correct and environment matches, these beat M0 and M1 directly.
    #   (b) Pl_analogy (DST analogy belief) blends toward POLICY_M1 as evidence
    #       suggests the analogy may be wrong.  Pl=1 → full analogy; Pl=0 → M1 safe.
    #       Falling back to M1 (not a midpoint) means M2 is conservative when uncertain —
    #       correct meta-reasoning: "if I don't trust my analogy, revert to proven-safe."
    #   (c) Real-time hazard from path DST (Pl_H) + entity proximity continuously
    #       modulates speed: fast when clear, slows toward M1 level near obstacles/entity.
    #   (d) Hard override: if Pl_H drops below threshold → full M1 conservatism.
    #
    #  medical_care, Pl_a=0.90 (domestic): speed≈0.56, inflate≈1.27, res≈0.04 — fast+precise
    #  medical_care, Pl_a=0.64 (warehouse): speed≈0.52, inflate≈1.44, res≈0.05 — conservative
    #  efficient_courier, Pl_a=0.90:        speed≈0.61, inflate≈0.94, res≈0.09 — courier-tuned

    # (a) Two-layer trust envelope:
    #   Pl_env:     does this environment warrant the analogy's caution?
    #               low → clear/empty room → relax toward M0 (efficient, no penalty)
    #               high → obstacle-rich → analogy caution is warranted → trust it fully
    #   Pl_analogy: is the analogy correct for this task?
    #               low → analogy failing → revert to M1 (proven-safe fallback)
    #
    # Extreme cases:
    #   Pl_analogy=1, Pl_env=1 → full analogy params        (ideal case)
    #   Pl_analogy=1, Pl_env=0 → M0 params (clear room, relax) (empty-room efficiency)
    #   Pl_analogy=0, Pl_env=* → M1 params (analogy failing → safe regardless of env)
    analogy_speed   = base_policy["speed"]
    analogy_inflate = base_policy["inflation"]
    analogy_res     = base_policy["resolution"]
    # Layer 1: env-relevance — blend analogy ↔ M0 based on how much env warrants caution
    env_speed   = Pl_env * analogy_speed   + (1.0 - Pl_env) * POLICY_M0["speed"]
    env_inflate = Pl_env * analogy_inflate + (1.0 - Pl_env) * POLICY_M0["inflation"]
    env_res     = Pl_env * analogy_res     + (1.0 - Pl_env) * POLICY_M0["resolution"]
    # Layer 2: analogy trust — blend env-result ↔ M1 based on analogy correctness
    envelope_speed   = Pl_analogy * env_speed   + (1.0 - Pl_analogy) * POLICY_M1["speed"]
    envelope_inflate = Pl_analogy * env_inflate + (1.0 - Pl_analogy) * POLICY_M1["inflation"]
    envelope_res     = Pl_analogy * env_res     + (1.0 - Pl_analogy) * POLICY_M1["resolution"]

    # (b) Familiarity relaxes resolution only (finer → coarser as env proves safe)
    familiarity  = min(1.0, consecutive_clean_runs / 8.0)
    envelope_res = envelope_res + familiarity * (POLICY_M0["resolution"] - envelope_res) * 0.35

    # (c) Real-time hazard: entity proximity + path uncertainty → reduce speed
    entity_prox = 0.0
    if anomaly_last_hits > 0:
        entity_prox = float(np.clip(1.0 - anomaly_last_dist / META_PARAMS["laser_range"], 0.0, 1.0))
    path_hazard = float(np.clip(1.0 - plausibility_H, 0.0, 1.0))
    hazard      = max(entity_prox, path_hazard * 0.7)  # entity is a sharper signal

    if plausibility_H <= dynamic_threshold:
        # (d) DST hard override: path blocked → full M1 conservatism
        target_v   = POLICY_M1["speed"]
        target_inf = POLICY_M1["inflation"]
        target_res = POLICY_M1["resolution"]
        active_model_label = f"M1-SAFE [{selected_analogy_name}]"
    else:
        # Clear path: analogy-guided parameters, speed continuously modulated by hazard.
        # Speed:     analogy ceiling → M1 floor as hazard rises (reactive speed control)
        # Inflation: set by Pl_analogy blend only — self-regulates across environments:
        #   domestic Pl≈0.90 → inflate≈1.27  (efficient paths, good margins)
        #   warehouse Pl≈0.50 → inflate≈1.53  (wider berths in dense environments)
        #   entity encounter Pl drops → inflate rises toward M1 (maximum safety)
        #
        # Example clear domestic (hazard≈0.10, Pl≈0.90): v=0.54, inf=1.27
        #         tight warehouse (hazard≈0.40, Pl≈0.64): v=0.47, inf=1.44
        #         near entity    (hazard≈0.80, Pl≈0.50): v=0.43, inf=1.53
        target_v   = max(POLICY_M1["speed"],
                         envelope_speed - hazard * (envelope_speed - POLICY_M1["speed"]))
        target_inf = envelope_inflate
        target_res = envelope_res
        active_model_label = f"M0-FAST [{selected_analogy_name}]"
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
    avoid_vec  = np.zeros(2)
    dist_obs   = float('inf')
    should_slow = False
    if not current_env_is_empty:
        closest_p, dist_obs = get_closest_grid_obstacle(robot_pos_est)
        if dist_obs < eff_inflation:
            radial    = ((robot_pos_est - closest_p) / dist_obs) * 6.5
            avoid_vec = radial + np.array([radial[1], -radial[0]]) * 5.2
            should_slow = True
    # Novel entity avoidance — M2 steers around what M0/M1 cannot see.
    # entity_known_pos persists once detected; avoidance doesn't drop when robot faces away.
    if anomaly_last_hits > 0 and np.linalg.norm(anomaly_last_dir) > 0.5:
        entity_known_pos = robot_pos_true + anomaly_last_dist * anomaly_last_dir
    if entity_known_pos is not None:
        dist_entity = max(0.05, np.linalg.norm(robot_pos_est - entity_known_pos))
        entity_zone = eff_inflation * 2.5
        if dist_entity < entity_zone:
            radial_e   = (robot_pos_est - entity_known_pos) / dist_entity
            prox_scale = min(3.0, entity_zone / dist_entity)
            avoid_vec += (radial_e * 9.0 + np.array([radial_e[1], -radial_e[0]]) * 7.0) * prox_scale
            should_slow = True
            if dist_entity < eff_inflation * 0.8:               # emergency stop — too close
                drive_vec *= 0.0
    if should_slow:
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
    # Novel entity physical collision (M2 can sense it and steer away — still possible at close range)
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

    # 6. Telemetry
    all_iterations_data.append({
        "Run": current_run, "Frame": frame_counter,
        "Mode": "Domestic" if current_run_mode == 0 else "Warehouse",
        "Active_Model": active_model_label,
        "Analogy": selected_analogy_name,
        "Novel_Entity_Active": int(novel_entity_active),
        "Novel_Entity_Type": novel_entity_type if novel_entity_active else "none",
        "Fuzzy_Threshold": round(dynamic_threshold, 2), "Pl_H": plausibility_H,
        "Pl_Analogy": round(Pl_analogy, 3),
        "Pl_Env": round(Pl_env, 3),
        "Familiarity": round(min(1.0, consecutive_clean_runs / 8.0), 3),
        "Consec_Clean": consecutive_clean_runs,
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
        time_factor = max(0.50, 1.15 - 0.20 * time_ratio)
        base_score = 100 - (run_spills * 25) - (run_collisions * 100)
        if run_spills == 0 and run_collisions == 0: base_score = 100
        final_se = max(0, base_score * time_factor * (battery_level / 100.0))
        historical_SE.append(final_se)
        had_collision = run_collisions > 0
        anomaly_ok    = anomaly_detected_in_run and not had_collision
        # Familiarity: reward consecutive clean runs, penalise failures
        if run_spills == 0 and not had_collision:
            consecutive_clean_runs += 1
        else:
            consecutive_clean_runs = max(0, consecutive_clean_runs - 2)
        update_analogy_belief(run_spills, had_collision, anomaly_ok)
        # Update Pl_env from avg Pl_H this run:
        #   high avg Pl_H → path always clear → environment doesn't need the analogy's caution
        #   low  avg Pl_H → many obstacles → environment IS warranting caution → trust analogy
        if cnt_pl_h_in_run > 0:
            avg_pl_h = sum_pl_h_in_run / cnt_pl_h_in_run
            if avg_pl_h > 0.82:
                Pl_env = max(0.05, Pl_env - 0.08)   # clear env → relax toward M0
            elif avg_pl_h < 0.60:
                Pl_env = min(0.95, Pl_env + 0.08)   # obstacle-rich → trust analogy more
        cold_start_data.append({
            "run": current_run, "se": final_se,
            "entity_active": int(novel_entity_active),
            "entity_type": novel_entity_type if novel_entity_active else "none",
            "analogy": selected_analogy_name,
            "spills": run_spills, "collisions": run_collisions,
            "Pl_analogy": round(Pl_analogy, 3),
            "Pl_env": round(Pl_env, 3),
        })
        analogy_history.append({
            "run":          current_run,
            "se":           final_se,
            "Pl_analogy":   round(Pl_analogy, 3),
            "Pl_env":       round(Pl_env, 3),
            "analogy":      selected_analogy_name,
            "spills":       run_spills,
            "collision":    int(had_collision),
            "entity_active": int(novel_entity_active),
        })
        print(f"    run {current_run} done | SE={final_se:.1f} | spills={run_spills} | "
              f"Pl_a={Pl_analogy:.2f} | Pl_env={Pl_env:.2f} | analogy={selected_analogy_name} | "
              f"entity={'ON' if novel_entity_active else 'off'}")
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
    label_map = {0: 'No Spill', 1: 'Spill'}
    spill_stats.index = [label_map.get(i, str(i)) for i in spill_stats.index]

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
    global novel_entity_active, cold_start_data
    global selected_analogy_name, base_policy, task_profile
    global analogy_history, anomaly_detected_in_run
    global anomaly_last_hits, anomaly_last_dist, anomaly_last_dir, entity_known_pos
    global consecutive_clean_runs, Pl_env, sum_pl_h_in_run, cnt_pl_h_in_run

    historical_SE     = []
    dynamic_threshold = 0.5
    mass_H_init       = 0.5
    mass_H, mass_notH, mass_Theta, plausibility_H = 0.5, 0.0, 0.5, 1.0
    # Derive initial analogy from task input; use cosine score to seed belief
    task_profile = extract_task_profile(ANALOGY_INPUT)
    selected_analogy_name, base_policy, cosine_score = select_analogy(task_profile)
    init_analogy_belief(max(0.60, cosine_score * 0.90))
    eff_speed          = base_policy["speed"]
    eff_inflation      = base_policy["inflation"]
    eff_resolution     = base_policy["resolution"]
    META_PARAMS["resolution"] = base_policy["resolution"]
    allowed_envs[:]    = cfg_envs
    current_run        = 1
    all_iterations_data     = []
    cold_start_data         = []
    analogy_history         = []
    anomaly_detected_in_run = False
    anomaly_last_hits       = 0
    anomaly_last_dist       = float('inf')
    anomaly_last_dir        = np.zeros(2)
    entity_known_pos        = None
    consecutive_clean_runs  = 0
    Pl_env                  = 0.50   # reset to neutral each config
    sum_pl_h_in_run         = 0.0
    cnt_pl_h_in_run         = 0
    novel_entity_active     = False
    is_playing         = True
    prev_v_final       = np.zeros(2)
    frames_since_spill = SPILL_COOLDOWN_FRAMES
    hits_in_path       = 0

# ─── THRESHOLD SWEEP ─────────────────────────────────────────────────────────

def _run_config(cfg_envs, n_runs, max_frames):
    """Run n_runs and return list of (se, spills, frames) tuples."""
    global _sweep_threshold, total_runs_to_execute
    orig_n = total_runs_to_execute
    total_runs_to_execute = n_runs
    init_config(cfg_envs)
    reset_run()
    results, total_f = [], 0
    while is_playing and total_f < max_frames:
        prev_run = current_run
        step_logic()
        total_f += 1
        if (current_run != prev_run or not is_playing) and historical_SE:
            df_tmp = pd.DataFrame(all_iterations_data)
            last_run_df = df_tmp[df_tmp['Run'] == (current_run - 1 if is_playing else current_run)]
            results.append((
                historical_SE[-1],
                int(last_run_df['Spill'].sum()),
                int(last_run_df['Frame'].max()),
            ))
        if len(results) >= n_runs:
            break
    total_runs_to_execute = orig_n
    return results


def _run_n_comparison(cfg_envs, n_runs, fixed_thresh=None):
    """Run n_runs, return (se_list, threshold_per_run). Used by comparison plot."""
    global _sweep_threshold, total_runs_to_execute
    orig_n = total_runs_to_execute
    total_runs_to_execute = n_runs
    _sweep_threshold = fixed_thresh
    init_config(cfg_envs)
    reset_run()
    total_f = 0
    while is_playing and total_f < n_runs * 4000:
        step_logic()
        total_f += 1
    se_list = list(historical_SE[:n_runs])
    thresh_list = [fixed_thresh if fixed_thresh is not None else 0.5] * len(se_list)
    if all_iterations_data and fixed_thresh is None:
        df_tmp = pd.DataFrame(all_iterations_data)
        if 'Fuzzy_Threshold' in df_tmp.columns:
            thresh_list = df_tmp.groupby('Run')['Fuzzy_Threshold'].first().tolist()
    _sweep_threshold = None
    total_runs_to_execute = orig_n
    return se_list, thresh_list


def compare_adaptive_vs_fixed(n_runs=20, out_dir="plots"):
    """
    Compare M2 with adaptive historical threshold vs two fixed thresholds.
    Plots SE per run + threshold evolution for warehouse and mix configs.
    Saves plots/fig7_adaptive_vs_fixed.png.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    configs_to_test = [([False, True], "warehouse"), ([True, True], "mix")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Adaptive Historical Threshold vs Fixed Threshold — SE Evolution over Runs\n"
        "(thick line = 5-run rolling mean)",
        fontsize=12, fontweight='bold')

    for col, (cfg_envs, cfg_name) in enumerate(configs_to_test):
        print(f"  Comparison [{cfg_name}]: adaptive...", end=" ", flush=True)
        se_adapt,   thresh_adapt = _run_n_comparison(cfg_envs, n_runs, fixed_thresh=None)
        print("fixed 0.50...", end=" ", flush=True)
        se_f50,     _            = _run_n_comparison(cfg_envs, n_runs, fixed_thresh=0.50)
        print("fixed 0.65...")
        se_f65,     _            = _run_n_comparison(cfg_envs, n_runs, fixed_thresh=0.65)

        runs = np.arange(1, n_runs + 1)
        color_a, color_50, color_65 = "#27AE60", "#E07B39", "#4C7EBE"

        # ── Row 0: SE per run ────────────────────────────────────────────────
        ax = axes[0, col]
        for se, color, label in [
            (se_adapt, color_a,   "Adaptive (historical SE)"),
            (se_f50,   color_50,  "Fixed threshold = 0.50"),
            (se_f65,   color_65,  "Fixed threshold = 0.65"),
        ]:
            r = runs[:len(se)]
            ax.plot(r, se, 'o-', color=color, lw=1.2, ms=3.5, alpha=0.45, label=label)
            if len(se) >= 5:
                roll = pd.Series(se).rolling(5, min_periods=1).mean().values
                ax.plot(r, roll, color=color, lw=2.8, alpha=0.9)

        ax.set_title(f"{cfg_name.capitalize()}: SE per Run", fontweight='bold')
        ax.set_xlabel("Run #"); ax.set_ylabel("SE Score (safety-first)")
        ax.set_ylim(0, 105); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

        # ── Row 1: threshold evolution ───────────────────────────────────────
        ax = axes[1, col]
        ax.plot(runs[:len(thresh_adapt)], thresh_adapt,
                'o-', color='navy', lw=2, ms=5, label='Adaptive threshold')
        ax.axhline(0.50, color=color_50, lw=1.8, linestyle='--', label='Fixed=0.50')
        ax.axhline(0.65, color=color_65, lw=1.8, linestyle='--', label='Fixed=0.65')
        ax.set_title(f"{cfg_name.capitalize()}: Threshold Evolution", fontweight='bold')
        ax.set_xlabel("Run #"); ax.set_ylabel("Threshold value")
        ax.set_ylim(0.20, 0.80); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = f"{out_dir}/fig7_adaptive_vs_fixed.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {out}")


def sweep_threshold_plot(n_runs=10, out_dir="plots"):
    """
    Sweep dynamic_threshold over warehouse config and plot spill rate + SE.
    Saves plots/fig6_threshold_sweep.png.
    """
    global _sweep_threshold
    import os
    os.makedirs(out_dir, exist_ok=True)

    thresholds = np.round(np.arange(0.10, 0.95, 0.10), 2)
    warehouse_envs = [False, True]

    avg_spills, avg_se, avg_frames = [], [], []

    print(f"\n{'='*55}")
    print(f"  THRESHOLD SWEEP (warehouse, {n_runs} runs each)")
    print('='*55)

    for thresh in thresholds:
        _sweep_threshold = float(thresh)
        runs = _run_config(warehouse_envs, n_runs, max_frames=n_runs * 3000)
        if not runs:
            avg_spills.append(np.nan); avg_se.append(np.nan); avg_frames.append(np.nan)
            continue
        s_arr = [r[1] for r in runs]
        e_arr = [r[0] for r in runs]
        f_arr = [r[2] for r in runs]
        avg_spills.append(np.mean(s_arr))
        avg_se.append(np.mean(e_arr))
        avg_frames.append(np.mean(f_arr))
        print(f"  threshold={thresh:.2f} | spills={np.mean(s_arr):.2f} | SE={np.mean(e_arr):.1f} | frames={np.mean(f_arr):.0f}")

    _sweep_threshold = None  # restore adaptive mode

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    color_spill = "#C0392B"
    color_se    = "#27AE60"

    l1, = ax1.plot(thresholds, avg_spills, 'o-', color=color_spill, lw=2.2,
                   markersize=7, label="Avg spills / run")
    ax1.fill_between(thresholds, avg_spills, alpha=0.12, color=color_spill)
    ax1.set_xlabel("DST Switching Threshold  (lower → switch to M1 sooner)",
                   fontsize=11)
    ax1.set_ylabel("Avg spills per run", color=color_spill, fontsize=11)
    ax1.tick_params(axis='y', labelcolor=color_spill)
    ax1.set_ylim(bottom=0)

    l2, = ax2.plot(thresholds, avg_se, 's--', color=color_se, lw=2.2,
                   markersize=7, label="Avg SE score")
    ax2.set_ylabel("Avg SE score", color=color_se, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color_se)
    ax2.set_ylim(0, 100)

    # Mark the sweet spot: threshold that maximises SE
    valid = [(t, s, e) for t, s, e in zip(thresholds, avg_spills, avg_se)
             if not np.isnan(e)]
    if valid:
        best_t, _, _ = max(valid, key=lambda x: x[2])
        ax1.axvline(best_t, color='navy', lw=1.5, linestyle=':', alpha=0.7)
        ax1.text(best_t + 0.01, ax1.get_ylim()[1] * 0.92,
                 f"Best SE\nthreshold={best_t:.2f}", fontsize=8.5,
                 color='navy', va='top')

    ax1.axvspan(0.10, 0.35, alpha=0.06, color='steelblue',
                label='Too cautious (slow)')
    ax1.axvspan(0.70, 0.90, alpha=0.06, color='tomato',
                label='Too reactive (risky)')

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=9)

    ax1.set_title("M2-DST-Hybrid: Effect of Switching Threshold on Safety & Performance\n"
                  "(Warehouse config, obstacles present 90% of runs)",
                  fontsize=11, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    plt.tight_layout()
    out = f"{out_dir}/fig6_threshold_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {out}")


def adaptation_figures(out_dir="plots"):
    """
    Generate fig8, fig9, fig10 from cold_start_data collected during the main runs.
    Must be called AFTER all 3 configs have been run.
    cold_start_data is accumulated across configs in a module-level list passed in.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

def run_all_configs_and_figures(out_dir="plots"):
    """Run 3 configs, collect cold_start across all, then emit fig8/9/10."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    all_cold = []   # aggregated across configs

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
                print(f"  [{cfg_name}] run {current_run}/{total_runs_to_execute} | "
                      f"frame {frame_counter} | env={mode} | bat={battery_level:.1f}%")
        if is_playing:
            print("  ⚠️  Safety cap hit — saving partial data")
        save_to_excel(cfg_name)
        for d in cold_start_data:
            d["config"] = cfg_name
        all_cold.extend(cold_start_data)

    _plot_fig8_adaptation(all_cold, out_dir)
    _plot_fig9_scope_collapse(all_cold, out_dir)
    _plot_fig10_cold_start_gap(all_cold, out_dir)


def _plot_fig8_adaptation(cold_data, out_dir):
    """SE evolution before/after novel entity appears (adaptation curve)."""
    if not cold_data:
        return
    df = pd.DataFrame(cold_data)
    spawn = NOVEL_ENTITY_SPAWN_RUN

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(
        "Fig 8 — Adaptation Curve: SE before/after Novel Entity Appears\n"
        f"(entity spawns at run {spawn}; M2 detects by analogy, M0/M1 are blind)",
        fontsize=11, fontweight='bold')

    for ax, cfg in zip(axes, ["domestic", "mix", "warehouse"]):
        sub = df[df["config"] == cfg].sort_values("run")
        if sub.empty:
            ax.set_title(cfg); continue
        runs = sub["run"].values
        ses  = sub["se"].values
        entity_on = sub["entity_active"].values.astype(bool)

        ax.plot(runs, ses, 'o-', color='#4C7EBE', lw=1.5, ms=4, alpha=0.6, label="SE per run")
        if len(ses) >= 3:
            roll = pd.Series(ses).rolling(3, min_periods=1).mean().values
            ax.plot(runs, roll, color='#4C7EBE', lw=2.5, alpha=0.9, label="3-run mean")

        # Shade entity-active region
        if entity_on.any():
            first_on = runs[entity_on][0]
            ax.axvspan(first_on - 0.5, runs[-1] + 0.5, alpha=0.10, color='tomato',
                       label="Novel entity active")
            ax.axvline(first_on - 0.5, color='tomato', lw=1.5, linestyle='--')
            ax.text(first_on, ax.get_ylim()[1] if ax.get_ylim()[1] > 10 else 95,
                    "entity\nappears", fontsize=7.5, color='tomato', va='top')

        ax.set_title(cfg.capitalize(), fontweight='bold')
        ax.set_xlabel("Run #"); ax.set_ylabel("SE Score")
        ax.set_ylim(0, 105); ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = f"{out_dir}/fig8_adaptation_curve.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {out}")


def _plot_fig9_scope_collapse(cold_data, out_dir):
    """Graceful Degradation Ratio: SE_entity / SE_pre for M2 (should stay high)."""
    if not cold_data:
        return
    df    = pd.DataFrame(cold_data)
    spawn = NOVEL_ENTITY_SPAWN_RUN
    rows  = []
    for cfg in ["domestic", "mix", "warehouse"]:
        sub  = df[df["config"] == cfg].sort_values("run")
        pre  = sub[sub["run"] < spawn]["se"].mean()
        post = sub[sub["entity_active"] == 1]["se"].mean()
        if pre > 0:
            rows.append({"config": cfg, "SE_pre": pre, "SE_post": post,
                         "GDR": post / pre if not np.isnan(post) else 0})

    if not rows:
        return
    gdr_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(gdr_df))
    bars = ax.bar(x, gdr_df["GDR"], color=["#27AE60","#E07B39","#4C7EBE"],
                  width=0.5, alpha=0.85, edgecolor='white')
    ax.axhline(1.0, color='gray', lw=1.5, linestyle='--', label="No degradation (GDR=1)")
    ax.axhline(0.5, color='tomato', lw=1.2, linestyle=':', alpha=0.7, label="Scope collapse threshold")
    for bar, row in zip(bars, gdr_df.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{row.GDR:.2f}", ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([r["config"].capitalize() for r in rows], fontsize=11)
    ax.set_ylabel("Graceful Degradation Ratio  (SE_entity / SE_pre)", fontsize=10)
    ax.set_ylim(0, 1.35)
    ax.set_title("Fig 9 — Scope Collapse Test: M2 Performance Ratio on Novel Entity\n"
                 "(GDR=1 → no degradation; GDR<0.5 → scope collapse)", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    out = f"{out_dir}/fig9_scope_collapse.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {out}")


def _plot_fig10_cold_start_gap(cold_data, out_dir):
    """Cold-start gap: difference in SE between first encounter and pre-entity baseline."""
    if not cold_data:
        return
    df    = pd.DataFrame(cold_data)
    spawn = NOVEL_ENTITY_SPAWN_RUN

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Fig 10 — Cold-Start Gap: How Much SE Drops on First Novel Entity Encounter\n"
        "and Recovery Trajectory (M2 with analogy vs M0/M1 which are blind)",
        fontsize=11, fontweight='bold')

    for cfg, ax in zip(["warehouse", "mix"], axes):
        sub  = df[df["config"] == cfg].sort_values("run")
        pre  = sub[sub["run"] < spawn]["se"].mean()
        post = sub[sub["entity_active"] == 1].sort_values("run")

        ax.axhline(pre, color='#4C7EBE', lw=2, linestyle='--', label=f"Pre-entity baseline ({pre:.1f})")
        if not post.empty:
            ax.plot(post["run"].values, post["se"].values,
                    'o-', color='#E07B39', lw=2, ms=5, label="SE with entity (M2)")
            if len(post) >= 3:
                roll = pd.Series(post["se"].values).rolling(3, min_periods=1).mean().values
                ax.plot(post["run"].values, roll, color='#E07B39', lw=3, alpha=0.5)
            cold_gap = pre - post["se"].iloc[0]
            ax.annotate(f"Cold-start\ngap = {cold_gap:.1f}",
                        xy=(post["run"].iloc[0], post["se"].iloc[0]),
                        xytext=(post["run"].iloc[0] + 0.8, post["se"].iloc[0] + 8),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                        fontsize=8.5, color='black')
        ax.set_title(f"{cfg.capitalize()}", fontweight='bold')
        ax.set_xlabel("Run #"); ax.set_ylabel("SE Score")
        ax.set_ylim(0, 105); ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = f"{out_dir}/fig10_cold_start_gap.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {out}")


WRONG_ANALOGY_INPUT   = "Urgent fast emergency rush, hurry hurry."   # → efficient_courier
CORRECT_ANALOGY_INPUT = ANALOGY_INPUT                                  # → medical_care


def _run_analogy_belief_test(analogy_input_override, cfg_envs, n_runs, scenario_label):
    """
    Run n_runs with analogy_input_override active, collect per-run analogy_history,
    tag each entry with scenario_label, and return the list.
    """
    global ANALOGY_INPUT, total_runs_to_execute, analogy_history
    orig_input = ANALOGY_INPUT
    orig_n     = total_runs_to_execute
    ANALOGY_INPUT            = analogy_input_override
    total_runs_to_execute    = n_runs
    try:
        init_config(cfg_envs)
        reset_run()
        total_f = 0
        while is_playing and total_f < n_runs * 5000:
            step_logic()
            total_f += 1
        return [dict(d, scenario=scenario_label) for d in analogy_history]
    finally:
        ANALOGY_INPUT         = orig_input
        total_runs_to_execute = orig_n


def _plot_fig11_analogy_belief(correct_data, wrong_data, out_dir):
    """
    Fig 11 — Analogy Belief Self-Correction.

    Panel A: Pl(analogy_correct) over runs for correct vs wrong analogy scenario.
    Panel B: SE per run for each scenario, showing partial self-correction.
    Vertical dashed line marks NOVEL_ENTITY_SPAWN_RUN.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    c_df = pd.DataFrame(correct_data)
    w_df = pd.DataFrame(wrong_data)

    color_correct = "#27AE60"
    color_wrong   = "#C0392B"
    spawn         = NOVEL_ENTITY_SPAWN_RUN

    fig, (ax_belief, ax_se) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    fig.suptitle(
        "Fig 11 — Analogy Belief Self-Correction (Second DST Layer)\n"
        "Correct analogy → belief stays high.  Wrong analogy → belief drops, robot self-corrects toward neutral.",
        fontsize=11, fontweight='bold')

    # ── Panel A: Pl(analogy_correct) ─────────────────────────────────────────
    for df, color, label in [
        (c_df, color_correct, f"Correct analogy ({c_df['analogy'].iloc[0] if not c_df.empty else '?'})"),
        (w_df, color_wrong,   f"Wrong analogy   ({w_df['analogy'].iloc[0] if not w_df.empty else '?'})"),
    ]:
        if df.empty:
            continue
        runs = df["run"].values
        pls  = df["Pl_analogy"].values
        ax_belief.plot(runs, pls, 'o-', color=color, lw=1.5, ms=4.5, alpha=0.55, label=label)
        if len(pls) >= 3:
            roll = pd.Series(pls).rolling(3, min_periods=1).mean().values
            ax_belief.plot(runs, roll, color=color, lw=2.8, alpha=0.92)

    ax_belief.axhline(0.5, color='gray', lw=1.2, linestyle=':', alpha=0.7,
                      label="Neutral threshold (Pl=0.5)")
    ax_belief.axvline(spawn - 0.5, color='tomato', lw=1.5, linestyle='--', alpha=0.6)
    ax_belief.text(spawn, 0.97, "entity\nappears", fontsize=7.5, color='tomato',
                   va='top', transform=ax_belief.get_xaxis_transform())
    ax_belief.axvspan(spawn - 0.5, max(
        (c_df["run"].max() if not c_df.empty else spawn),
        (w_df["run"].max() if not w_df.empty else spawn)
    ) + 0.5, alpha=0.06, color='tomato')
    ax_belief.set_ylabel("Pl(analogy_correct)", fontsize=10)
    ax_belief.set_ylim(0.0, 1.05)
    ax_belief.legend(fontsize=9); ax_belief.grid(alpha=0.3)
    ax_belief.spines[['top', 'right']].set_visible(False)

    # ── Panel B: SE per run ───────────────────────────────────────────────────
    for df, color, label in [
        (c_df, color_correct, "SE — correct analogy"),
        (w_df, color_wrong,   "SE — wrong analogy (self-correcting)"),
    ]:
        if df.empty:
            continue
        runs = df["run"].values
        ses  = df["se"].values
        ax_se.scatter(runs, ses, color=color, s=22, alpha=0.45, zorder=3)
        if len(ses) >= 3:
            roll = pd.Series(ses).rolling(3, min_periods=1).mean().values
            ax_se.plot(runs, roll, color=color, lw=2.8, alpha=0.9, label=label)

    ax_se.axvline(spawn - 0.5, color='tomato', lw=1.5, linestyle='--', alpha=0.6)
    ax_se.axvspan(spawn - 0.5, max(
        (c_df["run"].max() if not c_df.empty else spawn),
        (w_df["run"].max() if not w_df.empty else spawn)
    ) + 0.5, alpha=0.06, color='tomato')
    ax_se.set_xlabel("Run #", fontsize=10)
    ax_se.set_ylabel("SE Score (safety-first)", fontsize=10)
    ax_se.set_ylim(0, 105)
    ax_se.legend(fontsize=9); ax_se.grid(alpha=0.3)
    ax_se.spines[['top', 'right']].set_visible(False)

    # Annotate recovery area on wrong-analogy SE if visible
    if not w_df.empty and len(w_df) > 5:
        early_wrong = w_df[w_df["run"] <= w_df["run"].min() + 4]["se"].mean()
        late_wrong  = w_df[w_df["run"] >= w_df["run"].max() - 4]["se"].mean()
        if not np.isnan(early_wrong) and not np.isnan(late_wrong):
            recovery = late_wrong - early_wrong
            sign = "+" if recovery >= 0 else ""
            ax_se.text(0.72, 0.10,
                       f"Self-correction:\n{sign}{recovery:.1f} SE pts (late vs early)",
                       transform=ax_se.transAxes, fontsize=8.5,
                       color=color_wrong, va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    plt.tight_layout()
    out = f"{out_dir}/fig11_analogy_belief.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {out}")


def _run_env_belief_test(cfg_envs, init_pl_env, n_runs, force_empty, scenario_label):
    """
    Run n_runs with a given initial Pl_env, return per-run analogy_history list.
    force_empty=True forces every run to have no obstacles (tests clear-room adaptation).
    """
    global Pl_env, _force_empty_env, total_runs_to_execute
    orig_n = total_runs_to_execute
    total_runs_to_execute = n_runs
    _force_empty_env = force_empty
    init_config(cfg_envs)
    Pl_env = float(init_pl_env)   # override the 0.50 default set by init_config
    reset_run()
    total_f = 0
    while is_playing and total_f < n_runs * 5000:
        step_logic()
        total_f += 1
    result = [dict(d, scenario=scenario_label, init_pl_env=init_pl_env)
              for d in analogy_history]
    _force_empty_env = False
    total_runs_to_execute = orig_n
    return result


def run_fig12_env_belief_experiment(out_dir="plots"):
    """
    Fig 12: Test 3 initial Pl_env beliefs (0.20, 0.50, 0.80) in 2 environments
    (forced-empty warehouse, obstacle-rich warehouse).  Shows Pl_env evolving and SE
    trajectory — the meta-reasoner learns the environment's relevance from scratch.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    print("\n→ Fig 12: Pl_env initial belief experiment")
    init_beliefs = [0.20, 0.50, 0.80]

    results = {}
    for env_label, force_empty in [("empty", True), ("obstacles", False)]:
        results[env_label] = {}
        cfg = [False, True]   # warehouse base (90% obstacle for "obstacles", forced-empty for "empty")
        for ip in init_beliefs:
            label = f"{env_label}_pl{int(ip * 100)}"
            print(f"  [{env_label}] init_Pl_env={ip:.2f}...", end=" ", flush=True)
            data = _run_env_belief_test(cfg, init_pl_env=ip, n_runs=20,
                                        force_empty=force_empty, scenario_label=label)
            results[env_label][ip] = data
            avg_se = np.mean([d["se"] for d in data]) if data else float('nan')
            print(f"avg SE={avg_se:.1f}")

    _plot_fig12_env_belief(results, out_dir)


def _plot_fig12_env_belief(results, out_dir):
    """Plot Fig 12: Pl_env evolution + SE trajectory for 3 initial beliefs × 2 environments."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    colors = {0.20: "#4C7EBE", 0.50: "#E07B39", 0.80: "#27AE60"}
    env_titles = {
        "empty":     "Empty Room (no obstacles) — analogy should relax",
        "obstacles": "Obstacle-Rich Warehouse — analogy caution warranted",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Fig 12 — Environment Relevance Belief (Pl_env): Effect of Initial Prior\n"
        "Pl_env tracks whether the environment warrants the analogy's caution.\n"
        "Clear room → Pl_env drops → M2 relaxes toward M0.   Obstacles → Pl_env rises → M2 trusts analogy.",
        fontsize=11, fontweight='bold')

    for col, env_label in enumerate(["empty", "obstacles"]):
        ax_env = axes[0, col]
        ax_se  = axes[1, col]
        ax_env.set_title(env_titles[env_label], fontweight='bold', fontsize=9)
        ax_se.set_title(f"{env_label.capitalize()}: SE Trajectory", fontweight='bold', fontsize=9)

        for ip in [0.20, 0.50, 0.80]:
            data = results[env_label].get(ip, [])
            if not data:
                continue
            df    = pd.DataFrame(data)
            runs  = df["run"].values
            color = colors[ip]
            label = f"init Pl_env = {ip:.2f}"

            if "Pl_env" in df.columns:
                pl_vals = df["Pl_env"].values
                ax_env.plot(runs, pl_vals, 'o-', color=color, lw=1.2, ms=3.5, alpha=0.45)
                if len(pl_vals) >= 3:
                    roll = pd.Series(pl_vals).rolling(3, min_periods=1).mean().values
                    ax_env.plot(runs, roll, color=color, lw=2.8, alpha=0.92, label=label)

            se_vals = df["se"].values
            ax_se.scatter(runs, se_vals, color=color, s=22, alpha=0.45, zorder=3)
            if len(se_vals) >= 3:
                roll = pd.Series(se_vals).rolling(3, min_periods=1).mean().values
                ax_se.plot(runs, roll, color=color, lw=2.8, alpha=0.9, label=label)

        ax_env.axhline(0.5, color='gray', lw=1.2, linestyle=':', alpha=0.7, label="Neutral (0.5)")
        # Annotate convergence direction
        arrow_x = 18
        if env_label == "empty":
            ax_env.annotate("converges\n↓ toward M0",
                            xy=(arrow_x, 0.18), fontsize=7.5, color='steelblue',
                            ha='center', va='top')
        else:
            ax_env.annotate("converges\n↑ toward full\nanalogy trust",
                            xy=(arrow_x, 0.80), fontsize=7.5, color='#27AE60',
                            ha='center', va='bottom')

        ax_env.set_ylabel("Pl_env (environment relevance)", fontsize=9)
        ax_env.set_ylim(-0.05, 1.10)
        ax_env.legend(fontsize=8, loc='center right')
        ax_env.grid(alpha=0.3)
        ax_env.spines[['top', 'right']].set_visible(False)

        ax_se.set_xlabel("Run #", fontsize=9)
        ax_se.set_ylabel("SE Score (safety-first)", fontsize=9)
        ax_se.set_ylim(0, 105)
        ax_se.legend(fontsize=8)
        ax_se.grid(alpha=0.3)
        ax_se.spines[['top', 'right']].set_visible(False)

        # Annotate SE spread at run 20 to show belief doesn't hurt much
        late_ses = {}
        for ip in [0.20, 0.50, 0.80]:
            data = results[env_label].get(ip, [])
            if data:
                df = pd.DataFrame(data)
                tail = df[df["run"] >= df["run"].max() - 4]["se"].mean()
                late_ses[ip] = tail
        if len(late_ses) == 3:
            spread = max(late_ses.values()) - min(late_ses.values())
            ax_se.text(0.02, 0.06,
                       f"Late-run SE spread: {spread:.1f} pts",
                       transform=ax_se.transAxes, fontsize=8,
                       color='dimgray', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.7))

    plt.tight_layout()
    out = f"{out_dir}/fig12_env_belief.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {out}")


if __name__ == "__main__":
    # 1. Adaptive vs fixed comparison → fig7
    print("\n→ Fig 7: Adaptive vs Fixed threshold comparison")
    compare_adaptive_vs_fixed(n_runs=20)

    # 2. Threshold sweep (warehouse) → fig6
    sweep_threshold_plot(n_runs=10)

    # 3. Full 3-config simulation → Excel files + fig8/9/10
    print("\n→ Running 3 configs (novel entity from run", NOVEL_ENTITY_SPAWN_RUN, "onward)")
    run_all_configs_and_figures()

    # 4. Analogy belief self-correction test → fig11
    # Warehouse: 90% obstacle runs — efficient_courier's low inflation + high speed
    # causes frequent multi-spill events that falsify the analogy clearly.
    print("\n→ Fig 11: Analogy belief self-correction test (warehouse env, 20 runs each)")
    print(f"  Correct analogy input : \"{CORRECT_ANALOGY_INPUT}\"")
    print(f"  Wrong   analogy input : \"{WRONG_ANALOGY_INPUT}\"")
    print("  Running correct analogy scenario...")
    correct_hist = _run_analogy_belief_test(
        CORRECT_ANALOGY_INPUT, [False, True], n_runs=20, scenario_label="correct")
    print("  Running wrong analogy scenario...")
    wrong_hist   = _run_analogy_belief_test(
        WRONG_ANALOGY_INPUT,   [False, True], n_runs=20, scenario_label="wrong")
    _plot_fig11_analogy_belief(correct_hist, wrong_hist, out_dir="plots")

    # 5. Environment relevance belief experiment → fig12
    print("\n→ Fig 12: Environment relevance belief (Pl_env) experiment")
    print("  Tests 3 initial Pl_env priors × 2 environments (empty / obstacle-rich).")
    print("  Shows Pl_env converging and SE adapting regardless of initial belief.")
    run_fig12_env_belief_experiment(out_dir="plots")

    print("\n✅ All done.")
