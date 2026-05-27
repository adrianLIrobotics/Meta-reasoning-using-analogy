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
    "medical_care_sealed": {
        # Lid-closed variant: same scenario but sealed container → reduced spill risk.
        # Meta-uncertainty: robot reasons that the lid lowers urgency → slightly faster.
        "description": "Medical delivery with sealed container — reduced urgency, still careful",
        "signature": np.array([0.65, 0.35]),
        "policy": {"speed": 0.62, "inflation": 1.05, "resolution": 0.05},
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
_force_empty_env     = False
# Force every run to have obstacles (used by Fig 13 4-scenario experiment).
_force_cluttered_env = False
# Override M2 DST with a fixed policy (simulates M0/M1 within the M2 framework).
# Dict with keys: speed, inflation, resolution, label — or None for normal M2 DST.
_fixed_policy_override = None

# ─── TASK META-UNCERTAINTY (lid state) ───────────────────────────────────────
# "Hot coffee is dangerous" (lid open) vs "Sealed container" (lid closed).
# When lid is closed, spill probability is scaled down significantly, and the
# robot can reason its way to the 'medical_care_sealed' analogy — faster but
# still careful. The robot does NOT directly observe the lid: it infers via analogy.
TASK_LID_CLOSED   = False   # True = cup has lid → reduced spill probability
LID_CLOSED_FACTOR = 0.25    # p_spill × this when lid is closed

# ─── SWITCHING MECHANISM (Fig 15 comparison) ─────────────────────────────────
# "DST"       : path-cone rays only (smart — ignores irrelevant side obstacles)
# "FULL_ROOM" : all 36 rays (naive — reacts to room-wide density, not path-specific)
# "ATTENTION" : no Pl_analogy switching; scalar attention_weight modulates
#               inflation/resolution continuously. If spilling despite high attention
#               → reduce attention ('give up' on caution, accept risk).
SWITCH_MODE      = "DST"
attention_weight = 0.50     # legacy scalar kept for compat; ATTENTION mode now uses attn_vec
_attn_fail_count = 0        # spill-despite-high-attention runs (triggers reduction)

# ─── ATTENTIONAL VECTOR ───────────────────────────────────────────────────────
# Replaces the scalar attention_weight with a 3-component vector:
#   attn_vec[0] = w_speed      : how much to dampen speed (0=M0-fast, 1=M1-slow)
#   attn_vec[1] = w_inflation  : how much to widen obstacle clearance (0=M0, 1=M1)
#   attn_vec[2] = w_resolution : how much to sharpen path grid (0=M0-coarse, 1=M1-fine)
# Each component is updated independently: speed-dominant spills raise w_speed;
# jerk-dominant spills raise w_resolution (finer grid → smoother path → less jerk).
attn_vec            = np.array([0.50, 0.50, 0.50])
_run_speed_risk_acc = 0.0   # speed_risk accumulated this run (risk diagnosis)
_run_jerk_risk_acc  = 0.0   # jerk_risk accumulated this run

# ─── LIQUID FILL LEVEL ───────────────────────────────────────────────────────
# Continuous probability input to spill physics. 0=empty, 1.0=full cup (maximum risk).
# Half-full (0.5) halves p_spill vs full cup when lid is open.
# When TASK_LID_CLOSED=True, LID_CLOSED_FACTOR overrides this (sealed container).
LIQUID_FILL_LEVEL = 1.0   # module default: full cup, lid open

# ─── AGILITY LOGGING ─────────────────────────────────────────────────────────
_agility_log         = []   # per-run dicts for Fig 16 regime-change experiment
_regime_schedule     = {}   # {run_number: TASK_LID_CLOSED value} — injected externally

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

def update_belief_full_room(hits_all, total_all, min_dist):
    """DST update using ALL 36 rays (room-wide density) instead of path-cone only.
    Identical math to update_belief_dst but ignores whether obstacles are in the
    robot's path — a cluttered room with a clear direct route still triggers caution.
    This is the 'naive' baseline: smarter than a raw count but less path-aware than DST."""
    global mass_H, mass_notH, mass_Theta
    density_factor   = hits_all / max(1, total_all)
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
    return round(float(np.clip(pl, 0.0, 1.0)), 3)

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
    global _run_speed_risk_acc, _run_jerk_risk_acc

    anomaly_detected_in_run = False
    _run_speed_risk_acc = 0.0
    _run_jerk_risk_acc  = 0.0
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
    elif _force_cluttered_env:
        current_env_is_empty = False
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
    global _fixed_policy_override
    global TASK_LID_CLOSED, SWITCH_MODE, attention_weight, _attn_fail_count
    global attn_vec, _run_speed_risk_acc, _run_jerk_risk_acc, LIQUID_FILL_LEVEL
    global _agility_log, _regime_schedule

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
        # Dispatch sensing to the right belief-update mechanism
        if SWITCH_MODE == "FULL_ROOM":
            hits_all_rays  = sum(1 for _, t in laser_endpoints if t == "OBSTACLE")
            plausibility_H = update_belief_full_room(hits_all_rays, 36, min_dist)
        else:  # "DST" and "ATTENTION" both use path-cone DST for frame-level Pl_H
            plausibility_H = update_belief_dst(hits_in_path, total_rays_in_cone, min_dist)
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
    analogy_res     = base_policy["resolution"]
    # Layer 1: env-relevance
    #   Speed:     Pl_env × analogy ↔ M0  (clear room → analogy speed ↓ toward M0 efficiency)
    #   Inflation: Pl_env × max_safe ↔ M0  (clear → small margins; dense → wide berths)
    #              max_safe scales with ROOM_SIZE so domestic never over-inflates
    #              (warehouse: max=1.85; domestic 10x10: max≈1.18 — tight rooms need less).
    #   Resolution: Pl_env × analogy ↔ M0 (obstacle-rich → finer grid for precise planning)
    env_speed   = Pl_env * analogy_speed   + (1.0 - Pl_env) * POLICY_M0["speed"]
    room_scale  = min(1.0, ROOM_SIZE / 30.0)   # 1.0=warehouse 30m; 0.33=domestic 10m
    max_safe_inflate = POLICY_M0["inflation"] + (POLICY_M1["inflation"] - POLICY_M0["inflation"]) * room_scale
    env_inflate = Pl_env * max_safe_inflate + (1.0 - Pl_env) * POLICY_M0["inflation"]
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

    if SWITCH_MODE == "ATTENTION" and _fixed_policy_override is None:
        # ATTENTION mode: probabilistic attentional vector [w_speed, w_inflation, w_resolution].
        # Each component is updated independently from per-run spill risk diagnosis:
        #   speed-dominant spills (speed_risk > jerk_risk) → raise w_speed more
        #   jerk-dominant spills (tight corners, coarse path) → raise w_resolution more
        #   inflation always responds (broader clearance helps both risk types)
        # This decouples "I need to slow down" from "I need a smoother path".
        target_v   = POLICY_M0["speed"]      + attn_vec[0] * (POLICY_M1["speed"]      - POLICY_M0["speed"])
        target_inf = POLICY_M0["inflation"]  + attn_vec[1] * (POLICY_M1["inflation"]  - POLICY_M0["inflation"])
        target_res = POLICY_M0["resolution"] + attn_vec[2] * (POLICY_M1["resolution"] - POLICY_M0["resolution"])
        active_model_label = f"M2-ATTNv[{attn_vec[0]:.2f},{attn_vec[1]:.2f},{attn_vec[2]:.2f}]"
    elif plausibility_H <= dynamic_threshold:
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
    # Baseline override: simulate M0/M1 within M2 framework (bypasses DST envelope).
    if _fixed_policy_override is not None:
        target_v   = _fixed_policy_override["speed"]
        target_inf = _fixed_policy_override["inflation"]
        target_res = _fixed_policy_override["resolution"]
        active_model_label = _fixed_policy_override.get("label", "BASELINE")
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
    # Spill factor: sealed lid overrides fill-level; open cup risk scales with fill level
    # (half-full cup = 0.5 × spill probability vs full cup = 1.0).
    lid_factor = LID_CLOSED_FACTOR if TASK_LID_CLOSED else LIQUID_FILL_LEVEL
    p_spill    = (speed_risk + jerk_risk) * cooldown_factor * lid_factor
    # Accumulate risk per run for attentional vector diagnosis at run end
    _run_speed_risk_acc += speed_risk
    _run_jerk_risk_acc  += jerk_risk
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
        # Efficiency: time + battery reward. Collision → task failure (efficiency=0).
        # Spill penalty is flat (not diluted by time/battery): harm of a spill is
        # domain-constant — spilling medication does not hurt less because you were slow.
        efficiency = max(0.0, (100 - run_collisions * 100) * time_factor * (battery_level / 100.0))
        final_se   = max(0.0, efficiency - run_spills * 30)
        historical_SE.append(final_se)
        had_collision = run_collisions > 0
        anomaly_ok    = anomaly_detected_in_run and not had_collision
        # Familiarity: reward consecutive clean runs, penalise failures
        if run_spills == 0 and not had_collision:
            consecutive_clean_runs += 1
        else:
            consecutive_clean_runs = max(0, consecutive_clean_runs - 2)
        update_analogy_belief(run_spills, had_collision, anomaly_ok)

        # ATTENTION mode: update attentional vector from run outcomes.
        # Spill risk diagnosis: compare accumulated speed_risk vs jerk_risk to determine
        # which parameter to tighten most. This gives per-component attention updates
        # rather than moving all parameters together (scalar approach).
        if SWITCH_MODE == "ATTENTION":
            total_risk = _run_speed_risk_acc + _run_jerk_risk_acc + 1e-9
            speed_frac = _run_speed_risk_acc / total_risk   # 0–1: dominance of speed risk
            jerk_frac  = _run_jerk_risk_acc  / total_risk   # 0–1: dominance of jerk risk
            if run_spills > 0:
                step = 0.10 * min(run_spills, 3)             # cap at 3 spills worth of step
                # Speed component: responds to speed-dominant spills
                attn_vec[0] = min(1.0, attn_vec[0] + step * (0.30 + 0.70 * speed_frac))
                # Inflation: always responds — wider berths help both risk types
                attn_vec[1] = min(1.0, attn_vec[1] + step * 0.60)
                # Resolution component: responds to jerk-dominant spills (finer grid = smoother path)
                attn_vec[2] = min(1.0, attn_vec[2] + step * (0.30 + 0.70 * jerk_frac))
                if float(np.mean(attn_vec)) >= 0.70:
                    _attn_fail_count += 1
                if _attn_fail_count > 3:   # stuck at high attention, still spilling → reduce
                    attn_vec[:] = np.maximum(0.10, attn_vec - 0.20)
                    _attn_fail_count = 0
            else:
                _attn_fail_count = 0
                attn_vec[:] = np.maximum(0.0, attn_vec - 0.04)   # relax all after clean run
            attention_weight = float(np.mean(attn_vec))           # keep legacy scalar in sync
            _run_speed_risk_acc = 0.0
            _run_jerk_risk_acc  = 0.0

        # Lid-aware analogy selection: when lid is closed, bias toward sealed variant
        # (meta-level uncertainty — robot infers task risk from task description).
        if TASK_LID_CLOSED and selected_analogy_name == "medical_care":
            # Only switch if analogy belief still supports it (Pl_analogy high enough)
            if Pl_analogy > 0.65 and _fixed_policy_override is None:
                selected_analogy_name = "medical_care_sealed"
                base_policy = ANALOGY_LIBRARY["medical_care_sealed"]["policy"]
        elif not TASK_LID_CLOSED and selected_analogy_name == "medical_care_sealed":
            selected_analogy_name = "medical_care"
            base_policy = ANALOGY_LIBRARY["medical_care"]["policy"]

        # Regime schedule: externally injected lid changes at specific run numbers
        if (current_run + 1) in _regime_schedule:
            TASK_LID_CLOSED = _regime_schedule[current_run + 1]

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
            "frames": frame_counter,
            "battery": round(battery_level, 1),
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
        _agility_log.append({
            "run":       current_run,            "se":      final_se,
            "spills":    run_spills,             "Pl_a":    round(Pl_analogy, 3),
            "Pl_env":    round(Pl_env, 3),
            "attn":      round(float(np.mean(attn_vec)), 3),   # scalar mean (plot compat)
            "attn_v":    [round(float(x), 3) for x in attn_vec],  # full vector
            "lid":       TASK_LID_CLOSED,        "mode":    SWITCH_MODE,
            "eff_speed": round(eff_speed, 3),    "fill":    LIQUID_FILL_LEVEL,
        })
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
    global attention_weight, _attn_fail_count
    global attn_vec, _run_speed_risk_acc, _run_jerk_risk_acc

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
    # Smart Pl_env prior: warehouse is 90% cluttered, domestic 90% clear, mix neutral.
    # Seeding belief correctly eliminates the cold-start penalty in early runs.
    if cfg_envs == [False, True]:    # warehouse only
        Pl_env = 0.72
    elif cfg_envs == [True, False]:  # domestic only
        Pl_env = 0.28
    else:                            # mix
        Pl_env = 0.50
    sum_pl_h_in_run         = 0.0
    cnt_pl_h_in_run         = 0
    novel_entity_active     = False
    is_playing         = True
    prev_v_final       = np.zeros(2)
    frames_since_spill = SPILL_COOLDOWN_FRAMES
    # Reset attention and agility state for new config
    attention_weight    = 0.50
    _attn_fail_count    = 0
    attn_vec[:]         = [0.50, 0.50, 0.50]
    _run_speed_risk_acc = 0.0
    _run_jerk_risk_acc  = 0.0
    _agility_log.clear()
    _regime_schedule.clear()
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


# Scenario keys shared between runner and plot function — defined once.
_FIG13_SCENARIO_KEYS = [
    "domestic\nclear",
    "domestic\ncluttered",
    "warehouse\nclear",
    "warehouse\ncluttered",
]
_FIG13_XLABEL = "Run #"
_FIG13_YLABEL = "SE Score"


def _run_4scenario_model(model_label, policy_override, cfg_envs, n_runs,
                         force_empty, force_cluttered):
    """
    Run n_runs for one model in one scenario.
    policy_override=None → M2 DST; dict → fixed policy (M0/M1 baseline).
    Entity is disabled for a clean environment-only comparison.
    """
    global _force_empty_env, _force_cluttered_env, _fixed_policy_override
    global total_runs_to_execute, NOVEL_ENTITY_SPAWN_RUN
    global eff_speed, eff_inflation, eff_resolution

    orig_n     = total_runs_to_execute
    orig_spawn = NOVEL_ENTITY_SPAWN_RUN
    total_runs_to_execute  = n_runs
    NOVEL_ENTITY_SPAWN_RUN = 9999   # disable novel entity for clean env comparison
    _force_empty_env     = force_empty
    _force_cluttered_env = force_cluttered
    _fixed_policy_override = (
        {**policy_override, "label": model_label} if policy_override is not None else None
    )

    init_config(cfg_envs)
    # Smart Pl_env prior: if we know the scenario type up front, seed belief accordingly.
    # Cluttered → expect obstacles → start with elevated caution.
    # Empty     → expect clear paths → start relaxed toward M0.
    # This eliminates the cold-start penalty where Pl_env=0.50 causes early spills
    # before the environment has been experienced.
    if force_cluttered:
        Pl_env = 0.72
    elif force_empty:
        Pl_env = 0.22
    # For M0/M1 start from their own params (skip medical_care warm-up lag)
    if policy_override is not None:
        eff_speed      = policy_override["speed"]
        eff_inflation  = policy_override["inflation"]
        eff_resolution = policy_override["resolution"]
        META_PARAMS["resolution"] = policy_override["resolution"]
        build_obstacle_grid()
    reset_run()

    total_f = 0
    while is_playing and total_f < n_runs * 5000:
        step_logic()
        total_f += 1

    run_results = [{"se": d["se"], "spills": d["spills"], "model": model_label,
                    "frames": d.get("frames", 0), "battery": d.get("battery", 100)}
                   for d in cold_start_data]
    _force_empty_env       = False
    _force_cluttered_env   = False
    _fixed_policy_override = None
    NOVEL_ENTITY_SPAWN_RUN = orig_spawn
    total_runs_to_execute  = orig_n
    return run_results


def run_fig1_overview(n_runs=40, out_dir="plots"):
    """
    Fig 1 — Performance Overview (M0 vs M1 vs M2, 40 runs, no novel entity).

    Runs each model for n_runs in each of the 3 configs (domestic, mix, warehouse)
    using the natural environment mix for each config (no force_empty/cluttered).
    The entity is disabled so M2's belief has time to mature and results are clean.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    _OVERVIEW_CONFIGS = [
        ("domestic",  [True,  False]),
        ("mix",       [True,  True ]),
        ("warehouse", [False, True ]),
    ]
    _MODELS = [("M0", POLICY_M0), ("M1", POLICY_M1), ("M2", None)]

    # results[config][model] = list of run dicts
    results = {cfg: {} for cfg, _ in _OVERVIEW_CONFIGS}

    print("\n→ Fig 1: Performance overview (40 runs, no novel entity)")
    for cfg_name, cfg_envs in _OVERVIEW_CONFIGS:
        for model_label, policy_override in _MODELS:
            print(f"    {cfg_name:10s} | {model_label} …", end=" ", flush=True)
            runs = _run_4scenario_model(
                model_label, policy_override, cfg_envs, n_runs,
                force_empty=False, force_cluttered=False
            )
            results[cfg_name][model_label] = runs
            mean_se = np.mean([r["se"] for r in runs]) if runs else 0
            print(f"mean SE={mean_se:.1f}")

    _plot_fig1_overview(results, _OVERVIEW_CONFIGS, out_dir)


def _plot_fig1_overview(results, configs, out_dir):
    """4-panel grouped bar chart: SE, spills, frames, battery per model per config."""
    cfg_names  = [c for c, _ in configs]
    model_labels = ["M0", "M1", "M2"]
    colors = {"M0": "#E07B39", "M1": "#4C7EBE", "M2": "#27AE60"}
    bar_w  = 0.22
    x      = np.arange(len(cfg_names))
    offsets = [-bar_w - 0.03, 0, bar_w + 0.03]

    metrics = [
        ("se",      "SE Score (0–100)",        "Avg SE Score",          ".1f"),
        ("spills",  "Spills per run",           "Avg Spills per Run",    ".2f"),
        ("frames",  "Frames to goal",           "Avg Frames (lower=faster)", ".0f"),
        ("battery", "Battery % remaining",      "Avg Battery at Goal",   ".1f"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        "Fig 1 — Performance Overview: M0 vs M1 vs M2\n"
        f"(40 runs per configuration, no novel entity — clean comparison)",
        fontsize=12, fontweight='bold')

    for ax, (key, ylabel, title, fmt) in zip(axes.flat, metrics):
        for mi, model in enumerate(model_labels):
            vals, errs = [], []
            for cfg in cfg_names:
                run_list = results.get(cfg, {}).get(model, [])
                v = [r.get(key, r.get("se", 0)) for r in run_list]
                vals.append(np.mean(v) if v else 0)
                errs.append(np.std(v)  if len(v) > 1 else 0)
            bars = ax.bar(x + offsets[mi], vals, bar_w, yerr=errs, capsize=3,
                          label=model, color=colors[model],
                          edgecolor='white', linewidth=0.5, alpha=0.9,
                          error_kw={'linewidth': 1.2, 'capthick': 1.2})
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f'{v:{fmt}}', ha='center', va='bottom', fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in cfg_names])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = f"{out_dir}/fig1_performance_overview.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {out}")


def run_4scenario_experiment(n_runs=20, out_dir="plots"):
    """
    Fig 13: Compare M0, M1, M2 across 4 clearly separated scenarios.
    Shows M2 adapts: efficient in clear rooms, maximally safe in cluttered ones.

    domestic-clear     → M0 ≈ M2 >> M1  (no obstacles, caution is wasted)
    domestic-cluttered → M2 > M1 > M0   (furniture, analogy guides precise nav)
    warehouse-clear    → M0 ≈ M2 >> M1  (large empty space, M1 is too slow)
    warehouse-cluttered→ M2 ≈ M1 >> M0  (dense obstacles, safety critical)
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    _SCENARIOS = [
        (_FIG13_SCENARIO_KEYS[0], [True, False],  True,  False),
        (_FIG13_SCENARIO_KEYS[1], [True, False],  False, True),
        (_FIG13_SCENARIO_KEYS[2], [False, True],  True,  False),
        (_FIG13_SCENARIO_KEYS[3], [False, True],  False, True),
    ]
    _MODELS = [("M0", POLICY_M0), ("M1", POLICY_M1), ("M2", None)]

    results = {k: {} for k in _FIG13_SCENARIO_KEYS}

    print("\n→ Fig 13: 4-Scenario comparison (M0 vs M1 vs M2, no novel entity)")
    for scen_label, cfg_envs, force_empty, force_cluttered in _SCENARIOS:
        tag = scen_label.replace("\n", "_")
        for model_label, policy in _MODELS:
            print(f"  [{tag}] {model_label}...", end=" ", flush=True)
            data = _run_4scenario_model(model_label, policy, cfg_envs,
                                        n_runs, force_empty, force_cluttered)
            results[scen_label][model_label] = data
            avg_se = np.mean([d["se"] for d in data]) if data else float('nan')
            avg_sp = np.mean([d["spills"] for d in data]) if data else float('nan')
            print(f"avg SE={avg_se:.1f}  spills={avg_sp:.2f}")

    _plot_fig13_4scenarios(results, out_dir)


def _plot_fig13_4scenarios(results, out_dir):
    """Bar + trajectory plot for the 4-scenario M0/M1/M2 comparison."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    _COLORS   = {"M0": "#E07B39", "M1": "#4C7EBE", "M2": "#27AE60"}
    _MODELS   = ["M0", "M1", "M2"]
    _EXPECTED = {
        _FIG13_SCENARIO_KEYS[0]: "M0 ≈ M2 > M1",
        _FIG13_SCENARIO_KEYS[1]: "M2 > M1 > M0",
        _FIG13_SCENARIO_KEYS[2]: "M0 ≈ M2 > M1",
        _FIG13_SCENARIO_KEYS[3]: "M2 ≈ M1 >> M0",
    }

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        "Fig 13 — M2 Adaptive Meta-Reasoning: 4-Scenario Comparison (M0 vs M1 vs M2)\n"
        "M2 reads the environment via Pl_env: relaxes to M0 efficiency in clear rooms, "
        "rises to M1 safety margins in cluttered ones.",
        fontsize=11, fontweight='bold')

    for col, scen in enumerate(_FIG13_SCENARIO_KEYS):
        ax_bar  = axes[0, col]
        ax_line = axes[1, col]

        avgs = {m: np.mean([d["se"] for d in results[scen].get(m, [{"se": 0}])])
                for m in _MODELS}
        bars = ax_bar.bar(_MODELS, [avgs[m] for m in _MODELS],
                          color=[_COLORS[m] for m in _MODELS],
                          width=0.55, alpha=0.85, edgecolor='white', linewidth=1.5)
        for bar, m in zip(bars, _MODELS):
            h = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 1,
                        f"{h:.1f}", ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color=_COLORS[m])

        ax_bar.set_title(scen.replace("\n", " — ").title(), fontweight='bold', fontsize=9)
        ax_bar.set_ylabel("Avg SE Score", fontsize=9)
        ax_bar.set_ylim(0, 105)
        ax_bar.grid(axis='y', alpha=0.3)
        ax_bar.spines[['top', 'right']].set_visible(False)
        ax_bar.text(0.98, 0.96, _EXPECTED[scen],
                    transform=ax_bar.transAxes, ha='right', va='top',
                    fontsize=8, color='dimgray',
                    bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.8})

        for model in _MODELS:
            ses = [d["se"] for d in results[scen].get(model, [])]
            if not ses:
                continue
            runs = np.arange(1, len(ses) + 1)
            ax_line.plot(runs, ses, 'o-', color=_COLORS[model],
                         lw=1.0, ms=3, alpha=0.35)
            if len(ses) >= 4:
                roll = pd.Series(ses).rolling(4, min_periods=1).mean().values
                ax_line.plot(runs, roll, color=_COLORS[model],
                             lw=2.6, alpha=0.92, label=model)

        ax_line.set_xlabel(_FIG13_XLABEL, fontsize=8)
        ax_line.set_ylabel(_FIG13_YLABEL, fontsize=8)
        ax_line.set_ylim(0, 105)
        ax_line.legend(fontsize=8)
        ax_line.grid(alpha=0.3)
        ax_line.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = f"{out_dir}/fig13_4scenarios.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {out}")


# ─── HELPER: run M2 (no fixed policy) in one scenario ───────────────────────

def _run_m2_scenario(n_runs, cfg_envs, force_empty=False, force_cluttered=False,
                     switch_mode="DST", task_lid_closed=False, regime_sched=None):
    """Run M2 with given SWITCH_MODE and initial lid state for n_runs.
    Returns a copy of _agility_log (richer than cold_start_data — includes attn, mode, etc.)."""
    global _force_empty_env, _force_cluttered_env, _fixed_policy_override
    global total_runs_to_execute, NOVEL_ENTITY_SPAWN_RUN
    global SWITCH_MODE, TASK_LID_CLOSED, _regime_schedule, Pl_env

    orig_n     = total_runs_to_execute
    orig_spawn = NOVEL_ENTITY_SPAWN_RUN
    orig_mode  = SWITCH_MODE
    orig_lid   = TASK_LID_CLOSED

    total_runs_to_execute  = n_runs
    NOVEL_ENTITY_SPAWN_RUN = 9999
    _force_empty_env       = force_empty
    _force_cluttered_env   = force_cluttered
    _fixed_policy_override = None
    SWITCH_MODE            = switch_mode
    TASK_LID_CLOSED        = task_lid_closed

    init_config(cfg_envs)
    # init_config clears _regime_schedule — populate AFTER
    if regime_sched:
        _regime_schedule.update(regime_sched)
    # Override Pl_env for scenario type (init_config already sets warehouse → 0.72)
    if force_cluttered:
        Pl_env = 0.72
    elif force_empty:
        Pl_env = 0.22
    reset_run()

    total_f = 0
    while is_playing and total_f < n_runs * 5000:
        step_logic()
        total_f += 1

    log = list(_agility_log)  # snapshot before restore

    _force_empty_env       = False
    _force_cluttered_env   = False
    _fixed_policy_override = None
    SWITCH_MODE            = orig_mode
    TASK_LID_CLOSED        = orig_lid
    NOVEL_ENTITY_SPAWN_RUN = orig_spawn
    total_runs_to_execute  = orig_n
    return log


def _run_lid_model(model_label, policy_override, cfg_envs, n_runs,
                   force_cluttered, task_lid_closed):
    """Run one model (M0/M1/M2) with a specific lid state in one scenario.
    Returns list of {se, spills, model}. Used by Fig 14."""
    global _force_empty_env, _force_cluttered_env, _fixed_policy_override
    global total_runs_to_execute, NOVEL_ENTITY_SPAWN_RUN
    global SWITCH_MODE, TASK_LID_CLOSED, Pl_env
    global eff_speed, eff_inflation, eff_resolution

    orig_n     = total_runs_to_execute
    orig_spawn = NOVEL_ENTITY_SPAWN_RUN
    orig_lid   = TASK_LID_CLOSED
    orig_mode  = SWITCH_MODE

    total_runs_to_execute  = n_runs
    NOVEL_ENTITY_SPAWN_RUN = 9999
    _force_empty_env       = False
    _force_cluttered_env   = force_cluttered
    _fixed_policy_override = (
        {**policy_override, "label": model_label} if policy_override is not None else None
    )
    TASK_LID_CLOSED = task_lid_closed
    SWITCH_MODE     = "DST"  # M2 default; ignored when _fixed_policy_override is set

    init_config(cfg_envs)
    if force_cluttered:
        Pl_env = 0.72
    if policy_override is not None:
        eff_speed      = policy_override["speed"]
        eff_inflation  = policy_override["inflation"]
        eff_resolution = policy_override["resolution"]
        META_PARAMS["resolution"] = policy_override["resolution"]
        build_obstacle_grid()
    reset_run()

    total_f = 0
    while is_playing and total_f < n_runs * 5000:
        step_logic()
        total_f += 1

    run_results = [{"se": d["se"], "spills": d["spills"], "model": model_label}
                   for d in cold_start_data]

    _force_empty_env       = False
    _force_cluttered_env   = False
    _fixed_policy_override = None
    TASK_LID_CLOSED        = orig_lid
    SWITCH_MODE            = orig_mode
    NOVEL_ENTITY_SPAWN_RUN = orig_spawn
    total_runs_to_execute  = orig_n
    return run_results


# ─── FIG 14: Lid Uncertainty ──────────────────────────────────────────────────

def run_fig14_lid_uncertainty(n_runs=30, out_dir="plots"):
    """
    Fig 14: Task meta-uncertainty — open cup (lid open) vs sealed cup (lid closed).
    Warehouse-cluttered, 30 runs each condition × 3 models.

    M0/M1: benefit only from reduced spill physics (×0.25 lid factor), policy unchanged.
    M2:    additionally adapts via 'medical_care_sealed' analogy — recognises lid state
           changes the risk profile and adjusts speed/inflation/resolution.

    Expected: M2 lid_closed improvement > M0/M1 lid_closed improvement, because M2
    combines physical benefit (fewer spill events) with policy benefit (faster, still safe).
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    cfg_envs = [False, True]  # warehouse
    _MODELS  = [("M0", POLICY_M0), ("M1", POLICY_M1), ("M2", None)]
    _LIDS    = [("lid_open", False), ("lid_closed", True)]

    results = {m: {} for m, _ in _MODELS}
    print("\n→ Fig 14: Lid uncertainty (warehouse_cluttered, M0 vs M1 vs M2)")
    for model_label, policy in _MODELS:
        for lid_label, lid_val in _LIDS:
            print(f"  [{model_label}] {lid_label}...", end=" ", flush=True)
            data = _run_lid_model(model_label, policy, cfg_envs, n_runs,
                                  force_cluttered=True, task_lid_closed=lid_val)
            results[model_label][lid_label] = data
            avg_se = np.mean([d["se"] for d in data]) if data else float('nan')
            avg_sp = np.mean([d["spills"] for d in data]) if data else float('nan')
            print(f"avg SE={avg_se:.1f}  spills={avg_sp:.2f}")

    _plot_fig14(results, out_dir)


def _plot_fig14(results, out_dir):
    """3-row × 3-col plot: SE bars, SE over runs, spill bars per model."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    _MODELS    = ["M0", "M1", "M2"]
    _LIDS      = ["lid_open", "lid_closed"]
    _COLORS    = {"M0": "#E07B39", "M1": "#4C7EBE", "M2": "#27AE60"}
    _LID_ALPHA = {"lid_open": 0.50, "lid_closed": 0.92}
    _LID_LABEL = {"lid_open": "Lid Open", "lid_closed": "Lid Closed"}

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(
        "Fig 14 — Task Meta-Uncertainty: Lid Open vs Lid Closed (Warehouse-Cluttered)\n"
        "M2 adapts via 'medical_care_sealed' analogy (policy + physics). "
        "M0/M1 gain spill-physics benefit only — no policy adaptation.",
        fontsize=11, fontweight='bold')

    bar_x = np.arange(2)
    for col, m in enumerate(_MODELS):
        ax_bar  = axes[0, col]
        ax_line = axes[1, col]
        ax_sp   = axes[2, col]

        # Row 0: avg SE bars (open vs closed)
        bar_se = [np.mean([d["se"] for d in results[m].get(l, [{"se": 0}])]) for l in _LIDS]
        bars = ax_bar.bar(bar_x, bar_se, color=_COLORS[m],
                          width=0.45, edgecolor='white', linewidth=1.5)
        for bar, lid in zip(bars, _LIDS):
            bar.set_alpha(_LID_ALPHA[lid])
        for bar, v in zip(bars, bar_se):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, v + 1,
                        f"{v:.1f}", ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color=_COLORS[m])
        delta = bar_se[1] - bar_se[0]
        ax_bar.text(0.97, 0.96, f"Δ={delta:+.1f}",
                    transform=ax_bar.transAxes, ha='right', va='top',
                    fontsize=9, color='dimgray',
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.8))
        ax_bar.set_xticks(bar_x)
        ax_bar.set_xticklabels([_LID_LABEL[l] for l in _LIDS], fontsize=9)
        ax_bar.set_title(f"Model {m}", fontweight='bold', fontsize=10)
        ax_bar.set_ylabel("Avg SE Score", fontsize=9)
        ax_bar.set_ylim(0, 105)
        ax_bar.grid(axis='y', alpha=0.3)
        ax_bar.spines[['top', 'right']].set_visible(False)

        # Row 1: SE over runs (raw + rolling mean)
        for lid in _LIDS:
            ses = [d["se"] for d in results[m].get(lid, [])]
            if not ses:
                continue
            runs = np.arange(1, len(ses) + 1)
            ax_line.plot(runs, ses, 'o-', color=_COLORS[m],
                         alpha=_LID_ALPHA[lid] * 0.45, lw=0.8, ms=3)
            roll = pd.Series(ses).rolling(4, min_periods=1).mean().values
            ax_line.plot(runs, roll, color=_COLORS[m], lw=2.5,
                         alpha=_LID_ALPHA[lid], label=_LID_LABEL[lid])
        ax_line.set_xlabel("Run #", fontsize=8)
        ax_line.set_ylabel("SE Score", fontsize=8)
        ax_line.set_ylim(0, 105)
        ax_line.legend(fontsize=8)
        ax_line.grid(alpha=0.3)
        ax_line.spines[['top', 'right']].set_visible(False)

        # Row 2: avg spills/run bars
        bar_sp = [np.mean([d["spills"] for d in results[m].get(l, [{"spills": 0}])]) for l in _LIDS]
        bars2 = ax_sp.bar(bar_x, bar_sp, color=_COLORS[m],
                          width=0.45, edgecolor='white', linewidth=1.5)
        for bar, lid in zip(bars2, _LIDS):
            bar.set_alpha(_LID_ALPHA[lid])
        for bar, v in zip(bars2, bar_sp):
            ax_sp.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                       f"{v:.2f}", ha='center', va='bottom',
                       fontsize=9, fontweight='bold', color=_COLORS[m])
        ax_sp.set_xticks(bar_x)
        ax_sp.set_xticklabels([_LID_LABEL[l] for l in _LIDS], fontsize=9)
        ax_sp.set_ylabel("Avg Spills / Run", fontsize=9)
        ax_sp.set_ylim(0, max(max(bar_sp), 0.1) * 1.5)
        ax_sp.grid(axis='y', alpha=0.3)
        ax_sp.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = f"{out_dir}/fig14_lid_uncertainty.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {out}")


# ─── FIG 15: Switching Mechanism Comparison ───────────────────────────────────

def run_fig15_switching_comparison(n_runs=40, out_dir="plots"):
    """
    Fig 15: DST (path-cone) vs FULL_ROOM (all 36 rays) vs ATTENTION (scalar weight).
    All run M2 in warehouse_cluttered, lid_open, no regime change.

    DST: only samples rays in the forward 70° cone toward goal — ignores side clutter.
         Smart: a cluttered room with clear path = low obstacle density in cone = trust analogy.
    FULL_ROOM: samples all 36 rays — reacts to ambient room density.
         Naive: even when path is clear, dense side walls trigger extra caution → slower SE.
    ATTENTION: no hard policy switch. A single scalar (0=M0-like, 1=M1-like) modulates
         inflation continuously. If spilling despite high attention → reduce weight (give up).

    Expected ranking: DST > FULL_ROOM ≥ ATTENTION (path-awareness is the key differentiator).
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    cfg_envs = [False, True]
    _MODES   = ["DST", "FULL_ROOM", "ATTENTION"]

    results = {}
    print("\n→ Fig 15: Switching comparison (warehouse_cluttered, M2, lid_open)")
    for mode in _MODES:
        print(f"  [{mode}]...", end=" ", flush=True)
        log = _run_m2_scenario(n_runs, cfg_envs, force_cluttered=True,
                               switch_mode=mode, task_lid_closed=False)
        results[mode] = log
        avg_se = np.mean([e["se"] for e in log]) if log else float('nan')
        avg_sp = np.mean([e["spills"] for e in log]) if log else float('nan')
        print(f"avg SE={avg_se:.1f}  spills={avg_sp:.2f}")

    _plot_fig15(results, out_dir)


def _plot_fig15(results, out_dir):
    """3-panel vertical: SE over runs · Pl_analogy/attention trace · cumulative spills."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    _MODES  = ["DST", "FULL_ROOM", "ATTENTION"]
    _COLORS = {"DST": "#27AE60", "FULL_ROOM": "#E07B39", "ATTENTION": "#7B68EE"}
    _LABELS = {
        "DST":       "DST (path-cone, smart)",
        "FULL_ROOM": "Full-Room (all rays, naive)",
        "ATTENTION": "Attention scalar (no switch)",
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    fig.suptitle(
        "Fig 15 — Switching Mechanism Comparison: DST vs Full-Room vs Attention\n"
        "DST reasons over path-relevant obstacles only. "
        "Full-Room reacts to ambient room density. "
        "Attention modulates inflation continuously without hard policy switching.",
        fontsize=10, fontweight='bold')

    ax_se, ax_pl, ax_cum = axes

    for mode in _MODES:
        log    = results.get(mode, [])
        runs   = [e["run"] for e in log]
        ses    = [e["se"] for e in log]
        spills_cum = np.cumsum([e["spills"] for e in log]).tolist()
        # Pl_analogy for DST/FULL_ROOM; attention_weight for ATTENTION
        pl_vals = [e["Pl_a"] if mode != "ATTENTION" else e["attn"] for e in log]

        c = _COLORS[mode]
        lbl = _LABELS[mode]

        ax_se.plot(runs, ses, 'o-', color=c, alpha=0.20, lw=0.8, ms=3)
        if len(ses) >= 4:
            roll = pd.Series(ses).rolling(4, min_periods=1).mean().values
            ax_se.plot(runs, roll, color=c, lw=2.6, alpha=0.95, label=lbl)
        else:
            ax_se.plot(runs, ses, color=c, lw=2.0, label=lbl)

        ax_pl.plot(runs, pl_vals, '-', color=c, lw=2.0, alpha=0.90, label=lbl)
        ax_cum.plot(runs, spills_cum, '-', color=c, lw=2.4, alpha=0.90, label=lbl)

    ax_se.set_ylabel("SE Score (rolling-4 mean)", fontsize=10)
    ax_se.set_ylim(0, 105)
    ax_se.legend(fontsize=9)
    ax_se.grid(alpha=0.3)
    ax_se.spines[['top', 'right']].set_visible(False)

    ax_pl.set_ylabel("Pl_analogy / attention weight", fontsize=10)
    ax_pl.set_ylim(-0.05, 1.10)
    ax_pl.axhline(0.5, color='gray', lw=0.8, linestyle='--', alpha=0.5,
                  label="switch threshold (DST/FULL_ROOM)")
    ax_pl.legend(fontsize=9)
    ax_pl.grid(alpha=0.3)
    ax_pl.spines[['top', 'right']].set_visible(False)

    ax_cum.set_xlabel("Run #", fontsize=10)
    ax_cum.set_ylabel("Cumulative Spills", fontsize=10)
    ax_cum.legend(fontsize=9)
    ax_cum.grid(alpha=0.3)
    ax_cum.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = f"{out_dir}/fig15_switching.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {out}")


# ─── FIG 16: Agility Metrics Under Regime Change ──────────────────────────────

def _compute_reaction_latency(ses, change_idx, n_per_phase):
    """Runs after change_idx (0-indexed) until rolling-3 mean reaches 90% of phase steady-state.
    steady-state = mean of last 3 runs in destination phase."""
    phase_end  = change_idx + n_per_phase
    dest_ses   = ses[change_idx: min(phase_end, len(ses))]
    if len(dest_ses) < 3:
        return n_per_phase
    steady = np.mean(dest_ses[-3:])
    for k in range(len(dest_ses) - 2):
        window = np.mean(dest_ses[k: k + 3])
        if steady > 5 and window >= 0.90 * steady:
            return k + 1
    return n_per_phase  # never recovered within phase


def _compute_sample_efficiency(log, threshold=0.75):
    """Runs until Pl_analogy first crosses threshold for 3 consecutive runs.
    For ATTENTION mode (no Pl_a), uses attn_vec norm stability instead.
    Returns n_per_phase if never achieved within the log."""
    pla_vals = [e["Pl_a"] for e in log]
    if all(v == 0 for v in pla_vals):
        # ATTENTION mode: stable when attn mean changes < 0.05 over 3 consecutive runs
        attn_vals = [e["attn"] for e in log]
        for k in range(2, len(attn_vals)):
            if max(attn_vals[k-2:k+1]) - min(attn_vals[k-2:k+1]) < 0.05:
                return k + 1
        return len(log)
    for k in range(2, len(pla_vals)):
        if all(v >= threshold for v in pla_vals[k-2: k+1]):
            return k + 1
    return len(log)


def run_fig16_agility_metrics(n_per_phase=15, out_dir="plots"):
    """
    Fig 16: Agility of each switching mechanism under two regime changes.
    Phase 1 (runs 1..n): lid_open  →  Phase 2 (n+1..2n): lid_closed  →  Phase 3 (2n+1..3n): lid_open.

    Measures:
      Reaction latency : runs until rolling-3 SE recovers to 90% of destination-phase steady-state.
      Behavior stability: SE std dev within each phase (after 3-run warm-up).
      SE trace         : 3n-run performance curve with regime change markers.

    DST expected fastest: path-cone update immediately sees reduced obstacle density
    in the sealed-lid phase → analogy shifts to medical_care_sealed within 2-3 runs.
    ATTENTION expected slowest: scalar weight updates are conservative (+0.12/spill)
    so require more evidence before crossing the high-attention regime.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    total_runs = 3 * n_per_phase
    cfg_envs   = [False, True]  # warehouse
    regime     = {n_per_phase + 1: True, 2 * n_per_phase + 1: False}
    _MODES     = ["DST", "FULL_ROOM", "ATTENTION"]

    results = {}
    print(f"\n→ Fig 16: Agility metrics — {n_per_phase} runs × 3 phases per mechanism")
    for mode in _MODES:
        print(f"  [{mode}]...", end=" ", flush=True)
        log = _run_m2_scenario(total_runs, cfg_envs, force_cluttered=True,
                               switch_mode=mode, task_lid_closed=False,
                               regime_sched=regime)
        results[mode] = log
        avg_se = np.mean([e["se"] for e in log]) if log else float('nan')
        print(f"avg SE={avg_se:.1f}  ({len(log)} runs logged)")

    _plot_fig16(results, out_dir, n_per_phase)


def _plot_fig16(results, out_dir, n_per_phase):
    """4-panel: SE trace · reaction latency · sample efficiency · behavior stability."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    _MODES  = ["DST", "FULL_ROOM", "ATTENTION"]
    _COLORS = {"DST": "#27AE60", "FULL_ROOM": "#E07B39", "ATTENTION": "#7B68EE"}
    _LABELS = {"DST": "DST (path-cone)", "FULL_ROOM": "Full-Room", "ATTENTION": "Attn Vector"}

    fig, axes = plt.subplots(4, 1, figsize=(13, 19))
    fig.suptitle(
        "Fig 16 — Agility Metrics Under Regime Change\n"
        f"Phase 1 (runs 1–{n_per_phase}): lid open  →  "
        f"Phase 2 ({n_per_phase+1}–{2*n_per_phase}): lid closed  →  "
        f"Phase 3 ({2*n_per_phase+1}–{3*n_per_phase}): lid open",
        fontsize=10, fontweight='bold')

    ax_se, ax_lat, ax_seff, ax_std = axes

    change_idxs    = [n_per_phase, 2 * n_per_phase]
    latencies      = {m: [] for m in _MODES}
    stabilities    = {m: [] for m in _MODES}
    sample_effs    = {m: 0   for m in _MODES}

    for mode in _MODES:
        log   = results.get(mode, [])
        runs  = list(range(1, len(log) + 1))
        ses   = [e["se"] for e in log]
        c     = _COLORS[mode]
        lbl   = _LABELS[mode]

        # SE trace
        ax_se.plot(runs, ses, 'o-', color=c, alpha=0.18, lw=0.8, ms=3)
        if len(ses) >= 4:
            roll = pd.Series(ses).rolling(4, min_periods=1).mean().values
            ax_se.plot(runs, roll, color=c, lw=2.5, alpha=0.95, label=lbl)
        else:
            ax_se.plot(runs, ses, color=c, lw=2.0, label=lbl)

        # Reaction latency per regime change
        for cp in change_idxs:
            lat = _compute_reaction_latency(ses, cp, n_per_phase)
            latencies[mode].append(lat)

        # Sample efficiency: runs until stable belief (Pl_a ≥ 0.75 for 3 consecutive runs)
        sample_effs[mode] = _compute_sample_efficiency(log, threshold=0.75)

        # Behavior stability: SE std in each phase (skip first 3 runs)
        for start in [0, n_per_phase, 2 * n_per_phase]:
            ph_ses = ses[start + 3: start + n_per_phase] if start + 3 < len(ses) else []
            stabilities[mode].append(np.std(ph_ses) if len(ph_ses) >= 2 else float('nan'))

    # ── Panel 1: SE trace ──
    for cp in change_idxs:
        ax_se.axvline(cp + 0.5, color='gray', lw=1.5, linestyle='--', alpha=0.6)
    ax_se.axvspan(n_per_phase, 2 * n_per_phase, alpha=0.07, color='steelblue',
                  label=f"lid closed (runs {n_per_phase+1}–{2*n_per_phase})")
    ax_se.set_ylabel("SE Score (rolling-4 mean)", fontsize=10)
    ax_se.set_ylim(0, 105)
    ax_se.legend(fontsize=9, loc='lower right')
    ax_se.grid(alpha=0.3)
    ax_se.spines[['top', 'right']].set_visible(False)
    ax_se.set_title("SE trace over all phases (dashed = regime change)", fontsize=9, pad=4)

    # ── Panel 2: Reaction latency ──
    x  = np.arange(len(_MODES))
    w  = 0.30
    change_labels = [
        f"open→closed\n(run {n_per_phase + 1})",
        f"closed→open\n(run {2 * n_per_phase + 1})",
    ]
    for ph_i, (ch_lbl, alpha) in enumerate(zip(change_labels, [0.60, 0.95])):
        vals = [latencies[m][ph_i] for m in _MODES]
        bars = ax_lat.bar(x + (ph_i - 0.5) * w, vals, width=w,
                          color=[_COLORS[m] for m in _MODES], alpha=alpha,
                          edgecolor='white', linewidth=1.2, label=ch_lbl)
        for bar, v in zip(bars, vals):
            ax_lat.text(bar.get_x() + bar.get_width() / 2, v + 0.15,
                        str(int(v)), ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels([_LABELS[m] for m in _MODES], fontsize=9)
    ax_lat.set_ylabel("Reaction Latency (runs)", fontsize=10)
    ax_lat.set_ylim(0, n_per_phase + 2)
    ax_lat.legend(fontsize=9)
    ax_lat.grid(axis='y', alpha=0.3)
    ax_lat.spines[['top', 'right']].set_visible(False)
    ax_lat.set_title(
        "Reaction Latency per Regime Change  (lower = faster adaptation)", fontsize=9, pad=4)

    # ── Panel 3: Sample efficiency ──
    se_vals = [sample_effs[m] for m in _MODES]
    bars3   = ax_seff.bar(x, se_vals, color=[_COLORS[m] for m in _MODES],
                          width=0.45, edgecolor='white', linewidth=1.5, alpha=0.88)
    for bar, v in zip(bars3, se_vals):
        ax_seff.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
                     str(int(v)), ha='center', va='bottom',
                     fontsize=10, fontweight='bold')
    ax_seff.set_xticks(x)
    ax_seff.set_xticklabels([_LABELS[m] for m in _MODES], fontsize=9)
    ax_seff.set_ylabel("Runs to Stable Belief (lower = more efficient)", fontsize=10)
    ax_seff.set_ylim(0, 3 * n_per_phase + 2)
    ax_seff.grid(axis='y', alpha=0.3)
    ax_seff.spines[['top', 'right']].set_visible(False)
    ax_seff.set_title(
        "Sample Efficiency  (runs until Pl_analogy ≥ 0.75 for 3 consecutive runs)", fontsize=9, pad=4)

    # ── Panel 4: Behavior stability ──
    x3   = np.arange(len(_MODES))
    w3   = 0.22
    ph_names  = ["Ph1\nopen", "Ph2\nclosed", "Ph3\nopen"]
    ph_alphas = [0.50, 0.95, 0.70]
    for ph_i, (ph_name, ph_alpha) in enumerate(zip(ph_names, ph_alphas)):
        for mi, m in enumerate(_MODES):
            v = stabilities[m][ph_i]
            lbl_bar = ph_name if mi == 0 else ""
            if not np.isnan(v):
                ax_std.bar(x3[mi] + (ph_i - 1) * w3, v, width=w3,
                           color=_COLORS[m], alpha=ph_alpha,
                           edgecolor='white', linewidth=1.2, label=lbl_bar)
                ax_std.text(x3[mi] + (ph_i - 1) * w3, v + 0.3,
                            f"{v:.1f}", ha='center', va='bottom', fontsize=8)
    ax_std.set_xticks(x3)
    ax_std.set_xticklabels([_LABELS[m] for m in _MODES], fontsize=9)
    ax_std.set_ylabel("SE Std Dev (lower = more stable)", fontsize=10)
    ax_std.set_xlabel("Switching Mechanism", fontsize=10)
    ax_std.legend(fontsize=9, title="Phase")
    ax_std.grid(axis='y', alpha=0.3)
    ax_std.spines[['top', 'right']].set_visible(False)
    ax_std.set_title(
        "Behavior Stability per Phase  (SE std dev, first 3 runs excluded as warm-up)",
        fontsize=9, pad=4)

    plt.tight_layout()
    out = f"{out_dir}/fig16_agility.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {out}")


# ─── FIG 17: Fill-Level Sweep ────────────────────────────────────────────────

def run_fig17_fill_level(n_runs=20, out_dir="plots"):
    """
    Fig 17: How does liquid fill level (0.25 → 1.0) affect SE and spill rate per model?
    Models M0, M1, M2 × fill levels [0.25, 0.50, 0.75, 1.0] in warehouse-cluttered.
    TASK_LID_CLOSED=False throughout (open cup scenario).

    Insight expected:
      M0: steep SE degradation as fill rises (high jerk, lots of spill events).
      M1: gradual degradation (slow+wide margins, low jerk even when full).
      M2: moderate degradation but mitigated — analogy policy's finer resolution
          reduces jerk so the higher fill-level spill probability fires less often.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    global LIQUID_FILL_LEVEL, TASK_LID_CLOSED
    _FILLS  = [0.25, 0.50, 0.75, 1.00]
    _MODELS = [
        ("M0", {"speed": POLICY_M0["speed"], "inflation": POLICY_M0["inflation"],
                "resolution": POLICY_M0["resolution"], "label": "M0-Reactive"}),
        ("M1", {"speed": POLICY_M1["speed"], "inflation": POLICY_M1["inflation"],
                "resolution": POLICY_M1["resolution"], "label": "M1-Symbolic"}),
        ("M2", None),
    ]
    cfg_envs = [False, True]   # warehouse

    print(f"\n→ Fig 17: Fill-level sweep — {len(_FILLS)} levels × 3 models × {n_runs} runs")

    results = {m: {"se": [], "spills": []} for m, _ in _MODELS}

    orig_fill = LIQUID_FILL_LEVEL
    orig_lid  = TASK_LID_CLOSED

    for fill in _FILLS:
        LIQUID_FILL_LEVEL = fill
        TASK_LID_CLOSED   = False
        for model_label, policy_override in _MODELS:
            global _force_empty_env, _force_cluttered_env, _fixed_policy_override
            global total_runs_to_execute
            _force_empty_env     = False
            _force_cluttered_env = True
            _fixed_policy_override = policy_override
            total_runs_to_execute  = n_runs
            init_config(cfg_envs)
            reset_run()
            frame_count = 0
            while is_playing and frame_count < 60_000:
                step_logic()
                frame_count += 1
            run_ses    = [d["se"]     for d in cold_start_data]
            run_spills = [d["spills"] for d in cold_start_data]
            results[model_label]["se"].append(np.mean(run_ses)    if run_ses    else float('nan'))
            results[model_label]["spills"].append(np.mean(run_spills) if run_spills else float('nan'))
            print(f"    fill={fill:.2f}  {model_label}:  SE={results[model_label]['se'][-1]:.1f}  "
                  f"spills/run={results[model_label]['spills'][-1]:.2f}")

    LIQUID_FILL_LEVEL  = orig_fill
    TASK_LID_CLOSED    = orig_lid
    _fixed_policy_override = None
    _force_cluttered_env   = False
    _plot_fig17(results, _FILLS, out_dir)


def _plot_fig17(results, fills, out_dir):
    """2-panel: SE vs fill level (left) · spills/run vs fill level (right)."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    _COLORS = {"M0": "#E74C3C", "M1": "#3498DB", "M2": "#27AE60"}
    _LABELS = {"M0": "M0-Reactive", "M1": "M1-Symbolic", "M2": "M2-DST"}
    _MARKERS = {"M0": "s", "M1": "^", "M2": "o"}
    fill_pct = [int(f * 100) for f in fills]

    fig, (ax_se, ax_sp) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "Fig 17 — Liquid Fill Level vs Performance  (warehouse-cluttered, lid open, 20 runs/point)\n"
        "Fill level = 0.25 (quarter-full) → 1.0 (brimming) — scales p_spill continuously",
        fontsize=10, fontweight='bold')

    for model in ["M0", "M1", "M2"]:
        se_vals = results[model]["se"]
        sp_vals = results[model]["spills"]
        c, lbl, mk = _COLORS[model], _LABELS[model], _MARKERS[model]
        ax_se.plot(fill_pct, se_vals, mk + '-', color=c, lw=2.2, ms=8, label=lbl)
        for x, y in zip(fill_pct, se_vals):
            ax_se.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                           xytext=(0, 8), ha='center', fontsize=8, color=c)
        ax_sp.plot(fill_pct, sp_vals, mk + '-', color=c, lw=2.2, ms=8, label=lbl)
        for x, y in zip(fill_pct, sp_vals):
            ax_sp.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                           xytext=(0, 8), ha='center', fontsize=8, color=c)

    ax_se.axvline(50, color='gray', linestyle=':', alpha=0.5, label="half-full (50%)")
    ax_sp.axvline(50, color='gray', linestyle=':', alpha=0.5, label="half-full (50%)")

    ax_se.set_xlabel("Liquid Fill Level (%)", fontsize=10)
    ax_se.set_ylabel("Average SE Score", fontsize=10)
    ax_se.set_xticks(fill_pct)
    ax_se.set_ylim(0, 100)
    ax_se.legend(fontsize=9)
    ax_se.grid(alpha=0.3)
    ax_se.spines[['top', 'right']].set_visible(False)
    ax_se.set_title("Safety-Efficiency vs Fill Level", fontsize=9, pad=4)

    ax_sp.set_xlabel("Liquid Fill Level (%)", fontsize=10)
    ax_sp.set_ylabel("Spills per Run", fontsize=10)
    ax_sp.set_xticks(fill_pct)
    ax_sp.set_ylim(0, max(
        max(results[m]["spills"], default=0) for m in ["M0","M1","M2"]
    ) * 1.25 + 0.3)
    ax_sp.legend(fontsize=9)
    ax_sp.grid(alpha=0.3)
    ax_sp.spines[['top', 'right']].set_visible(False)
    ax_sp.set_title("Spills per Run vs Fill Level", fontsize=9, pad=4)

    plt.tight_layout()
    out = f"{out_dir}/fig17_fill_level.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {out}")


if __name__ == "__main__":
    # 0. Performance overview (M0 vs M1 vs M2, 40 runs, no entity) → fig1
    run_fig1_overview(n_runs=40)

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

    # 6. 4-scenario comparison (M0 vs M1 vs M2) → fig13
    print("\n→ Fig 13: 4-scenario comparison")
    print("  domestic/warehouse × clear/cluttered — shows M2 adaptive advantage.")
    run_4scenario_experiment(n_runs=40, out_dir="plots")

    # 7. Lid uncertainty (open vs closed cup) → fig14
    print("\n→ Fig 14: Task meta-uncertainty — lid open vs lid closed")
    print("  M2 adapts via medical_care_sealed analogy + physics. M0/M1 physics only.")
    run_fig14_lid_uncertainty(n_runs=30, out_dir="plots")

    # 8. Switching mechanism comparison → fig15
    print("\n→ Fig 15: DST vs Full-Room vs Attention switching (warehouse_cluttered)")
    print("  DST: path-cone only (smart). FULL_ROOM: all 36 rays (naive). ATTENTION: scalar weight.")
    run_fig15_switching_comparison(n_runs=40, out_dir="plots")

    # 9. Agility metrics under regime change → fig16
    print("\n→ Fig 16: Agility under regime change (3 × 15 runs, lid_open → closed → open)")
    print("  Measures reaction latency, sample efficiency, behavior stability, SE trace.")
    run_fig16_agility_metrics(n_per_phase=15, out_dir="plots")

    # 10. Fill-level sweep → fig17
    print("\n→ Fig 17: Liquid fill-level sweep (0.25 → 1.0) — continuous spill probability")
    print("  Shows M2's path-smoothness advantage under increasing open-cup risk.")
    run_fig17_fill_level(n_runs=20, out_dir="plots")

    print("\n✅ All done.")
