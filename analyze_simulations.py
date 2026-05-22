"""
analyze_simulations.py
Auto-discovers all model*.xlsx files in the current directory and
generates 5 insight figures saved to plots/.

Usage:  python analyze_simulations.py
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
STEP_SIZE   = 0.22
PLOTS_DIR   = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})

MODEL_COLORS = {
    "M0-Reactive":   "#E07B39",
    "M1-Symbolic":   "#4C7EBE",
    "M2-DST-Hybrid": "#27AE60",
}
CONFIG_COLORS = {"domestic": "#5B9BD5", "mix": "#ED7D31", "warehouse": "#A9373B"}
MODEL_ORDER   = ["M0-Reactive", "M1-Symbolic", "M2-DST-Hybrid"]
CONFIG_ORDER  = ["domestic", "mix", "warehouse"]

FILENAME_RE = re.compile(r'(model\d)_\w+_(domestic|mix|warehouse)_\d{8}_\d{4}\.xlsx$')
MODEL_MAP   = {'model0': 'M0-Reactive', 'model1': 'M1-Symbolic', 'model2': 'M2-DST-Hybrid'}

# ─── LOAD ────────────────────────────────────────────────────────────────────
def load_files():
    # Collect all matching files, then keep only the newest per (model, config)
    candidate = {}
    for fp in sorted(glob.glob("model*.xlsx")):
        m = FILENAME_RE.search(os.path.basename(fp))
        if not m:
            continue
        model  = MODEL_MAP.get(m.group(1), m.group(1))
        config = m.group(2)
        key    = (model, config)
        # mtime: keep the most recently written file for each key
        mtime  = os.path.getmtime(fp)
        if key not in candidate or mtime > candidate[key][1]:
            candidate[key] = (fp, mtime)

    records = []
    for (model, config), (fp, _) in sorted(candidate.items()):
        try:
            tele = pd.read_excel(fp, sheet_name='Telemetry')
            perf = pd.read_excel(fp, sheet_name='Performance_Report')
        except Exception as e:
            print(f"  ⚠  skip {fp}: {e}")
            continue
        tele['_model']  = model
        tele['_config'] = config
        perf_d = dict(zip(perf['Metric'].astype(str), perf['Result']))
        records.append(dict(model=model, config=config, file=fp, tele=tele, perf=perf_d))

    if not records:
        raise FileNotFoundError("No matching model*.xlsx files found in current directory.")

    models_found  = sorted({r['model']  for r in records})
    configs_found = sorted({r['config'] for r in records})
    print(f"Loaded {len(records)} files.")
    print(f"  Models : {models_found}")
    print(f"  Configs: {configs_found}")
    return records


def present(order, found):
    found_set = set(found)
    return [x for x in order if x in found_set]


def compute_run_stats(records):
    rows = []
    for r in records:
        for run_id, rdf in r['tele'].groupby('Run'):
            frames  = int(rdf['Frame'].max())
            battery = float(rdf['Bat'].iloc[-1])
            spills  = int(rdf['Spill'].sum())
            cols    = int(rdf['Collision'].sum())
            mode    = str(rdf['Mode'].iloc[-1])
            rs      = 30.0 if mode == 'Warehouse' else 10.0
            dist    = np.sqrt(2) * (rs - 2.0)
            ideal_f = dist / (STEP_SIZE * 0.85)
            tf      = max(0.2, 1.3 - 0.30 * frames / ideal_f)
            bs      = 100 if (spills == 0 and cols == 0) else max(0, 100 - spills * 15 - cols * 100)
            se      = max(0, bs * tf * battery / 100.0)
            rows.append(dict(model=r['model'], config=r['config'], run=run_id,
                             mode=mode, frames=frames, battery=battery,
                             spills=spills, collisions=cols, se=se))
    return pd.DataFrame(rows)


def save_fig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ {path}")


def grouped_bar(ax, data_df, metric, models, configs, ylabel, title, fmt='.1f'):
    bar_w = 0.22
    x = np.arange(len(configs))
    offsets = [(i - (len(models) - 1) / 2) * (bar_w + 0.05) for i in range(len(models))]
    for i, model in enumerate(models):
        vals, errs = [], []
        for config in configs:
            sub = data_df[(data_df['model'] == model) & (data_df['config'] == config)][metric]
            vals.append(sub.mean() if len(sub) else 0)
            errs.append(sub.std()  if len(sub) > 1 else 0)
        bars = ax.bar(x + offsets[i], vals, bar_w, yerr=errs, capsize=3,
                      label=model, color=MODEL_COLORS.get(model, 'gray'),
                      edgecolor='white', linewidth=0.5, alpha=0.9,
                      error_kw={'linewidth': 1.2, 'capthick': 1.2})
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(errs) * 0.1 + 0.3,
                    f'{v:{fmt}}', ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in configs])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)


def scatter_legend(ax, models, configs):
    model_patches = [mpatches.Patch(color=MODEL_COLORS.get(m, 'gray'), label=m)
                     for m in models]
    shape_map = {'domestic': 'o', 'mix': 's', 'warehouse': '^'}
    shape_artists = [plt.Line2D([0], [0], marker=shape_map.get(c, 'o'),
                                color='dimgray', label=c.capitalize(),
                                linestyle='None', markersize=7)
                     for c in configs]
    ax.legend(handles=model_patches + shape_artists, fontsize=7.5, ncol=2)


# ─── FIG 1: PERFORMANCE OVERVIEW ─────────────────────────────────────────────
def fig_performance(rs):
    models  = present(MODEL_ORDER,  rs['model'].unique())
    configs = present(CONFIG_ORDER, rs['config'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Performance Overview — M0-Reactive vs M1-Symbolic vs M2-DST-Hybrid",
                 fontsize=13, fontweight='bold')

    grouped_bar(axes[0, 0], rs, 'se',       models, configs, "SE Score (0–100)",   "Avg Subjective Experience (SE) Score")
    grouped_bar(axes[0, 1], rs, 'spills',   models, configs, "Spill count per run", "Avg Spills per Run")
    grouped_bar(axes[1, 0], rs, 'frames',   models, configs, "Frames",              "Avg Frames to Goal (lower=faster)", fmt='.0f')
    grouped_bar(axes[1, 1], rs, 'battery',  models, configs, "Battery %",           "Avg Battery Remaining at Goal", fmt='.1f')

    plt.tight_layout()
    save_fig(fig, "fig1_performance_overview.png")


# ─── FIG 2: SAFETY ANALYSIS ──────────────────────────────────────────────────
def fig_safety(records, rs):
    tele    = pd.concat([r['tele'] for r in records], ignore_index=True)
    models  = present(MODEL_ORDER,  tele['_model'].unique())
    configs = present(CONFIG_ORDER, tele['_config'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Safety Analysis — Spill Patterns & Risk Factors",
                 fontsize=13, fontweight='bold')

    # (a) Spill cause: speed vs jerk stacked bar
    ax = axes[0, 0]
    speed_c, jerk_c = [], []
    for m in models:
        mdf = tele[(tele['_model'] == m) & (tele['Spill'] == 1)]
        speed_c.append((mdf['Spill_Cause'] == 'speed').sum())
        jerk_c.append((mdf['Spill_Cause'] == 'jerk').sum())
    x = np.arange(len(models))
    ax.bar(x, speed_c, 0.45, label='Speed-caused',
           color='#C0392B', alpha=0.85, edgecolor='white')
    ax.bar(x, jerk_c, 0.45, bottom=speed_c, label='Jerk-caused',
           color='#E67E22', alpha=0.85, edgecolor='white')
    for i, (s, j) in enumerate(zip(speed_c, jerk_c)):
        tot = s + j
        if tot > 0:
            ax.text(i, tot + 0.2, str(tot), ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_title("Total Spill Count by Cause & Model", fontweight='bold')
    ax.set_ylabel("Number of spill events"); ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # (b) p_spill per-frame distribution (box, log scale)
    ax = axes[0, 1]
    data = [tele[tele['_model'] == m]['p_spill'].dropna().values for m in models]
    bp = ax.boxplot(data, labels=models, patch_artist=True,
                    medianprops={'color': 'black', 'lw': 2}, showfliers=False)
    for patch, model in zip(bp['boxes'], models):
        patch.set_facecolor(MODEL_COLORS.get(model, 'gray')); patch.set_alpha(0.75)
    ax.set_title("Spill Probability Distribution (per frame)", fontweight='bold')
    ax.set_ylabel("p_spill"); ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3); ax.tick_params(axis='x', labelsize=9)
    ax.set_xticklabels(models)

    # (c) Localization error vs spill rate scatter (per file)
    ax = axes[1, 0]
    shape_map = {'domestic': 'o', 'mix': 's', 'warehouse': '^'}
    for r in records:
        df = r['tele']
        loc_err  = df['Loc_Error'].mean()
        spill_rt = df['Spill'].mean() * 100
        ax.scatter(loc_err, spill_rt,
                   color=MODEL_COLORS.get(r['model'], 'gray'),
                   marker=shape_map.get(r['config'], 'o'),
                   s=130, zorder=3, edgecolors='white', linewidth=1.2)
    ax.set_xlabel("Avg Localisation Error (m)")
    ax.set_ylabel("Spill Rate (% of frames)")
    ax.set_title("Localisation Error vs Spill Rate\n(color=model, shape=config)",
                 fontweight='bold')
    scatter_legend(ax, models, configs)
    ax.grid(alpha=0.3)

    # (d) In-Avoidance rate: during spills vs normal frames
    ax = axes[1, 1]
    av_normal, av_spill = [], []
    for m in models:
        mdf = tele[tele['_model'] == m]
        sf  = mdf[mdf['Spill'] == 1]
        nf  = mdf[mdf['Spill'] == 0]
        av_normal.append(nf['In_Avoidance'].mean() * 100 if len(nf) else 0)
        av_spill.append(sf['In_Avoidance'].mean()  * 100 if len(sf) else 0)
    x = np.arange(len(models))
    ax.bar(x - 0.2, av_normal, 0.35, label='Normal frames',
           color='#2980B9', alpha=0.85, edgecolor='white')
    ax.bar(x + 0.2, av_spill,  0.35, label='Spill frames',
           color='#C0392B', alpha=0.85, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_title("In-Avoidance Rate: Normal vs Spill Frames", fontweight='bold')
    ax.set_ylabel("% frames in avoidance mode")
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "fig2_safety_analysis.png")


# ─── FIG 3: BEHAVIORAL PROFILES ──────────────────────────────────────────────
def fig_behavior(records):
    tele   = pd.concat([r['tele'] for r in records], ignore_index=True)
    models = present(MODEL_ORDER, tele['_model'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Behavioral Profiles — Physical Characteristics per Model",
                 fontsize=13, fontweight='bold')

    def box_by_model(ax, col, title, ylabel, log=False):
        data = [tele[tele['_model'] == m][col].dropna().values for m in models]
        bp = ax.boxplot(data, labels=models, patch_artist=True,
                        medianprops={'color': 'black', 'lw': 2}, showfliers=False)
        for patch, model in zip(bp['boxes'], models):
            patch.set_facecolor(MODEL_COLORS.get(model, 'gray')); patch.set_alpha(0.75)
        if log:
            ax.set_yscale('log')
        ax.set_title(title, fontweight='bold'); ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3); ax.tick_params(axis='x', labelsize=9)
        ax.set_xticklabels(models)

    box_by_model(axes[0, 0], 'Speed', "Speed Distribution (m/s · step)", "Actual Speed")
    box_by_model(axes[0, 1], 'Jerk',  "Jerk Distribution (smoothness)",  "Jerk magnitude", log=True)
    box_by_model(axes[1, 0], 'CPU',   "CPU Load per Frame",               "CPU %")

    # (d) Battery depletion: mean ± std over normalised progress
    ax = axes[1, 1]
    n_bins = 20
    bin_edges   = np.linspace(0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for model in models:
        run_curves = []
        for r in records:
            if r['model'] != model:
                continue
            for _, rdf in r['tele'].groupby('Run'):
                rdf = rdf.sort_values('Frame')
                max_f = rdf['Frame'].max()
                if max_f == 0:
                    continue
                norm = rdf['Frame'].values / max_f
                bat  = rdf['Bat'].values
                run_curves.append(np.interp(bin_centers, norm, bat))
        if not run_curves:
            continue
        arr  = np.array(run_curves)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        color = MODEL_COLORS.get(model, 'gray')
        ax.plot(bin_centers, mean, color=color, lw=2.5, label=model)
        ax.fill_between(bin_centers, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Normalised journey progress (0 = start, 1 = goal)")
    ax.set_ylabel("Battery %")
    ax.set_title("Battery Depletion Profile (mean ± std)", fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "fig3_behavior_profiles.png")


# ─── FIG 4: M2 DST DYNAMICS ──────────────────────────────────────────────────
def fig_dst_dynamics(records):
    m2 = [r for r in records if r['model'] == 'M2-DST-Hybrid']
    if not m2:
        print("  ⓘ  No M2-DST-Hybrid files — skipping fig 4.")
        return

    configs = present(CONFIG_ORDER, [r['config'] for r in m2])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("M2-DST-Hybrid: Adaptive Mechanism — Belief, Resolution & Policy Switching",
                 fontsize=13, fontweight='bold')

    # (a) Resolution over frame (thin per-run, thick per-config mean)
    ax = axes[0, 0]
    for r in m2:
        df = r['tele']
        if 'Resolution' not in df.columns:
            continue
        for _, rdf in df.groupby('Run'):
            rdf = rdf.sort_values('Frame')
            ax.plot(rdf['Frame'], rdf['Resolution'],
                    color=CONFIG_COLORS.get(r['config'], 'gray'), alpha=0.2, lw=0.7)
    for config in configs:
        cdf = pd.concat([r['tele'] for r in m2 if r['config'] == config], ignore_index=True)
        if 'Resolution' not in cdf.columns:
            continue
        mean_r = cdf.groupby('Frame')['Resolution'].mean()
        ax.plot(mean_r.index, mean_r.values,
                color=CONFIG_COLORS.get(config, 'gray'), lw=2.5, label=config.capitalize())
    ax.axhline(0.10, color=MODEL_COLORS['M0-Reactive'], lw=1.2, linestyle='--',
               alpha=0.7, label='M0 target (0.10)')
    ax.axhline(0.05, color=MODEL_COLORS['M1-Symbolic'], lw=1.2, linestyle='--',
               alpha=0.7, label='M1 target (0.05)')
    ax.set_title("Grid Resolution Switching over Time", fontweight='bold')
    ax.set_xlabel("Frame"); ax.set_ylabel("Resolution (m/cell)")
    ax.set_ylim(0.03, 0.12); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) DST Plausibility_H over frame
    ax = axes[0, 1]
    has_plh = any('Pl_H' in r['tele'].columns for r in m2)
    if has_plh:
        for r in m2:
            df = r['tele']
            if 'Pl_H' not in df.columns:
                continue
            for _, rdf in df.groupby('Run'):
                rdf = rdf.sort_values('Frame')
                ax.plot(rdf['Frame'], rdf['Pl_H'],
                        color=CONFIG_COLORS.get(r['config'], 'gray'), alpha=0.2, lw=0.7)
        for config in configs:
            cdf = pd.concat([r['tele'] for r in m2 if r['config'] == config], ignore_index=True)
            if 'Pl_H' not in cdf.columns:
                continue
            mean_pl = cdf.groupby('Frame')['Pl_H'].mean()
            ax.plot(mean_pl.index, mean_pl.values,
                    color=CONFIG_COLORS.get(config, 'gray'), lw=2.5, label=config.capitalize())
    ax.set_title("DST Plausibility of 'Safe' Hypothesis (Pl_H)", fontweight='bold')
    ax.set_xlabel("Frame"); ax.set_ylabel("Plausibility_H (0=danger, 1=safe)")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) Active policy fraction: M0 vs M1 per config (stacked bar)
    ax = axes[1, 0]
    m0_frac, m1_frac = [], []
    for config in configs:
        cdf = pd.concat([r['tele'] for r in m2 if r['config'] == config], ignore_index=True)
        if 'Active_Model' not in cdf.columns:
            m0_frac.append(50); m1_frac.append(50)
            continue
        m0 = cdf['Active_Model'].str.contains('REACTIVE', na=False).mean() * 100
        m0_frac.append(m0); m1_frac.append(100 - m0)
    x = np.arange(len(configs))
    b0 = ax.bar(x, m0_frac, 0.5, label='M0-REACTIVE (fast)',
                color=MODEL_COLORS['M0-Reactive'], alpha=0.85, edgecolor='white')
    b1 = ax.bar(x, m1_frac, 0.5, bottom=m0_frac, label='M1-SYMBOLIC (safe)',
                color=MODEL_COLORS['M1-Symbolic'], alpha=0.85, edgecolor='white')
    for i, (m0v, m1v) in enumerate(zip(m0_frac, m1_frac)):
        if m0v > 8:
            ax.text(i, m0v / 2, f'{m0v:.0f}%', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
        if m1v > 8:
            ax.text(i, m0v + m1v / 2, f'{m1v:.0f}%', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in configs])
    ax.set_title("Active Policy Time Fraction (M2-DST)", fontweight='bold')
    ax.set_ylabel("% of frames"); ax.set_ylim(0, 108)
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # (d) Adaptive fuzzy threshold over frames
    ax = axes[1, 1]
    has_ft = any('Fuzzy_Threshold' in r['tele'].columns for r in m2)
    if has_ft:
        for r in m2:
            df = r['tele']
            if 'Fuzzy_Threshold' not in df.columns:
                continue
            for _, rdf in df.groupby('Run'):
                rdf = rdf.sort_values('Frame')
                ax.plot(rdf['Frame'], rdf['Fuzzy_Threshold'],
                        color=CONFIG_COLORS.get(r['config'], 'gray'), alpha=0.25, lw=0.9)
        for config in configs:
            cdf = pd.concat([r['tele'] for r in m2 if r['config'] == config], ignore_index=True)
            if 'Fuzzy_Threshold' not in cdf.columns:
                continue
            mean_ft = cdf.groupby('Frame')['Fuzzy_Threshold'].mean()
            ax.plot(mean_ft.index, mean_ft.values,
                    color=CONFIG_COLORS.get(config, 'gray'), lw=2.5, label=config.capitalize())
        patches = [mpatches.Patch(color=CONFIG_COLORS.get(c, 'gray'), label=c.capitalize())
                   for c in configs]
        ax.legend(handles=patches, fontsize=8)
    ax.axhline(0.5, color='gray', lw=1, linestyle='--', alpha=0.6, label='Default threshold')
    ax.set_title("Adaptive Fuzzy Threshold Evolution", fontweight='bold')
    ax.set_xlabel("Frame"); ax.set_ylabel("Switching Threshold")
    ax.set_ylim(0.2, 0.8); ax.grid(alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "fig4_dst_dynamics.png")


# ─── FIG 5: EFFICIENCY & VARIANCE ────────────────────────────────────────────
def fig_efficiency(rs):
    models  = present(MODEL_ORDER,  rs['model'].unique())
    configs = present(CONFIG_ORDER, rs['config'].unique())
    shape_map = {'domestic': 'o', 'mix': 's', 'warehouse': '^'}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Efficiency & Run-Level Variance — All Configurations",
                 fontsize=13, fontweight='bold')

    # (a) Per-run SE distribution: box + strip per model (all configs pooled)
    ax = axes[0, 0]
    rng = np.random.default_rng(42)
    for i, model in enumerate(models):
        vals = rs[rs['model'] == model]['se'].values
        bp = ax.boxplot(vals, positions=[i], widths=0.45, patch_artist=True,
                        medianprops={'color': 'black', 'lw': 2.5}, showfliers=False)
        bp['boxes'][0].set_facecolor(MODEL_COLORS.get(model, 'gray'))
        bp['boxes'][0].set_alpha(0.55)
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(i + jitter, vals, color=MODEL_COLORS.get(model, 'gray'),
                   s=35, alpha=0.85, zorder=4, edgecolors='white', linewidth=0.6)
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, fontsize=9)
    ax.set_title("SE Score per Run (all configs pooled)", fontweight='bold')
    ax.set_ylabel("SE Score"); ax.grid(axis='y', alpha=0.3)

    # (b) Scatter: frames vs SE (speed-safety frontier)
    ax = axes[0, 1]
    for _, row in rs.iterrows():
        ax.scatter(row['frames'], row['se'],
                   color=MODEL_COLORS.get(row['model'], 'gray'),
                   marker=shape_map.get(row['config'], 'o'),
                   s=80, alpha=0.8, edgecolors='white', linewidth=0.8)
    scatter_legend(ax, models, configs)
    ax.set_xlabel("Frames to Goal  (fewer = faster)")
    ax.set_ylabel("SE Score  (higher = better)")
    ax.set_title("Speed–Safety Efficiency Frontier", fontweight='bold')
    ax.grid(alpha=0.3)

    # (c) SE by config (environment difficulty)
    ax = axes[1, 0]
    grouped_bar(ax, rs, 'se', models, configs,
                "SE Score (0–100)",
                "SE Score by Environment (mean ± std)\n"
                "↓ difficulty reveals adaptive advantage")

    # (d) Normalised trade-off spider (as grouped bar)
    ax = axes[1, 1]
    metric_labels = ['SE\n(norm)', 'Safety\n(1 – spill_rate)', 'Speed\n(1 – norm_frames)', 'Battery\n(remaining)']
    se_max   = rs['se'].max()    or 1
    fr_range = max(rs['frames'].max() - rs['frames'].min(), 1)
    fr_min   = rs['frames'].min()
    sp_max   = rs['spills'].max() or 1
    bt_range = max(rs['battery'].max() - rs['battery'].min(), 1)
    bt_min   = rs['battery'].min()

    bar_w   = 0.22
    x       = np.arange(len(metric_labels))
    offsets = [(i - (len(models) - 1) / 2) * (bar_w + 0.05) for i in range(len(models))]
    for i, model in enumerate(models):
        mdf = rs[rs['model'] == model]
        v = [
            mdf['se'].mean()       / se_max,
            1.0 - mdf['spills'].mean() / sp_max,
            1.0 - (mdf['frames'].mean() - fr_min) / fr_range,
            (mdf['battery'].mean() - bt_min) / bt_range,
        ]
        bars = ax.bar(x + offsets[i], v, bar_w, label=model,
                      color=MODEL_COLORS.get(model, 'gray'),
                      edgecolor='white', linewidth=0.5, alpha=0.9)
        for bar, val in zip(bars, v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    ax.axhline(1.0, color='gray', lw=0.8, linestyle='--', alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylim(0, 1.25)
    ax.set_title("Normalised Multi-Metric Trade-off (1.0 = best)", fontweight='bold')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "fig5_efficiency_variance.png")


# ─── CONSOLE SUMMARY ─────────────────────────────────────────────────────────
def print_summary(rs):
    print("\n" + "=" * 65)
    print("  SUMMARY TABLE — Avg SE | Spills | Frames | Battery")
    print("=" * 65)
    pivot = rs.groupby(['model', 'config']).agg(
        SE=('se', 'mean'), Spills=('spills', 'mean'),
        Frames=('frames', 'mean'), Battery=('battery', 'mean')
    ).round(1)
    print(pivot.to_string())
    print("=" * 65)


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  Robot Simulation Analysis")
    print("=" * 55)

    records  = load_files()
    tele_all = pd.concat([r['tele'] for r in records], ignore_index=True)
    rs       = compute_run_stats(records)

    n_runs   = len(rs)
    n_frames = len(tele_all)
    print(f"  {n_runs} runs · {n_frames:,} telemetry frames total\n")

    print("→ Fig 1: Performance overview")
    fig_performance(rs)

    print("→ Fig 2: Safety analysis")
    fig_safety(records, rs)

    print("→ Fig 3: Behavioral profiles")
    fig_behavior(records)

    print("→ Fig 4: M2 DST adaptive dynamics")
    fig_dst_dynamics(records)

    print("→ Fig 5: Efficiency & variance")
    fig_efficiency(rs)

    print_summary(rs)
    print(f"\n✅ All plots saved to ./{PLOTS_DIR}/\n")
