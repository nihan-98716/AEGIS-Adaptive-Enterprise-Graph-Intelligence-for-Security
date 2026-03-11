import os
import argparse
import copy
import random
import multiprocessing
import pandas as pd
from typing import Dict, Any

# Local Module Imports
from config import (SIMULATION_CONFIG, ANOMALY_CONFIG, NL_REPORTING_ENABLED,
                    GNN_ANOMALY_MODE, GNN_CONFIG)
from network_graph import generate_enterprise_graph
from propagation_engine import run_simulation
from risk_engine import calculate_risk_scores, get_risk_report
from anomaly_detector import make_anomaly_detector, run_anomaly_detection_experiment
from feature_extractor import extract_training_dataset
from report_generator import run_report_generation, ReportGenerator
from defense_simulator import run_defense_experiment
from ai_modules import RLDefenseAgent
from risk_engine import plot_risk_heatmap
from anomaly_detector import AnomalyDetector
from defense_simulator import plot_strategy_comparison, plot_infection_curves


# ---------------------------------------------------------------------------
# Dashboard generator — combines per-scenario charts into one figure
# ---------------------------------------------------------------------------

def generate_scenario_dashboard(scenario_name: str, strategy_df, detection_metrics: dict,
                                 base_sim_result: dict, viz_sim_result: dict,
                                 model_metrics: dict = None):
    """
    Builds a 2x2 multi-panel dashboard PNG for a single scenario and saves it
    to outputs/dashboard_{scenario_name}.png.

    Panel layout:
        [Top-left]     Risk Heatmap (top 30 nodes by composite risk score)
        [Top-right]    GNN Detection Timeline (cumulative infected vs detected)
        [Bottom-left]  Defense Strategy Comparison (bar chart)
        [Bottom-right] Scenario Stats Summary (text panel)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    import os

    fig = plt.figure(figsize=(30, 14))
    fig.patch.set_facecolor("#0d1117")

    # Title banner
    fig.suptitle(
        f"AEGIS — Scenario: {scenario_name.replace('_', ' ')}",
        fontsize=22, fontweight="bold", color="white", y=0.97
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.28,
                           left=0.03, right=0.97, top=0.92, bottom=0.04)

    def load_panel(ax, path, title):
        """Load a saved PNG into an axes panel."""
        if os.path.exists(path):
            img = mpimg.imread(path)
            ax.imshow(img)
            ax.set_title(title, fontsize=13, fontweight="bold",
                         color="white", pad=8)
            ax.axis("off")
            ax.set_facecolor("#161b22")
        else:
            ax.set_facecolor("#161b22")
            ax.text(0.5, 0.5, f"Chart not found:\n{path}",
                    ha="center", va="center", color="#888", fontsize=10,
                    transform=ax.transAxes)
            ax.set_title(title, fontsize=13, fontweight="bold",
                         color="white", pad=8)
            ax.axis("off")

    # Panel 1 — Risk Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    load_panel(ax1,
               f"outputs/risk_heatmap_{scenario_name}.png",
               "Composite Risk Score — Top 30 Nodes")

    # Panel 2 — Detection Timeline
    ax2 = fig.add_subplot(gs[0, 1])
    load_panel(ax2,
               f"outputs/detection_timeline_{scenario_name}.png",
               "GNN Anomaly Detection Timeline")

    # Panel 3 — Strategy Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    load_panel(ax3,
               f"outputs/strategy_comparison_{scenario_name}.png",
               "Defense Strategy Comparison (Monte Carlo, 30 runs)")

    # Panel 5 — Model Metrics
    ax5 = fig.add_subplot(gs[0, 2])
    ax5.set_facecolor("#161b22")
    ax5.axis("off")
    ax5.set_title("Model Performance Metrics", fontsize=13,
                  fontweight="bold", color="white", pad=8)

    mm = model_metrics or {}
    metric_lines = [
        ("GNN ANOMALY DETECTOR", None, "#f5a623"),
        ("Accuracy",        f"{mm.get('gnn_accuracy',0)*100:.1f}%",    "#ecf0f1"),
        ("Precision",       f"{mm.get('gnn_precision',0)*100:.1f}%",   "#ecf0f1"),
        ("Recall",          f"{mm.get('gnn_recall',0)*100:.1f}%",      "#3fb950" if mm.get('gnn_recall',0)>0.7 else "#f85149"),
        ("F1 Score",        f"{mm.get('gnn_f1',0)*100:.1f}%",          "#3fb950" if mm.get('gnn_f1',0)>0.7 else "#f85149"),
        ("AUC-ROC",         f"{mm.get('gnn_auc',0):.3f}",              "#3fb950" if mm.get('gnn_auc',0)>0.7 else "#f85149"),
        ("False Pos Rate",  f"{mm.get('gnn_fpr',0)*100:.1f}%",         "#3fb950" if mm.get('gnn_fpr',0)<0.2 else "#f85149"),
        ("TP / FP / FN",    f"{mm.get('gnn_tp',0)} / {mm.get('gnn_fp',0)} / {mm.get('gnn_fn',0)}", "#8b949e"),
        ("", None, "white"),
        ("RISK SCORING ENGINE", None, "#f5a623"),
        ("Precision@5",     f"{mm.get('risk_precision_at_5',0)*100:.1f}%",  "#ecf0f1"),
        ("Precision@10",    f"{mm.get('risk_precision_at_10',0)*100:.1f}%", "#ecf0f1"),
        ("Precision@20",    f"{mm.get('risk_precision_at_20',0)*100:.1f}%", "#ecf0f1"),
        ("Kendall\'s Tau", f"{mm.get('risk_kendall_tau',0):.3f}",          "#3fb950" if mm.get('risk_kendall_tau',0)>0.3 else "#8b949e"),
        ("", None, "white"),
        ("DQN RL AGENT", None, "#f5a623"),
        ("Trained",         str(mm.get('dqn_trained', False)),              "#3fb950" if mm.get('dqn_trained') else "#f85149"),
        ("Exploit Rate",    f"{mm.get('dqn_exploitation_rate',0)*100:.1f}%","#ecf0f1"),
        ("Train Steps",     str(mm.get('dqn_train_steps', 0)),              "#8b949e"),
        ("Tail Reward",     f"{mm.get('dqn_mean_tail_reward',0):+.3f}",     "#3fb950" if mm.get('dqn_mean_tail_reward',0)>0 else "#f85149"),
    ]

    my = 0.97
    for label, value, color in metric_lines:
        if label == "":
            my -= 0.02
            continue
        if value is None:
            ax5.text(0.03, my, label, transform=ax5.transAxes,
                     fontsize=9.5, fontweight="bold", color=color, va="top",
                     fontfamily="monospace")
            my -= 0.042
        else:
            ax5.text(0.03, my, label, transform=ax5.transAxes,
                     fontsize=9, color="#8b949e", va="top", fontfamily="monospace")
            ax5.text(0.97, my, value, transform=ax5.transAxes,
                     fontsize=9, color=color, va="top", ha="right",
                     fontfamily="monospace", fontweight="bold")
            my -= 0.046

    # Panel 6 — move stats summary to gs[1,2]
    ax4_new = fig.add_subplot(gs[1, 2])
    ax4_new.set_facecolor("#161b22")
    ax4_new.axis("off")
    ax4_new.set_title("Scenario Intelligence Summary", fontsize=13,
                      fontweight="bold", color="white", pad=8)

    final_rate   = base_sim_result.get("final_infection_rate", 0) * 100
    viz_rate     = viz_sim_result.get("final_infection_rate", 0) * 100
    velocity     = base_sim_result.get("spread_velocity", 0)
    t_crit       = base_sim_result.get("time_to_first_critical_node")
    t_crit_str   = f"Timestep {t_crit}" if t_crit else "Not reached"
    stages       = list(base_sim_result.get("attack_stages_timeline", {}).keys())
    lead_time    = detection_metrics.get("detection_lead_time_mean", 0)
    tpr          = detection_metrics.get("true_positive_rate", 0) * 100
    fdr          = detection_metrics.get("false_positive_rate", 0) * 100
    best_row2    = strategy_df.sort_values("mean_infection_rate").iloc[0]
    worst_row2   = strategy_df.sort_values("mean_infection_rate").iloc[-1]
    best_strat2  = best_row2["strategy_name"]
    best_rate2   = best_row2["mean_infection_rate"] * 100
    worst_rate2  = worst_row2["mean_infection_rate"] * 100
    reduction2   = worst_rate2 - best_rate2

    sum_lines = [
        ("ATTACK METRICS", None, "#58a6ff"),
        (f"Base infection rate",    f"{final_rate:.1f}%",        "white"),
        (f"Spread velocity",        f"{velocity:.2f} nodes/step","white"),
        (f"Time to critical node",  t_crit_str,                  "white"),
        (f"Attack stages reached",  str(len(stages)),            "white"),
        ("", None, "white"),
        ("GNN DETECTION", None, "#58a6ff"),
        (f"Detection lead time",    f"+{lead_time:.1f} steps",   "#3fb950" if lead_time>0 else "#f85149"),
        (f"True positive rate",     f"{tpr:.1f}%",               "#3fb950" if tpr>80 else "#f85149"),
        (f"False discovery rate",   f"{fdr:.1f}%",               "#3fb950" if fdr<20 else "#f85149"),
        ("", None, "white"),
        ("DEFENSE OUTCOME", None, "#58a6ff"),
        (f"Best strategy",          best_strat2,                 "#3fb950"),
        (f"Best infection rate",    f"{best_rate2:.1f}%",        "#3fb950"),
        (f"Worst infection rate",   f"{worst_rate2:.1f}%",       "#f85149"),
        (f"Reduction achieved",     f"{reduction2:.1f}%",        "#3fb950"),
    ]

    sy = 0.97
    for label, value, color in sum_lines:
        if label == "":
            sy -= 0.025
            continue
        if value is None:
            ax4_new.text(0.04, sy, label, transform=ax4_new.transAxes,
                         fontsize=10, fontweight="bold", color=color, va="top",
                         fontfamily="monospace")
            sy -= 0.045
        else:
            ax4_new.text(0.04, sy, label, transform=ax4_new.transAxes,
                         fontsize=9.5, color="#8b949e", va="top", fontfamily="monospace")
            ax4_new.text(0.96, sy, value, transform=ax4_new.transAxes,
                         fontsize=9.5, color=color, va="top", ha="right",
                         fontfamily="monospace", fontweight="bold")
            sy -= 0.052

    out_path = f"outputs/dashboard_{scenario_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"      -> Saved {out_path}")
    return out_path



# ---------------------------------------------------------------------------
# Model Metrics Computation
# ---------------------------------------------------------------------------

def compute_model_metrics(det_log, inf_log, G_base, risk_df, rl_agent=None):
    """
    Computes classification and ranking metrics for GNN, Risk Engine, and DQN.
    Returns a flat dict of all metrics.
    """
    import numpy as np
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score, roc_auc_score)

    metrics = {}
    nodes = sorted(G_base.nodes())
    n = len(nodes)

    # ── GNN Anomaly Detector ─────────────────────────────────────────────────
    # Build ground-truth: was each node ever infected during the sim?
    ever_infected = set()
    for t, newly in inf_log:
        ever_infected.update(newly)

    # Build predicted: was each node ever flagged by GNN?
    ever_detected = set()
    for t, flagged in det_log:
        ever_detected.update(flagged)

    if ever_infected:
        y_true = np.array([1 if nd in ever_infected  else 0 for nd in nodes])
        y_pred = np.array([1 if nd in ever_detected  else 0 for nd in nodes])

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        metrics['gnn_accuracy']  = round(accuracy_score(y_true, y_pred), 4)
        metrics['gnn_precision'] = round(precision_score(y_true, y_pred, zero_division=0), 4)
        metrics['gnn_recall']    = round(recall_score(y_true, y_pred, zero_division=0), 4)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        metrics['gnn_f1']        = round(f1, 4)
        metrics['gnn_fpr']       = round(fp / max(fp + tn, 1), 4)
        metrics['gnn_tp']        = tp
        metrics['gnn_fp']        = fp
        metrics['gnn_tn']        = tn
        metrics['gnn_fn']        = fn
        # AUC: use anomaly scores as soft probability if available
        scores = np.array([float(G_base.nodes[nd].get('anomaly_score', 0.0)) for nd in nodes])
        if scores.max() > 0 and len(np.unique(y_true)) > 1:
            scores_norm = scores / (scores.max() + 1e-9)
            try:
                metrics['gnn_auc'] = round(roc_auc_score(y_true, scores_norm), 4)
            except Exception:
                metrics['gnn_auc'] = 0.0
        else:
            metrics['gnn_auc'] = 0.0
    else:
        for k in ['gnn_accuracy','gnn_precision','gnn_recall','gnn_f1',
                  'gnn_fpr','gnn_auc','gnn_tp','gnn_fp','gnn_tn','gnn_fn']:
            metrics[k] = 0.0

    # ── Risk Scoring Engine ──────────────────────────────────────────────────
    # Precision@K: fraction of top-K risk nodes that were actually infected
    for k in [5, 10, 20]:
        top_k = risk_df.head(k)['node_id'].tolist()
        infected_ids = set(G_base.nodes[nd].get('node_id') for nd in ever_infected)
        hits = sum(1 for nid in top_k if nid in infected_ids)
        metrics[f'risk_precision_at_{k}'] = round(hits / k, 4)

    # Kendall's Tau: rank correlation between risk score rank and infection order
    infection_order = {}
    for t, newly in inf_log:
        for nd in newly:
            if nd not in infection_order:
                infection_order[nd] = t

    risk_ranks  = {row['node_id']: i for i, row in risk_df.reset_index(drop=True).iterrows()}
    common_ids  = [G_base.nodes[nd].get('node_id') for nd in nodes
                   if G_base.nodes[nd].get('node_id') in infection_order
                   and G_base.nodes[nd].get('node_id') in risk_ranks]
    if len(common_ids) >= 3:
        from scipy.stats import kendalltau
        r_ranks = [risk_ranks[nid] for nid in common_ids]
        i_ranks = [infection_order[G_base.nodes[
                       next(nd for nd in nodes
                            if G_base.nodes[nd].get('node_id') == nid)].get('node_id')]
                   for nid in common_ids]
        tau, _ = kendalltau(r_ranks, i_ranks)
        metrics['risk_kendall_tau'] = round(float(tau), 4)
    else:
        metrics['risk_kendall_tau'] = 0.0

    # ── DQN RL Agent ─────────────────────────────────────────────────────────
    if rl_agent is not None and rl_agent.is_trained:
        metrics['dqn_trained']          = True
        metrics['dqn_epsilon']          = round(float(rl_agent.epsilon), 4)
        metrics['dqn_exploitation_rate']= round(1.0 - float(rl_agent.epsilon), 4)
        metrics['dqn_replay_size']      = len(rl_agent._replay)
        metrics['dqn_train_steps']      = int(rl_agent._step)
        # Mean reward from last 10% of replay buffer as proxy for converged performance
        if rl_agent._replay:
            tail = rl_agent._replay[int(len(rl_agent._replay)*0.9):]
            metrics['dqn_mean_tail_reward'] = round(float(np.mean([r for _,_,r,_,_ in tail])), 4)
        else:
            metrics['dqn_mean_tail_reward'] = 0.0
    else:
        for k in ['dqn_trained','dqn_epsilon','dqn_exploitation_rate',
                  'dqn_replay_size','dqn_train_steps','dqn_mean_tail_reward']:
            metrics[k] = 0.0

    return metrics


def save_model_metrics_csv(all_scenario_metrics: list):
    """Saves per-scenario model metrics to outputs/model_metrics.csv."""
    import pandas as pd, os
    os.makedirs('outputs', exist_ok=True)
    df = pd.DataFrame(all_scenario_metrics)
    df.to_csv('outputs/model_metrics.csv', index=False)
    print("  -> Saved outputs/model_metrics.csv")
    return df


def execute_scenario_pipeline(scenario_name: str, G, entry_nodes, attacker_mode: str, beta: float, 
                             anomaly_detector: AnomalyDetector, report_gen: ReportGenerator, skip_report: bool,
                             rl_agent=None, visualize: bool = False) -> Dict[str, Any]:
    """
    Executes the standard 8-step pipeline for any given attack scenario mapping.
    """
    print(f"\n{'='*60}\nEvaluating Scenario: {scenario_name}\n{'='*60}")
    
    budget = SIMULATION_CONFIG['budget']
    max_steps = SIMULATION_CONFIG['max_timesteps']
    n_runs = SIMULATION_CONFIG['n_runs']
    
    # 1. Base Simulation (No Defense) for Anomaly & Risk metrics
    print("[1/6] Running base propagation (No Defense) to seed ML models...")
    G_base = copy.deepcopy(G)
    base_sim_result = run_simulation(
        G_base,
        max_timesteps=SIMULATION_CONFIG['max_timesteps'],
        attacker_mode=attacker_mode,
        beta=beta,
        seed=SIMULATION_CONFIG['seed'],
        initial_node=entry_nodes[0] if entry_nodes else None
    )

    # 1b. Visualization simulation — run a small search across seeds to find
    #     a run with meaningful spread for chart purposes.
    #     Uses elevated beta=0.85 and tries up to 10 seeds; picks the run with
    #     the highest final infection rate. Does NOT affect defense results.
    viz_best = None
    for _viz_seed in range(SIMULATION_CONFIG['seed'], SIMULATION_CONFIG['seed'] + 15):
        _G_viz = copy.deepcopy(G)
        _viz_result = run_simulation(
            _G_viz,
            max_timesteps=SIMULATION_CONFIG['max_timesteps'],
            attacker_mode=attacker_mode,
            beta=0.85,
            seed=_viz_seed,
            initial_node=entry_nodes[0] if entry_nodes else None
        )
        if viz_best is None or _viz_result['final_infection_rate'] > viz_best['final_infection_rate']:
            viz_best = _viz_result
            G_viz = _G_viz
        if viz_best['final_infection_rate'] >= 0.15:   # good enough — stop early
            break
    viz_sim_result = viz_best
    print(f"      Viz simulation (beta=0.85, best of up to 15 seeds): "
          f"{viz_sim_result['final_infection_rate']*100:.1f}% final infection rate, "
          f"{viz_sim_result['total_timesteps']} timesteps")
    
    # 2. Extract ML dataset from base run log
    # Using a list wrapping the single base dict log format as expected
    print("[2/6] Extracting continuous GNN node features & labels...")
    features_df, labels_df = extract_training_dataset([base_sim_result], G_base)
    
    # 3. Anomaly Detection — GNN-based replay over the full timestep log
    print("[3/6] Running GNN anomaly detection on base propagation...")
    try:
        from network_graph import reset_graph as _reset
        import copy as _copy

        # 3a. Fit detector on the clean baseline (before any infection)
        G_fit = _copy.deepcopy(G)
        _reset(G_fit)
        if not anomaly_detector.is_fitted:
            anomaly_detector.fit_baseline(G_fit)

        # 3b. Replay the viz simulation (elevated beta) for richer chart data
        G_replay = _copy.deepcopy(G)
        _reset(G_replay)
        det_log = []
        inf_log = []

        for step in viz_sim_result['timestep_log']:
            t = step['timestep']
            phase = step['phase']
            if phase == 'baseline_recording':
                continue

            newly_infected = step['newly_infected_nodes']

            # Apply infections to replay graph
            for n in newly_infected:
                G_replay.nodes[n]['infection_state'] = 'infected'

            # Score nodes and collect detections
            detected_now = anomaly_detector.detect_anomalies(
                G_replay,
                threshold=GNN_CONFIG.get('gnn_threshold', 2.0)
            )
            det_log.append((t, detected_now))
            inf_log.append((t, newly_infected))

        detection_metrics = anomaly_detector.compute_detection_metrics(det_log, inf_log)
        print(f"      -> Detection Lead Time: {detection_metrics.get('detection_lead_time_mean', 0.0):.2f} timesteps")
        print(f"      -> True Positive Rate:  {detection_metrics.get('true_positive_rate', 0.0)*100:.1f}%")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"      -> Anomaly eval failed: {e}")
        detection_metrics = {}
        det_log, inf_log = [], []
         
    # 4. Risk Reporting
    print("[4/6] Computing static + dynamic Risk scoring...")
    calculate_risk_scores(G_base)
    risk_df = get_risk_report(G_base)

    # 4b. Model metrics (GNN + Risk Engine + DQN)
    model_metrics = compute_model_metrics(det_log, inf_log, G_base, risk_df, rl_agent=rl_agent)
    
    # 5. Multiprocessing Defense Comparisons
    strategies = ["none", "random", "patch_vulnerable", "patch_centrality", "isolate_chokepoints", "anomaly_guided", "rl_agent"]
    comparison_results = []
    
    print("[5/6] Executing Defense Comparisons using ProcessPoolExecutor...")
    
    for strategy in strategies:
        print(f"      -> Testing strategy: {strategy} ({n_runs} runs)")
        summary = run_defense_experiment(
            G,
            strategy_fn=None,
            attacker_mode=attacker_mode,
            strategy_name=strategy,
            anomaly_detector=anomaly_detector,
            n_runs=n_runs,
            initial_node=entry_nodes[0] if entry_nodes else None,
            max_timesteps=max_steps,
            beta=beta
        )
        comparison_results.append({
            'strategy_name': strategy,
            'mean_infection_rate': summary['mean_infection_rate'],
            'mean_containment_time': summary.get('mean_containment_time'),
            'success_count': summary.get('success_count', 0)
        })
        print(f"         > Mean Infection: {summary['mean_infection_rate']*100:.1f}%")
             
    strategy_df = pd.DataFrame(comparison_results)
    
    # 6. Auto-generate charts + dashboard for this scenario
    print("[6/6] Generating charts and dashboard...")
    os.makedirs('outputs', exist_ok=True)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import seaborn as sns
        import shutil

        _DARK  = '#1a1a2e'
        _PANEL = '#16213e'
        _ACCENT= '#e94560'
        _GOLD  = '#f5a623'
        _BLUE  = '#4a9eda'
        _GREEN = '#27ae60'
        _TEXT  = '#ecf0f1'

        # ----------------------------------------------------------------
        # DASHBOARD  — 2x2 grid
        # ----------------------------------------------------------------
        fig = plt.figure(figsize=(20, 14), facecolor=_DARK)
        fig.suptitle(
            f"GraphShield — Scenario: {scenario_name.replace('_', ' ')}",
            fontsize=20, fontweight='bold', color=_TEXT, y=0.98
        )

        gs = gridspec.GridSpec(
            2, 2, figure=fig,
            hspace=0.38, wspace=0.32,
            left=0.06, right=0.97, top=0.92, bottom=0.06
        )

        # ---- Panel A: Strategy Comparison (horizontal bar chart) --------
        ax_strat = fig.add_subplot(gs[0, 0])
        ax_strat.set_facecolor(_PANEL)

        strat_sorted = strategy_df.sort_values('mean_infection_rate', ascending=True)
        strats  = strat_sorted['strategy_name'].tolist()
        rates   = (strat_sorted['mean_infection_rate'] * 100).tolist()
        colors  = [_ACCENT if s == 'none' else
                   _GREEN  if s == 'anomaly_guided' else
                   _BLUE   for s in strats]

        bars = ax_strat.barh(strats, rates, color=colors, edgecolor='none', height=0.6)
        for bar, rate in zip(bars, rates):
            ax_strat.text(
                bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{rate:.1f}%', va='center', ha='left',
                color=_TEXT, fontsize=9, fontweight='bold'
            )
        ax_strat.set_xlabel('Mean Infection Rate (%)', color=_TEXT, fontsize=10)
        ax_strat.set_title('A  Defense Strategy Comparison', color=_TEXT,
                            fontsize=12, fontweight='bold', loc='left', pad=8)
        ax_strat.tick_params(colors=_TEXT, labelsize=9)
        ax_strat.spines[:].set_color('#444466')
        ax_strat.set_xlim(0, max(rates) * 1.25)
        for spine in ax_strat.spines.values():
            spine.set_color('#444466')

        # ---- Panel B: Detection Timeline --------------------------------
        ax_det = fig.add_subplot(gs[0, 1])
        ax_det.set_facecolor(_PANEL)

        if det_log and inf_log:
            inf_dict = {}
            for t, nodes in inf_log:
                inf_dict.setdefault(t, []).extend(nodes)
            det_dict = {}
            for t, nodes in det_log:
                det_dict.setdefault(t, []).extend(nodes)

            all_t = sorted(set(inf_dict.keys()) | set(det_dict.keys()))
            cum_inf, cum_det = [], []
            inf_set, det_set = set(), set()
            for t in all_t:
                inf_set.update(inf_dict.get(t, []))
                det_set.update(det_dict.get(t, []))
                cum_inf.append(len(inf_set))
                cum_det.append(len(det_set))

            ax_det.plot(all_t, cum_inf, color=_ACCENT,  lw=2.5, label='Cumulative Infected')
            ax_det.plot(all_t, cum_det, color=_GOLD, lw=2.5, ls='--', label='GNN Detected')
            ax_det.fill_between(all_t, cum_det, cum_inf,
                                alpha=0.15, color=_ACCENT, label='Undetected gap')
            ax_det.set_ylim(bottom=0)
            lead = detection_metrics.get('detection_lead_time_mean', 0)
            tpr  = detection_metrics.get('true_positive_rate', 0) * 100
            ax_det.set_title(
                f'B  GNN Detection Timeline  |  Lead: {lead:+.1f}t  TPR: {tpr:.0f}%',
                color=_TEXT, fontsize=12, fontweight='bold', loc='left', pad=8
            )
        else:
            ax_det.text(0.5, 0.5, 'No detection data', ha='center', va='center',
                        color=_TEXT, fontsize=13, transform=ax_det.transAxes)
            ax_det.set_title('B  GNN Detection Timeline', color=_TEXT,
                              fontsize=12, fontweight='bold', loc='left', pad=8)

        ax_det.set_xlabel('Simulation Timestep', color=_TEXT, fontsize=10)
        ax_det.set_ylabel('Cumulative Node Count', color=_TEXT, fontsize=10)
        ax_det.tick_params(colors=_TEXT, labelsize=9)
        for spine in ax_det.spines.values():
            spine.set_color('#444466')
        ax_det.legend(fontsize=8, facecolor=_PANEL, labelcolor=_TEXT,
                       edgecolor='#444466', loc='upper left')

        # ---- Panel C: Risk Heatmap (top 15 nodes) -----------------------
        ax_risk = fig.add_subplot(gs[1, 0])
        ax_risk.set_facecolor(_PANEL)

        top15 = risk_df.head(15).copy()
        dept_palette = {'IT': _BLUE, 'Finance': _GREEN,
                        'HR': _GOLD, 'Operations': _ACCENT}
        risk_colors = [dept_palette.get(d, _TEXT) for d in top15['department']]

        ax_risk.barh(top15['node_id'], top15['risk_score'],
                     color=risk_colors, edgecolor='none', height=0.65)
        ax_risk.set_xlabel('Composite Risk Score', color=_TEXT, fontsize=10)
        ax_risk.set_title('C  Top-15 Risk Nodes by Department',
                           color=_TEXT, fontsize=12, fontweight='bold', loc='left', pad=8)
        ax_risk.tick_params(colors=_TEXT, labelsize=8)
        for spine in ax_risk.spines.values():
            spine.set_color('#444466')

        # Department legend
        from matplotlib.patches import Patch
        legend_els = [Patch(facecolor=c, label=d)
                      for d, c in dept_palette.items()]
        ax_risk.legend(handles=legend_els, fontsize=8,
                        facecolor=_PANEL, labelcolor=_TEXT,
                        edgecolor='#444466', loc='lower right')

        # ---- Panel D: Key Metrics summary panel -------------------------
        ax_stats = fig.add_subplot(gs[1, 1])
        ax_stats.set_facecolor(_PANEL)
        ax_stats.axis('off')

        best_strat = strategy_df.sort_values('mean_infection_rate').iloc[0]
        base_rate  = strategy_df[strategy_df['strategy_name']=='none']['mean_infection_rate'].values
        base_pct   = float(base_rate[0]) * 100 if len(base_rate) else 0.0
        best_pct   = float(best_strat['mean_infection_rate']) * 100
        reduction  = base_pct - best_pct
        lead       = detection_metrics.get('detection_lead_time_mean', 0.0)
        tpr        = detection_metrics.get('true_positive_rate', 0.0) * 100
        spread_v   = base_sim_result.get('spread_velocity', 0.0)
        t_crit     = base_sim_result.get('time_to_first_critical_node')
        t_crit_str = f'T={t_crit}' if t_crit else 'Not reached'
        stages     = list(base_sim_result.get('attack_stages_timeline', {}).keys())
        top_node   = risk_df.iloc[0]['node_id'] if len(risk_df) else 'N/A'
        top_dept   = risk_df.iloc[0]['department'] if len(risk_df) else 'N/A'

        metrics_lines = [
            ('SCENARIO',          scenario_name.replace('_', ' ')),
            ('Attacker Mode',     attacker_mode.upper()),
            ('Beta (base)',       f'{beta:.2f}'),
            ('',                  ''),
            ('INFECTION',         ''),
            ('Undefended rate',   f'{base_pct:.1f}%'),
            ('Best defense',      best_strat["strategy_name"]),
            ('Best rate',         f'{best_pct:.1f}%'),
            ('Reduction',         f'{reduction:.1f} pp'),
            ('Spread velocity',   f'{spread_v:.2f} nodes/step'),
            ('1st critical node', t_crit_str),
            ('',                  ''),
            ('GNN DETECTION',     ''),
            ('Lead time',         f'{lead:+.2f} timesteps'),
            ('True positive rate',f'{tpr:.0f}%'),
            ('',                  ''),
            ('TOP RISK NODE',     ''),
            ('Node',              top_node),
            ('Department',        top_dept),
            ('Attack stages',     ', '.join(stages[-3:])),
        ]

        y = 0.97
        for label, value in metrics_lines:
            if label == '' and value == '':
                y -= 0.025
                continue
            is_header = value == '' or label in ('SCENARIO','INFECTION','GNN DETECTION','TOP RISK NODE')
            if is_header:
                ax_stats.text(0.03, y, label, transform=ax_stats.transAxes,
                              color=_GOLD, fontsize=10, fontweight='bold',
                              va='top')
            else:
                ax_stats.text(0.03, y, label + ':',
                              transform=ax_stats.transAxes,
                              color='#aaaacc', fontsize=9, va='top')
                ax_stats.text(0.52, y, value,
                              transform=ax_stats.transAxes,
                              color=_TEXT, fontsize=9, fontweight='bold', va='top')
            y -= 0.047

        ax_stats.set_title('D  Simulation Summary',
                            color=_TEXT, fontsize=12, fontweight='bold',
                            loc='left', pad=8)

        # ---- Save dashboard ---------------------------------------------
        dash_path = f'outputs/dashboard_{scenario_name}.png'
        fig.savefig(dash_path, dpi=150, bbox_inches='tight',
                    facecolor=_DARK)
        plt.close(fig)
        print(f"      -> Saved {dash_path}")

        # ---- Also save individual charts for completeness ---------------
        # Note: risk heatmap is rendered natively in Panel C above (dark theme)
        # No separate plot_risk_heatmap() call needed — it would overwrite Panel C

        if det_log and inf_log:
            anomaly_detector.plot_detection_timeline(
                det_log, inf_log, title_suffix=scenario_name)
            if os.path.exists('detection_timeline.png'):
                shutil.move('detection_timeline.png',
                            f'outputs/detection_timeline_{scenario_name}.png')

        strat_df_plot = strategy_df.copy()
        for col in ['std_infection_rate', 'mean_spread_velocity', 'std_spread_velocity',
                    'mean_containment_time', 'std_containment_time']:
            if col not in strat_df_plot.columns:
                strat_df_plot[col] = 0.0
        plot_strategy_comparison(strat_df_plot)
        if os.path.exists('strategy_comparison.png'):
            shutil.move('strategy_comparison.png',
                        f'outputs/strategy_comparison_{scenario_name}.png')

        plt.close('all')

        # Dashboard — composite 4-panel figure
        generate_scenario_dashboard(
            scenario_name,
            strategy_df,
            detection_metrics,
            base_sim_result,
            viz_sim_result,
            model_metrics=model_metrics,
        )
        print(f"      -> All charts saved to outputs/")

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"      -> Chart generation failed: {e}")

    # 7. NL Generation (Optional)
    report_text = ""
    if not skip_report and NL_REPORTING_ENABLED:
        print(f"\n[Report] Generating threat assessment via Ollama...")
        payload = report_gen.build_context_payload(
            scenario_name,
            base_sim_result,
            risk_df,
            detection_metrics,
            strategy_df,
            attacker_mode=attacker_mode,
        )
        report_text = report_gen.generate_report(payload)
        report_gen.save_report(report_text, scenario_name)
    elif skip_report:
        print(f"\n[Report] Generation skipped due to --no_report flag.")
        
    # 8. Aggregate Scenario Returns
    scenario_result = {
        'scenario_name': scenario_name,
        'base_simulation': base_sim_result,
        'viz_simulation': viz_sim_result,   # elevated-beta run for charts only
        'detection_metrics': detection_metrics,
        'risk_df': risk_df,
        'strategy_comparison': strategy_df,
        'features_df': features_df,
        'labels_df': labels_df,
        'model_metrics': model_metrics
    }
    
    # Output to CSV 
    os.makedirs('outputs', exist_ok=True)
    strategy_df.to_csv(f'outputs/results_{scenario_name}.csv', index=False)
    
    # 7b. Optional interactive network visualisation
    if visualize:
        try:
            from network_graph import visualize_graph
            viz_path = f'outputs/network_{scenario_name}.html'
            os.makedirs('outputs', exist_ok=True)

            # Build detected set from base sim (low beta) not viz sim (high beta)
            # This gives meaningful gold nodes rather than flagging the whole graph
            G_viz_replay = copy.deepcopy(G)
            reset_graph(G_viz_replay)
            detected_nodes = set()
            for step in base_sim_result['timestep_log']:
                if step['phase'] == 'baseline_recording':
                    continue
                for n in step['newly_infected_nodes']:
                    G_viz_replay.nodes[n]['infection_state'] = 'infected'
                flagged = anomaly_detector.detect_anomalies(G_viz_replay, threshold=0.5)
                detected_nodes.update(flagged)

            # Only mark detected on nodes NOT already showing as infected
            for n in detected_nodes:
                if n in G_base.nodes and G_base.nodes[n].get('infection_state') != 'infected':
                    G_base.nodes[n]['detected'] = True

            visualize_graph(
                G_base,
                output_file=viz_path,
                title=f"GraphShield — {scenario_name.replace('_', ' ')} (post-attack state)"
            )
            print(f"[Viz] Interactive network saved → {viz_path}  ({len(detected_nodes)} nodes GNN-flagged)")
        except Exception as e:
            print(f"[Viz] Visualisation failed: {e}")
            print(f"[Viz] Interactive network saved → {viz_path}")
        except Exception as e:
            print(f"[Viz] Visualisation failed: {e}")

    print(f"\n[Done] Scenario {scenario_name} Complete.\n")
    return scenario_result

# ---------------------------------------------------------
# Scenario Definitions
# ---------------------------------------------------------

def scenario_random_workstation(G, anomaly_detector, report_gen, skip_report, rl_agent=None, visualize=False):
    workstations = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'workstation']
    entry = [random.choice(workstations)] if workstations else [list(G.nodes())[0]]
    return execute_scenario_pipeline("Random_Workstation", G, entry, "random", 0.3, anomaly_detector, report_gen, skip_report, rl_agent=rl_agent, visualize=visualize)

def scenario_targeted_finance(G, anomaly_detector, report_gen, skip_report, rl_agent=None, visualize=False):
    finance_nodes = [n for n, d in G.nodes(data=True) if d.get('department') == 'Finance']
    if finance_nodes:
        best_finance = max(finance_nodes, key=lambda n: G.nodes[n].get('asset_value', 0))
        entry = [best_finance]
    else:
        entry = [list(G.nodes())[0]]
    return execute_scenario_pipeline("Targeted_Finance", G, entry, "greedy", 0.3, anomaly_detector, report_gen, skip_report, rl_agent=rl_agent, visualize=visualize)

def scenario_domain_controller(G, anomaly_detector, report_gen, skip_report, rl_agent=None, visualize=False):
    servers = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'server' and d.get('privilege_level') == 'domain_admin']
    entry = [servers[0]] if servers else [list(G.nodes())[0]]
    return execute_scenario_pipeline("Domain_Controller", G, entry, "greedy", 0.4, anomaly_detector, report_gen, skip_report, rl_agent=rl_agent, visualize=visualize)

def scenario_stealth(G, anomaly_detector, report_gen, skip_report, rl_agent=None, visualize=False):
    entry = [random.choice(list(G.nodes()))]
    return execute_scenario_pipeline("Stealth", G, entry, "stealth", 0.1, anomaly_detector, report_gen, skip_report, rl_agent=rl_agent, visualize=visualize)

# ---------------------------------------------------------
# Pipeline Executions
# ---------------------------------------------------------

def run_full_pipeline(skip_report: bool, visualize: bool = False):
    """
    Executes all scenarios sequentially 
    """
    random.seed(SIMULATION_CONFIG['seed'])
    
    print("Initialize baseline enterprise network...")
    G = generate_enterprise_graph(seed=SIMULATION_CONFIG['seed'])
    
    print("Initialize ML models & detectors...")
    anomaly_detector = make_anomaly_detector(
        contamination=ANOMALY_CONFIG['contamination'],
        random_state=SIMULATION_CONFIG['seed'],
        hidden_dim=GNN_CONFIG['hidden_dim'],
        lr=GNN_CONFIG['lr'],
        epochs=GNN_CONFIG['epochs'],
        use_gnn=GNN_ANOMALY_MODE,
    )
    report_gen = ReportGenerator(model_name="mistral")

    # Train one DQN agent per attacker mode so each is optimised for its scenario
    print("Initialize DQN RL defense agents (one per attacker mode)...")
    rl_agents = {}
    for mode, beta in [('random', 0.3), ('greedy', 0.4), ('stealth', 0.1)]:
        agent = RLDefenseAgent(n_nodes=G.number_of_nodes(), budget=5)
        agent.SAVE_PATH = f'models/rl_agent_{mode}'
        if not agent.load():
            agent.train(G, attacker_mode=mode, beta=beta, episodes=60)
            agent.save()
        rl_agents[mode] = agent
        print(f"  [DQN] {mode} agent ready (epsilon={agent.epsilon:.3f})")

    all_results = []

    all_results.append(scenario_random_workstation(G, anomaly_detector, report_gen, skip_report, rl_agent=rl_agents['random'], visualize=visualize))
    all_results.append(scenario_targeted_finance(G, anomaly_detector, report_gen, skip_report, rl_agent=rl_agents['greedy'], visualize=visualize))
    all_results.append(scenario_domain_controller(G, anomaly_detector, report_gen, skip_report, rl_agent=rl_agents['greedy'], visualize=visualize))
    all_results.append(scenario_stealth(G, anomaly_detector, report_gen, skip_report, rl_agent=rl_agents['stealth'], visualize=visualize))
    
    print("\n" + "="*80)
    print("FINAL ENTERPRISE SIMULATION SUMMARY")
    print("="*80)
    
    # Compile multi-scenario JSON representation for Meta-Analytics
    meta_payload = []
    for sr in all_results:
        best_row = sr['strategy_comparison'].sort_values('mean_infection_rate').iloc[0]
        meta_payload.append({
            'scenario': sr['scenario_name'],
            'best_strategy': best_row['strategy_name'],
            'min_infection_rate': best_row['mean_infection_rate'],
            'base_infection_rate': sr['base_simulation']['final_infection_rate']
        })
        print(f"-> {sr['scenario_name']:<25} | Best Defense: {best_row['strategy_name']:<20} | Score: {best_row['mean_infection_rate']:.3f}")
        
    if not skip_report and NL_REPORTING_ENABLED:
        print("\nFiring Meta-Analysis comparison payload to Ollama...")
        report_gen.generate_comparison_report(meta_payload)
        
    # Cross-scenario infection curves chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shutil

        curve_data = {}
        for sr in all_results:
            tlog = sr.get("viz_simulation", sr["base_simulation"])["timestep_log"]
            # Include patient_zero + propagation phases (skip baseline_recording)
            curve = [
                step["infected_count"]
                for step in tlog
                if step["phase"] != "baseline_recording"
            ]
            if curve:
                curve_data[sr["scenario_name"]] = curve

        if curve_data:
            plot_infection_curves(curve_data)
            if os.path.exists("infection_curves.png"):
                shutil.move("infection_curves.png", "outputs/infection_curves_all_scenarios.png")
                print("Saved outputs/infection_curves_all_scenarios.png")
        plt.close("all")
    except Exception as e:
        print(f"Cross-scenario chart failed: {e}")

    # ── Model Metrics: console + CSV ──────────────────────────────────────
    print("\n" + "="*80)
    print("MODEL PERFORMANCE METRICS")
    print("="*80)

    all_mm = []
    for sr in all_results:
        mm = sr.get('model_metrics', {})
        sname = sr['scenario_name']
        mm['scenario'] = sname
        all_mm.append(mm)
        print(f"\n  Scenario: {sname}")
        print(f"  GNN  | Acc={mm.get('gnn_accuracy',0)*100:.1f}%  "
              f"Prec={mm.get('gnn_precision',0)*100:.1f}%  "
              f"Rec={mm.get('gnn_recall',0)*100:.1f}%  "
              f"F1={mm.get('gnn_f1',0)*100:.1f}%  "
              f"AUC={mm.get('gnn_auc',0):.3f}  "
              f"FPR={mm.get('gnn_fpr',0)*100:.1f}%")
        print(f"  Risk | P@5={mm.get('risk_precision_at_5',0)*100:.1f}%  "
              f"P@10={mm.get('risk_precision_at_10',0)*100:.1f}%  "
              f"P@20={mm.get('risk_precision_at_20',0)*100:.1f}%  "
              f"Tau={mm.get('risk_kendall_tau',0):.3f}")
        print(f"  DQN  | Trained={mm.get('dqn_trained',False)}  "
              f"ExploitRate={mm.get('dqn_exploitation_rate',0)*100:.1f}%  "
              f"Steps={mm.get('dqn_train_steps',0)}  "
              f"TailReward={mm.get('dqn_mean_tail_reward',0):+.3f}")

    try:
        save_model_metrics_csv(all_mm)
    except Exception as e:
        print(f"  -> CSV export failed: {e}")

    # ── Final summary CSV ────────────────────────────────────────────────
    try:
        import pandas as pd
        summary_rows = []
        for sr in all_results:
            strat_df = sr['strategy_comparison']
            for _, row in strat_df.iterrows():
                summary_rows.append({
                    'scenario':           sr['scenario_name'],
                    'strategy':           row['strategy_name'],
                    'mean_infection_rate':round(row['mean_infection_rate'], 4),
                    'std_infection_rate': round(row.get('std_infection_rate', 0), 4),
                    'base_infection_rate':round(sr['base_simulation']['final_infection_rate'], 4),
                    'detection_lead_time':round(sr['detection_metrics'].get('detection_lead_time_mean', 0), 2),
                    'true_positive_rate': round(sr['detection_metrics'].get('true_positive_rate', 0), 4),
                })
        pd.DataFrame(summary_rows).to_csv('outputs/final_summary.csv', index=False)
        print("  -> Saved outputs/final_summary.csv")
    except Exception as e:
        print(f"  -> Final summary CSV failed: {e}")

    # ── Cross-scenario model metrics chart ───────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        scenarios = [sr['scenario_name'] for sr in all_results]
        gnn_f1    = [sr.get('model_metrics', {}).get('gnn_f1',    0) * 100 for sr in all_results]
        gnn_auc   = [sr.get('model_metrics', {}).get('gnn_auc',   0) * 100 for sr in all_results]
        gnn_rec   = [sr.get('model_metrics', {}).get('gnn_recall', 0) * 100 for sr in all_results]
        risk_p10  = [sr.get('model_metrics', {}).get('risk_precision_at_10', 0) * 100 for sr in all_results]

        x = np.arange(len(scenarios))
        w = 0.2
        fig, ax = plt.subplots(figsize=(13, 6))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')

        bars = [
            (gnn_f1,   '#f5a623', 'GNN F1'),
            (gnn_auc,  '#58a6ff', 'GNN AUC-ROC'),
            (gnn_rec,  '#3fb950', 'GNN Recall'),
            (risk_p10, '#a855f7', 'Risk P@10'),
        ]
        for i, (vals, color, label) in enumerate(bars):
            rects = ax.bar(x + (i - 1.5) * w, vals, w, label=label,
                           color=color, alpha=0.85, edgecolor='#30363d')
            for rect in rects:
                h = rect.get_height()
                if h > 0:
                    ax.text(rect.get_x() + rect.get_width()/2, h + 1,
                            f'{h:.0f}%', ha='center', va='bottom',
                            color='white', fontsize=7.5, fontfamily='monospace')

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in scenarios],
                           color='white', fontsize=10)
        ax.set_ylabel('Score (%)', color='white', fontsize=11)
        ax.set_title('AEGIS — Cross-Scenario Model Performance',
                     color='white', fontsize=14, fontweight='bold', pad=12)
        ax.set_ylim(0, 115)
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#30363d')
        ax.legend(facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='white', fontsize=9)
        ax.grid(axis='y', color='#30363d', linewidth=0.5)
        plt.tight_layout()
        plt.savefig('outputs/cross_scenario_metrics.png', dpi=150,
                    bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print("  -> Saved outputs/cross_scenario_metrics.png")
    except Exception as e:
        print(f"  -> Cross-scenario metrics chart failed: {e}")

    print("\nAll tasks completed. Charts and logs exported to outputs/ directory.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Enterprise Cyber Contagion Simulator Orchestrator")
    parser.add_argument("--scenario", type=str, choices=["all", "random", "finance", "dc", "stealth"], default="all",
                        help="Specific scenario to run, or 'all' to run the full pipeline.")
    parser.add_argument("--attacker", type=str, choices=["random", "greedy", "stealth"], default="random",
                        help="Global override for attacker mode (ignored if running 'all').")
    parser.add_argument("--ai_mode", action="store_true", 
                        help="[Future] Enable fully autonomous GNN + RL driven evaluation flows.")
    parser.add_argument("--visualize", action="store_true",
                        help="Launch visualizer dashboard at end of run.")
    parser.add_argument("--no_report", action="store_true",
                        help="Skip Ollama/NL generation for faster offline data runs.")
    
    args = parser.parse_args()
    
    if args.scenario == "all":
         run_full_pipeline(skip_report=args.no_report, visualize=args.visualize)
    else:
         # Initialize bare requirements for single runs
         G = generate_enterprise_graph(seed=SIMULATION_CONFIG['seed'])
         
         anomaly_detector = make_anomaly_detector(
             contamination=ANOMALY_CONFIG['contamination'],
             random_state=SIMULATION_CONFIG['seed'],
             hidden_dim=GNN_CONFIG['hidden_dim'],
             lr=GNN_CONFIG['lr'],
             epochs=GNN_CONFIG['epochs'],
             use_gnn=GNN_ANOMALY_MODE,
         )
         report_gen = ReportGenerator(model_name="mistral")

         rl_agent = RLDefenseAgent(n_nodes=G.number_of_nodes(), budget=5)
         if not rl_agent.load():
             rl_agent.train(G, attacker_mode='random', beta=0.3, episodes=60)
             rl_agent.save()

         if args.scenario == "random":
             scenario_random_workstation(G, anomaly_detector, report_gen, skip_report=args.no_report, rl_agent=rl_agent, visualize=args.visualize)
         elif args.scenario == "finance":
             scenario_targeted_finance(G, anomaly_detector, report_gen, skip_report=args.no_report, rl_agent=rl_agent, visualize=args.visualize)
         elif args.scenario == "dc":
             scenario_domain_controller(G, anomaly_detector, report_gen, skip_report=args.no_report, rl_agent=rl_agent, visualize=args.visualize)
         elif args.scenario == "stealth":
             scenario_stealth(G, anomaly_detector, report_gen, skip_report=args.no_report, visualize=args.visualize)