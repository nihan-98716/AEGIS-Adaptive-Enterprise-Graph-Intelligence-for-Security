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
                                 base_sim_result: dict, viz_sim_result: dict):
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

    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor("#0d1117")

    # Title banner
    fig.suptitle(
        f"AEGIS — Scenario: {scenario_name.replace('_', ' ')}",
        fontsize=22, fontweight="bold", color="white", y=0.97
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25,
                           left=0.04, right=0.96, top=0.92, bottom=0.04)

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

    # Panel 4 — Stats Summary (text)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#161b22")
    ax4.axis("off")
    ax4.set_title("Scenario Intelligence Summary", fontsize=13,
                  fontweight="bold", color="white", pad=8)

    # Gather stats
    final_rate   = base_sim_result.get("final_infection_rate", 0) * 100
    viz_rate     = viz_sim_result.get("final_infection_rate", 0) * 100
    velocity     = base_sim_result.get("spread_velocity", 0)
    t_crit       = base_sim_result.get("time_to_first_critical_node")
    t_crit_str   = f"Timestep {t_crit}" if t_crit else "Not reached"
    stages       = list(base_sim_result.get("attack_stages_timeline", {}).keys())
    lead_time    = detection_metrics.get("detection_lead_time_mean", 0)
    tpr          = detection_metrics.get("true_positive_rate", 0) * 100
    fdr          = detection_metrics.get("false_positive_rate", 0) * 100

    best_row   = strategy_df.sort_values("mean_infection_rate").iloc[0]
    worst_row  = strategy_df.sort_values("mean_infection_rate").iloc[-1]
    best_strat = best_row["strategy_name"]
    best_rate  = best_row["mean_infection_rate"] * 100
    worst_rate = worst_row["mean_infection_rate"] * 100
    reduction  = worst_rate - best_rate

    lines = [
        ("ATTACK METRICS", None, "#58a6ff"),
        (f"Base infection rate",      f"{final_rate:.1f}%",       "white"),
        (f"Viz spread (beta=0.85)",   f"{viz_rate:.1f}%",         "#8b949e"),
        (f"Spread velocity",          f"{velocity:.2f} nodes/step","white"),
        (f"Time to critical node",    t_crit_str,                  "white"),
        (f"Attack stages reached",    str(len(stages)),            "white"),
        ("", None, "white"),
        ("GNN DETECTION", None, "#58a6ff"),
        (f"Detection lead time",      f"+{lead_time:.1f} steps",  "#3fb950" if lead_time > 0 else "#f85149"),
        (f"True positive rate",       f"{tpr:.1f}%",              "#3fb950" if tpr > 80 else "#f85149"),
        (f"False discovery rate",     f"{fdr:.1f}%",              "#3fb950" if fdr < 20 else "#f85149"),
        ("", None, "white"),
        ("DEFENSE OUTCOME", None, "#58a6ff"),
        (f"Best strategy",            best_strat,                 "#3fb950"),
        (f"Best infection rate",      f"{best_rate:.1f}%",        "#3fb950"),
        (f"Worst infection rate",     f"{worst_rate:.1f}%",       "#f85149"),
        (f"Reduction achieved",       f"{reduction:.1f}%",        "#3fb950"),
    ]

    y = 0.96
    for label, value, color in lines:
        if label == "" :
            y -= 0.025
            continue
        if value is None:
            # Section header
            ax4.text(0.04, y, label, transform=ax4.transAxes,
                     fontsize=10, fontweight="bold", color=color, va="top",
                     fontfamily="monospace")
            y -= 0.045
        else:
            ax4.text(0.04, y, label, transform=ax4.transAxes,
                     fontsize=9.5, color="#8b949e", va="top", fontfamily="monospace")
            ax4.text(0.96, y, value, transform=ax4.transAxes,
                     fontsize=9.5, color=color, va="top", ha="right",
                     fontfamily="monospace", fontweight="bold")
            y -= 0.052

    # Subtle border on stats panel
    for spine in ax4.spines.values():
        spine.set_edgecolor("#30363d")
        spine.set_linewidth(1)

    out_path = f"outputs/dashboard_{scenario_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"      -> Saved {out_path}")
    return out_path


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

        # Write final GNN anomaly scores back to G_base for tooltip display
        # Score from a clean base-sim replay (low beta) not viz sim (high beta)
        # so only truly anomalous nodes score high, not the whole infected graph.
        import copy as _sc; from network_graph import reset_graph as _rsg
        G_score_replay = _sc.deepcopy(G); _rsg(G_score_replay)
        for _step in base_sim_result['timestep_log']:
            if _step['phase'] == 'baseline_recording': continue
            for _n in _step['newly_infected_nodes']:
                G_score_replay.nodes[_n]['infection_state'] = 'infected'
        final_scores = anomaly_detector.score_nodes(G_score_replay)
        for n, score in final_scores.items():
            if n in G_base.nodes:
                G_base.nodes[n]['anomaly_score'] = round(float(score), 3)

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"      -> Anomaly eval failed: {e}")
        detection_metrics = {}
        det_log, inf_log = [], []
         
    # 4. Risk Reporting
    print("[4/6] Computing static + dynamic Risk scoring...")
    calculate_risk_scores(G_base)
    risk_df = get_risk_report(G_base)
    
    # 5. Multiprocessing Defense Comparisons
    strategies = ["none", "random", "patch_vulnerable", "patch_centrality", "isolate_bridges", "anomaly_guided", "rl_agent"]
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
            rl_agent=rl_agent,
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
            f"AEGIS — Scenario: {scenario_name.replace('_', ' ')}",
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
        'labels_df': labels_df
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

            # Gold nodes = direct susceptible neighbours of infected nodes in base sim.
            # Do NOT use GNN scoring — GraphSAGE floods the entire graph due to
            # neighbourhood aggregation. Topology-based at-risk is accurate and sparse.
            infected_in_base = set(
                n for n, d in G_base.nodes(data=True)
                if d.get('infection_state') == 'infected'
            )
            at_risk_nodes = set()
            for inf_node in infected_in_base:
                for nbr in G_base.neighbors(inf_node):
                    if G_base.nodes[nbr].get('infection_state') != 'infected':
                        at_risk_nodes.add(nbr)

            # Clear stale flags, apply fresh at-risk flags
            for n in G_base.nodes():
                G_base.nodes[n].pop('detected', None)
            for n in at_risk_nodes:
                G_base.nodes[n]['detected'] = True

            gold_count = len(at_risk_nodes)
            red_count  = len(infected_in_base)
            print(f"[Viz] Infected(red)={red_count}  At-risk neighbours(gold)={gold_count}")

            visualize_graph(
                G_base,
                output_file=viz_path,
                title=f"AEGIS — {scenario_name.replace('_', ' ')} (post-attack state)"
            )
            print(f"[Viz] Saved -> {viz_path}")

        except Exception as e:
            import traceback; traceback.print_exc()
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

    print("Initialize DQN RL defense agent...")
    rl_agent = RLDefenseAgent(n_nodes=G.number_of_nodes(), budget=5)
    if not rl_agent.load():
        rl_agent.train(G, attacker_mode='random', beta=0.3, episodes=60)
        rl_agent.save()
    print(f"  [DQN] Agent ready (epsilon={rl_agent.epsilon:.3f})") 

    all_results = []
    
    all_results.append(scenario_random_workstation(G, anomaly_detector, report_gen, skip_report, rl_agent=rl_agent, visualize=visualize))
    all_results.append(scenario_targeted_finance(G, anomaly_detector, report_gen, skip_report, rl_agent=rl_agent, visualize=visualize))
    all_results.append(scenario_domain_controller(G, anomaly_detector, report_gen, skip_report, rl_agent=rl_agent, visualize=visualize))
    all_results.append(scenario_stealth(G, anomaly_detector, report_gen, skip_report, rl_agent=rl_agent, visualize=visualize))
    
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
             scenario_stealth(G, anomaly_detector, report_gen, skip_report=args.no_report, rl_agent=rl_agent, visualize=args.visualize)