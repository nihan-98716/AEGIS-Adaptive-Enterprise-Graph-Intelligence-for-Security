import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from network_graph import reset_graph
from propagation_engine import run_simulation, calculate_infection_probability, update_global_attack_stage, select_target_nodes
from risk_engine import get_critical_nodes, calculate_risk_scores


# --- Defense Strategies ---

def strategy_none(G, budget):
    """No action taken."""
    return []

def strategy_random(G, budget):
    """Patch random susceptible nodes up to budget."""
    susceptible = [n for n, d in G.nodes(data=True) if d.get('infection_state') == 'susceptible']
    to_patch = random.sample(susceptible, min(budget, len(susceptible)))
    
    for n in to_patch:
        G.nodes[n]['infection_state'] = 'patched'
        G.nodes[n]['vulnerability_score'] = 0.0
        
    return to_patch

def strategy_patch_vulnerable(G, budget):
    """Patch highest vulnerability_score nodes that are not yet infected up to budget."""
    eligible = [n for n, d in G.nodes(data=True) if d.get('infection_state') == 'susceptible']
    # Sort eligible by vulnerability_score descending
    eligible.sort(key=lambda n: G.nodes[n].get('vulnerability_score', 0.0), reverse=True)
    
    to_patch = eligible[:budget]
    for n in to_patch:
        G.nodes[n]['infection_state'] = 'patched'
        G.nodes[n]['vulnerability_score'] = 0.0
        
    return to_patch

def strategy_patch_centrality(G, budget):
    """Patch highest risk_score nodes that are not yet infected up to budget."""
    # Ensure risk scores are computed and up to date
    calculate_risk_scores(G)
    
    eligible = [n for n, d in G.nodes(data=True) if d.get('infection_state') == 'susceptible']
    eligible.sort(key=lambda n: G.nodes[n].get('risk_score', 0.0), reverse=True)
    
    to_patch = eligible[:budget]
    for n in to_patch:
        G.nodes[n]['infection_state'] = 'patched'
        G.nodes[n]['vulnerability_score'] = 0.0
        
    return to_patch

def strategy_isolate_bridges(G, budget):
    """
    Identify bridge edges, remove edges connected to highest betweenness nodes, up to budget limit.
    A budget action here is an edge removal.
    """
    actions_taken = []
    
    # Needs metrics updated to know highest betweenness
    calculate_risk_scores(G)
    
    bridges = list(nx.bridges(G))
    if not bridges:
        return actions_taken
        
    # We want to prioritize bridges connected to high betweenness nodes.
    # We'll score each bridge by the sum of betweenness of its endpoints.
    scored_bridges = []
    for u, v in bridges:
        b_score = G.nodes[u].get('betweenness_centrality', 0.0) + G.nodes[v].get('betweenness_centrality', 0.0)
        scored_bridges.append(((u, v), b_score))
        
    scored_bridges.sort(key=lambda x: x[1], reverse=True)
    
    for (u, v), score in scored_bridges[:budget]:
        if G.has_edge(u, v):
            G.remove_edge(u, v)
            actions_taken.append(f"Removed bridge {u}-{v}")
            
    return actions_taken

def strategy_anomaly_guided(G, budget, anomaly_detector):
    """
    Prioritize patching nodes where detected=True and not yet infected.
    Fall back to centrality-based if no detected nodes available.
    """
    anomalous_susceptible = [
        n for n, d in G.nodes(data=True) 
        if d.get('detected', False) and d.get('infection_state') == 'susceptible'
    ]
    
    to_patch = anomalous_susceptible[:budget]
    remaining_budget = budget - len(to_patch)
    
    # Fallback to centrality if we have budget left
    if remaining_budget > 0:
        calculate_risk_scores(G)
        eligible = [
            n for n, d in G.nodes(data=True) 
            if d.get('infection_state') == 'susceptible' and n not in to_patch
        ]
        eligible.sort(key=lambda n: G.nodes[n].get('risk_score', 0.0), reverse=True)
        to_patch.extend(eligible[:remaining_budget])
        
    for n in to_patch:
        G.nodes[n]['infection_state'] = 'patched'
        G.nodes[n]['vulnerability_score'] = 0.0
        
    return to_patch

# --- Simulation Mechanics ---

def run_defense_experiment(G, strategy_fn, attacker_mode, strategy_name, anomaly_detector=None, n_runs=30, seed=42, initial_node=None, max_timesteps=50, beta=0.3):
    """
    Runs n_runs simulations with the given strategy applied each timestep.
    Returns dict with mean and std for performance metrics.
    """
    print(f"\nRunning tests: Strategy [{strategy_name}] runs={n_runs}")
    
    if seed is not None:
         random.seed(seed)
         np.random.seed(seed)
         
    metrics = {
        'infection_rate': [],
        'spread_velocity': [],
        'time_to_critical_node': [],
        'containment_time': [],
        'detection_lead_time': []
    }
    
    # We need to hack into the timestep loop of propagation_engine slightly
    # Since it runs a monolithic loop, to inject defense actions *every timestep*,
    # we realistically need our own runner wrapper or we pass a callback.
    # However, the spec says "Runs n_runs simulations with the given strategy applied each timestep"
    # Since propagation_engine `run_simulation` doesn't accept a defense callback,
    # we will extract its core logic here but inject the defense logic.
    
    # Pre-train anomaly detector BEFORE the n_runs loop.
    # Skip if already fitted — GNNAnomalyDetector is expensive to retrain and
    # main.py fits it once on the baseline graph before the strategy loop.
    if anomaly_detector and not anomaly_detector.is_fitted:
        sim_G_baseline = copy.deepcopy(G)
        reset_graph(sim_G_baseline)
        for n in sim_G_baseline.nodes():
            baseline = {}
            for neighbor in sim_G_baseline.neighbors(n):
                freq = sim_G_baseline[n][neighbor].get('traffic_frequency', 0.5)
                weight = sim_G_baseline[n][neighbor].get('trust_weight', 0.5)
                baseline[str(neighbor)] = freq * weight
            sim_G_baseline.nodes[n]['baseline_traffic'] = baseline
        anomaly_detector.fit_baseline(sim_G_baseline)
    elif anomaly_detector and anomaly_detector.is_fitted:
        print(f"  Detector already fitted — reusing trained model for [{strategy_name}]")

    # 2D list to store infected counts per timestep for all runs
    all_runs_infection_curves = []

    for run in range(n_runs):
        sim_G = copy.deepcopy(G)
        reset_graph(sim_G)
        random.seed(seed + run)
        # We need to hack into the timestep loop of propagation_engine slightly
        # Since it runs a monolithic loop, to inject defense actions *every timestep*,
        # we realistically need our own runner wrapper or we pass a callback.
        # However, the spec says "Runs n_runs simulations with the given strategy applied each timestep"
        # Since propagation_engine `run_simulation` doesn't accept a defense callback,
        # we will extract its core logic here but inject the defense logic.
        
        # --- Embedded Simulation Logic with Defense Injection ---
        
        # Phase 1: Baseline Recording
        # Assuming propagation_engine defined `record_baseline_traffic(G)`
        # We'll just manually recreate the necessary baseline attributes
        for n in sim_G.nodes():
            baseline = {}
            for neighbor in sim_G.neighbors(n):
                freq = sim_G[n][neighbor].get('traffic_frequency', 0.5)
                weight = sim_G[n][neighbor].get('trust_weight', 0.5)
                baseline[str(neighbor)] = freq * weight
            sim_G.nodes[n]['baseline_traffic'] = baseline
            
        actual_timesteps = 5
            
        # Phase 2: Patient Zero
        if initial_node is not None and initial_node in sim_G.nodes():
            patient_zero = initial_node
        else:
            workstations = [n for n, d in sim_G.nodes(data=True) if d.get('node_type') == 'workstation']
            patient_zero = random.choice(workstations) if workstations else random.choice(list(sim_G.nodes()))
        sim_G.nodes[patient_zero]['infection_state'] = 'infected'
        sim_G.nodes[patient_zero]['attack_stage'] = 'initial_access'
        
        actual_timesteps += 1
        
        current_attack_stage = update_global_attack_stage(sim_G, "none")
        time_to_first_critical_node = None
        total_new_infections = 0
        
        
        detection_log = []
        infection_log = []
        
        # Track infected counts per timestep for this single run
        run_infection_curve = [1] # T=6 contains 1 infected (patient zero)
        
        
        # Phase 3: Propagation
        for t in range(7, max_timesteps + 1):
            actual_timesteps += 1
            
    
            # 1. Defend FIRST before propagation
            if strategy_name == "anomaly_guided":
                strategy_anomaly_guided(sim_G, 5, anomaly_detector)
            elif strategy_name == "none":
                strategy_none(sim_G, 5)
            elif strategy_name == "random":
                strategy_random(sim_G, 5)
            elif strategy_name == "patch_vulnerable":
                strategy_patch_vulnerable(sim_G, 5)
            elif strategy_name == "patch_centrality":
                strategy_patch_centrality(sim_G, 5)
            elif strategy_name == "isolate_bridges":
                strategy_isolate_bridges(sim_G, 5)
            
            # 2. Propagate
            pending_infections = []
            infected_nodes = [n for n, d in sim_G.nodes(data=True) if d.get('infection_state') == 'infected']
            
            for source in infected_nodes:
                targets = select_target_nodes(sim_G, source, attacker_mode)
                for target in targets:
                    if sim_G.nodes[target].get('infection_state') == 'susceptible' and target not in pending_infections:
                        prob = calculate_infection_probability(sim_G, source, target, beta)
                        if random.random() < prob:
                            pending_infections.append(target)
                            
            # Apply infections
            for n in pending_infections:
                sim_G.nodes[n]['infection_state'] = 'infected'
                node_type = sim_G.nodes[n].get('node_type')
                privilege = sim_G.nodes[n].get('privilege_level')
                if time_to_first_critical_node is None and (node_type in ['server', 'controller'] or privilege == 'domain_admin'):
                    time_to_first_critical_node = t
                    
            total_new_infections += len(pending_infections)
            
            # Check early exit / logging
            current_infected_count = sum(1 for _, d in sim_G.nodes(data=True) if d.get('infection_state') == 'infected')
            run_infection_curve.append(current_infected_count)
            
            # 3. Detect
            if anomaly_detector:
                detected_now = anomaly_detector.detect_anomalies(sim_G, threshold=0.5)
                detection_log.append((t, detected_now))
                infection_log.append((t, pending_infections))
                
            # Only exit if fully infected OR no infected node has any susceptible neighbors left
            if current_infected_count == sim_G.number_of_nodes():
                while len(run_infection_curve) < (max_timesteps - 6 + 1):
                    run_infection_curve.append(current_infected_count)
                break

            infected_nodes_check = [n for n, d in sim_G.nodes(data=True) if d.get('infection_state') == 'infected']
            any_susceptible_neighbors = any(
                any(sim_G.nodes[nbr].get('infection_state') == 'susceptible' for nbr in sim_G.neighbors(n))
                for n in infected_nodes_check
            )
            if not any_susceptible_neighbors:
                while len(run_infection_curve) < (max_timesteps - 6 + 1):
                    run_infection_curve.append(current_infected_count)
                break
                
        all_runs_infection_curves.append(run_infection_curve)
        
        # --- End Embedded Logic ---
        
        # Evaluate metrics for this run
        final_infected = sum(1 for _, d in sim_G.nodes(data=True) if d.get('infection_state') == 'infected')
        final_infection_rate = final_infected / sim_G.number_of_nodes()
        containment_time = actual_timesteps - time_to_first_critical_node if time_to_first_critical_node else None
        prop_timesteps = actual_timesteps - 6
        spread_velocity = total_new_infections / prop_timesteps if prop_timesteps > 0 else 0.0
        
        metrics['infection_rate'].append(final_infection_rate)
        metrics['spread_velocity'].append(spread_velocity)
        
        if time_to_first_critical_node is not None:
             metrics['time_to_critical_node'].append(time_to_first_critical_node)
        if containment_time is not None:
             metrics['containment_time'].append(containment_time)
             
        if anomaly_detector:
             run_metrics = anomaly_detector.compute_detection_metrics(detection_log, infection_log)
             lead = run_metrics.get('detection_lead_time_mean', 0.0)
             metrics['detection_lead_time'].append(lead)
             
    # Calculate true mean infection curve
    mean_infection_curve = np.mean(all_runs_infection_curves, axis=0).tolist()
    
    # Aggregate stats over n_runs
    summary = {
        'strategy_name': strategy_name,
        'mean_infection_rate': np.nanmean(metrics['infection_rate']) if metrics['infection_rate'] else 0.0,
        'std_infection_rate': np.nanstd(metrics['infection_rate']) if metrics['infection_rate'] else 0.0,
        
        'mean_spread_velocity': np.nanmean(metrics['spread_velocity']) if metrics['spread_velocity'] else 0.0,
        'std_spread_velocity': np.nanstd(metrics['spread_velocity']) if metrics['spread_velocity'] else 0.0,
        
        'mean_time_to_critical': np.nanmean(metrics['time_to_critical_node']) if metrics['time_to_critical_node'] else 0.0,
        'std_time_to_critical': np.nanstd(metrics['time_to_critical_node']) if metrics['time_to_critical_node'] else 0.0,
        
        'mean_containment_time': np.nanmean(metrics['containment_time']) if metrics['containment_time'] else 0.0,
        'std_containment_time': np.nanstd(metrics['containment_time']) if metrics['containment_time'] else 0.0,
        
        'mean_detection_lead_time': np.nanmean(metrics['detection_lead_time']) if metrics['detection_lead_time'] else 0.0,
        'std_detection_lead_time': np.nanstd(metrics['detection_lead_time']) if metrics['detection_lead_time'] else 0.0,
        
        'mean_infection_curve': mean_infection_curve
    }
    
    return summary

def compare_all_strategies(G, attacker_mode, anomaly_detector=None):
    """
    Runs all 6 strategies using run_defense_experiment.
    Returns summary DataFrame.
    """
    strat_names = ['none', 'random', 'patch_vulnerable', 'patch_centrality', 'isolate_bridges', 'anomaly_guided']
    
    rows = []
    for s_name in strat_names:
        res = run_defense_experiment(
            G, 
            strategy_fn=None,  # Using name mapping inside runner
            attacker_mode=attacker_mode, 
            strategy_name=s_name,
            anomaly_detector=anomaly_detector, 
            n_runs=10,  # Keeping runs lower for execution speed demo
            seed=42
        )
        rows.append(res)
        
    df = pd.DataFrame(rows)
    df = df.sort_values(by='mean_infection_rate', ascending=True).reset_index(drop=True)
    return df

def plot_strategy_comparison(results_df):
    """
    Grouped bar chart comparing all strategies across all metrics.
    Colors anomaly_guided strategy differently.
    """
    categories = ['mean_infection_rate', 'mean_spread_velocity', 'mean_containment_time']
    strategies = results_df['strategy_name'].tolist()
    
    x = np.arange(len(strategies))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#1f77b4' if s != 'anomaly_guided' else '#d62728' for s in strategies]
    
    # Plot bars dynamically handling the three primary metrics
    for i, cat in enumerate(categories):
        means = results_df[cat].tolist()
        errors = results_df[cat.replace('mean_', 'std_')].tolist()
        
        # We normalize spread velocity and containment time against the maximums so they graph 
        # on a similar visual scale as infection_rate (0-1)
        max_val = max(means) if max(means) > 0 else 1
        normalized_means = [m / max_val for m in means]
        normalized_errors = [e / max_val for e in errors]
        
        ax.bar(
            x + (i * width) - width, 
            normalized_means, 
            width, 
            yerr=normalized_errors, 
            label=f"{cat} (normalized)",
            capsize=5,
            color=colors,
            alpha=[1.0, 0.7, 0.4][i] # Differentiating the metrics within the colors
        )

    ax.set_ylabel('Normalized Score', fontsize=12)
    ax.set_title('Defense Strategy Comparison (Lower is generally better)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    
    # Custom legend
    from matplotlib.patches import Patch
    custom_lines = [
        Patch(facecolor='grey', alpha=1.0, label='Standard Strategy'),
        Patch(facecolor='#d62728', alpha=1.0, label='AI Integration (Anomaly Guided)'),
        Patch(facecolor='black', alpha=1.0, label='Infection Rate'),
        Patch(facecolor='black', alpha=0.7, label='Spread Velocity'),
        Patch(facecolor='black', alpha=0.4, label='Containment Time')
    ]
    ax.legend(handles=custom_lines, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=300)
    print("\nSaved strategy_comparison.png")
    # plt.close()  # suppressed for non-interactive Agg backend

def plot_infection_curves(curve_dict):
    """
    curve_dict maps strategy_name to list of mean infected_count per timestep.
    Plot all strategies as lines on one chart.
    """
    plt.figure(figsize=(12, 7))
    
    for strategy, counts in curve_dict.items():
        timesteps = range(len(counts))
        linewidth = 3 if strategy == 'anomaly_guided' else 1.5
        linestyle = '-' if strategy == 'anomaly_guided' else '--'
        plt.plot(timesteps, counts, label=strategy, linewidth=linewidth, linestyle=linestyle)
        
    plt.title('Infection Curves by Defense Strategy', fontsize=16)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Mean Infected Count', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('infection_curves.png', dpi=300)
    print("Saved infection_curves.png")
    # plt.close()  # suppressed for non-interactive Agg backend

if __name__ == "__main__":
    from network_graph import generate_enterprise_graph
    from anomaly_detector import AnomalyDetector
    
    print("Generating simulation environment...")
    G = generate_enterprise_graph(seed=2048)
    detector = AnomalyDetector(contamination=0.1, random_state=42)
    
    # 1. Compare all strategies (This acts as both the logic test and data generator)
    print("\nExecuting multi-run defense analysis...")
    df_results = compare_all_strategies(G, attacker_mode="greedy", anomaly_detector=detector)
    
    print("\n" + "="*80)
    print("STRATEGY REPORT RANKINGS (Sorted by Infection Rate)")
    print("="*80)
    show_cols = ['strategy_name', 'mean_infection_rate', 'mean_time_to_critical', 'mean_spread_velocity']
    print(df_results[show_cols].to_string(index=False))
    
    # 2. Plotting comparisons
    plot_strategy_comparison(df_results)
    
    # 3. Plot Real Infection Curves Generated From Average Monte Carlo Results
    curve_data = {}
    for _, row in df_results.iterrows():
        curve_data[row['strategy_name']] = row['mean_infection_curve']
        
    plot_infection_curves(curve_data)