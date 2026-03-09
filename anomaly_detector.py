import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import copy

from network_graph import generate_enterprise_graph, reset_graph
from propagation_engine import run_simulation
from ai_modules import GNNPredictor


class AnomalyDetector:
    def __init__(self, contamination=0.1, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination, 
            random_state=self.random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        
    def extract_node_features(self, G, node_id):
        """
        Extracts features for a single node into a numpy array (1D).
        """
        data = G.nodes[node_id]
        
        # 1. degree (int)
        degree = G.degree(node_id)
        
        # Aggregate edge metrics
        neighbors = list(G.neighbors(node_id))
        if neighbors:
            mean_edge_trust_weight = np.mean([G[node_id][nbr].get('trust_weight', 0.5) for nbr in neighbors])
            mean_edge_traffic_frequency = np.mean([G[node_id][nbr].get('traffic_frequency', 0.5) for nbr in neighbors])
            mean_edge_exploitability = np.mean([G[node_id][nbr].get('exploitability', 0.5) for nbr in neighbors])
        else:
            mean_edge_trust_weight = 0.0
            mean_edge_traffic_frequency = 0.0
            mean_edge_exploitability = 0.0
            
        # 5. vulnerability_score (float)
        v_score = data.get('vulnerability_score', 0.0)
        
        # 6. privilege_level_numeric
        priv_map = {'user': 0.33, 'admin': 0.66, 'domain_admin': 1.0}
        priv_num = priv_map.get(data.get('privilege_level', 'user'), 0.33)
        
        # 7. asset_value (normalized 0-1) - assuming max is 10 as per network_graph.py
        asset_val = data.get('asset_value', 1) / 10.0
        
        # 8 & 9. neighbor states
        n_infected = sum(1 for nbr in neighbors if G.nodes[nbr].get('infection_state') == 'infected')
        n_detected = sum(1 for nbr in neighbors if G.nodes[nbr].get('detected', False))
        
        # 10. baseline_deviation
        # Current traffic is mapped as trust_weight * traffic_frequency as defined in propagation_engine.py
        # But wait, the standard edge attributes don't change during simulation exactly.
        # How do we compute deviation? We compare current behavior to baseline. 
        # For this simulator construct, we will emulate deviation scaling if nodes are infected.
        # So we calculate "current_traffic" dynamically and compare to baseline.
        baseline = data.get('baseline_traffic', {})
        total_deviation = 0.0
        
        if baseline:
            for nbr in neighbors:
                str_nbr = str(nbr)
                if str_nbr in baseline:
                    base_val = baseline[str_nbr]
                    
                    # Compute current dynamic traffic
                    # If neighbor is infected, traffic spikes (simulating propagation comms)
                    current_freq = G[node_id][nbr].get('traffic_frequency', 0.5)
                    current_weight = G[node_id][nbr].get('trust_weight', 0.5)
                    current_val = current_freq * current_weight
                    
                    if G.nodes[nbr].get('infection_state') == 'infected' or data.get('infection_state') == 'infected':
                        current_val *= 1.5  # Simulate 50% traffic spike during infection comms
                        
                    total_deviation += abs(current_val - base_val)
                    
            baseline_deviation = total_deviation / len(neighbors) if len(neighbors) > 0 else 0.0
        else:
            baseline_deviation = 0.0
            
        return np.array([
            degree,
            mean_edge_trust_weight,
            mean_edge_traffic_frequency,
            mean_edge_exploitability,
            v_score,
            priv_num,
            asset_val,
            n_infected,
            n_detected,
            baseline_deviation
        ])
        
    def build_feature_matrix(self, G):
        """
        Extract features for all nodes, return as matrix of shape (n_nodes, 10)
        """
        # Ensure consistent node ordering
        nodes = list(G.nodes())
        nodes.sort() 
        matrix = np.array([self.extract_node_features(G, n) for n in nodes])
        return matrix
        
    def fit_baseline(self, G):
        """
        Train IsolationForest on the current graph's feature matrix.
        Call this after baseline_traffic is recorded.
        """
        X = self.build_feature_matrix(G)
        self.model.fit(X)
        self.is_fitted = True
        print("AnomalyDetector: Isolation Forest trained on baseline traffic.")
        
    def score_nodes(self, G):
        """
        Score all nodes using the trained IsolationForest.
        anomaly_score is the negative of decision_function output (higher = more anomalous).
        Updates each node's anomaly_score attribute.
        Returns dict {node_id: anomaly_score}
        """
        if not self.is_fitted:
            print("WARNING: AnomalyDetector not trained. Call fit_baseline() first. Returning 0.0 scores.")
            return {n: 0.0 for n in G.nodes()}
            
        X = self.build_feature_matrix(G)
        
        # decision_function returns positive for inliers, negative for outliers
        # We invert it so higher = more anomalous
        scores = -self.model.decision_function(X)
        
        nodes = list(G.nodes())
        nodes.sort()
        
        result = {}
        for i, n in enumerate(nodes):
            score = float(scores[i])
            result[n] = score
            G.nodes[n]['anomaly_score'] = score
            
        return result
        
    def detect_anomalies(self, G, threshold=2.0):
        """
        Return list of nodes whose anomaly_score exceeds threshold.
        Sets detected=True on those nodes in the graph.
        """
        if not self.is_fitted:
            return []
            
        anomalies = []
        scores = self.score_nodes(G)
        
        for n, score in scores.items():
            if score > threshold:
                anomalies.append(n)
                G.nodes[n]['detected'] = True
            else:
                # Optionally reset false alarms if score drops, but usually detected stays True.
                # We'll leave it as True once tripped for the simulator construct unless overridden reset.
                pass
                
        return anomalies
        
    def compute_detection_metrics(self, detection_log, infection_log):
        """
        Computes performance metrics based on logging arrays.
        detection_log: list of (timestep, [detected_node_ids])
        infection_log: list of (timestep, [infected_node_ids])
        """
        # Flatten infection history to find actual infection time for each node
        true_infection_times = {}
        for t, nodes in infection_log:
            for n in nodes:
                if n not in true_infection_times:
                    true_infection_times[n] = t
                    
        # Flatten detection history
        first_detection_times = {}
        for t, nodes in detection_log:
            for n in nodes:
                if n not in first_detection_times:
                    first_detection_times[n] = t
                    
        # Calculate metric sets
        true_positives = set()
        false_positives = set()
        lead_times = []
        
        all_infected = set(true_infection_times.keys())
        all_detected = set(first_detection_times.keys())
        
        for n in all_detected:
            if n in all_infected:
                true_positives.add(n)
                # Lead time: Infection Time - Detection Time 
                # (If detection happened before infection, lead time is positive)
                lead = true_infection_times[n] - first_detection_times[n]
                lead_times.append(lead)
            else:
                false_positives.add(n)
                
        detection_coverage = len(true_positives) / len(all_infected) if all_infected else 0.0
        
        # TPR = TP / All Infected (Recall)
        tpr = len(true_positives) / len(all_infected) if all_infected else 0.0
        
        # FPR is tricky without knowing "all true negatives" properly, 
        # standard fallback is False Positives / All Never Infected
        all_never_infected = [n for _, nodes in detection_log for n in nodes if n not in all_infected] 
        # A more standard approach for this sim: False Alarms / Total Nodes 
        # But we'll follow standard ML approx for the count: ratio of incorrect alarms
        fpr = len(false_positives) / len(all_detected) if all_detected else 0.0
        
        metrics = {
            'detection_lead_time_mean': np.mean(lead_times) if lead_times else 0.0,
            'detection_lead_time_std': np.std(lead_times) if lead_times else 0.0,
            'true_positive_rate': tpr,
            'false_positive_rate': fpr,  # This is technically FDR (False Discovery Rate), but commonly requested this way
            'detection_coverage': detection_coverage,
            'total_infected': len(all_infected),
            'total_detected': len(all_detected)
        }
        
        return metrics

    def plot_detection_timeline(self, detection_log, infection_log, title_suffix=""):
        """
        Plots cumulative infected vs detected node counts over time.
        detection_log : list of (timestep, [detected_node_ids])
        infection_log : list of (timestep, [newly_infected_node_ids])
        """
        infection_dict = {}
        for t, nodes in infection_log:
            infection_dict.setdefault(t, []).extend(nodes)

        detection_dict = {}
        for t, nodes in detection_log:
            detection_dict.setdefault(t, []).extend(nodes)

        all_timesteps = sorted(set(infection_dict.keys()) | set(detection_dict.keys()))

        if not all_timesteps:
            print("  plot_detection_timeline: no data to plot.")
            return

        cum_infected = []
        cum_detected = []
        inf_set = set()
        det_set = set()

        for t in all_timesteps:
            inf_set.update(infection_dict.get(t, []))
            det_set.update(detection_dict.get(t, []))
            cum_infected.append(len(inf_set))
            cum_detected.append(len(det_set))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(all_timesteps, cum_infected, color="crimson",
                label="Cumulative Infected", linewidth=2.5)
        ax.plot(all_timesteps, cum_detected, color="goldenrod",
                label="Cumulative Detected (GNN)", linewidth=2.5, linestyle="--")

        # Shade the gap between detected and infected
        ax.fill_between(all_timesteps, cum_detected, cum_infected,
                        alpha=0.15, color="crimson", label="Undetected gap")

        title = "GNN Anomaly Detection Timeline vs Infection Spread"
        if title_suffix:
            title += f" — {title_suffix}"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Simulation Timestep", fontsize=12)
        ax.set_ylabel("Cumulative Node Count", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)
        plt.tight_layout()

        output_file = "detection_timeline.png"
        plt.savefig(output_file, dpi=150)
        print(f"  Detection timeline saved to {output_file}")
        plt.close(fig)

# ---------------------------------------------------------------------------
# GNNAnomalyDetector  — drop-in replacement for AnomalyDetector that uses
# the GraphSAGE autoencoder from ai_modules.py instead of IsolationForest.
#
# Exposes the identical public API:
#   fit_baseline(G)
#   score_nodes(G)     -> dict {node_id: score}
#   detect_anomalies(G, threshold) -> list of node_ids
#   compute_detection_metrics(detection_log, infection_log) -> dict
#   plot_detection_timeline(detection_log, infection_log)
# ---------------------------------------------------------------------------

class GNNAnomalyDetector:
    """
    Unsupervised graph anomaly detector backed by a two-layer GraphSAGE
    autoencoder (pure NumPy — no PyTorch required).

    Nodes are scored by their per-node reconstruction error relative to the
    baseline mean error.  A threshold > 1.0 selects nodes that look more
    anomalous than the average baseline node.  The default threshold of 1.5
    is a conservative starting point; tune upward to reduce false positives.
    """

    def __init__(self, hidden_dim=64, lr=0.01, epochs=200, random_state=42):
        self.hidden_dim   = hidden_dim
        self.lr           = lr
        self.epochs       = epochs
        self.random_state = random_state
        self.model        = GNNPredictor(
            feature_dim=10,
            hidden_dim=hidden_dim,
            lr=lr,
            epochs=epochs,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit_baseline(self, G):
        """
        Train the GraphSAGE autoencoder on the clean baseline graph state.
        Must be called before score_nodes / detect_anomalies.
        """
        print("GNNAnomalyDetector: Training GraphSAGE autoencoder on baseline graph...")
        self.model.train(G)
        self.is_fitted = True

    def score_nodes(self, G):
        """
        Score all nodes.  Returns dict {node_id: anomaly_score} and updates
        G.nodes[n]['anomaly_score'] in place.
        Score ~= 1.0 -> normal; score >> 1.0 -> anomalous.
        """
        if not self.is_fitted:
            print("WARNING: GNNAnomalyDetector not trained. "
                  "Call fit_baseline() first. Returning 0.0 scores.")
            return {n: 0.0 for n in G.nodes()}
        return self.model.score_nodes(G)

    def detect_anomalies(self, G, threshold=2.0):
        """
        Return list of node IDs whose anomaly_score exceeds `threshold`.
        Sets detected=True on those nodes in the graph.

        Default threshold of 1.5 means "50 % more anomalous than the
        average baseline node".  The IsolationForest equivalent used 0.5
        on a negated decision-function scale; both are tunable knobs.
        """
        if not self.is_fitted:
            return []

        scores    = self.score_nodes(G)
        anomalies = []
        for n, score in scores.items():
            if score > threshold:
                anomalies.append(n)
                G.nodes[n]['detected'] = True
        return anomalies

    # Delegate shared metric / plotting logic to the standalone functions
    # already defined on AnomalyDetector so we don't duplicate code.

    def compute_detection_metrics(self, detection_log, infection_log):
        """Identical metric computation to AnomalyDetector."""
        _helper = AnomalyDetector.__new__(AnomalyDetector)
        return AnomalyDetector.compute_detection_metrics(
            _helper, detection_log, infection_log)

    def plot_detection_timeline(self, detection_log, infection_log, title_suffix=""):
        """Identical timeline plot to AnomalyDetector."""
        _helper = AnomalyDetector.__new__(AnomalyDetector)
        return AnomalyDetector.plot_detection_timeline(
            _helper, detection_log, infection_log, title_suffix=title_suffix)


# ---------------------------------------------------------------------------
# Factory: returns the right detector based on config AI_MODE flag
# ---------------------------------------------------------------------------

def make_anomaly_detector(contamination=0.1, random_state=42,
                           hidden_dim=64, lr=0.01, epochs=200,
                           use_gnn=False):
    """
    Factory function that returns either a GNNAnomalyDetector or the
    classic IsolationForest-based AnomalyDetector.

    Parameters
    ----------
    use_gnn : bool
        Pass True (or set AI_MODE=True in config.py) to use the GNN backend.
        Defaults to False so existing pipelines are unaffected.

    Usage in main.py
    ----------------
        from config import AI_MODE
        from anomaly_detector import make_anomaly_detector
        detector = make_anomaly_detector(
            contamination=ANOMALY_CONFIG['contamination'],
            random_state=SIMULATION_CONFIG['seed'],
            use_gnn=AI_MODE,
        )
    """
    if use_gnn:
        print("[Config] AI_MODE=True -> using GNNAnomalyDetector (GraphSAGE autoencoder)")
        return GNNAnomalyDetector(
            hidden_dim=hidden_dim,
            lr=lr,
            epochs=epochs,
            random_state=random_state,
        )
    else:
        print("[Config] AI_MODE=False -> using AnomalyDetector (IsolationForest)")
        return AnomalyDetector(contamination=contamination, random_state=random_state)


def run_anomaly_detection_experiment(G, simulation_result):
    """
    Takes a completed simulation result dict from propagation_engine
    Replays the timestep log and scores nodes.
    Returns metrics and detection_log.
    """
    print("\n" + "="*50)
    print("ANOMALY DETECTION REPLAY EXPERIMENT")
    print("="*50)
    
    detector = AnomalyDetector(contamination=0.1, random_state=42)
    
    # We need a clean graph to replay state into.
    # The baseline_traffic is already populated on G from the original simulation run 
    # so fit_baseline will work correctly after we reset the infection states.
    sim_G = copy.deepcopy(G)
    reset_graph(sim_G)
    
    detection_log = []
    infection_log = []
    
    timestep_data = simulation_result['timestep_log']
    
    # 1. Identify baseline phase and train
    # We train the detector now using the clean Baseline state of sim_G
    detector.fit_baseline(sim_G)
    
    # 2. Replay remaining phases
    action_steps = [log for log in timestep_data if log['phase'] != 'baseline_recording']
    
    for log in action_steps:
        t = log['timestep']
        newly_infected = log['newly_infected_nodes']
        
        # Apply infections to sim graph
        for n in newly_infected:
            sim_G.nodes[n]['infection_state'] = 'infected'
            
        # Run detection
        # An IsolationForest decision_function roughly bounds -0.5 to 0.5. 
        # Negated, higher than 0.5 is anomaly based on spec.
        detected_now = detector.detect_anomalies(sim_G, threshold=0.5)
        
        # We will log the current active anomaly state list per timestep.
        detection_log.append((t, detected_now))
        
        # Log newly infected nodes directly from timestep data
        infection_log.append((t, newly_infected))
        
    # 3. Calculate metrics
    metrics = detector.compute_detection_metrics(detection_log, infection_log)
    
    return metrics, detection_log, infection_log, detector


if __name__ == "__main__":
    print("Generating network map...")
    G = generate_enterprise_graph(seed=999)
    
    print("Running initial contagion simulation...")
    # Beta=0.6 for an aggressive spread so the anomaly detector has enough data
    sim_result = run_simulation(G, max_timesteps=30, attacker_mode="greedy", beta=0.6, seed=999)
    print(f"Simulation complete. Total timesteps: {sim_result['total_timesteps']}")
    
    print("\nRunning anomaly detector analysis...")
    metrics, det_log, inf_log, detector = run_anomaly_detection_experiment(G, sim_result)
    
    print("\n--- Detection Performance Metrics ---")
    print(f"Detection Coverage:   {metrics['detection_coverage']*100:.1f}%")
    print(f"True Positive Rate:   {metrics['true_positive_rate']*100:.1f}%")
    print(f"False Discovery Rate: {metrics['false_positive_rate']*100:.1f}%")
    
    lead_mean = metrics['detection_lead_time_mean']
    lead_std = metrics['detection_lead_time_std']
    
    if lead_mean > 0:
         print(f"Mean Lead Time:       +{lead_mean:.1f} timesteps (std ±{lead_std:.1f}) [SUCCESS: Detected before infection]")
    else:
         print(f"Mean Lead Time:       {lead_mean:.1f} timesteps (std ±{lead_std:.1f}) [LATE: Detected after infection]")
         
    # Plot timeline
    detector.plot_detection_timeline(det_log, inf_log)