import numpy as np

# We import the baseline logic here. It should gracefully import assuming defense_simulator exists
# or can be supplied as a fallback mapping.
try:
    from defense_simulator import strategy_patch_centrality
except ImportError:
    def strategy_patch_centrality(G, budget):
        return []


# ---------------------------------------------------------------------------
# Utility: pure-NumPy GraphSAGE helpers
# ---------------------------------------------------------------------------

def _relu(x):
    return np.maximum(0, x)


def _normalize_rows(x, eps=1e-8):
    """L2-normalise each row of a 2-D matrix."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _build_adj_list(G):
    """
    Returns a list-of-lists adjacency structure indexed by sorted node order.
    Index i in the list corresponds to the i-th entry in sorted(G.nodes()).
    """
    nodes = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    adj = [[] for _ in nodes]
    for u, v in G.edges():
        adj[node_to_idx[u]].append(node_to_idx[v])
        adj[node_to_idx[v]].append(node_to_idx[u])
    return adj, node_to_idx


def _sage_aggregate(H, adj):
    """
    One GraphSAGE mean-aggregation step.
    H   : (n, d) node embedding matrix
    adj : list-of-lists adjacency
    Returns aggregated (n, d) matrix — for each node, mean of neighbour embeddings.
    Falls back to the node's own embedding when degree == 0.
    """
    n, d = H.shape
    agg = np.zeros_like(H)
    for i, neighbours in enumerate(adj):
        if neighbours:
            agg[i] = H[neighbours].mean(axis=0)
        else:
            agg[i] = H[i]
    return agg


def _sage_layer(H, adj, W_self, W_neigh, b):
    """
    Single GraphSAGE layer:
        h_new = ReLU( h_self @ W_self^T + h_agg @ W_neigh^T + b )
    """
    agg = _sage_aggregate(H, adj)
    return _relu(H @ W_self.T + agg @ W_neigh.T + b)


# ---------------------------------------------------------------------------
# GNNPredictor  (GraphSAGE autoencoder for unsupervised anomaly detection)
# ---------------------------------------------------------------------------

class GNNPredictor:
    """
    Two-layer GraphSAGE encoder -> linear decoder, trained as a node-feature
    autoencoder on baseline (clean) graph state.

    Anomaly score for a node == its per-node mean-squared reconstruction error.
    Nodes that look unusual relative to the baseline will reconstruct poorly
    and therefore receive high anomaly scores -- no infection labels needed.

    Architecture
    ------------
    Input  : (n_nodes, feature_dim)
    Layer 1: GraphSAGE  feature_dim  -> hidden_dim     (ReLU)
    Layer 2: GraphSAGE  hidden_dim   -> hidden_dim//2  (ReLU)
    Decoder: Linear     hidden_dim//2 -> feature_dim   (linear)

    Training
    --------
    Loss  : mean per-node MSE between input and reconstruction
    Optim : SGD with momentum (pure NumPy, no external ML deps)
    """

    def __init__(self, feature_dim=10, hidden_dim=64, lr=0.01, epochs=200,
                 random_state=42):
        self.feature_dim = feature_dim
        self.hidden_dim  = hidden_dim
        self.latent_dim  = hidden_dim // 2
        self.lr          = lr
        self.epochs      = epochs
        self.rng         = np.random.default_rng(random_state)
        self.is_fitted   = False
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation  (He init for ReLU layers)
    # ------------------------------------------------------------------

    def _init_weights(self):
        fd, hd, ld = self.feature_dim, self.hidden_dim, self.latent_dim

        def he(fan_in, shape):
            std = np.sqrt(2.0 / fan_in)
            return self.rng.normal(0, std, shape).astype(np.float64)

        # Encoder layer 1  (feature_dim -> hidden_dim)
        self.W1_self  = he(fd, (hd, fd))
        self.W1_neigh = he(fd, (hd, fd))
        self.b1       = np.zeros(hd, dtype=np.float64)

        # Encoder layer 2  (hidden_dim -> latent_dim)
        self.W2_self  = he(hd, (ld, hd))
        self.W2_neigh = he(hd, (ld, hd))
        self.b2       = np.zeros(ld, dtype=np.float64)

        # Decoder  (latent_dim -> feature_dim, linear)
        self.W_dec = he(ld, (fd, ld))
        self.b_dec = np.zeros(fd, dtype=np.float64)

        # Momentum buffers (SGD with momentum)
        self._mom = {k: np.zeros_like(v) for k, v in self._params().items()}

    def _params(self):
        return {
            'W1_self': self.W1_self,  'W1_neigh': self.W1_neigh,  'b1': self.b1,
            'W2_self': self.W2_self,  'W2_neigh': self.W2_neigh,  'b2': self.b2,
            'W_dec':   self.W_dec,    'b_dec':    self.b_dec,
        }

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, X, adj):
        """Returns (reconstruction, H1, H2, agg1, agg2)."""
        # Encoder L1
        agg1   = _sage_aggregate(X, adj)
        H1     = _relu(X @ self.W1_self.T + agg1 @ self.W1_neigh.T + self.b1)

        # Encoder L2
        agg2   = _sage_aggregate(H1, adj)
        H2     = _relu(H1 @ self.W2_self.T + agg2 @ self.W2_neigh.T + self.b2)

        # Decoder (linear)
        recon  = H2 @ self.W_dec.T + self.b_dec

        return recon, H1, H2, agg1, agg2

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, G, simulation_logs=None):
        """
        Train the autoencoder on the CURRENT (baseline) graph state.

        Parameters
        ----------
        G               : NetworkX graph with node features populated
        simulation_logs : ignored (kept for API compatibility)
        """
        X = self.extract_features(G)  # float64 from extract_features — no cast needed

        # Column-wise min-max normalisation; save scale for inference
        self._X_min = X.min(axis=0)
        self._X_max = X.max(axis=0)
        scale  = np.maximum(self._X_max - self._X_min, 1e-8)
        X_norm = (X - self._X_min) / scale

        adj, _ = _build_adj_list(G)
        mu = 0.9   # momentum coefficient
        n  = X_norm.shape[0]

        loss_history = []

        for epoch in range(self.epochs):
            recon, H1, H2, agg1, agg2 = self._forward(X_norm, adj)

            # MSE loss
            diff = recon - X_norm                           # (n, fd)
            loss = float(np.mean(diff ** 2))
            loss_history.append(loss)

            # ---- Decoder gradients ----
            d_recon    = (2.0 / n) * diff                   # (n, fd)
            d_W_dec    = d_recon.T @ H2                     # (fd, ld)
            d_b_dec    = d_recon.sum(axis=0)                # (fd,)
            d_H2       = d_recon @ self.W_dec               # (n, ld)

            # ---- Encoder L2 gradients (through ReLU) ----
            d_H2_pre   = d_H2 * (H2 > 0)                   # (n, ld)
            d_W2_self  = d_H2_pre.T @ H1                    # (ld, hd)
            d_W2_neigh = d_H2_pre.T @ agg2                  # (ld, hd)
            d_b2       = d_H2_pre.sum(axis=0)               # (ld,)
            d_H1       = d_H2_pre @ self.W2_self             # (n, hd)
            # Gradient flows back through the neighbour aggregation as well
            d_H1      += _sage_aggregate(d_H2_pre @ self.W2_neigh, adj)

            # ---- Encoder L1 gradients (through ReLU) ----
            d_H1_pre   = d_H1 * (H1 > 0)                   # (n, hd)
            d_W1_self  = d_H1_pre.T @ X_norm                # (hd, fd)
            d_W1_neigh = d_H1_pre.T @ agg1                  # (hd, fd)
            d_b1       = d_H1_pre.sum(axis=0)               # (hd,)

            # ---- SGD + momentum parameter update ----
            grads = {
                'W1_self': d_W1_self,  'W1_neigh': d_W1_neigh,  'b1': d_b1,
                'W2_self': d_W2_self,  'W2_neigh': d_W2_neigh,  'b2': d_b2,
                'W_dec':   d_W_dec,    'b_dec':    d_b_dec,
            }
            params = self._params()
            for name, grad in grads.items():
                self._mom[name]  = mu * self._mom[name] + self.lr * grad
                params[name]    -= self._mom[name]

            if (epoch + 1) % 50 == 0:
                print(f"  GNN Autoencoder | epoch {epoch+1:3d}/{self.epochs}"
                      f" | loss {loss:.6f}")

        self.is_fitted = True
        self._baseline_errors = self._per_node_errors(X_norm, adj)
        print(f"  GNN training complete. "
              f"Baseline mean recon error: {self._baseline_errors.mean():.4f} "
              f"(std: {self._baseline_errors.std():.4f})")
        return loss_history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _per_node_errors(self, X_norm, adj):
        """
        Returns (n,) array of per-node MSE reconstruction errors.
        Clips X_norm to [-5, 5] before forward pass to prevent activation
        explosion on out-of-distribution (post-infection) features.
        """
        X_safe = np.clip(X_norm, -5.0, 5.0)
        recon, *_ = self._forward(X_safe, adj)
        diff   = recon.astype(np.float64) - X_safe.astype(np.float64)
        errors = np.mean(diff ** 2, axis=1)
        errors = np.where(np.isfinite(errors), errors, 1e6)
        return errors

    def score_nodes(self, G):
        """
        Compute anomaly scores for all nodes relative to each node's OWN
        baseline reconstruction error.

        Score = log1p(max(node_error/node_baseline - 1, 0))  — log-scaled per-node ratio

            0.0       normal (at or below baseline)
            > 0.5     mildly anomalous
            > 1.5     strongly anomalous (likely infected / compromised)

        Capped at 20 for readability. Updates G.nodes[n]['anomaly_score'].
        Returns dict {node_id: score}.
        """
        if not self.is_fitted:
            raise RuntimeError("GNNPredictor: call train() before score_nodes().")

        X      = self.extract_features(G)
        scale  = np.maximum(self._X_max - self._X_min, 1e-8)
        X_norm = (X - self._X_min) / scale

        adj, _ = _build_adj_list(G)
        errors = self._per_node_errors(X_norm, adj)

        # Per-node normalised ratio, then log-scale to compress the range
        # log1p(ratio - 1): baseline=0, mild anomaly=~0.5-1.5, strong=2.0+
        per_node_baseline = np.maximum(self._baseline_errors, 1e-8)
        ratio  = np.clip(errors / per_node_baseline, 0.0, 1e6)
        scores = np.log1p(np.maximum(ratio - 1.0, 0.0))   # 0 at baseline, grows with anomaly

        result = {}
        for i, node in enumerate(sorted(G.nodes())):
            s = round(float(scores[i]), 3)
            result[node] = s
            G.nodes[node]['anomaly_score'] = s

        return result

    def predict_infection_probability(self, G):
        """
        Returns per-node infection probability proxy (0–1).
        Uses normalised anomaly score if trained; falls back to risk_score.
        """
        if self.is_fitted:
            raw = self.score_nodes(G)
            max_s = max(raw.values()) or 1.0
            return {n: min(s / max_s, 1.0) for n, s in raw.items()}

        max_rs = max([d.get('risk_score', 0.0)
                      for _, d in G.nodes(data=True)] + [1.0])
        return {n: d.get('risk_score', 0.0) / max_rs
                for n, d in G.nodes(data=True)}

    # ------------------------------------------------------------------
    # Feature extraction  (same contract as original stub)
    # ------------------------------------------------------------------

    def extract_features(self, G):
        """
        Extracts 10 node features from the graph.
        Returns numpy array of shape (n_nodes, 10).
        Node order: sorted(G.nodes()).
        """
        nodes    = sorted(G.nodes())
        features = np.zeros((len(nodes), 10), dtype=np.float64)

        for i, node_id in enumerate(nodes):
            data   = G.nodes[node_id]
            degree = G.degree(node_id)

            if degree > 0:
                nbrs = list(G.neighbors(node_id))
                mean_trust_weight      = np.mean([G[node_id][nb].get('trust_weight', 0.5)      for nb in nbrs])
                mean_traffic_frequency = np.mean([G[node_id][nb].get('traffic_frequency', 0.5) for nb in nbrs])
                mean_exploitability    = np.mean([G[node_id][nb].get('exploitability', 0.5)     for nb in nbrs])
            else:
                mean_trust_weight = mean_traffic_frequency = mean_exploitability = 0.5

            vulnerability_score     = data.get('vulnerability_score', 0.0)
            priv_level              = data.get('privilege_level', 'user')
            privilege_level_numeric = {'domain_admin': 1.0, 'admin': 0.66}.get(priv_level, 0.33)
            asset_value_normalized  = data.get('asset_value', 1.0) / 10.0
            n_infected_neighbors    = sum(1 for nb in G.neighbors(node_id)
                                          if G.nodes[nb].get('infection_state') == 'infected')
            n_detected_neighbors    = sum(1 for nb in G.neighbors(node_id)
                                          if G.nodes[nb].get('detected', False))
            anomaly_score           = data.get('anomaly_score', 0.0)

            features[i] = [
                degree,
                mean_trust_weight,
                mean_traffic_frequency,
                mean_exploitability,
                vulnerability_score,
                privilege_level_numeric,
                asset_value_normalized,
                n_infected_neighbors,
                n_detected_neighbors,
                anomaly_score,
            ]

        return features


# ---------------------------------------------------------------------------
# RLDefenseAgent  (stub — wiring into defense loop is a separate roadmap item)
# ---------------------------------------------------------------------------

class RLDefenseAgent:
    def __init__(self, state_dim=5, action_dim=100, budget=5):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.budget     = budget

    def get_state(self, G):
        n_nodes = G.number_of_nodes()
        if n_nodes == 0:
            return np.zeros(self.state_dim)

        infected_nodes   = [n for n, d in G.nodes(data=True) if d.get('infection_state') == 'infected']
        infection_rate   = len(infected_nodes) / n_nodes
        mean_risk_score  = np.mean([d.get('risk_score', 0.0) for _, d in G.nodes(data=True)])
        detected_nodes   = [n for n, d in G.nodes(data=True) if d.get('detected', False)]
        n_det_norm       = len(detected_nodes) / n_nodes

        critical_nodes   = [n for n, d in G.nodes(data=True)
                            if d.get('node_type') in ['server', 'controller', 'database']]
        crit_infected    = [n for n in critical_nodes if n in infected_nodes]
        n_crit_norm      = len(crit_infected) / max(len(critical_nodes), 1)

        max_t            = G.graph.get('max_timesteps', 50)
        current_t        = G.graph.get('timestep', 0)
        t_norm           = current_t / float(max_t) if max_t > 0 else 0.0

        return np.array([infection_rate, mean_risk_score, n_det_norm,
                         n_crit_norm, t_norm])

    def select_action(self, state, G):
        """
        Real implementation: PPO policy network
        State -> 2-layer MLP -> action logits over all nodes
        Sample top-budget nodes from softmax distribution
        """
        return strategy_patch_centrality(G, self.budget)

    def compute_reward(self, G, prev_infection_rate):
        n_nodes = G.number_of_nodes()
        if n_nodes == 0:
            return 0.0
        infected_nodes         = [n for n, d in G.nodes(data=True) if d.get('infection_state') == 'infected']
        current_infection_rate = len(infected_nodes) / n_nodes
        new_infections         = current_infection_rate - prev_infection_rate
        critical_nodes         = [n for n, d in G.nodes(data=True)
                                  if d.get('node_type') in ['server', 'controller', 'database']]
        crit_new               = sum(1 for n in critical_nodes
                                     if G.nodes[n].get('newly_infected', False))
        nodes_patched          = sum(1 for n, d in G.nodes(data=True)
                                     if d.get('patched_this_timestep', False))
        return float((-10 * new_infections * 100) + (-50 * crit_new) + (5 * nodes_patched))

    def update(self, state, action, reward, next_state):
        """PPO update — stub for future wiring."""
        pass