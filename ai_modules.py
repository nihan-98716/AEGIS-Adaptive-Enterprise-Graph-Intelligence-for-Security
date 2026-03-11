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
# DQN Defense Agent — pure NumPy, no PyTorch required
# ---------------------------------------------------------------------------

class _DQNNetwork:
    """
    Lightweight 2-layer MLP: state_dim -> 128 -> 64 -> action_dim
    Trained with SGD + experience replay entirely in NumPy.
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, seed=42):
        rng = np.random.default_rng(seed)
        self.lr = lr
        # Xavier initialisation
        def xavier(fan_in, fan_out):
            lim = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-lim, lim, (fan_in, fan_out))

        self.W1 = xavier(state_dim, 128);  self.b1 = np.zeros(128)
        self.W2 = xavier(128, 64);          self.b2 = np.zeros(64)
        self.W3 = xavier(64, action_dim);   self.b3 = np.zeros(action_dim)

    def forward(self, x):
        self._x  = x
        self._h1 = np.maximum(0, x @ self.W1 + self.b1)
        self._h2 = np.maximum(0, self._h1 @ self.W2 + self.b2)
        self._q  = self._h2 @ self.W3 + self.b3
        return self._q

    def backward(self, loss_grad):
        """Backprop through 2-layer ReLU MLP, SGD update."""
        dq   = loss_grad                              # (action_dim,)
        dW3  = np.outer(self._h2, dq)
        db3  = dq
        dh2  = dq @ self.W3.T
        dh2 *= (self._h2 > 0)
        dW2  = np.outer(self._h1, dh2)
        db2  = dh2
        dh1  = dh2 @ self.W2.T
        dh1 *= (self._h1 > 0)
        dW1  = np.outer(self._x, dh1)
        db1  = dh1

        self.W3 -= self.lr * dW3;  self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1

    def save(self, path):
        import os; os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)

    def load(self, path):
        d = np.load(path + '.npz')
        self.W1, self.b1 = d['W1'], d['b1']
        self.W2, self.b2 = d['W2'], d['b2']
        self.W3, self.b3 = d['W3'], d['b3']


class RLDefenseAgent:
    """
    DQN agent that learns which nodes to patch each timestep.

    State  : 5 features per node (infection, risk, anomaly, vulnerability, degree)
             flattened to (n_nodes * 5,) vector — padded/truncated to state_dim.
    Action : integer node index to patch (applied budget times greedily).
    Reward : -10 * new_infections_pct - 50 * critical_infected + 5 * patched_count
    Training: experience replay, epsilon-greedy exploration, target network sync.
    """

    SAVE_PATH = 'models/rl_agent'

    def __init__(self, n_nodes=103, budget=5, lr=5e-4,
                 gamma=0.95, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.97, replay_size=2000,
                 batch_size=64, target_sync=10, seed=42):

        self.n_nodes      = n_nodes
        self.budget       = budget
        self.state_dim    = n_nodes * 5          # 5 features per node
        self.action_dim   = n_nodes
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.epsilon_min  = epsilon_min
        self.epsilon_decay= epsilon_decay
        self.batch_size   = batch_size
        self.target_sync  = target_sync
        self.is_trained   = False
        self._step        = 0
        self._rng         = np.random.default_rng(seed)

        self.online  = _DQNNetwork(self.state_dim, self.action_dim, lr=lr, seed=seed)
        self.target  = _DQNNetwork(self.state_dim, self.action_dim, lr=lr, seed=seed+1)
        self._sync_target()

        # Replay buffer: list of (s, a, r, s', done) tuples
        self._replay  = []
        self._max_rep = replay_size

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------
    def get_state(self, G):
        """
        Returns a flat (n_nodes * 5,) array.
        Features per node: [infection_flag, risk_score, anomaly_score,
                            vulnerability_score, degree_norm]
        Nodes are visited in sorted order for consistency.
        """
        nodes   = sorted(G.nodes())
        n       = len(nodes)
        max_deg = max(dict(G.degree()).values()) or 1
        feats   = np.zeros((n, 5), dtype=np.float32)
        for i, nd in enumerate(nodes):
            d = G.nodes[nd]
            feats[i, 0] = 1.0 if d.get('infection_state') == 'infected'  else 0.0
            feats[i, 1] = float(d.get('risk_score',        0.0))
            feats[i, 2] = min(float(d.get('anomaly_score', 0.0)) / 10.0, 1.0)
            feats[i, 3] = float(d.get('vulnerability_score', 0.0))
            feats[i, 4] = G.degree(nd) / max_deg
        flat = feats.flatten()
        # Pad or truncate to self.state_dim
        if len(flat) < self.state_dim:
            flat = np.concatenate([flat, np.zeros(self.state_dim - len(flat))])
        return flat[:self.state_dim]

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, G):
        """
        Returns list of up to `budget` node IDs to patch.
        Uses epsilon-greedy: random susceptible nodes vs Q-network top-k.
        """
        susceptible = [n for n, d in G.nodes(data=True)
                       if d.get('infection_state') == 'susceptible']
        if not susceptible:
            return []

        if self._rng.random() < self.epsilon:
            # Explore: random susceptible nodes
            k = min(self.budget, len(susceptible))
            chosen = list(self._rng.choice(susceptible, size=k, replace=False))
        else:
            # Exploit: Q-network scores over all node indices
            state = self.get_state(G)
            q     = self.online.forward(state)
            nodes = sorted(G.nodes())
            # Mask out non-susceptible nodes
            mask  = np.full(len(nodes), -1e9)
            susc_set = set(susceptible)
            for i, nd in enumerate(nodes):
                if nd in susc_set:
                    mask[i] = q[i] if i < len(q) else 0.0
            top_idx = np.argsort(mask)[::-1][:self.budget]
            chosen  = [nodes[i] for i in top_idx if mask[i] > -1e9]

        return chosen

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def compute_reward(self, prev_infected_set, G):
        nodes           = list(G.nodes())
        n               = len(nodes)
        infected_now    = set(nd for nd in nodes
                              if G.nodes[nd].get('infection_state') == 'infected')
        new_infections  = len(infected_now - prev_infected_set)
        critical        = [nd for nd in nodes
                           if G.nodes[nd].get('node_type') in ('server', 'controller')]
        crit_infected   = sum(1 for nd in critical
                              if G.nodes[nd].get('infection_state') == 'infected')
        patched         = sum(1 for nd in nodes
                              if G.nodes[nd].get('infection_state') == 'patched')
        reward = (-10.0 * new_infections / max(n, 1)
                  - 5.0  * crit_infected  / max(len(critical), 1)
                  + 2.0  * patched        / max(n, 1))
        return float(reward)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def _sync_target(self):
        self.target.W1 = self.online.W1.copy()
        self.target.b1 = self.online.b1.copy()
        self.target.W2 = self.online.W2.copy()
        self.target.b2 = self.online.b2.copy()
        self.target.W3 = self.online.W3.copy()
        self.target.b3 = self.online.b3.copy()

    def store(self, s, a, r, s2, done):
        if len(self._replay) >= self._max_rep:
            self._replay.pop(0)
        self._replay.append((s, a, r, s2, done))

    def learn(self):
        if len(self._replay) < self.batch_size:
            return
        idx     = self._rng.choice(len(self._replay), self.batch_size, replace=False)
        batch   = [self._replay[i] for i in idx]
        total_loss = 0.0
        for s, a, r, s2, done in batch:
            q_vals  = self.online.forward(s)
            q_next  = self.target.forward(s2)
            target_val = r if done else r + self.gamma * np.max(q_next)
            error   = q_vals[a] - target_val
            grad    = np.zeros_like(q_vals)
            grad[a] = 2.0 * error / self.batch_size
            self.online.backward(grad)
            total_loss += error ** 2
        self._step += 1
        if self._step % self.target_sync == 0:
            self._sync_target()
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)
        return total_loss / self.batch_size

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------
    def train(self, G, attacker_mode='random', beta=0.3, episodes=60,
              max_timesteps=50, seed=42):
        """
        Train the DQN over `episodes` full simulation episodes on graph G.
        Uses the propagation engine directly for environment dynamics.
        """
        import copy as _cp
        from network_graph import reset_graph as _rg
        from propagation_engine import (run_simulation,
                                        calculate_infection_probability,
                                        select_target_nodes,
                                        update_global_attack_stage)

        print(f"  [DQN] Training for {episodes} episodes ...")
        nodes = sorted(G.nodes())

        for ep in range(episodes):
            sim_G = _cp.deepcopy(G)
            _rg(sim_G)

            # Baseline traffic
            for nd in sim_G.nodes():
                bl = {}
                for nb in sim_G.neighbors(nd):
                    bl[str(nb)] = (sim_G[nd][nb].get('traffic_frequency', 0.5)
                                   * sim_G[nd][nb].get('trust_weight', 0.5))
                sim_G.nodes[nd]['baseline_traffic'] = bl

            # Patient zero
            rng_ep = np.random.default_rng(seed + ep)
            wks = [n for n, d in sim_G.nodes(data=True) if d.get('node_type') == 'workstation']
            pz  = int(rng_ep.choice(wks)) if wks else int(rng_ep.choice(nodes))
            sim_G.nodes[pz]['infection_state'] = 'infected'
            atk_stage = update_global_attack_stage(sim_G, 'none')

            state        = self.get_state(sim_G)
            ep_reward    = 0.0

            for t in range(7, max_timesteps + 1):
                sim_G.graph['timestep'] = t
                prev_infected = set(n for n in nodes
                                    if sim_G.nodes[n].get('infection_state') == 'infected')

                # Agent acts — patch budget nodes
                chosen = self.select_action(sim_G)
                for nd in chosen:
                    sim_G.nodes[nd]['infection_state'] = 'patched'
                    sim_G.nodes[nd]['vulnerability_score'] = 0.0

                # Propagate
                pending = []
                for src in list(prev_infected):
                    for tgt in select_target_nodes(sim_G, src, attacker_mode):
                        if (sim_G.nodes[tgt].get('infection_state') == 'susceptible'
                                and tgt not in pending):
                            p = calculate_infection_probability(sim_G, src, tgt, beta)
                            if rng_ep.random() < p:
                                pending.append(tgt)
                for nd in pending:
                    sim_G.nodes[nd]['infection_state'] = 'infected'

                atk_stage = update_global_attack_stage(sim_G, atk_stage)

                # Reward & next state
                reward     = self.compute_reward(prev_infected, sim_G)
                next_state = self.get_state(sim_G)
                ep_reward += reward

                # Map chosen node IDs to first action index (store one transition per patch)
                for nd in chosen:
                    a_idx = nodes.index(nd) if nd in nodes else 0
                    done  = (t == max_timesteps)
                    self.store(state, a_idx, reward, next_state, done)

                self.learn()
                state = next_state

                inf_count = sum(1 for n in nodes
                                if sim_G.nodes[n].get('infection_state') == 'infected')
                if inf_count == len(nodes):
                    break
                infected_check = [n for n in nodes
                                  if sim_G.nodes[n].get('infection_state') == 'infected']
                if not any(any(sim_G.nodes[nb].get('infection_state') == 'susceptible'
                               for nb in sim_G.neighbors(n))
                           for n in infected_check):
                    break

            if (ep + 1) % 10 == 0:
                inf_rate = sum(1 for n in nodes
                               if sim_G.nodes[n].get('infection_state') == 'infected') / len(nodes)
                print(f"  [DQN] ep {ep+1:3d}/{episodes} | "
                      f"reward={ep_reward:+.2f} | "
                      f"inf={inf_rate*100:.1f}% | "
                      f"eps={self.epsilon:.3f}")

        self.is_trained = True
        self.epsilon = self.epsilon_min  # pure exploitation at inference
        print(f"  [DQN] Training complete.")

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self):
        self.online.save(self.SAVE_PATH)
        meta = np.array([self.epsilon, float(self.is_trained), float(self._step)])
        np.save(self.SAVE_PATH + '_meta.npy', meta)
        print(f"  [DQN] Model saved to {self.SAVE_PATH}.npz")

    def load(self):
        try:
            self.online.load(self.SAVE_PATH)
            self._sync_target()
            meta = np.load(self.SAVE_PATH + '_meta.npy')
            self.epsilon   = float(meta[0])
            self.is_trained= bool(meta[1])
            self._step     = int(meta[2])
            print(f"  [DQN] Loaded model from {self.SAVE_PATH}.npz "
                  f"(eps={self.epsilon:.3f})")
            return True
        except Exception:
            return False