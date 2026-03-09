"""
Configuration module for the Enterprise Cyber Contagion Simulator.
"""

# Flag to enable/disable advanced AI modes (GNN/RL). 
# Enabling this requires PyTorch and PyTorch Geometric to be installed.
AI_MODE = False

# Flag to enable the GraphSAGE autoencoder as the anomaly detection backend.
# When True, GNNAnomalyDetector replaces IsolationForest — no extra deps needed
# (pure NumPy implementation).  Set False to keep the original IsolationForest.
GNN_ANOMALY_MODE = True

# Flag to enable anomaly detection subsystem.
# Enabling this requires scikit-learn (IsolationForest) to be installed.
ANOMALY_DETECTION_ENABLED = True

# Flag to enable Natural Language threat reporting.
# Enabling this requires ollama or google-genai to be installed.
NL_REPORTING_ENABLED = True

# Dictionary containing all default parameters for the simulation engine.
SIMULATION_CONFIG = {
    "n_nodes": 100,
    "beta": 0.6,
    "max_timesteps": 50,
    "budget": 5,
    "n_runs": 30,
    "seed": 42,
    "attacker_mode": "random"
}

# Weights used for calculating the combined risk score across different dimensions.
ALPHA_WEIGHTS = {
    "alpha1": 0.4,
    "alpha2": 0.35,
    "alpha3": 0.25
}

# Configuration for the Anomaly Detection subsystem.
ANOMALY_CONFIG = {
    "contamination": 0.1,
    "threshold": 0.5
}

# Configuration for the GNN autoencoder (used when GNN_ANOMALY_MODE=True).
GNN_CONFIG = {
    "hidden_dim": 64,       # neurons per GraphSAGE layer
    "lr": 0.01,             # SGD learning rate
    "epochs": 200,          # training epochs on baseline graph
    "gnn_threshold": 2.0,   # log-scale anomaly score — only top outliers flagged
}

# File paths for saving/loading trained AI models.
MODEL_PATHS = {
    "gnn_model": "models/gnn.pt",
    "rl_model": "models/rl_agent.pt"
}