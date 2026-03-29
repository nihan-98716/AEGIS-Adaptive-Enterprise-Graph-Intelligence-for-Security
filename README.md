# AEGIS

<div align="center">

**Adaptive Enterprise Graph Intelligence for Security**

*An advanced cyber contagion simulation and defense analysis framework*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Basic Simulation](#basic-simulation)
  - [Advanced Options](#advanced-options)
- [Modules](#modules)
- [Defense Strategies](#defense-strategies)
- [Anomaly Detection](#anomaly-detection)
- [AI Integration](#ai-integration)
- [Output & Visualization](#output--visualization)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🔍 Overview

**AEGIS** is a comprehensive cybersecurity research and simulation framework that models, analyzes, and visualizes the propagation of cyber contagions (malware, vulnerabilities, attacks) across complex enterprise network architectures. Built on graph-based modeling and enhanced with artificial intelligence, AEGIS enables security teams, researchers, and students to:

- Understand attack propagation dynamics in realistic network topologies
- Evaluate the effectiveness of various defensive strategies
- Detect anomalous behaviors through machine learning and graph neural networks
- Generate automated threat intelligence reports using natural language processing
- Train reinforcement learning agents for adaptive cyber defense

The framework simulates real-world enterprise networks with departmental structures, varying privilege levels, and heterogeneous node types (workstations, servers, IoT devices) to provide insights into cybersecurity risk management.

---

## ✨ Key Features

### 🌐 Network Graph Propagation
- **Scale-free network topology** using Barabási–Albert model for realistic enterprise structures
- **Departmental clustering** (IT, Finance, HR, Operations) with 100+ nodes
- **Multi-stage attack modeling**: Initial Access → Lateral Movement → Privilege Escalation → Data Exfiltration → Impact
- **Dynamic infection propagation** based on:
  - Node vulnerability scores
  - Edge trust weights and traffic frequency
  - Privilege levels (user, admin, domain admin)
  - Network exploitability metrics

### 🛡️ Defense Simulation
Test and compare multiple defensive strategies:
- **No Defense** (baseline measurement)
- **Random Patching** (control strategy)
- **Vulnerability-Based Patching** (target highest vulnerability scores)
- **Centrality-Based Patching** (protect high-risk critical nodes)
- **GNN-Powered Predictive Defense** (AI-driven node prioritization)
- **Reinforcement Learning Agent** (adaptive strategy learning)

Track metrics:
- Final infection rate
- Average infection duration
- Defense budget utilization
- Strategy effectiveness over time

### 🔎 Anomaly Detection
Two detection modes available:

1. **IsolationForest Backend** (scikit-learn)
   - Traditional machine learning approach
   - Fast training and inference
   - Baseline anomaly detection

2. **GraphSAGE Autoencoder** (Pure NumPy implementation)
   - Graph Neural Network (GNN) architecture
   - Captures topological and feature-based anomalies
   - Reconstruction loss-based outlier detection
   - No PyTorch dependencies required

Features extracted per node:
- Degree centrality
- Edge trust weights, traffic frequency, exploitability
- Vulnerability scores
- Privilege levels
- Asset values
- Traffic patterns and behaviors

### 🤖 AI-Powered Intelligence

#### Graph Neural Networks (GNN)
- **GNN Predictor**: Forecasts next infection wave using GraphSAGE architecture
- **GNN Autoencoder**: Detects anomalous nodes through reconstruction error analysis
- Pure NumPy implementation available (no GPU required)

#### Reinforcement Learning (RL)
- **Q-Learning based defense agent**
- Learns optimal patching policies through simulation
- State representation: [infected_count, patched_count, avg_vulnerability]
- Actions: Patch specific node types or centrality-based targets

#### Natural Language Processing
- Automated threat report generation using:
  - **Ollama** (local LLM deployment)
  - **Google Gemini API** (cloud-based)
- Generates executive summaries, technical details, and actionable recommendations
- JSON-structured report outputs for integration with SIEM/SOAR platforms

### 📊 Risk Assessment Engine
Comprehensive risk scoring with configurable weights:
- **α₁**: Vulnerability impact (default 0.4)
- **α₂**: Privilege exposure (default 0.35)
- **α₃**: Asset criticality (default 0.25)

Features:
- Per-node risk calculation
- Critical node identification (top 10%)
- Network-wide risk heatmaps
- Temporal risk evolution tracking
- Kendall's Tau correlation analysis between risk and infection

---

## 🏗️ Architecture

AEGIS follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                        Main Orchestrator                     │
│                         (main.py)                            │
└────────────┬──────────────────────────────────┬─────────────┘
             │                                  │
    ┌────────▼────────┐                ┌───────▼────────┐
    │  Network Layer  │                │  Analysis Layer │
    │                 │                │                 │
    │ • Graph Gen     │                │ • Risk Engine   │
    │ • Propagation   │                │ • Anomaly Det.  │
    │ • Visualization │                │ • AI Modules    │
    └────────┬────────┘                └───────┬────────┘
             │                                  │
    ┌────────▼────────┐                ┌───────▼────────┐
    │  Defense Layer  │                │  Report Layer   │
    │                 │                │                 │
    │ • Strategies    │                │ • NL Reports    │
    │ • Simulation    │                │ • Metrics       │
    │ • RL Agent      │                │ • Visualization │
    └─────────────────┘                └─────────────────┘
```

**Data Flow:**
1. Generate enterprise network graph with realistic attributes
2. Simulate contagion propagation with configurable beta (infection rate)
3. Calculate risk scores and identify critical nodes
4. Apply defense strategies within budget constraints
5. Detect anomalies using ML/GNN models
6. Generate comprehensive threat reports
7. Visualize results through interactive graphs and plots

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning repository)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/AEGIS.git
cd AEGIS
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment (Optional)
For natural language report generation with Google Gemini:
```bash
# Create .env file
echo GEMINI_API_KEY=your_api_key_here > .env
```

For Ollama (local LLM):
```bash
# Install Ollama from https://ollama.ai/
# Pull a model
ollama pull llama3
```

---

## ⚙️ Configuration

Edit `config.py` to customize simulation parameters:

### Feature Flags
```python
AI_MODE = False                    # Enable GNN/RL (requires PyTorch)
GNN_ANOMALY_MODE = True           # Use GNN autoencoder for anomaly detection
ANOMALY_DETECTION_ENABLED = True  # Enable anomaly detection subsystem
NL_REPORTING_ENABLED = True       # Enable natural language reports
```

### Simulation Parameters
```python
SIMULATION_CONFIG = {
    "n_nodes": 100,          # Number of network nodes
    "beta": 0.6,             # Infection rate (0.0-1.0)
    "max_timesteps": 50,     # Simulation duration
    "budget": 5,             # Defense budget (nodes to patch)
    "n_runs": 30,            # Monte Carlo runs for averaging
    "seed": 42,              # Random seed for reproducibility
    "attacker_mode": "random"  # Attack strategy: random, targeted, adaptive
}
```

### Risk Weights
```python
ALPHA_WEIGHTS = {
    "alpha1": 0.4,   # Vulnerability impact weight
    "alpha2": 0.35,  # Privilege exposure weight
    "alpha3": 0.25   # Asset criticality weight
}
```

### Anomaly Detection
```python
ANOMALY_CONFIG = {
    "contamination": 0.1,  # Expected outlier percentage
    "threshold": 0.5       # Classification threshold
}

GNN_CONFIG = {
    "hidden_dim": 64,      # GNN hidden layer size
    "lr": 0.01,            # Learning rate
    "epochs": 200,         # Training epochs
    "gnn_threshold": 2.0   # Anomaly score threshold
}
```

---

## 🚀 Usage

### Basic Simulation

Run a standard cyber contagion simulation:
```bash
python main.py
```

This will:
1. Generate a 100-node enterprise network
2. Simulate contagion propagation
3. Calculate risk scores
4. Output results to console and `outputs/` directory

### Advanced Options

#### Run with Custom Parameters
```bash
python main.py --n_nodes 200 --beta 0.7 --budget 10 --n_runs 50
```

#### Enable Specific Modules
```bash
# Run with anomaly detection
python main.py --anomaly

# Run with defense simulation
python main.py --defense

# Run with AI report generation
python main.py --report

# Combine multiple modules
python main.py --anomaly --defense --report
```

#### Defense Strategy Comparison
```bash
python main.py --defense --strategy all
```

Strategies available:
- `none`: No defense (baseline)
- `random`: Random node patching
- `vulnerable`: Target highest vulnerability nodes
- `centrality`: Protect high-risk nodes
- `gnn`: AI-driven predictive defense
- `rl`: Reinforcement learning agent
- `all`: Compare all strategies

#### Extract Training Data
```bash
python feature_extractor.py --n_samples 500 --output datasets/training_data.csv
```

#### Visualize Network
```bash
python network_graph.py --interactive
```

---

## 📚 Modules

### `network_graph.py`
Generates and manages enterprise network topologies:
- **`generate_enterprise_graph(seed)`**: Creates Barabási–Albert scale-free graph with departmental structure
- **`reset_graph(G)`**: Resets infection states for re-simulation
- **`visualize_graph(G, output_path)`**: Generates interactive HTML visualization using Plotly

Node attributes:
- `node_id`, `node_type`, `vulnerability_score`, `privilege_level`
- `asset_value`, `infection_state`, `risk_score`, `department`
- `detected`, `anomaly_score`, `attack_stage`

Edge attributes:
- `trust_weight`, `traffic_frequency`, `exploitability`

### `propagation_engine.py`
Simulates contagion spread across the network:
- **`run_simulation(G, beta, max_timesteps)`**: Main simulation loop
- **`calculate_infection_probability(G, node, beta)`**: Compute per-node infection chance
- **`update_global_attack_stage(G)`**: Advance kill chain stages
- **`select_target_nodes(G, mode)`**: Choose next attack targets

Attack stages modeled:
1. Initial Access
2. Lateral Movement
3. Privilege Escalation
4. Data Exfiltration
5. Impact

### `risk_engine.py`
Evaluates network-wide cybersecurity risk:
- **`calculate_risk_scores(G)`**: Computes weighted risk per node
- **`get_critical_nodes(G, top_k)`**: Identifies highest-risk assets
- **`get_risk_report(G)`**: Generates summary statistics
- **`plot_risk_heatmap(G, output_path)`**: Visualizes risk distribution

Risk formula:
```
Risk = α₁ × vulnerability + α₂ × privilege + α₃ × (asset_value / 10)
```

### `anomaly_detector.py`
Detects abnormal network behaviors:
- **`AnomalyDetector`**: IsolationForest-based detector
- **`GNNAnomalyDetector`**: GraphSAGE autoencoder detector
- **`make_anomaly_detector(mode)`**: Factory function for detector selection
- **`run_anomaly_detection_experiment(G, detector)`**: Evaluation pipeline

Features per node: 15+ dimensions including degree, edge metrics, vulnerability, privilege, asset value, and traffic patterns.

### `defense_simulator.py`
Tests defensive strategies against attacks:
- **`run_defense_experiment(G, strategies, budget)`**: Compare multiple strategies
- **`strategy_none(G, budget)`**: Baseline (no action)
- **`strategy_random(G, budget)`**: Random patching
- **`strategy_patch_vulnerable(G, budget)`**: Target high vulnerability
- **`strategy_patch_centrality(G, budget)`**: Protect critical nodes
- **`strategy_gnn_predict(G, budget)`**: AI-powered prioritization
- **`plot_strategy_comparison(results)`**: Visualize effectiveness

### `ai_modules.py`
AI/ML models for predictive and adaptive defense:
- **`GNNPredictor`**: Graph neural network for infection forecasting
- **`RLDefenseAgent`**: Q-learning based adaptive defense
- **`GNNAnomalyDetector`**: Pure NumPy GraphSAGE autoencoder

### `report_generator.py`
Automated threat intelligence reporting:
- **`ReportGenerator`**: NLP-based report creation
- **`run_report_generation(G, model)`**: Generate comprehensive reports
- Supports Ollama (local) and Google Gemini (cloud)
- Output formats: JSON, Markdown, HTML

Report sections:
1. Executive Summary
2. Technical Analysis
3. Risk Assessment
4. Affected Assets
5. Attack Timeline
6. Recommendations

### `feature_extractor.py`
Dataset generation for ML training:
- **`extract_training_dataset(n_samples, output_path)`**: Create labeled datasets
- Multiple simulation runs with varied parameters
- Features + labels (infection outcome)
- CSV export for external ML pipelines

---

## 🛡️ Defense Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **None** | No defensive action | Baseline measurement |
| **Random** | Patch random susceptible nodes | Control group, resource-limited |
| **Vulnerable** | Patch highest vulnerability nodes | Known CVE exploitation |
| **Centrality** | Protect high-degree/risk nodes | Network disruption attacks |
| **GNN** | AI predicts next infection wave | Zero-day, APT scenarios |
| **RL** | Learns optimal policy via Q-learning | Adaptive adversaries |

### Performance Metrics
- **Infection Rate**: % of nodes infected at simulation end
- **Defense Efficiency**: (Prevented infections) / (Budget spent)
- **Time to Containment**: Timesteps until infection plateaus
- **Critical Node Protection**: % of top-10% risk nodes patched

---

## 🔍 Anomaly Detection

### IsolationForest Mode
Traditional ML approach using scikit-learn:
- Fast training (< 1 second for 100 nodes)
- Low memory footprint
- Effective for high-dimensional feature spaces
- Contamination parameter controls outlier sensitivity

### GNN Autoencoder Mode
Graph-aware deep learning:
- Captures structural + feature anomalies
- Pure NumPy implementation (no GPU needed)
- Training: 200 epochs on baseline graph
- Detection: Reconstruction loss > threshold → anomaly

**Anomaly Types Detected:**
- Unusual traffic patterns
- Privilege escalation attempts
- Lateral movement behaviors
- Data exfiltration activities
- Zero-day exploitation (high reconstruction error)

---

## 🤖 AI Integration

### Graph Neural Networks (GNN)
**Purpose**: Leverage graph structure for prediction

**Architecture**:
```
Input Graph → GraphSAGE Layer 1 (64 dim) → ReLU 
           → GraphSAGE Layer 2 (32 dim) → Output
```

**Training**:
- Supervised learning on historical infection data
- Input: Node features + neighborhood aggregation
- Output: Infection probability next timestep

**Inference**: Top-k nodes with highest infection probability are patched

### Reinforcement Learning
**Purpose**: Learn adaptive defense policies

**State Space**: [infected_count, patched_count, avg_vulnerability, timestep]

**Action Space**: [patch_server, patch_admin, patch_vulnerable, patch_critical, do_nothing]

**Reward Function**:
```
Reward = -1 × (new_infections) + 0.1 × (patched_nodes) - 0.05 × (budget_used)
```

**Training**: 1000 episodes with ε-greedy exploration (ε=0.1)

### Natural Language Processing
**Purpose**: Automated threat reporting

**Models Supported**:
- Ollama: llama3, mistral, gemma2 (local deployment)
- Google Gemini: gemini-pro, gemini-1.5-flash (API)

**Report Generation Pipeline**:
1. Extract network state, infection stats, risk scores
2. Construct structured JSON context
3. Query LLM with prompt template
4. Parse and format response
5. Save to `outputs/reports/`

---

## 📊 Output & Visualization

### Directory Structure
```
outputs/
├── graphs/
│   ├── network_initial.html        # Interactive network graph
│   ├── network_infected.html       # Post-simulation state
│   └── risk_heatmap.png           # Risk distribution
├── defense/
│   ├── strategy_comparison.png    # Defense effectiveness
│   └── infection_curves.png       # Temporal infection trends
├── anomaly/
│   ├── anomaly_scores.csv         # Per-node anomaly scores
│   └── detection_report.json      # Detected anomalies
├── reports/
│   ├── threat_report.json         # Structured report
│   └── threat_report.md           # Human-readable markdown
└── metrics/
    ├── risk_scores.csv            # Node-level risk data
    └── simulation_summary.json    # Aggregate statistics
```

### Visualization Features
- **Interactive graphs**: Hover for node details, zoom, pan (Plotly)
- **Color coding**: Infection states (susceptible, infected, recovered, patched)
- **Size scaling**: Node size reflects asset value or risk score
- **Edge styling**: Trust weights shown via opacity/thickness
- **Heatmaps**: Seaborn-based risk and anomaly visualizations

---

## 📂 Project Structure

```
AEGIS/
├── README.md                  # This file
├── LICENSE                    # Apache 2.0 license
├── requirements.txt           # Python dependencies
├── config.py                  # Configuration parameters
├── main.py                    # Main orchestrator
├── network_graph.py           # Graph generation & visualization
├── propagation_engine.py      # Contagion simulation
├── risk_engine.py             # Risk assessment
├── anomaly_detector.py        # Anomaly detection
├── defense_simulator.py       # Defense strategies
├── ai_modules.py              # GNN, RL models
├── report_generator.py        # NLP report generation
├── feature_extractor.py       # Dataset generation
├── test_run.py                # Integration tests
├── inspect_ollama.py          # Ollama diagnostic tool
├── list_gemini_models.py      # Gemini model listing
├── models/                    # Trained model storage
│   ├── gnn.pt
│   └── rl_agent.pt
├── outputs/                   # Simulation results
├── venv/                      # Virtual environment (gitignored)
└── __pycache__/               # Python cache (gitignored)
```

---

## 📋 Requirements

### Core Dependencies
- `networkx >= 3.0` - Graph data structures and algorithms
- `numpy >= 1.24` - Numerical computing
- `pandas >= 2.0` - Data manipulation and analysis
- `matplotlib >= 3.7` - Plotting and visualization
- `seaborn >= 0.12` - Statistical data visualization
- `scipy >= 1.10` - Scientific computing

### Optional Dependencies
**Machine Learning**:
- `scikit-learn >= 1.3` - IsolationForest anomaly detection
- `torch >= 2.0` - Deep learning (if AI_MODE=True)
- `torch-geometric >= 2.3` - Graph neural networks (if AI_MODE=True)

**Natural Language Processing**:
- `ollama >= 0.1.0` - Local LLM integration
- `google-generativeai >= 0.3` - Google Gemini API
- `python-dotenv >= 1.0` - Environment variable management

**Visualization**:
- `plotly >= 5.14` - Interactive graph visualizations

See `requirements.txt` for complete list with version constraints.

---

## 🤝 Contributing

We welcome contributions from the cybersecurity and machine learning communities! 

### How to Contribute
1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Contribution Areas
- 🐛 Bug fixes and issue reports
- 📝 Documentation improvements
- 🆕 New defense strategies
- 🧪 Additional attack models (ransomware, DDoS, APT)
- 🤖 Enhanced AI models (GAT, GIN, Transformer)
- 🌐 Real-world network topologies
- 📊 Visualization enhancements
- ⚡ Performance optimizations

### Code Standards
- Follow PEP 8 style guide
- Add docstrings for all functions/classes
- Include unit tests for new features
- Update README for user-facing changes

---

## 📄 License

This project is licensed under the **Apache License 2.0**.

```
Copyright 2026 nihan

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

See [LICENSE](LICENSE) file for full license text.

---

## 🙏 Acknowledgments

### Research & Foundations
- **NetworkX** for graph algorithms and data structures
- **scikit-learn** for machine learning primitives
- **PyTorch** and **PyTorch Geometric** for deep learning capabilities
- **Barabási-Albert model** for scale-free network generation
- **MITRE ATT&CK Framework** for attack stage modeling

### AI/ML Models
- **GraphSAGE** (Hamilton et al., 2017) for graph neural network architecture
- **Isolation Forest** (Liu et al., 2008) for anomaly detection
- **Q-Learning** (Watkins, 1989) for reinforcement learning

### Tools & Libraries
- **Ollama** for local LLM deployment
- **Google Gemini** for cloud-based generative AI
- **Plotly** for interactive visualizations
- **Seaborn** and **Matplotlib** for statistical plotting

---

## 📧 Contact

For questions, suggestions, or collaboration inquiries:
- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/AEGIS/issues)
- **Discussions**: [Join the community conversation](https://github.com/yourusername/AEGIS/discussions)

---

## 🔮 Future Roadmap

### Planned Features
- [ ] Multi-objective RL (balance cost, time, effectiveness)
- [ ] Real-time attack detection dashboard
- [ ] Integration with SIEM platforms (Splunk, ELK)
- [ ] Advanced attacker models (APT, insider threats)
- [ ] Cloud infrastructure support (AWS, Azure, GCP)
- [ ] Blockchain/DeFi network simulations
- [ ] Distributed computing for large-scale simulations
- [ ] Web-based GUI for non-technical users
- [ ] Mobile app for monitoring active simulations
- [ ] Integration with threat intelligence feeds (MISP, TAXII)

### Research Directions
- Transfer learning from one network topology to another
- Adversarial machine learning (evading GNN detectors)
- Explainable AI for defense decisions (LIME, SHAP)
- Game-theoretic attack/defense modeling
- Quantum-resistant cryptography simulations

---

<div align="center">

**Built with ❤️ for the cybersecurity research community**

⭐ Star this repo if you find it useful!

</div>
