import os
import pandas as pd
import numpy as np

def extract_training_dataset(simulation_logs, G):
    """
    Extracts node features and target labels from simulation logs to build a dataset for GNN training.
    
    This dataset can be loaded into PyTorch Geometric Data objects. For instance:
    ```python
    import torch
    from torch_geometric.data import Data
    
    # Load the saved CSVs
    features_df = pd.read_csv("outputs/node_features.csv")
    labels_df = pd.read_csv("outputs/infection_labels.csv")
    
    # Convert to PyTorch tensors
    x = torch.tensor(features_df.drop(columns=['timestep', 'node_id', 'run_id']).values, dtype=torch.float)
    y = torch.tensor(labels_df['infected_next'].values, dtype=torch.float)
    
    # Assuming edge_index is created from networkx graph G
    edges = list(G.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    data = Data(x=x, edge_index=edge_index, y=y)
    ```
    
    Args:
        simulation_logs: List of simulation result dicts/logs containing timestep data.
        G: NetworkX graph object containing base node attributes.
        
    Returns:
        tuple: (node_features_df, infection_labels_df)
    """
    all_features = []
    all_labels = []
    
    # Iterate over multiple simulation runs if present
    for run_idx, run_log in enumerate(simulation_logs):
        timestep_log = run_log.get('timestep_log', [])
        
        for t in range(len(timestep_log) - 1):
            current_step = timestep_log[t]
            next_step = timestep_log[t + 1]
            
            # Identify sets of currently infected and newly infected nodes
            newly_infected_next = set(next_step.get('newly_infected_nodes', []))
            
            for node_id, data in G.nodes(data=True):
                # Construct features for this node at timestep t
                # Note: In a real implementation, dynamic attributes like anomaly_score,
                # n_infected_neighbors should be retrieved from the timestep_log.
                # Here we use a generic placeholder construction based on graph attributes.
                
                features = {
                    'run_id': run_idx,
                    'timestep': t,
                    'node_id': node_id,
                    'degree': G.degree[node_id],
                    'mean_trust_weight': np.mean([d.get('trust_weight', 0.5) for _, _, d in G.edges(node_id, data=True)]) if G.degree[node_id] > 0 else 0,
                    'mean_traffic_frequency': data.get('traffic_frequency', 0.5), # Placeholder dynamic attribute
                    'mean_exploitability': data.get('vulnerability_score', 0.5),
                    'vulnerability_score': data.get('vulnerability_score', 0.5),
                    'privilege_level_numeric': 1.0 if data.get('privilege_level') == 'domain_admin' else 0.66 if data.get('privilege_level') == 'admin' else 0.33,
                    'asset_value_normalized': data.get('asset_value', 1) / 10.0,
                    'n_infected_neighbors': sum(1 for neighbor in G.neighbors(node_id) if neighbor in set(current_step.get('newly_infected_nodes', []))), # Simplified approx
                    'n_detected_neighbors': 0, # requires deeper log parsing
                    'anomaly_score': 0.0 # From risk engine/anomaly detector logs
                }
                all_features.append(features)
                
                # Label is 1 if node gets infected in the next timestep, else 0
                label = 1 if node_id in newly_infected_next else 0
                all_labels.append({
                    'run_id': run_idx,
                    'timestep': t,
                    'node_id': node_id,
                    'infected_next': label
                })
                
    features_df = pd.DataFrame(all_features)
    labels_df = pd.DataFrame(all_labels)
    
    # Save the dataframes to CSV
    os.makedirs('outputs', exist_ok=True)
    features_df.to_csv('outputs/node_features.csv', index=False)
    labels_df.to_csv('outputs/infection_labels.csv', index=False)
    
    return features_df, labels_df


def get_rl_training_episodes(simulation_logs):
    """
    Formats raw simulation logs into formalized state-action-reward-next_state tuples 
    required for training Reinforcement Learning algorithms.
    
    Args:
        simulation_logs: List of simulation result dicts encompassing the episode.
        
    Returns:
        List of tuples: (state, action, reward, next_state)
    """
    episodes = []
    
    for run_log in simulation_logs:
        timestep_log = run_log.get('timestep_log', [])
        
        # We need at least 2 timesteps to form a state -> next_state pair
        for t in range(len(timestep_log) - 1):
            current_step = timestep_log[t]
            next_step = timestep_log[t + 1]
            
            # Using placeholders for state vectors parsing from logs
            state = np.array([
                current_step.get('infection_rate', 0.0),
                0.5, # mean risk score
                0.0, # detected nodes
                0.0, # critical nodes infected
                t / float(run_log.get('total_timesteps', 50))
            ])
            
            next_state = np.array([
                next_step.get('infection_rate', 0.0),
                0.5, # mean risk score
                0.0, # detected nodes
                0.0, # critical nodes infected
                (t + 1) / float(run_log.get('total_timesteps', 50))
            ])
            
            # Note: The raw timestep log from `propagation_engine` doesn't track `nodes_patched`.
            # This is a defense simulator concept. Unless injected retrospectively during the
            # simulation loop, this will return an empty list. 
            action = current_step.get('nodes_patched', [])
            reward = next_step.get('reward', 0.0) # Evaluated via compute_reward 
            
            episodes.append((state, action, reward, next_state))
            
    return episodes
