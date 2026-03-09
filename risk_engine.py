import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau
from network_graph import generate_enterprise_graph

def compute_centrality_metrics(G):
    """
    Computes degree, betweenness, eigenvector (with fallback),
    and closeness centrality for all nodes in the graph.
    Returns dictionaries of these metrics.
    """
    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G)
    clos_cent = nx.closeness_centrality(G)
    
    try:
        eig_cent = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("WARNING: Eigenvector centrality failed to converge. Falling back to degree_centrality.")
        eig_cent = deg_cent
        
    return deg_cent, bet_cent, eig_cent, clos_cent

def calculate_risk_scores(G, alpha1=0.4, alpha2=0.35, alpha3=0.25):
    """
    Computes composite risk score for each node and stores it in the graph.
    Risk = a1 * betweenness_centrality + a2 * vulnerability_score + a3 * privilege_level_numeric
    """
    # 1. Compute metrics
    deg_cent, bet_cent, eig_cent, clos_cent = compute_centrality_metrics(G)
    
    privilege_mapping = {
        'user': 0.33,
        'admin': 0.66,
        'domain_admin': 1.0
    }
    
    # 2. Iterate and apply formula
    for node, data in G.nodes(data=True):
        b_cent = bet_cent.get(node, 0.0)
        v_score = data.get('vulnerability_score', 0.0)
        
        privilege_str = data.get('privilege_level', 'user')
        p_num = privilege_mapping.get(privilege_str, 0.33)
        
        risk_score = (alpha1 * b_cent) + (alpha2 * v_score) + (alpha3 * p_num)
        
        # Store all computed metrics back into the node attributes
        G.nodes[node]['degree_centrality'] = deg_cent.get(node, 0.0)
        G.nodes[node]['betweenness_centrality'] = b_cent
        G.nodes[node]['eigenvector_centrality'] = eig_cent.get(node, 0.0)
        G.nodes[node]['closeness_centrality'] = clos_cent.get(node, 0.0)
        G.nodes[node]['risk_score'] = risk_score

def get_risk_report(G):
    """
    Returns a pandas DataFrame containing detailed info for all nodes,
    sorted by risk_score descending.
    """
    rows = []
    for node, data in G.nodes(data=True):
        row = {
            'node_id': data.get('node_id', str(node)),
            'department': data.get('department', 'Unknown'),
            'node_type': data.get('node_type', 'Unknown'),
            'degree_centrality': data.get('degree_centrality', 0.0),
            'betweenness_centrality': data.get('betweenness_centrality', 0.0),
            'eigenvector_centrality': data.get('eigenvector_centrality', 0.0),
            'closeness_centrality': data.get('closeness_centrality', 0.0),
            'risk_score': data.get('risk_score', 0.0),
            'vulnerability_score': data.get('vulnerability_score', 0.0),
            'privilege_level': data.get('privilege_level', 'user'),
            'asset_value': data.get('asset_value', 0),
            'anomaly_score': data.get('anomaly_score', 0.0),
            'detected': data.get('detected', False)
        }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    # Sort descending by risk score
    df = df.sort_values(by='risk_score', ascending=False).reset_index(drop=True)
    return df

def get_critical_node_details(G, top_n=10):
    """
    Returns the top N nodes by risk_score as a list of dictionaries.
    Useful for reporting and dashboards.
    """
    df = get_risk_report(G)
    top_nodes = df.head(top_n)[['node_id', 'risk_score', 'department', 'node_type']].to_dict('records')
    return top_nodes

def get_critical_nodes(G, top_n=10):
    """
    Returns the top N node integer IDs by risk_score.
    Useful for downstream modules (e.g., defense simulator).
    """
    df = get_risk_report(G)
    # Reconstruct the integer node ID by finding the graph node 
    # that matches the generated node_id string.
    top_node_ids = df.head(top_n)['node_id'].tolist()
    
    # Map back string node_id ('IT-12') to integer graph node (12)
    int_ids = []
    for str_id in top_node_ids:
        for n, d in G.nodes(data=True):
            if d.get('node_id') == str_id:
                int_ids.append(n)
                break
                
    return int_ids

def get_combined_priority_nodes(G, top_n=10):
    """
    Ranks nodes by combined_score = risk_score * 0.6 + anomaly_score * 0.4
    Returns top N as a clean list of integer node IDs.
    Used by report generator to integrate risk and anomaly signals.
    """
    scored = []
    for n, d in G.nodes(data=True):
        combined = (d.get('risk_score', 0.0) * 0.6) + (d.get('anomaly_score', 0.0) * 0.4)
        scored.append((n, combined))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in scored[:top_n]]

def plot_risk_heatmap(G):
    """
    Visualizes nodes as a heatmap ranked by risk score using matplotlib,
    with department color coding.
    """
    df = get_risk_report(G)
    
    # We will plot Node IDs on the Y-axis, Risk Score on X-axis using a bar plot 
    # structured like a heatmap/dashboard for readability, coloring by department.
    
    # To keep the plot readable, cap it at the top 30 nodes
    plot_df = df.head(30)
    
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="whitegrid")
    
    # Define a color palette for departments
    dept_colors = {
        'IT': '#1f77b4',       # blue
        'Finance': '#2ca02c',  # green
        'HR': '#ff7f0e',       # orange
        'Operations': '#d62728'# red
    }
    
    ax = sns.barplot(
        data=plot_df, 
        x='risk_score', 
        y='node_id', 
        hue='department',
        palette=dept_colors,
        dodge=False
    )
    
    plt.title('Top 30 Nodes by Composite Risk Score', fontsize=16)
    plt.xlabel('Risk Score', fontsize=12)
    plt.ylabel('Node ID', fontsize=12)
    plt.legend(title='Department', loc='lower right')
    plt.tight_layout()
    
    # Save the plot
    output_file = "risk_heatmap.png"
    plt.savefig(output_file, dpi=300)
    print(f"\nRisk heatmap saved to {output_file}")
    # plt.close()  # suppressed for non-interactive Agg backend

def run_sensitivity_analysis(G):
    """
    Runs risk scoring with 5 alpha combinations and returns how much node rankings change.
    [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6], [0.33, 0.33, 0.33], [0.4, 0.35, 0.25]
    """
    print("\n" + "="*50)
    print("RISK SENSITIVITY ANALYSIS")
    print("="*50)
    
    # First, calculate baseline (default configuration)
    calculate_risk_scores(G, alpha1=0.4, alpha2=0.35, alpha3=0.25)
    default_df = get_risk_report(G)
    
    # We need to map node_id -> rank to calculate Kendall's tau properly
    default_ranks = {row['node_id']: rank for rank, row in default_df.iterrows()}
    # List of node_ids in order to keep standard axis for correlation array comparison
    node_order = list(default_ranks.keys()) 
    default_array = [default_ranks[n] for n in node_order]
    
    alpha_combinations = [
        {'name': 'High Centrality', 'alphas': (0.6, 0.2, 0.2)},
        {'name': 'High Vulnerability', 'alphas': (0.2, 0.6, 0.2)},
        {'name': 'High Privilege', 'alphas': (0.2, 0.2, 0.6)},
        {'name': 'Equal Weights', 'alphas': (0.33, 0.33, 0.33)},
        {'name': 'Default Config', 'alphas': (0.4, 0.35, 0.25)}
    ]
    
    results = {}
    
    for combo in alpha_combinations:
        a1, a2, a3 = combo['alphas']
        name = combo['name']
        print(f"\nEvaluating: {name} (α1={a1}, α2={a2}, α3={a3})")
        
        # Apply alphas to graph
        calculate_risk_scores(G, alpha1=a1, alpha2=a2, alpha3=a3)
        df = get_risk_report(G)
        
        # Print top 10 nodes
        print("Top 10 Nodes:")
        for idx, row in df.head(10).iterrows():
            print(f"  {idx+1}. {row['node_id']} (Type: {row['node_type']}, Dept: {row['department']}) - Risk: {row['risk_score']:.4f}")
            
        # Compute Kendall's tau against default array
        combo_ranks = {row['node_id']: rank for rank, row in df.iterrows()}
        combo_array = [combo_ranks[n] for n in node_order]
        
        tau, p_value = kendalltau(default_array, combo_array)
        
        print(f"Kendall's Tau (Rank Correlation vs Default): {tau:.4f}")
        results[name] = tau
        
    # Reset back to default for safety before ending function
    calculate_risk_scores(G, alpha1=0.4, alpha2=0.35, alpha3=0.25)
    
    return results

if __name__ == "__main__":
    # 1. Load sample network graph
    print("Generating simulated enterprise network...")
    G = generate_enterprise_graph(seed=99)
    
    # 2. Compute normal risk mechanics and load report
    calculate_risk_scores(G)
    report_df = get_risk_report(G)
    
    print("\n" + "="*50)
    print("DEFAULT COMPOSITE RISK REPORT (Top 15 Nodes)")
    print("="*50)
    columns_to_show = ['node_id', 'department', 'node_type', 'risk_score', 'betweenness_centrality', 'vulnerability_score', 'privilege_level']
    print(report_df[columns_to_show].head(15).to_string(index=False))
    
    # 3. Retrieve just critical nodes mapping for demo
    crit_nodes_details = get_critical_node_details(G, top_n=5)
    print(f"\nTop 5 Critical Nodes (Details extract): {[n['node_id'] for n in crit_nodes_details]}")
    
    crit_nodes = get_critical_nodes(G, top_n=5)
    print(f"Top 5 Critical Nodes (Integer IDs): {crit_nodes}")
    
    # 4. Perform sensitivity analysis test
    run_sensitivity_analysis(G)
    
    # 5. Plot risk heatmap
    plot_risk_heatmap(G)