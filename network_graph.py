import networkx as nx
import random
import os
import webbrowser
import webbrowser  # kept for reference, not used in visualize_graph

def generate_enterprise_graph(seed=None):
    if seed is not None:
        random.seed(seed)
    
    # 1. Generate Barabasi-Albert scale-free graph with 100 nodes
    # m is the number of edges to attach from a new node to existing nodes
    G = nx.barabasi_albert_graph(n=100, m=2, seed=seed)
    
    # 2. Create 4 departmental clusters (25 nodes each)
    departments = ['IT', 'Finance', 'HR', 'Operations']
    
    # We assign departments in chunks to ensure exactly 25 each
    nodes = list(G.nodes())
    random.shuffle(nodes)
    
    for i, node in enumerate(nodes):
        dept = departments[i // 25]  # 0-24: IT, 25-49: Finance, 50-74: HR, 75-99: Operations
        
        # Assign attributes based on department loosely
        if dept == 'IT':
            node_type = random.choices(['workstation', 'server'], weights=[0.8, 0.2])[0]
            privilege_level = random.choices(['user', 'admin'], weights=[0.8, 0.2])[0]
        else:
            node_type = 'workstation'
            privilege_level = 'user'
            
        nx.set_node_attributes(G, {
            node: {
                'node_id': f"{dept}-{node}",
                'node_type': node_type,
                'vulnerability_score': round(random.uniform(0.0, 1.0), 2),
                'privilege_level': privilege_level,
                'asset_value': random.randint(1, 10),
                'infection_state': 'susceptible',
                'risk_score': 0.0,
                'department': dept,
                'detected': False,
                'anomaly_score': 0.0,
                'baseline_traffic': {},
                'attack_stage': 'none'
            }
        })
        
    # 3. Add 3 high-connectivity hub nodes
    hubs = [
        {'id': 100, 'node_id': 'DC-1', 'node_type': 'controller', 'privilege_level': 'domain_admin', 'dept': 'IT', 'asset_value': 10},
        {'id': 101, 'node_id': 'CORE-1', 'node_type': 'server', 'privilege_level': 'admin', 'dept': 'IT', 'asset_value': 9},
        {'id': 102, 'node_id': 'AUTH-1', 'node_type': 'server', 'privilege_level': 'domain_admin', 'dept': 'IT', 'asset_value': 10}
    ]
    
    for hub in hubs:
        G.add_node(hub['id'],
            node_id=hub['node_id'],
            node_type=hub['node_type'],
            vulnerability_score=round(random.uniform(0.1, 0.4), 2),
            privilege_level=hub['privilege_level'],
            asset_value=hub['asset_value'],
            infection_state='susceptible',
            risk_score=0.0,
            department=hub['dept'],
            detected=False,
            anomaly_score=0.0,
            baseline_traffic={},
            attack_stage='none'
        )
        
        # Connect to all department clusters (to multiple nodes in each to ensure high connectivity)
        for dept in departments:
            # Find nodes in this department
            dept_nodes = [n for n, d in G.nodes(data=True) if d.get('department') == dept and n < 100]
            # Connect the hub to 5 random nodes in each department
            targets = random.sample(dept_nodes, min(5, len(dept_nodes)))
            for target in targets:
                G.add_edge(hub['id'], target)
                
    # 4. Assign edge attributes
    for u, v in G.edges():
        G[u][v].update({
            'trust_weight': round(random.uniform(0.0, 1.0), 2),
            'traffic_frequency': round(random.uniform(0.0, 1.0), 2),
            'exploitability': round(random.uniform(0.0, 1.0), 2)
        })
        
    return G

def get_graph_summary(G):
    print("=== Network Graph Summary ===")
    print(f"Node Count: {G.number_of_nodes()}")
    print(f"Edge Count: {G.number_of_edges()}")
    
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / G.number_of_nodes()
    print(f"Average Degree: {avg_degree:.2f}")
    
    print("\nTop 5 Highest Degree Nodes:")
    sorted_degrees = sorted(degrees.items(), key=lambda item: item[1], reverse=True)
    for i in range(min(5, len(sorted_degrees))):
        node, degree = sorted_degrees[i]
        node_id = G.nodes[node]['node_id']
        node_type = G.nodes[node].get('node_type', 'unknown')
        print(f"  {i+1}. {node_id} (Type: {node_type}) - Degree: {degree}")
    print("=============================")

def reset_graph(G):
    """Resets the graph states to allow clean multiple simulation runs."""
    for n in G.nodes():
        G.nodes[n]['infection_state'] = 'susceptible'
        G.nodes[n]['detected'] = False
        G.nodes[n]['anomaly_score'] = 0.0
        G.nodes[n]['attack_stage'] = 'none'
        G.nodes[n]['baseline_traffic'] = {}
        G.nodes[n]['risk_score'] = 0.0

def get_nodes_by_department(G, department):
    return [n for n, d in G.nodes(data=True) 
            if d.get('department') == department]

def get_hub_nodes(G):
    return [n for n, d in G.nodes(data=True) 
            if d.get('node_type') == 'controller' 
            or d.get('node_id') in ['DC-1', 'CORE-1', 'AUTH-1']]

def visualize_graph(G, output_file="cyber_network.html", title="AEGIS — Enterprise Network"):
    """
    Generates a standalone vis.js HTML network visualisation — no pyvis
    physics quirks, works in any browser without a loading bar hang.

    Colour scheme
    ─────────────
    Infected     → #e94560 (red)
    Patched      → #27ae60 (green)
    Anomaly det. → #f5a623 (gold)
    IT           → #4a9eda (blue)
    Finance      → #2ecc71 (green)
    HR           → #f39c12 (amber)
    Operations   → #e67e22 (orange)
    Hub nodes    → diamond shape, white border, size=38
    """
    import json

    dept_colors = {
        'IT':         '#4a9eda',   # steel blue
        'Finance':    '#a855f7',   # purple  — was green (clashed with Patched)
        'HR':         '#06b6d4',   # cyan    — was amber (clashed with Operations+Anomaly)
        'Operations': '#f97316',   # bright orange
    }
    hub_ids = {'DC-1', 'CORE-1', 'AUTH-1'}

    nodes_js = []
    for node, data in G.nodes(data=True):
        state   = data.get('infection_state', 'susceptible')
        dept    = data.get('department', 'IT')
        nid     = data.get('node_id', str(node))
        risk    = float(data.get('risk_score') or 0.0)
        anomaly = data.get('anomaly_score')
        vuln    = float(data.get('vulnerability_score') or 0.0)
        priv    = data.get('privilege_level', 0)
        ntype   = data.get('node_type', '')

        if state == 'infected':
            bg, border = '#e94560', '#ff6b6b'
        elif state == 'patched':
            bg, border = '#27ae60', '#2ecc71'
        elif data.get('detected'):
            bg, border = '#f5a623', '#f9ca24'
        else:
            bg = dept_colors.get(dept, '#7f8c8d')
            border = '#ffffff' if nid in hub_ids else '#aaaacc'

        if nid in hub_ids:
            border = '#ffffff'
            size   = 38
            shape  = 'diamond'
        else:
            size  = max(12, min(28, 12 + int(risk * 40)))
            shape = 'dot'

        if anomaly is None:
            anomaly_str = "N/A"
        elif isinstance(anomaly, str):
            anomaly_str = anomaly
        else:
            anomaly_str = f"{float(anomaly):.3f}"
        tooltip = (
            f"<b>{nid}</b><br>"
            f"Type: {ntype} | Dept: {dept}<br>"
            f"State: <b>{state}</b><br>"
            f"Risk Score: <b>{risk:.3f}</b><br>"
            f"Anomaly Score: {anomaly_str}<br>"
            f"Vulnerability: {vuln:.2f} | Privilege: {priv}"
        )

        nodes_js.append({
            "id":    node,
            "label": nid,
            "title": tooltip,
            "color": {"background": bg, "border": border,
                      "highlight": {"background": "#ffffff", "border": border}},
            "size":  size,
            "shape": shape,
            "font":  {"color": "#ecf0f1", "size": 11}
        })

    edges_js = []
    for u, v, edata in G.edges(data=True):
        trust   = edata.get('trust_weight', 0.5)
        traffic = edata.get('traffic_frequency', 0.5)
        exploit = edata.get('exploitability', 0.5)
        width   = round(1 + exploit * 3, 1)
        r = int(80 + exploit * 175)
        g = max(0, int(80 - exploit * 60))
        b = max(0, int(80 - exploit * 60))
        ecolor = f'#{r:02x}{g:02x}{b:02x}'
        edges_js.append({
            "from":  u,
            "to":    v,
            "width": width,
            "color": {"color": ecolor, "highlight": "#ffffff"},
            "title": f"Trust: {trust:.2f} | Traffic: {traffic:.2f} | Exploit: {exploit:.2f}"
        })

    nodes_json = json.dumps(nodes_js)
    edges_json = json.dumps(edges_js)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#1a1a2e; font-family: monospace; overflow:hidden; }}
  #network {{ width:100vw; height:100vh; }}
  #legend {{
    position:fixed; top:12px; left:12px; z-index:9999;
    background:#16213e; border:1px solid #444466; border-radius:8px;
    padding:12px 16px; font-size:12px; color:#ecf0f1; min-width:200px;
  }}
  #legend .title {{ font-size:13px; font-weight:bold; color:#f5a623; margin-bottom:10px; }}
  #legend .row {{ display:flex; align-items:center; gap:6px; margin:3px 0; }}
  #legend .dot {{ width:11px; height:11px; border-radius:50%; flex-shrink:0; }}
  #legend .diamond {{ width:11px; height:11px; transform:rotate(45deg); flex-shrink:0; }}
  #legend .note {{ margin-top:8px; font-size:10px; color:#888; }}
  #stats {{
    position:fixed; bottom:12px; left:12px; z-index:9999;
    background:#16213e; border:1px solid #444466; border-radius:8px;
    padding:8px 14px; font-size:11px; color:#aaaacc;
  }}
</style>
</head>
<body>
<div id="legend">
  <div class="title">{title}</div>
  <div class="row"><div class="dot" style="background:#e94560"></div> Infected</div>
  <div class="row"><div class="dot" style="background:#27ae60"></div> Patched</div>
  <div class="row"><div class="dot" style="background:#f5a623"></div> Anomaly Detected</div>
  <div class="row"><div class="dot" style="background:#4a9eda"></div> IT (susceptible)</div>
  <div class="row"><div class="dot" style="background:#a855f7"></div> Finance</div>
  <div class="row"><div class="dot" style="background:#06b6d4"></div> HR</div>
  <div class="row"><div class="dot" style="background:#f97316"></div> Operations</div>
  <div class="row"><div class="diamond" style="background:#ecf0f1"></div> Hub node (DC/CORE/AUTH)</div>
  <div class="note">Node size = risk score &nbsp;|&nbsp; Edge width = exploitability</div>
</div>
<div id="stats">Scroll to zoom &nbsp;·&nbsp; Drag to pan &nbsp;·&nbsp; Hover node for details</div>
<div id="network"></div>
<script>
  var nodes = new vis.DataSet({nodes_json});
  var edges = new vis.DataSet({edges_json});
  var container = document.getElementById('network');
  var options = {{
    nodes: {{ borderWidth: 2, borderWidthSelected: 4 }},
    edges: {{ arrows: {{ to: {{ enabled: false }} }}, smooth: {{ type: 'continuous' }} }},
    physics: {{
      enabled: true,
      barnesHut: {{
        gravitationalConstant: -8000,
        centralGravity: 0.3,
        springLength: 120,
        springConstant: 0.04,
        damping: 0.09,
        avoidOverlap: 0.2
      }},
      stabilization: {{ iterations: 300, updateInterval: 10 }}
    }},
    interaction: {{ hover: true, tooltipDelay: 80, hideEdgesOnDrag: true }}
  }};
  var network = new vis.Network(container, {{ nodes: nodes, edges: edges }}, options);
  network.once('stabilizationIterationsDone', function() {{
    network.setOptions({{ physics: {{ enabled: false }} }});
  }});
</script>
</body>
</html>"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  -> Network visualisation saved to {output_file}")

if __name__ == "__main__":
    # Create the reproducible graph
    seed = 42
    G = generate_enterprise_graph(seed=seed)
    
    # Output the graph summary
    get_graph_summary(G)
    
    # Intentionally infect some nodes for demonstration
    # Note: 'high_risk' and 'anomaly_detected' are not valid SIR states 
    # but are used here for visualization demonstration purposes.
    nodes = list(G.nodes())
    random.seed(seed)
    
    # Demo state changes
    infected_nodes = random.sample(nodes, 5)
    for n in infected_nodes:
        G.nodes[n]['infection_state'] = 'infected'
        
    patched_nodes = random.sample([n for n in nodes if n not in infected_nodes], 3)
    for n in patched_nodes:
        G.nodes[n]['infection_state'] = 'patched'
        
    high_risk_nodes = random.sample([n for n in nodes if n not in infected_nodes + patched_nodes], 4)
    for n in high_risk_nodes:
        G.nodes[n]['infection_state'] = 'high_risk'
        
    anomaly_nodes = random.sample([n for n in nodes if n not in infected_nodes + patched_nodes + high_risk_nodes], 2)
    for n in anomaly_nodes:
        G.nodes[n]['infection_state'] = 'susceptible'
        G.nodes[n]['detected'] = True
    
    # Visualize the graph
    visualize_graph(G, "demo_network.html")
    
    # Demonstrate reset
    print("\nDemonstrating reset_graph()...")
    reset_graph(G)
    states = set(data['infection_state'] for _, data in G.nodes(data=True))
    print(f"Node states after reset: {states}")