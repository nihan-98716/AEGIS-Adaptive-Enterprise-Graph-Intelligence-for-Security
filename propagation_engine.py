import networkx as nx
import random
from network_graph import generate_enterprise_graph, reset_graph

def get_privilege_multiplier(privilege_level):
    """Returns the multiplier L(i) based on privilege level."""
    if privilege_level == 'domain_admin':
        return 2.0
    elif privilege_level == 'admin':
        return 1.5
    else:  # user or default
        return 1.0

def calculate_infection_probability(G, source, target, beta):
    """
    Calculates P(i->j) = beta * W(i,j) * V(j) * L(i)
    """
    # L(i) = privilege multiplier of source node i
    source_priv = G.nodes[source].get('privilege_level', 'user')
    L_i = get_privilege_multiplier(source_priv)
    
    # V(j) = vulnerability_score of target node j
    V_j = G.nodes[target].get('vulnerability_score', 0.5)
    
    # W(i,j) = trust_weight of edge between i and j
    W_ij = G[source][target].get('trust_weight', 0.5)
    
    return beta * W_ij * V_j * L_i

def update_global_attack_stage(G, current_stage):
    """
    Evaluates the overall graph state to determine the highest attack stage reached.
    Stages progress strictly in this order:
    "none" -> "initial_access" -> "credential_harvesting" -> 
    "lateral_movement" -> "target_compromise" -> "ransomware_execution"
    """
    stages = ["none", "initial_access", "credential_harvesting", 
              "lateral_movement", "target_compromise", "ransomware_execution"]
    
    current_idx = stages.index(current_stage) if current_stage in stages else 0
    new_idx = current_idx
    
    infected_nodes = [n for n, d in G.nodes(data=True) if d.get('infection_state') == 'infected']
    num_infected = len(infected_nodes)
    
    if num_infected > 0 and new_idx < 1:
        new_idx = max(new_idx, 1)  # initial_access
        
    if num_infected >= 2 and new_idx < 2:
        new_idx = max(new_idx, 2)  # credential_harvesting
        
    if num_infected >= 3 and new_idx < 3:
        new_idx = max(new_idx, 3)  # lateral_movement
        
    # Check for target_compromise (server or controller infected)
    if new_idx < 4:
        for n in infected_nodes:
            if G.nodes[n].get('node_type') in ['server', 'controller']:
                new_idx = max(new_idx, 4)
                break
                
    # Check for ransomware_execution (domain_admin infected)
    if new_idx < 5:
        for n in infected_nodes:
            if G.nodes[n].get('privilege_level') == 'domain_admin':
                new_idx = max(new_idx, 5)
                break
                
    # Update node-level attack_stage attributes for all infected nodes
    new_stage = stages[new_idx]
    for n in infected_nodes:
        # Only upgrade node stage, don't downgrade
        node_stage = G.nodes[n].get('attack_stage', 'none')
        node_idx = stages.index(node_stage) if node_stage in stages else 0
        if new_idx > node_idx:
            G.nodes[n]['attack_stage'] = new_stage
            
    return new_stage

def record_baseline_traffic(G):
    """
    Records baseline_traffic for each node: neighbor_id -> mean_communication_weight.
    Computed as trust_weight * traffic_frequency.
    """
    for n in G.nodes():
        baseline = {}
        for neighbor in G.neighbors(n):
            freq = G[n][neighbor].get('traffic_frequency', 0.5)
            weight = G[n][neighbor].get('trust_weight', 0.5)
            baseline[str(neighbor)] = freq * weight
        G.nodes[n]['baseline_traffic'] = baseline

def select_target_nodes(G, source, mode):
    """
    Selects target neighbors based on attacker mode.
    Only returns susceptible neighbors.
    """
    neighbors = list(G.neighbors(source))
    susceptible = [n for n in neighbors if G.nodes[n].get('infection_state') == 'susceptible']
    
    if not susceptible:
        return []
        
    if mode == "stealth":
        # Avoids edges where traffic_frequency > 0.7
        valid_targets = [n for n in susceptible if G[source][n].get('traffic_frequency', 0.5) <= 0.7]
        return valid_targets
        
    elif mode == "greedy":
        # Always targets highest asset_value neighbor
        # If there's a tie, max() returns the first one encountered, which is fine
        highest_target = max(susceptible, key=lambda n: G.nodes[n].get('asset_value', 1))
        # Return as list to match other modes, even if choosing greedy targeting means only 1 attempt
        return [highest_target]
        
    else:  # random (default)
        # Spreads to random eligible neighbor - we'll interpret this as attempting to infect 
        # a RANDOM sample of neighbors, or perhaps just 1 random neighbor to keep spread constrained
        # The prompt says "spreads to random eligible neighbor" (singular), so we return 1.
        return [random.choice(susceptible)]

def run_simulation(G, max_timesteps=50, attacker_mode="random", beta=0.3, seed=42, initial_node=None):
    """
    Runs the cyber contagion simulation.
    """
    if seed is not None:
        random.seed(seed)
        
    # Ensure fresh start
    reset_graph(G)
    
    timestep_log = []
    attack_stages_timeline = {"none": 0}
    current_attack_stage = "none"
    time_to_first_critical_node = None
    total_new_infections = 0
    actual_timesteps = 0
    
    print(f"Starting simulation. Mode: {attacker_mode}, Beta: {beta}, Seed: {seed}")
    
    # Phase 1: Baseline Traffic Recording (Timesteps 1-5)
    # We do this before any infection.
    record_baseline_traffic(G)
    for t in range(1, 6):
        actual_timesteps += 1
        timestep_log.append({
            'timestep': t,
            'phase': 'baseline_recording',
            'infected_count': 0,
            'newly_infected_nodes': [],
            'detected_nodes': 0,
            'current_attack_stage': "none"
        })
        
    if initial_node is not None and initial_node in G.nodes():
        patient_zero = initial_node
    else:
        workstations = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'workstation']
        patient_zero = random.choice(workstations) if workstations else random.choice(list(G.nodes()))
    
    G.nodes[patient_zero]['infection_state'] = 'infected'
    G.nodes[patient_zero]['attack_stage'] = 'initial_access'
    current_attack_stage = update_global_attack_stage(G, "none")
    attack_stages_timeline[current_attack_stage] = 6
    
    actual_timesteps += 1
    timestep_log.append({
        'timestep': 6,
        'phase': 'patient_zero',
        'infected_count': 1,
        'newly_infected_nodes': [patient_zero],
        'detected_nodes': 0,
        'current_attack_stage': current_attack_stage
    })
    
    print(f"Patient Zero: {G.nodes[patient_zero].get('node_id', patient_zero)}")
    
    # Phase 3: Active Propagation
    for t in range(7, max_timesteps + 1):
        actual_timesteps += 1
        pending_infections = []
        
        # 1. Collect all infections first (mid-timestep state immutability)
        infected_nodes = [n for n, d in G.nodes(data=True) if d.get('infection_state') == 'infected']
        
        for source in infected_nodes:
            targets = select_target_nodes(G, source, attacker_mode)
            for target in targets:
                # Target must still be susceptible in the graph
                # (Though technically it might be in pending_infections multiple times, we just infect it once)
                if G.nodes[target].get('infection_state') == 'susceptible' and target not in pending_infections:
                    prob = calculate_infection_probability(G, source, target, beta)
                    if random.random() < prob:
                        pending_infections.append(target)
                        
        # 2. Apply all state changes after iterating
        for n in pending_infections:
            G.nodes[n]['infection_state'] = 'infected'
            
            # Check critical node timing
            node_type = G.nodes[n].get('node_type')
            privilege = G.nodes[n].get('privilege_level')
            if time_to_first_critical_node is None and (node_type in ['server', 'controller'] or privilege == 'domain_admin'):
                time_to_first_critical_node = t
                
        total_new_infections += len(pending_infections)
        
        # 3. Update global attack stage
        new_stage = update_global_attack_stage(G, current_attack_stage)
        if new_stage != current_attack_stage:
            if new_stage not in attack_stages_timeline:
                attack_stages_timeline[new_stage] = t
            current_attack_stage = new_stage
            
        # Count detected (currently static/manual in this module without the defense simulator)
        detected_count = sum(1 for _, d in G.nodes(data=True) if d.get('detected', False))
        
        # 4. Record log
        current_infected_count = sum(1 for _, d in G.nodes(data=True) if d.get('infection_state') == 'infected')
        timestep_log.append({
            'timestep': t,
            'phase': 'propagation',
            'infected_count': current_infected_count,
            'newly_infected_nodes': pending_infections.copy(),
            'detected_nodes': detected_count,
            'current_attack_stage': current_attack_stage
        })
        
        # Early exit if fully infected or no new infections possible
        if current_infected_count == G.number_of_nodes() or (not pending_infections and current_infected_count > 0):
            break

    # Final calculations
    final_infected = sum(1 for _, d in G.nodes(data=True) if d.get('infection_state') == 'infected')
    final_infection_rate = final_infected / G.number_of_nodes()
    
    containment_time = actual_timesteps - time_to_first_critical_node if time_to_first_critical_node else None
    
    # Spread velocity: mean new infections per propagation timestep (excluding the 5 baseline + 1 patient zero)
    prop_timesteps = actual_timesteps - 6
    spread_velocity = total_new_infections / prop_timesteps if prop_timesteps > 0 else 0.0

    return {
        'timestep_log': timestep_log,
        'final_infection_rate': final_infection_rate,
        'time_to_first_critical_node': time_to_first_critical_node,
        'containment_time': containment_time,
        'total_timesteps': actual_timesteps,
        'attack_stages_timeline': attack_stages_timeline,
        'spread_velocity': spread_velocity
    }

if __name__ == "__main__":
    print("Generating network map...")
    G = generate_enterprise_graph(seed=1337)
    
    # Run a test simulation
    print("\n--- Running GREEDY Mode Simulation ---")
    results = run_simulation(G, max_timesteps=20, attacker_mode="greedy", beta=0.8, seed=1337)
    
    print("\n--- Timeline Results ---")
    for stage, t in results['attack_stages_timeline'].items():
        print(f"Reached '{stage}' at Timestep {t}")
        
    print(f"\nFinal Infection Rate: {results['final_infection_rate']*100:.1f}%")
    print(f"Spread Velocity: {results['spread_velocity']:.2f} infections/timestep")
    if results['time_to_first_critical_node']:
         print(f"Time to first critical node: Timestep {results['time_to_first_critical_node']}")
         
    print("\n--- Partial Timestep Log (Last 5 steps) ---")
    for log in results['timestep_log'][-5:]:
        print(f"T={log['timestep']} | Infected: {log['infected_count']} | New: {len(log['newly_infected_nodes'])} | Stage: {log['current_attack_stage']}")
