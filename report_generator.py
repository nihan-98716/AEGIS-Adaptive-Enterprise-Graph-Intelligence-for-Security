import os
import json
import pandas as pd
import ollama

class ReportGenerator:
    def __init__(self, model_name="llama3"):
        """
        Initialize the Ollama connection.
        If no model is specified, it defaults to llama3.
        """
        self.model_name = model_name
        try:
            # Check if ollama is reachable and get available models
            models_info = ollama.list()
            if hasattr(models_info, 'models'):
                available_models = [m.model for m in models_info.models]
            elif isinstance(models_info, dict):
                available_models = [m.get('name', m.get('model')) for m in models_info.get('models', [])]
            else:
                available_models = []
            
            if not available_models:
                print(f"Warning: No models found in Ollama. You may need to run 'ollama pull {self.model_name}'.")
            elif not any(m.startswith(self.model_name) for m in available_models) and available_models:
                print(f"Warning: Model '{self.model_name}' not found. Defaulting to '{available_models[0]}'.")
                self.model_name = available_models[0]
            else:
                for m in available_models:
                    if m.startswith(self.model_name):
                        self.model_name = m
                        break
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            self.model_name = model_name

    def build_context_payload(self, scenario_name, simulation_result, risk_report_df,
                              detection_metrics, strategy_comparison_df,
                              attacker_mode="unknown"):
        """
        Assembles all discrete simulation and tracking data into a structured payload for LLM analysis.
        """
        
        # 1. Base Variables
        # A bit hacky if using our current logs. The spec says simulation_result has final_infection_rate directly:
        final_infection_rate = simulation_result.get('final_infection_rate', 0.0) * 100
        
        time_to_crit = simulation_result.get('time_to_first_critical_node')
        time_to_crit_val = time_to_crit if time_to_crit is not None else "not reached"
        
        # Extract stages reached by looking at the timeline dict
        stages_reached = list(simulation_result.get('attack_stages_timeline', {}).keys())

        # 2. Extract Top 5 Risk Nodes dynamically
        top_risk_df = risk_report_df.head(5)
        top_5_risk_nodes = top_risk_df[['node_id', 'department', 'node_type', 'risk_score', 'vulnerability_score', 'privilege_level']].to_dict('records')

        # 3. Extract Top 5 Anomaly Nodes
        # For this requirement, we look at the top rows where anomaly_score is highest
        # We can also check if they were literally detected during the sim (detected=True).
        anomaly_sorted_df = risk_report_df.sort_values(by='anomaly_score', ascending=False)
        top_anomaly_raw = anomaly_sorted_df.head(5)
        top_5_anomaly_nodes = []
        for _, row in top_anomaly_raw.iterrows():
            top_5_anomaly_nodes.append({
                'node_id': row['node_id'],
                'anomaly_score': float(row['anomaly_score']),
                # 'detected_at_timestep' requires checking the timeline array which isn't directly passed here easily.
                # Usually we just say "Yes" or pull from a specific struct. We will default to a boolean flag representing it.
                'detected_at_timestep': 'Yes' if row['detected'] else 'No'
            })

        # 4. Extract Combined Priority (60/40)
        # Using the spec math explicitly as demanded logic:
        # Alternatively, we calculate it dynamically here:
        risk_report_df['combined_score'] = (risk_report_df['risk_score'] * 0.6) + (risk_report_df['anomaly_score'] * 0.4)
        combined_priority_nodes = risk_report_df.sort_values(by='combined_score', ascending=False).head(5)['node_id'].tolist()

        # 5. Extract Defense Comparison Metrics
        # Sorting is guaranteed by the generation function passing it
        best_strat = strategy_comparison_df.iloc[0]['strategy_name'] 
        worst_strat = strategy_comparison_df.iloc[-1]['strategy_name']
        
        strat_summary = []
        for _, row in strategy_comparison_df.iterrows():
             strat_summary.append({
                 'strategy_name': row['strategy_name'],
                 'mean_infection_rate': float(row['mean_infection_rate']),
                 'mean_containment_time': float(row['mean_containment_time']) if pd.notna(row['mean_containment_time']) else None
             })
             
        # 6. Calc Baseline
        baseline_rate_raw = strategy_comparison_df[strategy_comparison_df['strategy_name'] == 'none']['mean_infection_rate'].values
        baseline_rate = float(baseline_rate_raw[0]) if len(baseline_rate_raw) > 0 else (final_infection_rate/100)
        
        resilience_score = max(0.0, 1.0 - ((final_infection_rate/100) / baseline_rate)) if baseline_rate > 0 else 0.0

        # Construct Master Dict
        payload = {
            'scenario_name': scenario_name,
            'attacker_mode': attacker_mode,
            'final_infection_rate': final_infection_rate,
            'time_to_critical_node': time_to_crit_val,
            'spread_velocity': float(simulation_result.get('spread_velocity', 0.0)),
            'total_timesteps': int(simulation_result.get('total_timesteps', 0)),
            'attack_stages_reached': stages_reached,
            'top_5_risk_nodes': top_5_risk_nodes,
            'top_5_anomaly_nodes': top_5_anomaly_nodes,
            'combined_priority_nodes': combined_priority_nodes,
            'detection_lead_time_mean': float(detection_metrics.get('detection_lead_time_mean', 0.0)),
            'detection_lead_time_std': float(detection_metrics.get('detection_lead_time_std', 0.0)),
            'true_positive_rate': float(detection_metrics.get('true_positive_rate', 0.0)),
            'false_positive_rate': float(detection_metrics.get('false_positive_rate', 0.0)),
            'best_defense_strategy': best_strat,
            'worst_defense_strategy': worst_strat,
            'strategy_comparison_summary': strat_summary,
            'network_resilience_score': resilience_score
        }
        
        return payload

    def generate_report(self, context_payload):
        """
        Fires an API call to local Ollama compiling the structured payload into a final intelligence text object.
        """
        system_instruction = (
            "You are a senior cybersecurity analyst generating threat "
            "assessment reports for enterprise security teams. Your reports "
            "are clear, actionable, and written for both technical and "
            "non-technical stakeholders. Always structure reports with these "
            "sections: Executive Summary, Attack Timeline Analysis, "
            "Anomaly Detection Findings, Vulnerability Assessment, "
            "Defense Strategy Evaluation, Critical Recommendations. "
            "Be specific — reference actual node IDs, departments, "
            "and metrics from the data provided."
        )

        user_prompt = f"""Generate a complete threat assessment report for the following 
simulation results: {json.dumps(context_payload, indent=2)}

The report must:
- Reference specific node IDs and departments by name
- Explain what the detection_lead_time means for real defenders
- Compare defense strategies with specific percentage differences
- Give exactly 5 prioritized remediation recommendations
- End with a network resilience rating: Critical/High/Medium/Low 
  based on the resilience score"""

        print(f"\nSubmitting Payload to Ollama ({self.model_name}) for Scenario: {context_payload['scenario_name']}...")
        print(f"  (This may take 1-3 minutes while Mistral loads — please wait...)")
        
        try:
             response = ollama.chat(
                 model=self.model_name,
                 messages=[
                     {'role': 'system', 'content': system_instruction},
                     {'role': 'user', 'content': user_prompt},
                 ],
                 options={'num_predict': 1024},
             )
             return response.message.content
        except AttributeError:
             return response['message']['content']
        except Exception as e:
             import traceback
             traceback.print_exc()
             return f"Ollama Call aborted - Initialization failed or model not found: {e}"

    def save_report(self, report_text, scenario_name, output_dir="outputs/"):
        """
        Saves the generated AI text into a local directory in both text format and syntax-highlighted markdown.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        txt_path = os.path.join(output_dir, f"report_{scenario_name}.txt")
        md_path = os.path.join(output_dir, f"report_{scenario_name}.md")
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        print(f"Report saved locally to:\n -> {txt_path}\n -> {md_path}")

    def generate_comparison_report(self, all_scenario_results):
        """
        Generates overarching meta-analytics looking across all scenarios arrayed out during testing.
        """
        system_instruction = (
            "You are a CISO-level cybersecurity strategist. Provide a meta-analysis "
            "across multiple simulation scenarios, evaluating macro trends over single "
            "localized breaches."
        )
        
        user_prompt = f"""Here are the payload results extracted from 4 distinct simulation runs:
{json.dumps(all_scenario_results, indent=2)}

Generate a cross-scenario executive summary comparing:
1. Which scenario caused the fastest overall lateral spread.
2. Which defense strategy performed best overall across all variables.
3. Whether the anomaly detection implementation improved outcomes consistently.
4. Top 3 systemic vulnerabilities apparent across all scenarios.

Structure the output cleanly using Markdown headers."""

        print(f"\nSubmitting Meta-Analysis Payload to Ollama ({self.model_name})...")
        print(f"  (This may take 1-3 minutes while Mistral loads — please wait...)")
             
        try:
             response = ollama.chat(
                 model=self.model_name,
                 messages=[
                     {'role': 'system', 'content': system_instruction},
                     {'role': 'user', 'content': user_prompt},
                 ],
                 options={'num_predict': 1024},
             )
             report_text = response.message.content
        except AttributeError:
             report_text = response['message']['content']
        except Exception as e:
             report_text = f"Ollama Meta-Analysis aborted - Initialization failed or model not found: {e}"
        
        output_dir = "outputs/"
        os.makedirs(output_dir, exist_ok=True)
        md_path = os.path.join(output_dir, "report_full_comparison.md")
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        print(f"Meta-Analysis Report saved to: {md_path}")
        return report_text

def run_report_generation(scenario_name, simulation_result, risk_report_df, detection_metrics, strategy_comparison_df, model_name="mistral"):
    """
    Convenience wrapper instantiating the generator and orchestrating the full pipeline logic inline.
    """
    generator = ReportGenerator(model_name=model_name)
    
    # Optional: Automatically try to pull model if it doesn't exist
    try:
        models_info = ollama.list()
        if hasattr(models_info, 'models'):
            available_models = [m.model for m in models_info.models]
        elif isinstance(models_info, dict):
            available_models = [m.get('name', m.get('model')) for m in models_info.get('models', [])]
        else:
            available_models = []
            
        if not any(m.startswith(generator.model_name) for m in available_models):
            print(f"Model '{generator.model_name}' not found locally. Attempting to pull...")
            ollama.pull(generator.model_name)
    except Exception:
        pass
    
    payload = generator.build_context_payload(
        scenario_name, 
        simulation_result, 
        risk_report_df, 
        detection_metrics, 
        strategy_comparison_df
    )
    
    report_text = generator.generate_report(payload)
    generator.save_report(report_text, scenario_name)
    
    return report_text


if __name__ == "__main__":
    import pandas as pd
    
    print("Mocking test data for ReportGenerator execution module...")
    
    # 1. Mock simulation logic returns
    mock_sim_result = {
        'final_infection_rate': 0.85,
        'time_to_first_critical_node': 12,
        'spread_velocity': 2.4,
        'total_timesteps': 45,
        'attack_stages_timeline': {'none': 0, 'initial_access': 6, 'lateral_movement': 12, 'ransomware_execution': 40}
    }
    
    # 2. Mock risk_engine arrays
    mock_risk_df = pd.DataFrame([
        {'node_id': 'IT-42', 'department': 'IT', 'node_type': 'server', 'risk_score': 0.95, 'vulnerability_score': 0.8, 'privilege_level': 'domain_admin', 'anomaly_score': 0.88, 'detected': True},
        {'node_id': 'HR-12', 'department': 'HR', 'node_type': 'workstation', 'risk_score': 0.85, 'vulnerability_score': 0.9, 'privilege_level': 'user', 'anomaly_score': 0.91, 'detected': True},
        {'node_id': 'Finance-01', 'department': 'Finance', 'node_type': 'workstation', 'risk_score': 0.80, 'vulnerability_score': 0.6, 'privilege_level': 'admin', 'anomaly_score': 0.45, 'detected': False},
        {'node_id': 'IT-18', 'department': 'IT', 'node_type': 'workstation', 'risk_score': 0.77, 'vulnerability_score': 0.5, 'privilege_level': 'user', 'anomaly_score': 0.12, 'detected': False},
        {'node_id': 'Ops-99', 'department': 'Operations', 'node_type': 'controller', 'risk_score': 0.60, 'vulnerability_score': 0.7, 'privilege_level': 'user', 'anomaly_score': 0.66, 'detected': True}
    ])
    
    # 3. Mock anomaly metrics
    mock_detect_metrics = {
        'detection_lead_time_mean': 4.5,
        'detection_lead_time_std': 1.2,
        'true_positive_rate': 0.92,
        'false_positive_rate': 0.08
    }
    
    # 4. Mock summary dataframe
    mock_compare_df = pd.DataFrame([
        {'strategy_name': 'anomaly_guided', 'mean_infection_rate': 0.15, 'mean_containment_time': 8},
        {'strategy_name': 'patch_centrality', 'mean_infection_rate': 0.35, 'mean_containment_time': 15},
        {'strategy_name': 'isolate_bridges', 'mean_infection_rate': 0.55, 'mean_containment_time': 22},
        {'strategy_name': 'none', 'mean_infection_rate': 0.85, 'mean_containment_time': None}
    ])
    
    print("\nExecuting Pipeline against local Ollama...")
    try:
        report = run_report_generation(
            "Demo_Cyber_Scenario", 
             mock_sim_result, 
             mock_risk_df, 
             mock_detect_metrics, 
             mock_compare_df,
             model_name="mistral"
        )
        print("\n" + "="*80)
        print("GENERATED REPORT PREVIEW (Top 500 chars):")
        print("="*80)
        print(report[:500] + "...\n[CONTINUED IN FILE]")
    except Exception as e:
        import traceback
        print(f"API Execution Error - Verify key validity: {e}")
        traceback.print_exc()