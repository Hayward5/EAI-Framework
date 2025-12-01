import pandas as pd
import os
import json
import argparse
import re
from typing import Tuple, Optional, Dict, Any

def parse_split_decision(decision_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parses 'number1;number2' for Dictator/Ultimatum Proposer.
    """
    try:
        # Filter strictly to keep digits and split char
        cleaned_str = ''.join(filter(lambda c: c.isdigit() or c in ';.', str(decision_str)))
        parts = cleaned_str.split(';')
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    except (ValueError, AttributeError, TypeError):
        pass
    return None, None

def parse_responder_decision(decision_str: str) -> Optional[int]:
    """
    Parses 'ACCEPT'/'REJECT' for Ultimatum Responder.
    Returns 1 for Accept, 0 for Reject, None for error.
    """
    if not isinstance(decision_str, str):
        return None
    
    clean_decision = decision_str.strip().upper()
    
    # Handle common variations found in logs
    if "ACCEPT" in clean_decision:
        return 1
    elif "REJECT" in clean_decision:
        return 0
    return None

def analyze_log_directory(log_dir_path: str) -> Optional[Dict[str, Any]]:
    # --- File Paths ---
    decisions_file = os.path.join(log_dir_path, 'div_decisions.csv')
    agent1_config_file = os.path.join(log_dir_path, 'agent1_config.json')
    agent2_config_file = os.path.join(log_dir_path, 'agent2_config.json')
    game_config_file = os.path.join(log_dir_path, 'game_config.json')

    if not all(os.path.exists(f) for f in [decisions_file, agent1_config_file, agent2_config_file, game_config_file]):
        return None

    try:
        with open(agent1_config_file, 'r') as f: cfg1 = json.load(f)
        with open(agent2_config_file, 'r') as f: cfg2 = json.load(f)
        with open(game_config_file, 'r') as f: game_cfg = json.load(f)
            
        # --- Determine Role and Agent ---
        is_responder_game = game_cfg.get("do_second_step", False) == True
        
        llm_agent_config = None
        llm_decision_col = None
        
        if cfg1.get("agent_name") == "llm":
            llm_agent_config = cfg1
            llm_decision_col = 0 # agent1 is column 1 (0-indexed after timestamp)
            # If LLM is Agent 1, usually Proposer
        elif cfg2.get("agent_name") == "llm":
            llm_agent_config = cfg2
            llm_decision_col = 1 # agent2 is column 2
            # If LLM is Agent 2, usually Responder in Ultimatum
        else:
            return None

        # Extract Emotion
        has_emotion = llm_agent_config.get('has_emotion', False)
        emotion_value = llm_agent_config.get('emotion_prompt_file', '')
        emotion = emotion_value.split('/')[0] if (has_emotion and emotion_value) else "no_emotion"
        
        llm_name = llm_agent_config.get('llm_name')
        game_name = game_cfg.get('name', 'unknown')

        # --- Load Data ---
        # CSV structure: Timestamp, Agent1_Decision, Agent2_Decision
        df = pd.read_csv(decisions_file, header=None)
        # Select only the decision columns (drop timestamp at col 0)
        df_decisions = df.iloc[:, 1:3] 
        
        total_sum = float(game_cfg.get('total_sum', 1.0))
        result_metrics = {
            "llm": llm_name,
            "game": game_name,
            "role": "Unknown",
            "emotion": emotion,
            "num_samples": 0,
            "metric_value": 0.0,
            "metric_type": "N/A"
        }

        # --- Logic Branch: Responder vs Proposer ---
        
        # Case 1: Ultimatum Responder (LLM is Agent 2 and do_second_step is True)
        if game_name == "ultimatum" and is_responder_game and llm_decision_col == 1:
            result_metrics["role"] = "Responder"
            result_metrics["metric_type"] = "Accept Rate (%)"
            
            # Parse as Boolean (1/0)
            decisions = df_decisions.iloc[:, llm_decision_col].apply(parse_responder_decision)
            valid_decisions = decisions.dropna()
            
            if not valid_decisions.empty:
                result_metrics["num_samples"] = len(valid_decisions)
                result_metrics["metric_value"] = round(valid_decisions.mean() * 100, 2)
        
        # Case 2: Proposer / Dictator (Splitter)
        else:
            result_metrics["role"] = "Proposer"
            result_metrics["metric_type"] = "Kept Share (%)"
            
            # Parse as Split (Keep;Give)
            decisions = df_decisions.iloc[:, llm_decision_col].apply(parse_split_decision)
            
            # Extract 'kept' part (index 0 of tuple)
            kept_values = decisions.apply(lambda x: x[0] if x else None)
            valid_decisions = kept_values.dropna()
            
            if not valid_decisions.empty:
                result_metrics["num_samples"] = len(valid_decisions)
                # Calculate percentage kept
                avg_kept = valid_decisions.mean()
                result_metrics["metric_value"] = round((avg_kept / total_sum) * 100, 2)

        if result_metrics["num_samples"] == 0:
            return None
            
        return result_metrics

    except Exception as e:
        print(f"Error in {os.path.basename(log_dir_path)}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_root", nargs='?', default="EAI-Framework/logs")
    args = parser.parse_args()

    if not os.path.isdir(args.log_root):
        print("Log directory not found.")
        return

    all_results = []
    for dir_name in sorted(os.listdir(args.log_root)):
        full_dir_path = os.path.join(args.log_root, dir_name)
        if os.path.isdir(full_dir_path):
            res = analyze_log_directory(full_dir_path)
            if res: all_results.append(res)

    if not all_results:
        print("No results found.")
        return

    df = pd.DataFrame(all_results)
    
    # Output 1: Proposer Stats
    print("\n=== Proposer Behavior (Avg % Kept) ===")
    proposer_df = df[df['role'] == 'Proposer']
    if not proposer_df.empty:
        piv = proposer_df.pivot_table(
            index=['llm', 'emotion'], columns='game', values='metric_value', aggfunc='mean'
        )
        print(piv.fillna("-"))
        # Save to CSV in the parent directory
        piv.to_csv('../proposer_analysis.csv')
    
    # Output 2: Responder Stats
    print("\n=== Responder Behavior (Accept Rate %) ===")
    responder_df = df[df['role'] == 'Responder']
    if not responder_df.empty:
        # Game is always ultimatum for responder here
        piv = responder_df.pivot_table(
            index=['llm'], columns='emotion', values='metric_value', aggfunc='mean'
        )
        print(piv.fillna("-"))
        # Save to CSV in the parent directory
        piv.to_csv('../responder_analysis.csv')

if __name__ == "__main__":
    main()
