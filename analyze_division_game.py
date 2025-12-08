import re  # 添加此行以使用正則表達式
import csv  # 添加此行以使用 csv.reader
from typing import Tuple, Optional, Dict, Any
import os
import json
import pandas as pd
import argparse

def parse_split_decision(decision_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parses 'number1;number2' for Dictator/Ultimatum Proposer.
    """
    try:
        # 使用正則表達式提取第一個 'num;num' 模式，避免重複字串干擾
        match = re.search(r'(\d+);(\d+)', str(decision_str))
        if match:
            num1 = float(match.group(1))
            num2 = float(match.group(2))
            return num1, num2
        return None
    except (ValueError, AttributeError, TypeError):
        return None

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
        # 使用 csv.reader 手動解析 CSV，以正確處理帶引號的多行字串
        with open(decisions_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        if not rows:
            return None
        
        # Assume first row is the decision row: timestamp, agent1_decision, agent2_decision
        row = rows[0]
        if len(row) < 3:
            return None
        
        agent1_decision = row[1]
        agent2_decision = row[2]
        
        total_sum = float(game_cfg.get('total_sum', 1.0))
        result_metrics = {
            "llm": llm_name,
            "game": game_name,
            "role": "Unknown",
            "emotion": emotion,
            "num_samples": 1,  # Assuming one sample per log
            "metric_value": 0.0,
            "metric_type": "N/A"
        }

        # --- Logic Branch: Responder vs Proposer ---
        
        # Case 1: Ultimatum Responder (LLM is Agent 2 and do_second_step is True)
        if game_name == "ultimatum" and is_responder_game and llm_decision_col == 1:
            result_metrics["role"] = "Responder"
            result_metrics["metric_type"] = "Accept Rate (%)"
            
            # Parse as Boolean (1/0)
            decision_parsed = parse_responder_decision(agent2_decision)
            if decision_parsed is not None:
                result_metrics["metric_value"] = round(decision_parsed * 100, 2)
            else:
                return None
        
        # Case 2: Proposer / Dictator (Splitter)
        else:
            result_metrics["role"] = "Proposer"
            result_metrics["metric_type"] = "Kept Share (%)"
            
            # Parse as Split (Keep;Give)
            decision_to_parse = agent1_decision if llm_decision_col == 0 else agent2_decision
            split_parsed = parse_split_decision(decision_to_parse)
            if split_parsed and split_parsed[0] is not None:
                kept = split_parsed[0]
                result_metrics["metric_value"] = round((kept / total_sum) * 100, 2)
            else:
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
        print(f"Error: {args.log_root} is not a directory.")
        return

    all_results = []
    llm_order = []  # 記錄 LLM 出現順序
    for dir_name in sorted(os.listdir(args.log_root)):
        log_dir_path = os.path.join(args.log_root, dir_name)
        if os.path.isdir(log_dir_path):
            result = analyze_log_directory(log_dir_path)
            if result:
                all_results.append(result)
                # 記錄 LLM 順序（避免重複）
                if result['llm'] not in llm_order:
                    llm_order.append(result['llm'])

    if not all_results:
        print("No valid results found.")
        return

    df = pd.DataFrame(all_results)
    
    # Output 1: Proposer Stats
    proposer_df = df[df['role'] == 'Proposer']
    if not proposer_df.empty:
        proposer_pivot = proposer_df.pivot_table(values='metric_value', index=['game', 'llm'], columns='emotion', aggfunc='mean')
        # 按 LLM 處理順序排序
        proposer_pivot = proposer_pivot.reindex(llm_order, level='llm')
        print("=== Proposer Behavior (Avg % Kept) ===")
        print(proposer_pivot)
        proposer_pivot.to_csv('../proposer_analysis.csv')

    # Output 2: Responder Stats
    responder_df = df[df['role'] == 'Responder']
    if not responder_df.empty:
        responder_pivot = responder_df.pivot_table(values='metric_value', index=['llm'], columns='emotion', aggfunc='mean')
        # 按 LLM 處理順序排序
        responder_pivot = responder_pivot.reindex(llm_order)
        print("=== Responder Behavior (Accept Rate %) ===")
        print(responder_pivot)
        responder_pivot.to_csv('../responder_analysis.csv')

if __name__ == "__main__":
    main()
