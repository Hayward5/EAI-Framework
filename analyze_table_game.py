import pandas as pd
import os
import json
import argparse
from typing import Optional, Dict, Any

def analyze_log_directory(log_dir_path: str) -> Optional[Dict[str, Any]]:
    # --- File Paths ---
    decisions_file = os.path.join(log_dir_path, 'decisions.csv')
    agent1_config_file = os.path.join(log_dir_path, 'agent1_config.json')
    
    # [修改點 1] 改讀取 config.json
    config_file = os.path.join(log_dir_path, 'config.json')

    # Debug: 檢查檔案是否存在
    missing_files = []
    if not os.path.exists(decisions_file): missing_files.append("decisions.csv")
    if not os.path.exists(agent1_config_file): missing_files.append("agent1_config.json")
    # [修改點 2] 檢查 config.json
    if not os.path.exists(config_file): missing_files.append("config.json")
    
    if missing_files:
        print(f"   ⚠️  Skipping {os.path.basename(log_dir_path)}: Missing files -> {missing_files}")
        return None

    try:
        with open(agent1_config_file, 'r') as f: cfg1 = json.load(f)
        
        # [修改點 3] 從 config.json 中讀取 game_config
        with open(config_file, 'r') as f: full_config = json.load(f)
        # 兼容性處理：有時候是直接存 game_config，有時候是包在裡面
        game_cfg = full_config.get('game_config', full_config)
        
        llm_name = cfg1.get('llm_name', 'unknown')
        has_emotion = cfg1.get('has_emotion', False)
        emotion_full = cfg1.get('emotion', '')
        
        if has_emotion and emotion_full:
            emotion = emotion_full.split('/')[0]
        else:
            emotion = "no_emotion"
        if not emotion: emotion = "no_emotion"

        # --- Load Data ---
        try:
            df = pd.read_csv(decisions_file, header=None)
        except pd.errors.EmptyDataError:
            print(f"   ⚠️  Skipping {os.path.basename(log_dir_path)}: decisions.csv is completely empty.")
            return None
        except Exception as e:
            print(f"   ⚠️  Skipping {os.path.basename(log_dir_path)}: CSV read error ({e})")
            return None

        if df.empty: 
            print(f"   ⚠️  Skipping {os.path.basename(log_dir_path)}: DataFrame is empty.")
            return None

        if df.shape[1] < 2:
            print(f"   ⚠️  Skipping {os.path.basename(log_dir_path)}: CSV has too few columns ({df.shape[1]}). Expected >= 2.")
            return None

        agent1_moves = df.iloc[:, 1].astype(str).str.strip().str.upper()
        total_rounds = len(agent1_moves)
        
        if total_rounds == 0: 
            print(f"   ⚠️  Skipping {os.path.basename(log_dir_path)}: No rounds found.")
            return None

        # 計算合作率 (模糊匹配 J)
        COOPERATE_MOVE = "J"
        coop_count = agent1_moves.apply(lambda x: COOPERATE_MOVE in x).sum()
        coop_rate = (coop_count / total_rounds) * 100

        print(f"   ✅ Processed {os.path.basename(log_dir_path)}: {total_rounds} rounds, Coop Rate={coop_rate:.1f}%")

        return {
            "llm": llm_name,
            "emotion": emotion,
            "game": game_cfg.get('name', 'prisoner_dilemma'),
            "rounds": total_rounds,
            "coop_rate": round(coop_rate, 2)
        }

    except json.JSONDecodeError as e:
        print(f"   ⚠️  Skipping {os.path.basename(log_dir_path)}: JSON Decode Error ({e})")
        return None
    except Exception as e:
        print(f"   ⚠️  Skipping {os.path.basename(log_dir_path)}: Unexpected Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_root", nargs='?', default="logs") 
    args = parser.parse_args()

    if not os.path.isdir(args.log_root):
        print(f"❌ Error: Log directory '{args.log_root}' not found.")
        return

    all_results = []
    
    print(f"🔍 Scanning directory: {os.path.abspath(args.log_root)}")
    print("-" * 50)

    items = sorted(os.listdir(args.log_root))
    for dir_name in items:
        if not dir_name.startswith("prisoner_dilemma"):
            continue

        full_dir_path = os.path.join(args.log_root, dir_name)
        if os.path.isdir(full_dir_path):
            res = analyze_log_directory(full_dir_path)
            if res: 
                all_results.append(res)

    print("-" * 50)
    if not all_results:
        print("❌ No valid Prisoner's Dilemma results found.")
        return

    df = pd.DataFrame(all_results)
    
    print("\n=== Analysis Result ===")
    piv = df.pivot_table(
        index=['llm'], 
        columns='emotion', 
        values='coop_rate', 
        aggfunc='mean'
    )
    
    cols = sorted(piv.columns.tolist())
    if 'no_emotion' in cols:
        cols.insert(0, cols.pop(cols.index('no_emotion')))
    piv = piv[cols]

    print(piv.fillna("-"))
    
    output_csv = os.path.join(args.log_root, '../prisoner_dilemma_analysis.csv')
    piv.to_csv(output_csv)
    print(f"\n💾 Analysis saved to: {output_csv}")

if __name__ == "__main__":
    main()
