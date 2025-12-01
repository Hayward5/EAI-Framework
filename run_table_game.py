import argparse
from tqdm import tqdm
import os
import json
import traceback

# 引用模組
from src.agent.init_agent import init_agent
from src.config_utils.table_utils import prepare_game_description, prepare_agent_config
from src.dirs import LOG_PATH
from src.game import RepeatedTableGame
from src.evaluation import DecisionStatistics
from src.utils import TwoAgentsLogger, save_readable_config

# ===========================
# 1. 基礎設定 (Templates)
# ===========================

game_basic_config = {
    "name": "prisoner_dilemma", 
    "n_steps": 5,  # 保持 5 回合以節省成本
    "need_check_emotions": True,
    "need_demonstrate_emotions": False,
    "memorize_seen_emotions": False,
    "memorize_demonstrated_emotions": False,
}

naming_config = {
    "currency": "dollars",
    "coplayer": "coplayer",
    "move1": "J", # 合作
    "move2": "F", # 背叛
}

# Agent 1 (LLM) - 預設值，稍後會在迴圈中被覆蓋
agent1_basic_config = {
    "agent_name": "emotion_reflection_llm",
    "llm_name": "placeholder", 
    "has_emotion": False,
    "emotion": "",
    "do_scratchpad_step": False,
    "memory_update_addintional_keys": {
        'currency': naming_config["currency"]
    },
    "game_setting": {
        "round_question": "round_question",
        "general_template": "basic_template", 
        "environment": "experiment", 
        "emotions_info": "with_emotions_affect", 
        "final_instruction": "instruction",
    },
}

# Agent 2 (Rule-based: Tit-for-Tat)
agent2_basic_config = {
    "agent_name": "alterating", 
    "llm_name": "rule_based", 
    "has_emotion": False,
    "emotion": "none",
    "memory_update_addintional_keys": {
        'currency': naming_config["currency"]
    },
}

# ===========================
# 2. 執行邏輯
# ===========================

def run_game(game_config, naming_config, agent1_config, agent2_config, logger):
    game = RepeatedTableGame(
        reward_map=game_config["reward_map"],
        n_steps=game_config["n_steps"],
        need_check_emotions=game_config["need_check_emotions"],
        need_demonstrate_emotions=game_config["need_demonstrate_emotions"],
        memorize_demonstrated_emotions=game_config["memorize_demonstrated_emotions"],
        memorize_seen_emotions=game_config["memorize_seen_emotions"],
    )

    agent1 = init_agent(agent1_config["agent_name"], agent1_config)
    agent2 = init_agent(agent2_config["agent_name"], agent2_config)

    full_config = {
        "game_config": game_config,
        "naming_config": naming_config,
        "agent1_config": agent1_config,
        "agent2_config": agent2_config
    }
    logger.log_json({"config": full_config})
    # 兼容分析程式
    logger.log_json({"agent1_config": agent1_config}) 

    game.run(agent1, agent2, logger)


if __name__ == "__main__":
    
    # ==========================================
    # 3. 批量測試設定
    # ==========================================
    
    # 你提供的模型列表 (請確保 AWS Bedrock 有開通這些模型的權限)
    # 把想跑的模型取消註解即可
    llm_name_range = [
        "mistral.mistral-7b-instruct-v0:2",      # Mistral 7B (開源/小型)
        "mistral.mixtral-8x7b-instruct-v0:1",
        "meta.llama3-8b-instruct-v1:0",       # Llama 3 8B (US Profile, 若上面那個失敗通常這個會成功)
        "us.meta.llama3-1-70b-instruct-v1:0",
        "amazon.titan-text-lite-v1",
        "amazon.titan-text-express-v1",          # Amazon Titan (小型/閉源)
        "openai.gpt-oss-20b-1:0", # 
        "openai.gpt-oss-120b-1:0", 
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "anthropic.claude-3-5-sonnet-20240620-v1:0", 
        "cohere.command-r-v1:0",          # Cohere Command R (不同架構)
    ]
    
    # 檢查列表是否為空
    if not llm_name_range:
        print("⚠️  Warning: No models selected in 'llm_name_range'. Please uncomment at least one.")
        exit()

    experiments = [
        ("anger/simple", True),
        ("happiness/simple", True),
        ("no_emotion", False)
    ]
    
    print(f"📋 Total Models to Test: {len(llm_name_range)}")
    print(f"📋 Total Emotions per Model: {len(experiments)}")
    print("="*60)

    # --- 外層迴圈：遍歷每個模型 ---
    for model_idx, target_llm in enumerate(llm_name_range):
        print(f"\n🚀 [{model_idx+1}/{len(llm_name_range)}] Starting Experiments for Model: {target_llm}")
        
        # 加入 try-except，確保單一模型失敗不影響其他模型
        try:
            # --- 內層迴圈：遍歷每種情緒 ---
            # 使用 tqdm 顯示該模型的進度
            for emotion_name, has_emotion_flag in tqdm(experiments, desc=f"Testing {target_llm.split('.')[1] if '.' in target_llm else target_llm}"):
                
                # 1. 複製並設定 Agent 1
                current_agent1 = agent1_basic_config.copy()
                current_agent1["llm_name"] = target_llm
                current_agent1["has_emotion"] = has_emotion_flag
                current_agent1["emotion"] = emotion_name if has_emotion_flag else ""
                
                # 2. 準備遊戲設定
                final_game_config = prepare_game_description(
                    config=game_basic_config, 
                    naming_config=naming_config
                )
                
                # 3. 準備 Agent 1 (讀取 Prompt)
                final_agent1_config = prepare_agent_config(
                    config=current_agent1,
                    game_name=final_game_config["name"],
                    naming_config=naming_config,
                    agent_ind=1,
                )
                
                # ======================================================
                # [關鍵修正] 強制將 emotion 字串寫回設定檔，以免被 prepare 函式弄丟
                # ======================================================
                final_agent1_config["emotion"] = emotion_name if has_emotion_flag else ""
                # ======================================================
                
                # 4. 準備 Agent 2
                final_agent2_config = prepare_agent_config(
                    config=agent2_basic_config,
                    game_name=final_game_config["name"],
                    naming_config=naming_config,
                    agent_ind=2,
                )
                
                # 5. 初始化 Logger
                # 從模型名稱提取簡短版本 (例如 "meta.llama3-8b-instruct-v1:0" -> "llama3-8b")
                model_short_name = target_llm.split('.')[-1].split('-instruct')[0].split('-v')[0]
                logger = TwoAgentsLogger.construct_from_configs(
                    final_agent1_config, 
                    final_agent2_config, 
                    LOG_PATH, 
                    game_name=final_game_config['name'],
                    model_suffix=model_short_name
                )
                
                # 6. 執行遊戲
                run_game(final_game_config, naming_config, final_agent1_config, final_agent2_config, logger)
                
                # 7. 統計
                evaluate_statistics = DecisionStatistics(logger.run_name, LOG_PATH)
                decision_stats, count_combinations = evaluate_statistics.get_metric()
                
                save_readable_config(
                    {"decision_stats": decision_stats, "count_combinations": count_combinations},
                    logger.run_name,
                    LOG_PATH,
                )

        except Exception as e:
            print(f"\n❌ Critical Error with model {target_llm}: {str(e)}")
            print("Skipping to next model...")
            # traceback.print_exc() # 如果想看詳細錯誤訊息可打開這行

    print("\n" + "="*60)
    print("✅ All experiments finished!")
