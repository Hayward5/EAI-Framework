from copy import deepcopy

from tqdm import tqdm

from src.agent.init_agent import init_agent
from src.config_utils.division_utils import prepare_agent_config, prepare_division_game_config
from src.dirs import LOG_PATH
from src.division_game import DivisionGame
from src.utils import (
    TwoAgentsLogger,
    save_readable_config,
    print_config, read_text,
)
import argparse
import pickle

# The new get_bedrock_client in utils.py will handle credentials.
# No need to load .env or initialize the client here.


# Removed caching logic as it's not needed for simplified, fresh runs.


def generate_emotions(possible_emotions, possible_prompts):
    res = []
    for emotion in possible_emotions:
        for prompt_file in possible_prompts:
            res.append(f'{emotion}/{prompt_file}')
    return res


# --- START MODIFIED PARAMETERS FOR REPRESENTATIVE SAMPLING ---

# 1. 縮減金額情境 (Total Sums)：僅包含 [10**3]
total_sums = [10**3]

# 2. 鎖定對手角色 (Coplayers)：僅包含 ["opponent"]
coplayers = ["opponent"]

# 6. 設定 AWS Bedrock 模型 ID (Models)：閉源/強大代表 & 開源/權重代表
llms = [
    #"mistral.mistral-7b-instruct-v0:2",      # Mistral 7B (開源/小型)
    #"mistral.mixtral-8x7b-instruct-v0:1",
    #"meta.llama3-8b-instruct-v1:0",       # Llama 3 8B (US Profile, 若上面那個失敗通常這個會成功)
    #"us.meta.llama3-1-70b-instruct-v1:0",
    #"amazon.titan-text-lite-v1",
    #"amazon.titan-text-express-v1",          # Amazon Titan (小型/閉源)
    "openai.gpt-oss-20b-1:0", # 
    "openai.gpt-oss-120b-1:0", 
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0", 
    "cohere.command-r-v1:0",          # Cohere Command R (不同架構)
]

has_emotions = [True, False] # 保持不變，因為要對比有無情感的差異

# 3. 精簡情感變數 (Emotions)：僅包含 ["anger", "happiness"]
# 'no_emotion' 會在 generate_agent_configs 函式中自動處理
possible_emotions = ["anger", "happiness"]

# 4. 簡化情境提示 (Prompts)：僅包含 ["simple"]
possible_prompts = ["simple"]

emotions = generate_emotions(possible_emotions, possible_prompts)
do_scratchpad_steps = [True, False] # 保持不變，因為這會影響模型行為


# 5. 聚焦關鍵分配比例 (Split Ratios)：[0.2]
predefined_split_ratios = [0.2] 

# --- END MODIFIED PARAMETERS FOR REPRESENTATIVE SAMPLING ---


agent_basic_config = {
    "agent_name": "llm",
    "llm_name": "default_llm_agent", # This will be overwritten by llms list
    "has_emotion": False,
    "emotion": "surprise/situation_coplayer",
    "do_scratchpad_step": False,
    "memory_update_addintional_keys": {},
    "summary_step": "",
    "ratio": 0,
}


def generate_game_configs(game_name):
    return [ {"name": game_name, "total_sum": total_sum, "do_second_step": False} for total_sum in total_sums]


def generate_name_configs():
    return [ {"currency": "dollars", "coplayer": coplayer} for coplayer in coplayers]


def generate_agent_configs():
    cur_agent_basic_config = deepcopy(agent_basic_config)
    agent_configs = []
    for llm in llms:
        cur_agent_basic_config['llm_name'] = llm
        for do_scratchpad_step in do_scratchpad_steps:
            cur_agent_basic_config['do_scratchpad_step'] = do_scratchpad_step
            cur_agent_basic_config['has_emotion'] = True
            for emotion in emotions:
                cur_agent_basic_config['emotion'] = emotion
                agent_configs.append(deepcopy(cur_agent_basic_config))
            cur_agent_basic_config['has_emotion'] = False
            cur_agent_basic_config['emotion'] = ''
            agent_configs.append(deepcopy(cur_agent_basic_config))
    return agent_configs


def generate_predefined_agent_configs():
    res = []
    for ratio in predefined_split_ratios:
        cur_config = deepcopy(agent_basic_config)
        cur_config['agent_name'] = 'ratio_division'
        cur_config['summary_step'] = 'summary_step1'
        cur_config['ratio'] = ratio
        res.append(cur_config)
    return res

def run_game(game_config, naming_config, agent1_config, agent2_config, logger):
    game = DivisionGame(
        name=game_config['name'],
        total_sum=game_config['total_sum'],
        do_second_step=game_config['do_second_step'],
        coplayer_name=naming_config['coplayer']
    )

    agent1 = init_agent(agent1_config["agent_name"], agent1_config)
    agent2 = init_agent(agent2_config["agent_name"], agent2_config)

    logger.log_json(
        {
            "agent1_config": agent1_config,
            "agent2_config": agent2_config,
            "naming_config": naming_config,
            "game_config": game_config,
        }
    )

    game.run(agent1, agent2, logger)


def run_pipeline(game_basic_config, naming_config, agent1_basic_config, agent2_basic_config):
    game_config = prepare_division_game_config(game_basic_config)
    agent1_config = prepare_agent_config(
        config=agent1_basic_config,
        game_config=game_basic_config,
        naming_config=naming_config,
        agent_ind=1,
    )
    agent2_config = prepare_agent_config(
        config=agent2_basic_config,
        game_config=game_basic_config,
        naming_config=naming_config,
        agent_ind=2,
    )

    logger = TwoAgentsLogger.construct_from_configs(
        agent1_config, agent2_config, LOG_PATH, game_name=game_basic_config['name']
    )
    if args.verbose:
        print_config(game_config)
        print("==================")
        print_config(agent1_config)
        print("==================")
        print_config(agent2_config)
        raise Exception()

    run_game(game_config, naming_config, agent1_config, agent2_config, logger)

    save_readable_config(game_config, logger.run_name, LOG_PATH)
    save_readable_config(agent1_config, logger.run_name, LOG_PATH)
    save_readable_config(agent2_config, logger.run_name, LOG_PATH)




parser = argparse.ArgumentParser(description='Run single experiment')
parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose mode")
args = parser.parse_args()

if __name__ == "__main__":

    # cached_configs_1, cached_configs_2 = read_cached_configs()
    cached_configs_1, cached_configs_2 = [], []

    game_dictator_configs = generate_game_configs("dictator")
    game_ultimatum_configs = generate_game_configs("ultimatum")
    name_configs = generate_name_configs()
    agent_configs = generate_agent_configs()
    predefined_agent_configs = generate_predefined_agent_configs()

    print(len(game_dictator_configs), len(game_ultimatum_configs))
    print(len(name_configs))
    print(len(agent_configs), len(predefined_agent_configs))

    # ----dictator & ultimatum for 1st----

    all_configs_1 = []

    for game_basic_config in tqdm(game_dictator_configs + game_ultimatum_configs, desc='Iterating GAME conf'):
        game_basic_config["do_second_step"] = False
        for naming_config in tqdm(name_configs, desc='Iterating NAME conf'):
            for cur_agent_basic_config in tqdm(agent_configs, desc='Iterating AGENT conf'):
                cur_agent_basic_config1 = deepcopy(cur_agent_basic_config)
                cur_agent_basic_config2 = deepcopy(cur_agent_basic_config)
                cur_agent_basic_config1["summary_step"] = "summary_step1"
                cur_agent_basic_config2["summary_step"] = "summary_step2"
                #     configs.append((row.game_name, row.total_sum, row.coplayer, row.has_emotion, row.llm_name, row.emotion, row.emotion_prompt_file, row.do_scratchpad_step))
                emotion_prompt_file_parts = cur_agent_basic_config1["emotion"].split('/')
                if len(emotion_prompt_file_parts) != 2:
                    if cur_agent_basic_config1["emotion"] != '':
                        raise Exception()
                    emotion_prompt_file_parts = ['no_emotion', '']
                cur_config = (
                    game_basic_config['name'],
                    game_basic_config['total_sum'],
                    naming_config['coplayer'],
                    cur_agent_basic_config1['has_emotion'],
                    cur_agent_basic_config1['llm_name'],
                    emotion_prompt_file_parts[0],
                    emotion_prompt_file_parts[1],
                    cur_agent_basic_config1['do_scratchpad_step']
                )
                # if cur_config not in cached_configs_1:
                run_pipeline(game_basic_config, naming_config, cur_agent_basic_config1, cur_agent_basic_config2)
                # run_pipeline(game_basic_config, naming_config, cur_agent_basic_config1, cur_agent_basic_config2)

    # ----ultimatum----


    print('ULTIMATUM 2')
    all_configs_2 = []

    # --2--
    for game_basic_config in tqdm(game_ultimatum_configs, desc='Iterating GAME conf'):
        game_basic_config["do_second_step"] = True
        for naming_config in tqdm(name_configs, desc='Iterating NAME conf'):
            for cur_agent_basic_config in tqdm(agent_configs, desc='Iterating AGENT conf'):
                cur_agent_basic_config2 = deepcopy(cur_agent_basic_config)
                cur_agent_basic_config2["summary_step"] = "summary_step2"
                emotion_prompt_file_parts = cur_agent_basic_config2["emotion"].split('/')
                if len(emotion_prompt_file_parts) != 2:
                    if cur_agent_basic_config2["emotion"] != '':
                        raise Exception()
                    emotion_prompt_file_parts = ['no_emotion', '']
                for predefined_agent_config in predefined_agent_configs:
                    cur_config = (
                        game_basic_config['name'],
                        game_basic_config['total_sum'],
                        naming_config['coplayer'],
                        cur_agent_basic_config2['has_emotion'],
                        cur_agent_basic_config2['llm_name'],
                        emotion_prompt_file_parts[0],
                        emotion_prompt_file_parts[1],
                        cur_agent_basic_config2['do_scratchpad_step'],
                        predefined_agent_config['ratio']
                    )
                    all_configs_2.append(cur_config)
                    # if cur_config not in cached_configs_2:
                    run_pipeline(game_basic_config, naming_config, predefined_agent_config, cur_agent_basic_config2)
    #
    # with open('testing/possible_configs_1.pkl', 'wb') as f:
    #     pickle.dump(all_configs_1, f)

    with open('testing/possible_configs_22.pkl', 'wb') as f:
        pickle.dump(all_configs_2, f)
