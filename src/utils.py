import json
import os
import sys
from collections import defaultdict
import csv
from datetime import datetime
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

# ==========================================
# AWS Bedrock Client and Credentials Logic
# ==========================================

# --- AWS 憑證設定 ---
# 策略：優先讀取環境變數，若無則使用下方的備用憑證
AWS_ACCESS_KEY_FALLBACK = "01234567890"
AWS_SECRET_KEY_FALLBACK = "09876543210"
AWS_REGION = "us-east-1"

# Global boto3 client to be reused
bedrock_client: Optional[boto3.client] = None

def get_bedrock_client() -> boto3.client:
    """
    Initializes and returns a reusable AWS Bedrock Runtime client.
    Handles credential logic, prioritizing environment variables.
    """
    global bedrock_client
    if bedrock_client is not None:
        return bedrock_client

    # 1. 嘗試從環境變數取得 (最安全)
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", AWS_ACCESS_KEY_FALLBACK)
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", AWS_SECRET_KEY_FALLBACK)
    
    if "XXXX" in access_key or not access_key:
        print("🔴 錯誤: 未設定有效的 AWS 憑證。請設定環境變數 AWS_ACCESS_KEY_ID 和 AWS_SECRET_ACCESS_KEY，或修改 utils.py 中的 FALLBACK 變數。")
        sys.exit(1)

    print("🚀 Initializing AWS Bedrock client...")
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    return bedrock_client

def get_llm_response(model_name: str, messages: List[Dict[str, str]]) -> str:
    """
    Gets a response from an AWS Bedrock model using the Converse API.
    Handles 'system' parameter support and parses both Text and Reasoning blocks.
    """
    client = get_bedrock_client()

    # 1. Adapt message format
    system_prompts = [{"text": msg['content']} for msg in messages if msg['role'] == 'system']
    conversation = [
        {"role": msg['role'], "content": [{"text": msg['content']}]} 
        for msg in messages if msg['role'] != 'system'
    ]

    # Llama 3.1 check
    if "llama3-1" in model_name and not model_name.startswith("us.") and AWS_REGION == "us-east-1":
        print(f"⚠️  警告: 您可能需要為 {model_name} 使用 'us.' 前綴。")

    # 設定推論參數 (維持 512 以避免 Mistral/Llama 錯誤)
    inf_config = {"temperature": 0.0, "maxTokens": 1024}

    try:
        # First attempt: Use 'system' parameter
        response = client.converse(
            modelId=model_name,
            messages=conversation,
            system=system_prompts,
            inferenceConfig=inf_config
        )
    except ClientError as e:
        err_msg = e.response.get("Error", {}).get("Message", "").lower()
        
        # Mistral Fallback Logic
        if "system" in err_msg and "support" in err_msg:
            # print(f"🔄 Model {model_name} does not support system prompts. Merging...") # Optional logging
            if system_prompts and conversation and conversation[0]['role'] == 'user':
                all_sys_text = "\n".join([s['text'] for s in system_prompts])
                full_first_message = f"System Instructions:\n{all_sys_text}\n\nUser Request:\n{conversation[0]['content'][0]['text']}"
                conversation[0]['content'][0]['text'] = full_first_message
            
            try:
                response = client.converse(
                    modelId=model_name,
                    messages=conversation,
                    inferenceConfig=inf_config
                )
            except ClientError as fallback_e:
                print(f"🔴 AWS Bedrock Error on fallback: {fallback_e}")
                return "Error: API Fallback Failed"
        else:
            print(f"🔴 AWS Bedrock Error: {e}")
            return "Error: API Call Failed"
            
    except Exception as e:
        print(f"🔴 Unexpected error: {e}")
        return f"Error: {e}"

    # --- Parse Response (增強版) ---
    output_message = response.get('output', {}).get('message', {})
    final_text = ""
    
    for block in output_message.get('content', []):
        # 1. 標準文字區塊
        if 'text' in block:
            final_text += block['text']
        
        # 2. 推理內容區塊 (DeepSeek R1 / Nova 可能會用到)
        elif 'reasoningContent' in block:
            r_content = block['reasoningContent']
            if 'reasoningText' in r_content:
                # 將思考過程包在標籤中，或直接加入(視您的分析需求而定)
                # 這裡我們選擇加入，以免回應為空
                thought = r_content['reasoningText'].get('text', '')
                final_text += f"<think>{thought}</think>\n"

    final_text = final_text.strip()

    # --- 防呆機制：絕對不回傳空字串 ---
    if not final_text:
         print(f"⚠️  警告: 模型 {model_name} 回傳了完全空白的內容。")
         # 回傳一個空白格，防止下一輪對話崩潰 (ValidationException)
         # 同時標記錯誤，讓 Agent 知道出事了
         return "Error: Empty Response from Model"

    return final_text

# ==========================================
# Original Utility Functions (Preserved)
# ==========================================

def print_config(config):
    for k in config:
        print(k)
        print(config[k])
        print('------------')

def save_readable_config(config, run_name, log_path):
    with open(os.path.join(log_path, run_name, 'readable_summary.txt'), 'a') as f:
        for k in config:
            f.write(f'{k}\n')
            f.writelines(f'{config[k]}\n')
            f.write('------------\n')
        f.write('============\n')

class EmotionBuffer:
    def __init__(self):
        self.agent_id2emotions = defaultdict(list)

    def add_emotion(self, agent_id, emotion):
        self.agent_id2emotions[agent_id].append(emotion)


class BasicLogger:
    def __init__(
        self, keys: List[str], logs_path: str = "", run_name: str = None, game_name: str = '', model_suffix: str = ''
    ) -> None:
        self.keys: List[str] = keys
        self.logs_path: str = logs_path

        if run_name is None:
            self.run_name = datetime.now().strftime("%d_%m_%H%M%S")
            if game_name != '':
                self.run_name = game_name + '_' + self.run_name
            # 將模型名稱附加到資料夾名稱末端
            if model_suffix != '':
                self.run_name = self.run_name + '_' + model_suffix
        else:
            self.run_name = run_name

        os.makedirs(os.path.join(logs_path, self.run_name), exist_ok=True)

    def log_json(self, configs: Dict[str, Dict[str, any]]) -> None:
        for config_name, config in configs.items():
            self._write_config_to_file(config, config_name)

    def _write_config_to_file(self, config: Dict[str, any], filename: str) -> None:
        file_path = os.path.join(self.logs_path, self.run_name, f"{filename}.json")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(config, json_file, indent=4)

    @classmethod
    def construct_from_configs(
        cls,
        agent1_config: dict,
        agent2_config: dict,
        logs_path: str = "",
        run_name: str = None,
        game_name: str = None,
        model_suffix: str = '',
    ):
        keys = ["decisions", "memory", "decisions_scratchpad", "inner_emotions", "outer_emotions", "div_decisions",
                "div_decisions_scratchpad"]
        return cls(keys=keys, logs_path=logs_path, run_name=run_name, game_name=game_name, model_suffix=model_suffix)

    def log(self, values_dict: Dict[str, str], format='csv') -> None:
        pass


class TwoAgentsLogger(BasicLogger):
    def __init__(
        self, keys: List[str], logs_path: str = "", run_name: str = None, game_name: str = '', model_suffix: str = ''
    ) -> None:
        super().__init__(keys=keys, logs_path=logs_path, run_name=run_name, game_name=game_name, model_suffix=model_suffix)
    
    def log(self, values_dict: Dict[str, Dict[str, str]]) -> None:
        for key, agents in values_dict.items():
            if key in self.keys:
                file_path = os.path.join(
                    self.logs_path, self.run_name, f"{key}.csv"
                )
                with open(file_path, "a", newline="") as file:
                    writer = csv.writer(file)
                    if isinstance(agents, dict) and 'agent1' in agents and 'agent2' in agents:
                        writer.writerow(
                            [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), agents['agent1'], agents['agent2']]
                        )
                    else:
                        raise ValueError(f'Value for key {key} must be a dictionary with "agent1" and "agent2"')
            else:
                raise ValueError(f'Uninitialized key: {key}')


def print_emotion_evolution(emotion_buffer):
    pass

def read_text(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        return f.read()

def read_json(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        return json.load(f)
