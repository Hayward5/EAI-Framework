from pathlib import Path
import os

LANGUAGE = 'english'
BASE_PATH = Path(__file__).parent.parent
PROMPT_PATH = BASE_PATH / 'prompts' / LANGUAGE
AGENT_PATH = PROMPT_PATH / 'agent'
GAME_SET_PATH = AGENT_PATH / 'game_settings'
INSTRUCT_PATH = GAME_SET_PATH / 'final_instruction'

LOG_PATH = BASE_PATH / "logs"

os.makedirs(LOG_PATH, exist_ok=True)
