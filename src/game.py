from src.utils import EmotionBuffer, print_emotion_evolution
from tqdm import tqdm
import re

class TableGame:
    def __init__(self, reward_map):
        self.moves_to_rewards = reward_map
        # 自動從獎勵表中解析出合法的動作 (例如 ['J', 'F'])
        # reward_map keys 像是 "JJ", "JF", "FJ", "FF"
        # 我們取 key 的第一個字元當作合法動作集合
        self.valid_moves = list(set([k[0] for k in reward_map.keys()]))

    def run(self, agent1, agent2, logger):
        pass

    def parse_move(self, raw_step, valid_moves):
        """
        強健的動作解析器：從 LLM 的廢話中提取出合法的單一字母動作。
        """
        if raw_step in valid_moves:
            return raw_step
            
        # 1. 嘗試轉大寫並去除空白
        clean_step = str(raw_step).strip().upper()
        if clean_step in valid_moves:
            return clean_step
            
        # 2. 使用 Regex 尋找合法的動作字母 (例如 J 或 F)
        # 這會尋找字串中是否包含合法字母，並優先匹配獨立的字母
        for move in valid_moves:
            # 檢查是否有獨立的字母 (例如 "I choose J" 中的 J)
            if re.search(fr'\b{move}\b', clean_step):
                return move
        
        # 3. 如果還是找不到，放寬檢查，只要字串包含該字母就接受
        # (風險：如果 valid_moves 是 C, D，且回答 "Choose", 則會誤判為 C)
        # 為了安全，這裡只在上面 Regex 失敗時才做簡單包含檢查
        for move in valid_moves:
            if move in clean_step:
                return move
                
        return None


class RepeatedTableGame(TableGame):
    def __init__(
            self,
            reward_map,
            n_steps,
            need_check_emotions,
            need_demonstrate_emotions,
            memorize_demonstrated_emotions,
            memorize_seen_emotions,
    ):
        super().__init__(reward_map)
        self.n_steps = n_steps
        self.need_check_emotions = need_check_emotions
        self.need_demonstrate_emotions = need_demonstrate_emotions
        self.memorize_demonstrated_emotions = memorize_demonstrated_emotions
        self.memorize_seen_emotions = memorize_seen_emotions

    def run(self, agent1, agent2, logger):
        agent1.init_memory()
        agent2.init_memory()

        for step_num in range(self.n_steps): # tqdm(
            self._run_step(agent1, agent2, step_num, logger)

    def check_for_scatchpad(self, step):
        if isinstance(step, tuple):
            step, scratchpad_step = step
        else:
            step, scratchpad_step = step, None
        return step, scratchpad_step

    def _run_step(self, agent1, agent2, step_num, logger):
        # make current step
        raw_step1, scratchpad_step1 = self.check_for_scatchpad(agent1.make_step(step_num)) 
        raw_step2, scratchpad_step2 = self.check_for_scatchpad(agent2.make_step(step_num))
        
        # --- 修改重點：加入解析邏輯 ---
        step1 = self.parse_move(raw_step1, self.valid_moves)
        step2 = self.parse_move(raw_step2, self.valid_moves)
        
        # 防呆：如果解析失敗，記錄錯誤並使用預設值或跳過 (這裡選擇跳過並報錯)
        if step1 is None:
            print(f"⚠️ Agent 1 Move Parse Error: '{raw_step1}' not in {self.valid_moves}")
            step1 = self.valid_moves[0] # Fallback: 隨機選一個以免程式崩潰，或者你可以選擇 return
        if step2 is None:
            print(f"⚠️ Agent 2 Move Parse Error: '{raw_step2}' not in {self.valid_moves}")
            step2 = self.valid_moves[0]
        # ---------------------------

        try:
            reward1, reward2 = self.moves_to_rewards[step1 + step2]
        except KeyError as err:
            print(f"❌ Critical Logic Error: Combination '{step1+step2}' not found in reward map. Raw: {raw_step1}, {raw_step2}")
            return
            
        logger.log({"decisions": {"agent1": step1, "agent2": step2}})
        logger.log({"decisions_scratchpad": {"agent1": scratchpad_step1, "agent2": scratchpad_step2}})
        
        # reflect on self emotions
        additional_args1 = dict.fromkeys(
            ["inner_emotion", "outer_emotion", "opponent_outer_emotion"]
        )
        additional_args2 = dict.fromkeys(
            ["inner_emotion", "outer_emotion", "opponent_outer_emotion"]
        )
        if self.need_check_emotions:
            inner_emotion1 = agent1.get_inner_emotion()
            # 檢查 Agent 2 是否有情緒功能 (如果是 Rule-based agent 可能沒有這個方法)
            inner_emotion2 = agent2.get_inner_emotion() if hasattr(agent2, 'get_inner_emotion') else "neutral"
            
            agent1.update_emotion_memory(inner_emotion1)
            if hasattr(agent2, 'update_emotion_memory'):
                agent2.update_emotion_memory(inner_emotion2)
                
            additional_args1["inner_emotion"] = inner_emotion1
            additional_args2["inner_emotion"] = inner_emotion2
            logger.log(
                {
                    "inner_emotions": {
                        "agent1": inner_emotion1,
                        "agent2": inner_emotion2,
                    }
                }
            )
        # perceive and demonstrate emotions
        if self.need_demonstrate_emotions:
            outer_emotion1 = agent1.get_outer_emotion()
            outer_emotion2 = agent2.get_outer_emotion() if hasattr(agent2, 'get_outer_emotion') else "neutral"
            # agent1.perceive_opponent_emotion(outer_emotion2)
            # agent2.perceive_opponent_emotion(outer_emotion1)
            logger.log(
                {"outer_emotions": {"agent1": outer_emotion1, "agent2": outer_emotion2}}
            )
            if self.memorize_demonstrated_emotions:
                additional_args1["outer_emotion"] = outer_emotion1
                additional_args2["outer_emotion"] = outer_emotion2
            if self.memorize_seen_emotions:
                additional_args1["opponent_outer_emotion"] = outer_emotion1
                additional_args2["opponent_outer_emotion"] = outer_emotion2
                
        # Update memory: reformatted according to memory_update.txt answer
        # (optional) + self emotions + seen emotions
        memory_update1 = agent1.update_memory(
            step1,
            step2,
            reward1,
            reward2,
            step_num,
            inner_emotion=additional_args1["inner_emotion"],
            outer_emotion=additional_args1["outer_emotion"],
            outer_opponent_emotion=additional_args2["opponent_outer_emotion"],
        )
        
        # 針對 Agent 2 (如果是 Rule-based) 的相容性處理
        if hasattr(agent2, 'update_memory'):
            memory_update2 = agent2.update_memory(
                step2,
                step1,
                reward2,
                reward1,
                step_num,
                inner_emotion=additional_args2["inner_emotion"],
                outer_emotion=additional_args2["outer_emotion"],
                outer_opponent_emotion=additional_args1["opponent_outer_emotion"],
            )
        else:
            memory_update2 = "Rule-based Agent: No memory update"

        # print(step1, step2)
        # print(f"agent1: {memory_update1}")
        # print(f"agent2: {memory_update2}")
        logger.log({"memory": {"agent1": memory_update1, "agent2": memory_update2}})
