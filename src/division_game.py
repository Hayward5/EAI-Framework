import re

class DivisionGame:
    def __init__(self, name, total_sum, do_second_step, coplayer_name):
        self.name = name
        self.total_sum = total_sum
        self.do_second_step = do_second_step
        self.coplayer_name = coplayer_name
        self.update_format = """Your {coplayer} splitted {total_sum} as following: he keeps {keep_sum} dollars and gives you {give_sum} dollars.
Do you accept the split?"""

    def run(self, agent1, agent2, logger):
        agent1.init_memory()
        agent2.init_memory()
        agent1._round_question_format = agent1._round_question_format.format(total_sum=self.total_sum,
                                                                             coplayer=self.coplayer_name)
        step1, scratchpad_step1 = self.check_for_scatchpad(agent1.make_step(""))
        
        # Gracefully handle parsing failure
        parsed_answer = self.parse_answer1(step1)
        if parsed_answer is None:
            print(f"⚠️  Warning: Could not parse Agent 1's decision. Raw response: '{step1}'")
            logger.log({"div_decisions": {"agent1": f"PARSE_FAILED: {step1}", "agent2": ""}})
            logger.log({"div_decisions_scratchpad": {"agent1": scratchpad_step1, "agent2": ""}})
            return # Exit this specific game run gracefully

        agent1_sum_part, agent2_sum_part = parsed_answer
        step2, scratchpad_step2 = '', ''

        if self.name == 'ultimatum' and self.do_second_step:
            cur_update = self.update_format.format(coplayer=self.coplayer_name,
                                                  total_sum=self.total_sum,
                                                  keep_sum=agent1_sum_part,
                                                  give_sum=agent2_sum_part)
            agent2._round_question_format = agent2._round_question_format.format(coplayer=self.coplayer_name,
                                                                                  total_sum=self.total_sum,
                                                                                  keep_sum=agent1_sum_part,
                                                                                  give_sum=agent2_sum_part)
            agent2._add_to_history("user", cur_update)
            step2, scratchpad_step2 = self.check_for_scatchpad(agent2.make_step(""))
            # We don't use the result of parse_answer2, so we just call it
            self.parse_answer2(step2)

        logger.log({"div_decisions": {"agent1": step1, "agent2": step2}})
        logger.log({"div_decisions_scratchpad": {"agent1": scratchpad_step1, "agent2": scratchpad_step2}})

    def parse_answer1(self, answer: str):
        """Robustly parses the answer to find a 'number;number' pattern."""
        # Search for the pattern using regex
        match = re.search(r'(\d+);(\d+)', str(answer))
        
        if match:
            try:
                agent1_sum_part = float(match.group(1))
                agent2_sum_part = float(match.group(2))
                return agent1_sum_part, agent2_sum_part
            except (ValueError, TypeError):
                return None # Return None if conversion fails
        
        # Return None if pattern is not found
        return None

    def parse_answer2(self, answer: str):
        """Parses the 'ACCEPT' or 'REJECT' response."""
        answer = str(answer).strip(' .,:\t\n$').upper()

        if 'ACCEPT' in answer:
            return True
        elif 'REJECT' in answer:
            return False
        
        print(f"⚠️  Warning: Could not parse Agent 2's ACCEPT/REJECT decision. Raw response: '{answer}'")
        return None # Return None if parsing fails

    def check_for_scatchpad(self, step):
        if isinstance(step, tuple):
            step, scratchpad_step = step
        else:
            step, scratchpad_step = step, None
        return step, scratchpad_step
