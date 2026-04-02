from typing import Iterable, List, Tuple
import random
from typing import Optional, Dict
from enum import Enum
from copy import deepcopy

class ReplaceStrategy(Enum):
    REPLACE_CORRECT=0,
    REPLACE_WRONG=1,
    NO_REPLACE=2

class FreeFormQuestionFormatter:

    def __init__(self, question_template: str):
        self.question_template = question_template

    def format_question(
            self, 
            template_completions: Iterable[str]) -> str:
                
        question = self.question_template.format(*template_completions)
        return question + "\n"

class MCQuestionFormatter:

    def __init__(self, question_template: str, 
                 none_as_correct_prob: Optional[float]=None,
                 none_as_wrong_prob: Optional[float]=None):
        self.question_template = question_template
        self.PREFIX_LEN = len("(A) ")
        self.NONE = "none of the other answers is correct"
        if(none_as_correct_prob is not None 
           and (none_as_wrong_prob > 1 
                or none_as_correct_prob <= 0)):
            raise ValueError("none_as_correct_prob should be in (0, 1]")
        if(none_as_wrong_prob is not None 
           and (none_as_wrong_prob > 1 
                or none_as_wrong_prob <= 0)):
            raise ValueError("none_as_wrong_prob should be in (0, 1]")
        if((none_as_correct_prob is not None 
            and none_as_wrong_prob is not None) and
            (none_as_wrong_prob + none_as_correct_prob > 1)):
            raise ValueError(
                "none_as_correct_prob + none_as_wrong_prob "
                "should be in (0, 1]")
        self.none_above_as_correct_prob = none_as_correct_prob
        self.none_above_as_wrong_prob = none_as_wrong_prob

    def format_i_th_answer(self, answer: str, i: int):
        return "({0}) {1}\n".format(chr(ord('A') + i), answer)

    def format_to_question_answer(
            self, 
            template_completions: Iterable[str],
            gt_answer: str, distractors: List[str],
            num_distractors_to_pick: int=None) -> Tuple[str, str, List[str]]:
        assert (len(set(distractors)) == len(distractors)), \
            ("Provided list of distractors contains duplicates: {0}".
             format(distractors))
        replace_strategy = [ReplaceStrategy.NO_REPLACE]
        replace_probs = [1]
        if(self.none_above_as_correct_prob is not None):
            replace_strategy.append(ReplaceStrategy.REPLACE_CORRECT)
            replace_probs.append(self.none_above_as_correct_prob)
            replace_probs[0] -= self.none_above_as_correct_prob
        if(self.none_above_as_wrong_prob is not None):
            replace_strategy.append(ReplaceStrategy.REPLACE_WRONG)
            replace_probs.append(self.none_above_as_wrong_prob)
            replace_probs[0] -= self.none_above_as_wrong_prob
        if(len(replace_strategy) > 1):
            chosen = random.choices(replace_strategy, replace_probs, k=1)[0]
            if(chosen == ReplaceStrategy.REPLACE_CORRECT):
                gt_answer = self.NONE
            elif(chosen == ReplaceStrategy.REPLACE_WRONG):
                distractors = deepcopy(distractors)
                ind = random.randint(0, len(distractors) - 1)
                distractors[ind] = self.NONE

        question = self.question_template.format(*template_completions)
        distractors = random.sample(distractors, len(distractors))
        if(num_distractors_to_pick is not None):
            num_distractors_to_pick = min(num_distractors_to_pick,
                                          len(distractors))
            distractors = distractors[:num_distractors_to_pick]
        num_possible_answers = len(distractors) + 1
        gt_pos = random.randint(0, num_possible_answers - 1)
        ans = ([self.format_i_th_answer(_, i) 
                for i, _ in enumerate(distractors[:gt_pos])]
                + [self.format_i_th_answer(gt_answer, gt_pos)] 
                + [self.format_i_th_answer(_, i + gt_pos + 1) 
                   for i, _ in enumerate(distractors[gt_pos:])])
        str_splitter = "({0}) ".format(chr(ord('A') + gt_pos))
        assert (gt_answer.rstrip() == 
                ans[gt_pos].split(str_splitter)[1].rstrip()), \
            ("Ground truth answer does not appear in expected position ({0})."
             "\nQuestion: {1}\nGround truth: {2}Answers: {3}"
             .format(gt_pos, question, gt_answer, ans))
        assert (sum([gt_answer.rstrip() == 
                    _.split("({0}) ".format(chr(ord('A') + i)))[1] .rstrip()
                    for i, _ in enumerate(ans)]) == 1), \
               ("Ground truth answer appears more than once.\nQuestion: {0}\n"
                "Ground truth: {1}\n Answers: {2}".format(question, gt_answer,
                                                          ans))
        if(num_distractors_to_pick is None):
            num_distractors_to_pick = len(distractors)
        assert (
            set(_[1] for _ in ans) == set([chr(ord('A')+ _) for _ in range(
                num_distractors_to_pick + 1)])), \
               ("Badly formatted multiple-choice letters.\nAnswers: {0}"
                .format(ans))
        
        return question + "\n" + "".join(ans), ans[gt_pos], ans
    
class MCQFormatterWithSynonyms:
    formatters: List[MCQuestionFormatter]

    def __init__(self, base_question_template: str, 
                 synonym_question_templates: List[str], 
                 synonym_ans_maps: List[Dict[str, str]],
                 none_as_correct_prob: Optional[float]=None,
                 none_as_wrong_prob: Optional[float]=None
                 ):
        self.formatters = [MCQuestionFormatter(
            base_question_template, none_as_correct_prob, none_as_wrong_prob)
            ] + [MCQuestionFormatter(
                _, none_as_correct_prob, none_as_wrong_prob)
                for _ in synonym_question_templates]
        self.synonym_ans_maps = synonym_ans_maps

    def format_to_question_answer(
            self, 
            template_completions: Iterable[str],
            base_q_gt_answer: str, base_q_distractors: List[str],
            num_distractors_to_pick: int=None) -> Tuple[str, str, List[str]]:
        formatter_idx = random.randint(0, len(self.formatters) - 1)
        if(formatter_idx != 0):
            gt_answer = self.synonym_ans_maps[formatter_idx - 1][
                base_q_gt_answer]
            distractors = [self.synonym_ans_maps[formatter_idx - 1].get(_, _)
                           for _ in base_q_distractors]
        else:
            gt_answer = base_q_gt_answer
            distractors = base_q_distractors

        return self.formatters[formatter_idx].format_to_question_answer(
            template_completions, gt_answer, distractors, 
            num_distractors_to_pick)