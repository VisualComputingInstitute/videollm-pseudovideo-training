from enum import Enum

class KnownDatasets(str, Enum):
    TVBENCH="tvbench"
    ONE_OBJ_LIN="one_obj_lin"
    ONE_OBJ_LIN_BUGGY="one_obj_lin_buggy"
    MORE_OBJS_LIN="more_objs_lin"
    FOUR_NEW_COLORS="four_new_colors"
    MORE_OBJS_LIN_COUNT="more_objs_lin_count"

class TVBenchStyleEvalMode(str, Enum):
    TVBENCH_ORIGINAL="tvbench_original_eval"
    MOTION_EVAL_SIMPLE="motion_eval_simple"
    MOTION_EVAL_COMPLETE="motion_eval_complete"
    COUNTING="counting"

class CompleteEvalSubtasks(str, Enum):
    MOV_DIRECTION="Moving Direction"
    CORRUPTED_ATTR="Corrupted Attributes"
    EASY_DISTR="Easy Distractors"
    MEDIUM_DISTR="Medium Distractors"
    HARD_DISTR="Hard Distractors"
    ALL_8_CHOICES="All 8 possible choices"
    PERM_BY_1="Moving Direction permuted by 1"
    PERM_BY_2="Moving Direction permuted by 2"
    PERM_BY_3="Moving Direction permuted by 3"
    SINGLE_BLACK_FR="Single black frame given to model"
    REPEATED_FIRST_FR="Repeated first frame given to model"

COMPLETE_EVAL_SUBTASK_JSON = {
    CompleteEvalSubtasks.MOV_DIRECTION: "movement_direction.json",
    CompleteEvalSubtasks.CORRUPTED_ATTR: "corrupted_attr.json",
    CompleteEvalSubtasks.EASY_DISTR: "easy.json",
    CompleteEvalSubtasks.MEDIUM_DISTR: "medium.json",
    CompleteEvalSubtasks.HARD_DISTR: "hard.json",
    CompleteEvalSubtasks.ALL_8_CHOICES: "all_8_choices.json",
    CompleteEvalSubtasks.PERM_BY_1: "perm_1.json",
    CompleteEvalSubtasks.PERM_BY_2: "perm_2.json",
    CompleteEvalSubtasks.PERM_BY_3: "perm_3.json",
    CompleteEvalSubtasks.SINGLE_BLACK_FR: "movement_direction.json",
    CompleteEvalSubtasks.REPEATED_FIRST_FR: "movement_direction.json"
}