from os import PathLike
from typing import List, Tuple, Dict
import random
import json
from enum import Enum
import torch
from datetime import datetime
import os
import shutil
from torch.utils.data import DataLoader

from ..dataset_instantiation import MultiQuestionLinMotDataset
from ..counting_question_generators import (
    HowManyObjsQGen,
    HowManyStillObjsQGen,
    HowManyMovingObjsQGen,
    HowManyObjsPresentColorQGen,
    HowManyObjsAbsentColorQGen,
    HowManyObjsPresentShapeQGen,
    HowManyObjsAbsentShapeQGen)
from .common import format_ans_to_tvbench_style, unformat_from_tvbench_style

class CountingTasks(str, Enum):
    TOTAL_OBJS="total objects"
    MOV_OBJS="moving objects"
    STILL_OBJS="still objects"
    PRESENT_COL="objects of color present"
    ABSENT_COL="objects of color absent"
    PRESENT_SHP="objects of shape present"
    ABSENT_SHP="objects of shape absent"

def val_set_is_valid(questions: List[Dict], num_expected_answers: int) -> bool:
    is_valid = True
    for q in questions:
        gt_ans = q["answer"]
        all_ans = q["candidates"]
        found = [_ == gt_ans for _ in all_ans]
        for _ in all_ans:
            #only an additional period should be added at this point
            if(format_ans_to_tvbench_style(_, 0)[:-1] != _):
                is_valid = False
                break
            if(not unformat_from_tvbench_style(_).isdigit()):
                is_valid = False
                break
        if(sum(found) != 1 or len(all_ans) != num_expected_answers):
            is_valid = False
        if(not is_valid):
            print("Invalid question: " + json.dumps(q))
            break
    return is_valid

def create_val_set(
        gt_jsonl_file: PathLike, 
        num_distractors: int) -> List[Tuple[str, List[Dict]]]:
    
    
    all_gens = [
        (CountingTasks.TOTAL_OBJS, HowManyObjsQGen()), 
        (CountingTasks.MOV_OBJS, HowManyMovingObjsQGen()), 
        (CountingTasks.STILL_OBJS, HowManyStillObjsQGen()),
        (CountingTasks.PRESENT_COL, HowManyObjsPresentColorQGen()),
        (CountingTasks.ABSENT_COL, HowManyObjsAbsentColorQGen()),
        (CountingTasks.PRESENT_SHP, HowManyObjsPresentShapeQGen()),
        (CountingTasks.ABSENT_SHP, HowManyObjsAbsentShapeQGen())]

    all_questions = []
    q_id = 0
    for task, gen in all_gens:
        helper_dataset = MultiQuestionLinMotDataset("", gt_jsonl_file,
                                                    num_distractors)    
        num_videos = len(helper_dataset.all_videos)  
        NUM_Q_PER_TYPE = 2*num_videos/len(all_gens)  
        helper_dataset.setup_with_question_generators([gen])

        prefix_len = gen.formatter.PREFIX_LEN
        loader = DataLoader(helper_dataset, shuffle=True,
                            collate_fn=lambda _:_)
        i = 0
        all_questions.append((task, []))
        for batch in loader:
            if(i > NUM_Q_PER_TYPE):
                break
            sample = batch[0]
            all_questions[-1][1].append({
                "video": sample[0],
                "question": sample[1].split("\n")[0],
                "question_id": q_id,
                "answer": format_ans_to_tvbench_style(
                    sample[2], prefix_len),
                "candidates": [format_ans_to_tvbench_style(_, prefix_len)
                                for _ in sample[3]],
            })
            i += 1
            q_id += 1        
    #print("Total number of questions: " + str(len(all_questions)))
    return all_questions

def create_complete_val_set(gt_jsonl_file: PathLike,
                            output_directory: PathLike):
    random.seed(0)
    torch.manual_seed(0)
    if(os.path.isdir(output_directory) 
       and len(os.listdir(output_directory)) > 0):
        tmp = os.path.split(output_directory)
        output_directory = os.path.join(
            tmp[0], tmp[1] + datetime.now().strftime("%d_%m_%Y_%H_%M_%s"))
    os.makedirs(output_directory, exist_ok=True)

    all_q = create_val_set(gt_jsonl_file, 3)
    json_names = {
        CountingTasks.TOTAL_OBJS: "total_objects.json",
        CountingTasks.MOV_OBJS: "moving_objects.json",
        CountingTasks.STILL_OBJS: "still_objects.json",
        CountingTasks.PRESENT_COL: "objs_of_col_present.json",
        CountingTasks.ABSENT_COL: "objs_of_col_absent.json",
        CountingTasks.PRESENT_SHP: "objs_of_shape_present",
        CountingTasks.ABSENT_SHP: "objs_of_shape_absent"        
    }
    for task, questions in all_q:
        assert val_set_is_valid(questions, 4)
        output_json = os.path.join(output_directory, json_names[task])
        with(open(output_json, "w") as f):
            json.dump(questions, f)
        shutil.copy(gt_jsonl_file, os.path.join(output_directory, 
                                                "corresponding_gt.jsonl"))    