from os import PathLike
import os
import re
import shutil
import json
from copy import deepcopy
from typing import List, Dict, Tuple, Iterable, Callable, Set
from math import floor, ceil
import numpy as np
import random
from itertools import chain, product
from datetime import datetime

from longvu.custom_datasets.val_set_creation.common import (
    format_ans_to_tvbench_style, 
    unformat_from_tvbench_style)

from ..datasets import LinearMotion1QTypeDataset

DIRECTION_TO_DEGREES: Dict[str, float] = {
    "right": 0, 
    "up and to the right": 45,
    "up": 90, 
    "up and to the left": 135, 
    "left": 180,
    "down and to the left": 225, 
    "down": 270, 
    "down and to the right": 315
}

def convert_angle_to_0_360(angle: float) -> float:
    k = floor(angle/360)
    angle_0_360 = angle - k*360
    assert angle_0_360 >= 0 and angle_0_360 < 360
    return angle_0_360 

def smallest_vec_angle(v1_angle: float, v2_angle: float) -> float:
    diff = v1_angle - v2_angle
    diff = convert_angle_to_0_360(diff)
    return min(diff, 360 - diff)

ANGULAR_DIFFS_SORTED: Dict[str, List[Tuple[str, float]]] = {
    dir1: sorted([(dir2, smallest_vec_angle(deg1, deg2))
           for dir2, deg2 in DIRECTION_TO_DEGREES.items()], key=lambda _: _[1])
    for dir1, deg1 in DIRECTION_TO_DEGREES.items()}
#sanity check
for d, diffs in ANGULAR_DIFFS_SORTED.items():
    minimum = diffs.pop(0)
    assert minimum[0] == d and minimum[1] == 0
    assert (diffs[-1][1] == 180
            and (len(diffs) == len(DIRECTION_TO_DEGREES)- 1))
    
def create_val_set(
        gt_jsonl_file: PathLike, 
        num_distractors: int) -> Tuple[List[Dict], Set[str], Set[str], 
                                       Dict[str, Set[Tuple[str, str]]]]:
    helper_dataset = LinearMotion1QTypeDataset(
        "", gt_jsonl_file, num_distractors, exclude_still_not_present=True)
    
    prefix_len = helper_dataset.q_gen.formatter.PREFIX_LEN
    all_questions = []
    q_id = 0
    for sample in helper_dataset:
        all_questions.append({
            "video": sample[0],
            "question": sample[1].split("\n")[0],
            "question_id": q_id,
            "answer": format_ans_to_tvbench_style(
                sample[2], prefix_len),
            "candidates": [format_ans_to_tvbench_style(_, prefix_len)
                            for _ in sample[3]],
            "question_id": q_id
        })
        q_id += 1
    
    all_colors = set()
    all_shapes = set()
    moving_colors_shapes_per_vid = {}
    for s in helper_dataset.wrapped_dataset.moving_samples:
        if(s.obj_color not in all_colors):
            all_colors.add(s.obj_color)
        if(s.obj_shape not in all_shapes):
            all_shapes.add(s.obj_shape)
        if(s.video_path not in moving_colors_shapes_per_vid):
            moving_colors_shapes_per_vid[s.video_path] = set()
        moving_colors_shapes_per_vid[s.video_path].add(
            (s.obj_color, s.obj_shape))
    for s in helper_dataset.wrapped_dataset.still_samples:
        if(s.obj_color not in all_colors):
            all_colors.add(s.obj_color)
        if(s.obj_shape not in all_shapes):
            all_shapes.add(s.obj_shape)

    return all_questions, all_colors, all_shapes, moving_colors_shapes_per_vid

def loop_over_directions(
        directions: List[Tuple[float, str]],
        examine_order: Iterable[int], pick_n: int) -> List[str]:
    return [directions[_] for _ in examine_order[:pick_n]]

def sort_by_angle_shuffle_ties(
        angles_directions: List[Tuple[str, float]]) -> List[str]:
    ang_to_dirs = {}
    for dir, a in angles_directions:
        if(a not in ang_to_dirs):
            ang_to_dirs[a] = []
        ang_to_dirs[a].append(dir)
    to_return = []
    for a in sorted(list(ang_to_dirs.keys())):
        to_return += random.sample(ang_to_dirs[a], len(ang_to_dirs[a]))
    assert len(to_return) == len(angles_directions)
    return to_return

def criterion_easy(other_directions: List[Tuple[str, float]],
                   num_distractors: int) -> List[str]:
    dirs = sort_by_angle_shuffle_ties(other_directions)
    order = range(len(dirs) - 1, -1, -1) #larger angles => easier
    return loop_over_directions(dirs, order, num_distractors)

def criterion_medium(other_directions: List[Tuple[str, float]],
                   num_distractors: int) -> List[str]:
    dirs = sort_by_angle_shuffle_ties(other_directions)
    #alternate between hard and easy directions
    N = len(dirs)
    order = [[i, N - i - 1] for i in range(floor(N/2))]
    order = list(chain.from_iterable(order))
    if(len(order) < N): #odd length
        missing = floor(N/2)
        assert (len(order) == N - 1 and missing not in order)
        order.append(missing)
        assert(set(order) == set(range(N)))
    return loop_over_directions(dirs, order, num_distractors)

def criterion_hard(other_directions: List[Tuple[str, float]],
                   num_distractors: int) -> List[str]:
    dirs = sort_by_angle_shuffle_ties(other_directions)
    order = range(len(dirs))
    return loop_over_directions(dirs, order, num_distractors)

def create_difficulty_val_set(
        existing_questions: List[Dict], 
        num_distractors: int,
        criterion: Callable[[List[Tuple[str, float]], int], List[str]]
        ) -> List[Dict]:
    new_q = deepcopy(existing_questions)
    if(num_distractors > len(DIRECTION_TO_DEGREES) - 1):
        raise ValueError(
            "{0} distractors is impossible with {1} possible directions"
            .format(num_distractors, len(DIRECTION_TO_DEGREES)))
    
    for q in new_q:
        gt_direction = unformat_from_tvbench_style(q["answer"])
        if(gt_direction not in DIRECTION_TO_DEGREES):
            raise KeyError("Unknown direction \"" + gt_direction + "\"")
        
        other_directions = ANGULAR_DIFFS_SORTED[gt_direction]
        distractor_dirs = criterion(other_directions, num_distractors)
        q["candidates"] = [format_ans_to_tvbench_style(_, 0)
                            for _ in distractor_dirs] + [q["answer"]]
        q["candidates"] = random.sample(q["candidates"], len(q["candidates"]))
    return new_q

def create_permuted_val_set(existing_questions: List[Dict],
                            permute_by: int) -> List[Dict]:
    num_answers = len(existing_questions[0]["candidates"])
    if(permute_by == 0 or permute_by > num_answers - 1):
        raise ValueError("An answer can only be permuted by a non-negative & "
                         "smaller than the number of answers quantity")
    new_q = deepcopy(existing_questions)
    for q in new_q:
        gt_ans = q["answer"]
        all_ans = q["candidates"]
        found = [_ == gt_ans for _ in all_ans]
        assert sum(found) == 1
        gt_ind = found.index(True)
        new_ind = (gt_ind + permute_by) % num_answers
        tmp = all_ans[new_ind]
        all_ans[new_ind] = gt_ans
        all_ans[gt_ind] = tmp
    return new_q

def extract_color_and_shape(question: str) -> Tuple[str, str]:
    pattern = r"(.*?) the (\w+) (\w+) move (.*?)"
    match = re.match(pattern, question)
    if match:
        _, color, shape, _ = match.groups()
        return color, shape
    else:
        raise Exception("Color and/or shape could not be extracted from "
                        "question \"" + question + "\"")

def create_corrupted_attr_val_set(
        existing_questions: List[Dict],
        moving_colors_shapes_per_vid: Dict[str, Set[Tuple[str, str]]],
        all_colors: Set[str],
        all_shapes: Set[str]) -> List[Dict]:
    new_q = deepcopy(existing_questions)

    for q, original_q in zip(new_q, existing_questions):
        if(moving_colors_shapes_per_vid[q["video"]]
           == set(product(all_colors, all_shapes))):
            raise Exception(
                "Video \"" + q["video"] +"\" appears to contain all "
                "combinations of colors and shapes as moving objects. "
                "Attribute corruption cannot be performed.")
        color, shape = extract_color_and_shape(q["question"])
        other_color = random.choice(sorted(list(
            all_colors.difference(set([color])))))
        other_shape = random.choice(sorted(list(
            all_shapes.difference(set([shape])))))
        while((other_color, other_shape) 
              in moving_colors_shapes_per_vid[q["video"]]):
            other_color = random.choice(sorted(list(
                all_colors.difference(set([color])))))
            other_shape = random.choice(sorted(list(
                all_shapes.difference(set([shape])))))            
        assert (color != other_color and shape != other_shape 
                and color is not None and shape is not None 
                and other_color is not None and other_shape is not None)  

        q["question"] = q["question"].replace(color, other_color)
        q["question"] = q["question"].replace(shape, other_shape)

        assert q["question"] != original_q["question"]

    return new_q

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
            if(unformat_from_tvbench_style(_) not in DIRECTION_TO_DEGREES):
                is_valid = False
                break
        if(sum(found) != 1 or len(all_ans) != num_expected_answers):
            is_valid = False
        if(not is_valid):
            print("Invalid question: " + json.dumps(q))
            break
    return is_valid
        
def create_complete_val_set(gt_jsonl_file: PathLike,
                            output_directory: PathLike):
    random.seed(0)
    if(os.path.isdir(output_directory) 
       and len(os.listdir(output_directory)) > 0):
        tmp = os.path.split(output_directory)
        output_directory = os.path.join(
            tmp[0], tmp[1] + datetime.now().strftime("%d_%m_%Y_%H_%M_%s"))
    os.makedirs(output_directory, exist_ok=True)

    all_questions, all_colors, all_shapes, moving_colors_shapes_per_vid = (
        create_val_set(gt_jsonl_file, 3))
    easy = create_difficulty_val_set(all_questions, 3, criterion_easy)
    medium = create_difficulty_val_set(all_questions, 3, criterion_medium)
    hard = create_difficulty_val_set(all_questions, 3, criterion_hard)
    all_8_choices, _, _, _ = create_val_set(gt_jsonl_file, 8)
    perm_1 = create_permuted_val_set(all_questions, 1)
    perm_2 = create_permuted_val_set(all_questions, 2)
    perm_3 = create_permuted_val_set(all_questions, 3)
    corrupted_attr = create_corrupted_attr_val_set(
        all_questions, moving_colors_shapes_per_vid, all_colors, all_shapes)

    all_datasets = [
        (all_questions, "movement_direction.json", 4), (easy, "easy.json", 4),
        (medium, "medium.json", 4), (hard, "hard.json", 4), 
        (all_8_choices, "all_8_choices.json", 8), (perm_1, "perm_1.json",4 ),
        (perm_2, "perm_2.json", 4), (perm_3, "perm_3.json", 4),
        (corrupted_attr, "corrupted_attr.json", 4)]
    
    all_equal_len = all(
        [len(_[0]) == len(all_datasets[0][0]) for _ in all_datasets])
    assert all_equal_len

    for d in all_datasets:
        assert val_set_is_valid(d[0], d[2])
        output_json = os.path.join(output_directory, d[1])
        with(open(output_json, "w") as f):
            json.dump(d[0], f)
    shutil.copy(gt_jsonl_file, os.path.join(output_directory, 
                                            "corresponding_gt.jsonl"))
    
def create_complete_val_set_for_tvbench(tvbench_json_file: PathLike,
                                        output_directory: PathLike):
    random.seed(0)
    if(os.path.isdir(output_directory) 
       and len(os.listdir(output_directory)) > 0):
        tmp = os.path.split(output_directory)
        output_directory = (
            tmp[0] + tmp[1] + datetime.now().strftime("%d_%m_%Y_%H_%M_%s"))
    os.makedirs(output_directory, exist_ok=True)

    all_questions = []
    all_questions_8_directions = []
    all_colors = set()
    all_shapes = set()
    moving_colors_shapes_per_vid = {}
    with(open(tvbench_json_file, "r") as f):
        contents = json.load(f)
        for i, c in enumerate(contents):
            c["question_id"] = i
            color, shape = extract_color_and_shape(c["question"])
            all_colors.add(color)
            all_shapes.add(shape)
            if(c["video"] not in moving_colors_shapes_per_vid):
                moving_colors_shapes_per_vid[c["video"]] = set()
            moving_colors_shapes_per_vid[c["video"]].add((color, shape))
            all_questions.append(c)
            question_copy = deepcopy(c)
            for direction in DIRECTION_TO_DEGREES.keys():
                formatted_dir = format_ans_to_tvbench_style(direction, 0)
                if formatted_dir not in question_copy["candidates"]:
                    question_copy["candidates"].append(formatted_dir)
            random.shuffle(question_copy["candidates"])
            all_questions_8_directions.append(question_copy)

    perm_1 = create_permuted_val_set(all_questions, 1)
    perm_2 = create_permuted_val_set(all_questions, 2)
    perm_3 = create_permuted_val_set(all_questions, 3)
    easy = create_difficulty_val_set(all_questions, 3, criterion_easy)
    medium = create_difficulty_val_set(all_questions, 3, criterion_medium)
    hard = create_difficulty_val_set(all_questions, 3, criterion_hard)    
    corrupted_attr = create_corrupted_attr_val_set(
        all_questions, moving_colors_shapes_per_vid, all_colors, all_shapes)
    
    all_datasets = [
        (all_questions, "movement_direction.json", 4), (easy, "easy.json", 4),
        (medium, "medium.json", 4), (hard, "hard.json", 4), 
        (all_questions_8_directions, "all_8_choices.json", 8), 
        (perm_1, "perm_1.json",4 ), (perm_2, "perm_2.json", 4), 
        (perm_3, "perm_3.json", 4), (corrupted_attr, "corrupted_attr.json", 4)]
    
    all_equal_len = all(
        [len(_[0]) == len(all_datasets[0][0]) for _ in all_datasets])
    assert all_equal_len

    for d in all_datasets:
        assert val_set_is_valid(d[0], d[2])
        output_json = os.path.join(output_directory, d[1])
        with(open(output_json, "w") as f):
            json.dump(d[0], f)    