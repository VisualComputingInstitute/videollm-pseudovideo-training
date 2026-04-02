from __future__ import annotations
from longvu.custom_datasets.qa_dataset import QADataset
from os import PathLike
import os
from typing import List, Tuple, Set, Optional
from itertools import product
from dataclasses import dataclass

from custom_datasets.common import parse_jsonl
from custom_datasets.question_formatting import MCQuestionFormatter

@dataclass
class MotionSample:
    video_path: PathLike
    obj_color: str
    obj_shape: str
    move_direction: str

class LinearMotionAndReverseQuestions(QADataset):
    samples: List[MotionSample]

    def __init__(self, videos_root_dir: PathLike, 
                 ground_truth_jsonl_path: PathLike,
                 num_distractors_to_use: int,
                 directions_to_include: Optional[Set[str]]=None):
        super().__init__()

        self.MOVING_OBJECTS = "moving_objects"
        self.STILL_OBJECTS = "still_objects"
        self.COLOR = "color"
        self.SHAPE = "shape"
        self.PATH = "path"
        self.DIRECTION = "direction"
        self.VIDEO_NAME = "video_name"
        
        self.videos_root_dir = videos_root_dir
        self.directions_to_include = directions_to_include
        self.num_distractors = num_distractors_to_use
        self.ground_truth_jsonl_path=  ground_truth_jsonl_path
        self.ground_truth = parse_jsonl(ground_truth_jsonl_path)

        self.samples = []
        self.possible_color_shape_combos = set()
        self.possible_directions = set()
        
        self.q_formatter_ask_for_motion = MCQuestionFormatter(
            "Which direction does the {0} {1} move in the video?")
        self.q_formatter_ask_for_object = MCQuestionFormatter(
            "Which object moves {0} in the video?")
        
        possible_colors = set()
        possible_shapes = set()
        for d in self.ground_truth:
            for o in d[self.MOVING_OBJECTS]:
                direction = o[self.PATH][self.DIRECTION]
                if(self.directions_to_include is None or 
                   direction in self.directions_to_include):
                    self.samples.append(MotionSample(
                        os.path.join(self.videos_root_dir, d[self.VIDEO_NAME]),
                        o[self.COLOR], o[self.SHAPE], direction))
                self.possible_directions.add(direction)
                possible_colors.add(o[self.COLOR])
                possible_shapes.add(o[self.SHAPE])
            for o in d[self.STILL_OBJECTS]:
                possible_colors.add(o[self.COLOR])
                possible_shapes.add(o[self.SHAPE])
        self.possible_color_shape_combos = set([ c + " " + s for c, s in
            product(possible_colors, possible_shapes)])
        self.num_samples = len(self.samples)
                
    def __len__(self):
        #each moving object corresponds to two questions
        return self.num_samples*2
    
    def __getitem__(self, i: int) -> Tuple[PathLike, str, str, List[str]]:
        if(i < self.num_samples):
            sample = self.samples[i]
            distractor_directions = list(self.possible_directions.difference(
                        [sample.move_direction]))
            return (
                sample.video_path,
                *self.q_formatter_ask_for_motion.format_to_question_answer(
                    [sample.obj_color, sample.obj_shape], 
                    sample.move_direction, distractor_directions, 
                    self.num_distractors))
        elif(i < len(self)):
            sample = self.samples[i % self.num_samples]
            correct_color_shape = sample.obj_color + " " + sample.obj_shape
            #ensure same order of conversion of set to list
            distractor_combos = sorted(list([_ for _ in (
                self.possible_color_shape_combos.difference(
                    [correct_color_shape]))]))
            return (
                sample.video_path, 
                *self.q_formatter_ask_for_object.format_to_question_answer(
                    [sample.move_direction], correct_color_shape, 
                    distractor_combos,self.num_distractors))
        else:
            raise IndexError