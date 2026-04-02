from __future__ import annotations
from os import PathLike
import os
from typing import Set, Optional, List, Tuple, Dict
from dataclasses import dataclass
import random

from .qa_dataset import QADataset
from .common import parse_jsonl

class DistractorRandomizerMixin:

    def __init__(self, min_distr: int, max_distr: int):
        if(max_distr < min_distr):
            raise ValueError("`min_distr` should be <= `max_distr`")
        if(min_distr < 1):
            raise ValueError("`min_distr` should be at least 1")
        self.min_distr = min_distr
        self.max_distr = max_distr

    def get_random_num_distr(self) -> int:
        return random.randint(self.min_distr, self.max_distr)

@dataclass(frozen=True, order=True)
class MotionSample:
    video_path: PathLike
    obj_color: str
    obj_shape: str
    move_direction: str

class MultiQuestionLinMotDataset(QADataset):
    
    moving_samples: List[MotionSample]
    still_samples: List[MotionSample]
    synonym_map: Dict[str, List[str]]
    sampling_prob_per_sample: List[float]
    all_videos: List[PathLike]

    def __init__(self, videos_root_dir: PathLike, 
                 ground_truth_jsonl_path: PathLike,
                 directions_to_include: Optional[Set[str]]=None,
                 **parent_kwargs):
        super().__init__(**parent_kwargs)

        self.MOVING_OBJECTS = "moving_objects"
        self.STILL_OBJECTS = "still_objects"
        self.COLOR = "color"
        self.SHAPE = "shape"
        self.PATH = "path"
        self.DIRECTION = "direction"
        self.VIDEO_NAME = "video_name"

        self.videos_root_dir = videos_root_dir
        self.directions_to_include = directions_to_include
        self.ground_truth_jsonl_path=  ground_truth_jsonl_path
        self.ground_truth = parse_jsonl(ground_truth_jsonl_path)

        self.moving_samples = []
        self.still_samples = []
        self.NO_MOVEMENT = "none; it does not move"
        self.NOT_PRESENT = "no such object is present in the video"
        self.possible_directions = set()

        self.synonym_map = {
            self.NO_MOVEMENT: ["it remains still", "it does not move"],
            self.NOT_PRESENT: ["this object does not appear in the video"]
        }
        self.all_videos = []
        for d in self.ground_truth:
            if(d[self.VIDEO_NAME] not in self.all_videos):
                self.all_videos.append(d[self.VIDEO_NAME])
            for o in d[self.MOVING_OBJECTS]:
                direction = o[self.PATH][self.DIRECTION]
                if(self.directions_to_include is None or 
                   direction in self.directions_to_include):
                    self.moving_samples.append(MotionSample(
                        os.path.join(self.videos_root_dir, d[self.VIDEO_NAME]),
                        o[self.COLOR], o[self.SHAPE], direction))
                self.possible_directions.add(direction)
            for o in d[self.STILL_OBJECTS]:
                self.still_samples.append(MotionSample(
                    os.path.join(self.videos_root_dir, d[self.VIDEO_NAME]),
                    o[self.COLOR], o[self.SHAPE], self.NO_MOVEMENT))

    def get_random_synonym(self, move_direction: str) -> str:
        if(move_direction not in self.synonym_map):
            return move_direction
        else:
            return random.choice(self.synonym_map[move_direction])

    def __getitem__(self, i: int) -> Tuple[
        PathLike, str, str, Optional[List[str]]]:
        """"The first element of the tuple is the video path 
        corresponding to the question, second is the question, third the
        answer and optional fourth is the list of candidate answers (in
        case of multiple-choice format).
        """
        return super().__getitem__(i)