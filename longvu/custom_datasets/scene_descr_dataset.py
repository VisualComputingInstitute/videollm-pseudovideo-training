from os import PathLike
import os
import json
from typing import List, Tuple, TypeVar, Type
from .question_formatting import MCQuestionFormatter
from .qa_dataset import QADataset

T = TypeVar('T', bound='SceneDescrDataset')

class SceneDescrDataset(QADataset):
    samples: List[Tuple[PathLike, str, Tuple[int, int]]]

    def __init__(self, videos_base_path: PathLike, jsonl_path: PathLike,
                 num_distractors_to_use: int, **mc_formatter_kwargs):
        self.videos_base_path = videos_base_path
        self.SAMPLE_RANGE = "sample_range_secs"
        self.VIDEO_NAME = "video_name"
        self.SCENE = "scene"
        self.num_distractors_to_use = num_distractors_to_use
        
        with(open(jsonl_path, "r") as f):
            contents = [json.loads(_) for _ in f.readlines()]

        self.samples = [
            (c[self.VIDEO_NAME], c[self.SCENE], c[self.SAMPLE_RANGE])
              for c in contents]
        self.all_scenes = set([_[1] for _ in self.samples])
        self.q_formatter = MCQuestionFormatter(
            "Which of the options below best describes the video scene?",
            **mc_formatter_kwargs)

    def __len__(self):
        return len(self.samples)

    @classmethod
    def from_config_KVs(cls: Type[T], *args, **kw_args) -> T:
        return cls(*args, **kw_args)
    
    def __getitem__(self, i: int) -> Tuple[PathLike, Tuple[int, int], str, str, 
                                           List[str]]:
        item = self.samples[i]
        q, a, a_list = self.q_formatter.format_to_question_answer(
            [], item[1], sorted(list(self.all_scenes.difference([item[1]]))), 
            self.num_distractors_to_use)
        return (os.path.join(self.videos_base_path, item[0]), item[2],
                q, a, a_list)
    
