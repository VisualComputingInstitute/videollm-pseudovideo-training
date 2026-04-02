from os import PathLike
from typing import Optional, Set
from .multiq_linear_motion_dataset import MultiQuestionLinMotDataset
from .qa_dataset import QADataset
from .motion_question_generators import (
    MovementDirectionQGen, ObservationQGen, StillObjectQGen, 
    NotPresentObjectQGen)

class LinearMotion1QTypeDataset(QADataset):

    def __init__(self, videos_root_dir: PathLike, 
                 ground_truth_jsonl_path: PathLike,
                 num_distractors_to_use: int,
                 directions_to_include: Optional[Set[str]]=None,
                 none_as_correct_prob: Optional[float]=None,
                 none_as_wrong_prob: Optional[float]=None,
                 exclude_still_not_present: Optional[bool]=False):
        
        self.wrapped_dataset = MultiQuestionLinMotDataset(
            videos_root_dir, ground_truth_jsonl_path, num_distractors_to_use,
            directions_to_include, also_return_answers=True)
        self.q_gen = MovementDirectionQGen(
            exclude_still_not_present=exclude_still_not_present,
            none_as_correct_prob=none_as_correct_prob,
            none_as_wrong_prob=none_as_wrong_prob)
        self.wrapped_dataset.setup_with_question_generators([self.q_gen])

    def __len__(self):
        return len(self.wrapped_dataset)

    def __getitem__(self, i):
        return self.wrapped_dataset[i]
    
class LinearMotion4QTypesDataset(QADataset):

    def __init__(self, videos_root_dir: PathLike, 
                 ground_truth_jsonl_path: PathLike,
                 num_distractors_to_use: int,
                 directions_to_include: Optional[Set[str]]=None,
                 none_as_correct_prob: Optional[float]=None,
                 none_as_wrong_prob: Optional[float]=None):
        
        self.wrapped_dataset = MultiQuestionLinMotDataset(
            videos_root_dir, ground_truth_jsonl_path, num_distractors_to_use,
            directions_to_include)
        self.q_gen = [
            MovementDirectionQGen(
                none_as_correct_prob=none_as_correct_prob,
                none_as_wrong_prob=none_as_wrong_prob),
            ObservationQGen(
                self.wrapped_dataset, 
                none_as_correct_prob=none_as_correct_prob,
                none_as_wrong_prob=none_as_wrong_prob),
            StillObjectQGen(
                self.wrapped_dataset, 1,
                none_as_correct_prob=none_as_correct_prob,
                none_as_wrong_prob=none_as_wrong_prob), 
            NotPresentObjectQGen(
                self.wrapped_dataset, 1,
                none_as_correct_prob=none_as_correct_prob,
                none_as_wrong_prob=none_as_wrong_prob)]
        self.wrapped_dataset.setup_with_question_generators(self.q_gen)

    def __len__(self):
        return len(self.wrapped_dataset)

    def __getitem__(self, i):
        return self.wrapped_dataset[i]