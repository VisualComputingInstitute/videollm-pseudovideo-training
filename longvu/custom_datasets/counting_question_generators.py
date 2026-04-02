from typing import Set, Dict
from math import ceil
from itertools import chain

from .qa_dataset import Question, QuestionGenerator

from .multiq_linear_motion_dataset import (
    MultiQuestionLinMotDataset, DistractorRandomizerMixin)
from .question_formatting import MCQuestionFormatter


class DatasetCountInfo:
    total_objs_per_video: Dict[str, int]
    mov_objs_per_video: Dict[str, int]
    still_objs_per_video: Dict[str, int]
    objs_per_color_in_vid: Dict[str, Dict[str, int]]
    objs_per_shape_in_vid: Dict[str, Dict[str, int]]
    all_colors: Set[str]
    all_shapes: Set[str]
    max_objs_in_video: int
    max_mov_objs_in_video: int
    max_still_objs_in_video: int

    def __init__(self, dataset: MultiQuestionLinMotDataset):
        self.total_objs_per_video = {}
        self.mov_objs_per_video = {}
        self.still_objs_per_video = {}
        self.objs_per_color_in_vid = {}
        self.objs_per_shape_in_vid = {}
        self.all_colors = set()
        self.all_shapes = set()
        self.max_objs_in_video = -1
        self.max_mov_objs_in_video = -1
        self.max_still_objs_in_video = -1

        for idx, sample in chain(((0, _) for _ in dataset.moving_samples),
            ((1, _) for _ in dataset.still_samples)):
            path = sample.video_path
            color = sample.obj_color
            shape = sample.obj_shape
            if(path not in self.total_objs_per_video):
                self.total_objs_per_video[path] = 0
            if(path not in self.mov_objs_per_video):
                self.mov_objs_per_video[path] = 0
            if(path not in self.still_objs_per_video):
                self.still_objs_per_video[path] = 0

            if(path not in self.objs_per_color_in_vid):
                self.objs_per_color_in_vid[path] = {}
            if(path not in self.objs_per_shape_in_vid):
                self.objs_per_shape_in_vid[path] = {}
            if(color not in self.objs_per_color_in_vid[path]):
                self.objs_per_color_in_vid[path][color] = 0
            if(shape not in self.objs_per_shape_in_vid[path]):
                self.objs_per_shape_in_vid[path][shape] = 0

            self.all_colors.add(color)
            self.all_shapes.add(shape)

            self.total_objs_per_video[path] += 1
            if(idx == 0):
                self.mov_objs_per_video[path] += 1
            if(idx == 1):
                self.still_objs_per_video[path] += 1
            self.objs_per_color_in_vid[path][color] += 1
            self.objs_per_shape_in_vid[path][shape] += 1

        all_videos = set(self.total_objs_per_video.keys())
        assert (all_videos == set(self.still_objs_per_video.keys()))
        assert (all_videos == set(self.mov_objs_per_video.keys()))
        assert (all_videos == set(self.objs_per_color_in_vid.keys()))
        assert (all_videos == set(self.objs_per_shape_in_vid.keys()))
        self.all_videos_sorted = sorted(list(all_videos))
        
        for v in self.all_videos_sorted:
            if(self.total_objs_per_video[v] > self.max_objs_in_video):
                self.max_objs_in_video = self.total_objs_per_video[v]
            if(self.mov_objs_per_video[v] > self.max_mov_objs_in_video):
                self.max_mov_objs_in_video = self.mov_objs_per_video[v]
            if(self.still_objs_per_video[v] > self.max_still_objs_in_video):
                self.max_still_objs_in_video = self.still_objs_per_video[v]

            assert (self.still_objs_per_video[v] + self.mov_objs_per_video[v]
                    == self.total_objs_per_video[v])
            assert (self.total_objs_per_video[v] == sum(
                self.objs_per_shape_in_vid[v].values()))
            assert (self.total_objs_per_video[v] == sum(
                self.objs_per_color_in_vid[v].values()))

class CountingQuestionGenABC(QuestionGenerator):
    precomputed_info: DatasetCountInfo = None
    dataset_info: DatasetCountInfo
    
    def __init__(self, try_use_cached_info: bool=True):
        self.try_use_cached_info = try_use_cached_info
        self.dataset_info = None

    def setup_on_dataset(self, dataset: MultiQuestionLinMotDataset):
        if(self.precomputed_info is None or not self.try_use_cached_info):
            CountingQuestionGenABC.precomputed_info = DatasetCountInfo(dataset)
        self.dataset_info = self.precomputed_info

class HowManyObjsQGen(CountingQuestionGenABC, DistractorRandomizerMixin):

    def __init__(self, min_distr: int, max_distr: int, **mc_formatter_kwargs):
        DistractorRandomizerMixin.__init__(self, min_distr, max_distr)
        super().__init__(True)
        self.formatter = MCQuestionFormatter(
            "How many objects are there in the video?", **mc_formatter_kwargs)
        
    def setup_on_dataset(self, dataset: MultiQuestionLinMotDataset):
        self.dataset = dataset
        super().setup_on_dataset(dataset)
        self.distractors = list(range(
            1, self.dataset_info.max_objs_in_video + 4))

    def get_length(self):
        return len(self.dataset_info.all_videos_sorted)
        
    def form_question(self, question_index):
        video_path = self.dataset_info.all_videos_sorted[question_index]
        ans = self.dataset_info.total_objs_per_video[video_path]
        #note that 0 is not possible here, so we need to offset distractor 
        # indices by 1
        q, ans, candidates = self.formatter.format_to_question_answer(
            [], str(ans), 
            self.distractors[:ans - 1] + self.distractors[ans:], 
            self.get_random_num_distr())
        return Question(video_path, q, ans, candidates)
    
class HowManyMovingObjsQGen(CountingQuestionGenABC, DistractorRandomizerMixin):

    def __init__(self, min_distr: int, max_distr: int, **mc_formatter_kwargs):
        DistractorRandomizerMixin.__init__(self, min_distr, max_distr)
        super().__init__(True)
        self.formatter = MCQuestionFormatter(
            "How many moving objects are there in the video?", 
            **mc_formatter_kwargs)
        
    def setup_on_dataset(self, dataset: MultiQuestionLinMotDataset):
        self.dataset = dataset
        super().setup_on_dataset(dataset)
        self.distractors = list(range(
            0, self.dataset_info.max_mov_objs_in_video + 3))

    def get_length(self):
        return len(self.dataset_info.all_videos_sorted)
        
    def form_question(self, question_index):
        video_path = self.dataset_info.all_videos_sorted[question_index]
        ans = self.dataset_info.mov_objs_per_video[video_path]
        q, ans, candidates = self.formatter.format_to_question_answer(
            [], str(ans), self.distractors[:ans] + self.distractors[ans + 1:], 
            self.get_random_num_distr())
        return Question(video_path, q, ans, candidates)
    
class HowManyStillObjsQGen(CountingQuestionGenABC, DistractorRandomizerMixin):

    def __init__(self, min_distr: int, max_distr: int, **mc_formatter_kwargs):
        DistractorRandomizerMixin.__init__(self, min_distr, max_distr)
        super().__init__(True)
        self.formatter = MCQuestionFormatter(
            "How many objects that do not move are there in the video?", 
            **mc_formatter_kwargs)
        
    def setup_on_dataset(self, dataset: MultiQuestionLinMotDataset):
        self.dataset = dataset
        super().setup_on_dataset(dataset)
        self.distractors = list(range(
            0, self.dataset_info.max_still_objs_in_video + 3))

    def get_length(self):
        return len(self.dataset_info.all_videos_sorted)
        
    def form_question(self, question_index):
        video_path = self.dataset_info.all_videos_sorted[question_index]
        ans = self.dataset_info.still_objs_per_video[video_path]
        q, ans, candidates = self.formatter.format_to_question_answer(
            [], str(ans), self.distractors[:ans] + self.distractors[ans + 1:], 
            self.get_random_num_distr())
        return Question(video_path, q, ans, candidates)
    
class HowManyObjsPresentColorQGen(CountingQuestionGenABC, 
                                  DistractorRandomizerMixin):

    def __init__(self, min_distr: int, max_distr: int, **mc_formatter_kwargs):
        DistractorRandomizerMixin.__init__(self, min_distr, max_distr)
        super().__init__(True)
        self.formatter = MCQuestionFormatter(
            "How many {0} objects are there in the video?", 
            **mc_formatter_kwargs)
        
    def setup_on_dataset(self, dataset: MultiQuestionLinMotDataset):
        self.dataset = dataset
        super().setup_on_dataset(dataset)
        self.distractors = list(range(
            0, self.dataset_info.max_objs_in_video + 3))
        self.q_idx_to_video_and_color = []
        for v in self.dataset_info.all_videos_sorted:
            colors = self.dataset_info.objs_per_color_in_vid[v]
            self.q_idx_to_video_and_color += [(v, _) for _ in colors]
        self.distractors = list(range(
            ceil(self.dataset_info.max_objs_in_video 
            / len(self.dataset_info.all_colors))+ 3))
        
    def get_length(self):
        return len(self.q_idx_to_video_and_color)
        
    def form_question(self, question_index):
        video_path, color = self.q_idx_to_video_and_color[question_index]
        ans = self.dataset_info.objs_per_color_in_vid[video_path][color]
        q, ans, candidates = self.formatter.format_to_question_answer(
            [color], str(ans), 
            self.distractors[:ans] + self.distractors[ans + 1:], 
            self.get_random_num_distr())
        return Question(video_path, q, ans, candidates)

class HowManyObjsAbsentColorQGen(CountingQuestionGenABC, 
                                 DistractorRandomizerMixin):

    def __init__(self, min_distr: int, max_distr: int, **mc_formatter_kwargs):
        DistractorRandomizerMixin.__init__(self, min_distr, max_distr)
        super().__init__(True)
        self.formatter = MCQuestionFormatter(
            "How many {0} objects are there in the video?", 
            **mc_formatter_kwargs)
        
    def setup_on_dataset(self, dataset: MultiQuestionLinMotDataset):
        self.dataset = dataset
        super().setup_on_dataset(dataset)
        self.distractors = list(range(
            0, self.dataset_info.max_objs_in_video + 3))
        self.q_idx_to_video_and_color = []
        for v in self.dataset_info.all_videos_sorted:
            colors = self.dataset_info.all_colors.difference(
                self.dataset_info.objs_per_color_in_vid[v])
            self.q_idx_to_video_and_color += [(v, _) for _ in sorted(
                list(colors))]
        self.distractors = list(range(
            ceil(self.dataset_info.max_objs_in_video 
            / len(self.dataset_info.all_colors))+ 3))
        
    def get_length(self):
        return len(self.q_idx_to_video_and_color)
        
    def form_question(self, question_index):
        video_path, color = self.q_idx_to_video_and_color[question_index]
        ans = 0
        q, ans, candidates = self.formatter.format_to_question_answer(
            [color], str(ans), 
            self.distractors[:ans] + self.distractors[ans + 1:], 
            self.get_random_num_distr())
        return Question(video_path, q, ans, candidates)    
    
class HowManyObjsPresentShapeQGen(CountingQuestionGenABC, 
                                  DistractorRandomizerMixin):

    def __init__(self, min_distr: int, max_distr: int, **mc_formatter_kwargs):
        DistractorRandomizerMixin.__init__(self, min_distr, max_distr)
        super().__init__(True)
        self.formatter = MCQuestionFormatter(
            "How many {0}-shaped objects are there in the video?", 
            **mc_formatter_kwargs)
        
    def setup_on_dataset(self, dataset: MultiQuestionLinMotDataset):
        self.dataset = dataset
        super().setup_on_dataset(dataset)
        self.distractors = list(range(
            0, self.dataset_info.max_objs_in_video + 3))
        self.q_idx_to_video_and_shape = []
        for v in self.dataset_info.all_videos_sorted:
            shapes = self.dataset_info.objs_per_shape_in_vid[v]
            self.q_idx_to_video_and_shape += [(v, _) for _ in shapes]
        self.distractors = list(range(
            ceil(self.dataset_info.max_objs_in_video 
            / len(self.dataset_info.all_shapes))+ 3))
        
    def get_length(self):
        return len(self.q_idx_to_video_and_shape)
        
    def form_question(self, question_index):
        video_path, shape = self.q_idx_to_video_and_shape[question_index]
        ans = self.dataset_info.objs_per_shape_in_vid[video_path][shape]
        q, ans, candidates = self.formatter.format_to_question_answer(
            [shape], str(ans), 
            self.distractors[:ans] + self.distractors[ans + 1:], 
            self.get_random_num_distr())
        return Question(video_path, q, ans, candidates)
    
class HowManyObjsAbsentShapeQGen(CountingQuestionGenABC, 
                                 DistractorRandomizerMixin):

    def __init__(self, min_distr: int, max_distr: int, **mc_formatter_kwargs):
        DistractorRandomizerMixin.__init__(self, min_distr, max_distr)
        super().__init__(True)
        self.formatter = MCQuestionFormatter(
            "How many {0}-shaped objects are there in the video?", 
            **mc_formatter_kwargs)
        
    def setup_on_dataset(self, dataset: MultiQuestionLinMotDataset):
        self.dataset = dataset
        super().setup_on_dataset(dataset)
        self.distractors = list(range(
            0, self.dataset_info.max_objs_in_video + 3))
        self.q_idx_to_video_and_shape = []
        for v in self.dataset_info.all_videos_sorted:
            shapes = self.dataset_info.all_shapes.difference(
                self.dataset_info.objs_per_shape_in_vid[v])
            self.q_idx_to_video_and_shape += [(v, _) for _ in sorted(
                list(shapes))]
        self.distractors = list(range(
            ceil(self.dataset_info.max_objs_in_video 
            / len(self.dataset_info.all_shapes))+ 3))
        
    def get_length(self):
        return len(self.q_idx_to_video_and_shape)
        
    def form_question(self, question_index):
        video_path, shape = self.q_idx_to_video_and_shape[question_index]
        ans = 0
        q, ans, candidates = self.formatter.format_to_question_answer(
            [shape], str(ans), 
            self.distractors[:ans] + self.distractors[ans + 1:], 
            self.get_random_num_distr())
        return Question(video_path, q, ans, candidates)
        


        
            