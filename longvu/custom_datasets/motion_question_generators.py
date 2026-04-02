import random
from typing import Tuple, Set, List, Dict
import itertools

from .qa_dataset import Question, QuestionGenerator

from .multiq_linear_motion_dataset import (
    MultiQuestionLinMotDataset, MotionSample,
    DistractorRandomizerMixin)
from .question_formatting import MCQuestionFormatter, MCQFormatterWithSynonyms


class MovementDirectionQGen(QuestionGenerator, DistractorRandomizerMixin):

    def __init__(self, min_distr: int, max_distr: int, 
                 exclude_still_not_present: bool=False, 
                 enable_synonyms: bool=False, **mc_formatter_kwargs):
        DistractorRandomizerMixin.__init__(self, min_distr, max_distr)
        self.exclude_still_not_present = exclude_still_not_present
        if(not enable_synonyms):
            self.formatter = MCQuestionFormatter(
                "Which direction does the {0} {1} move in the video?",
                **mc_formatter_kwargs)
        else:
            self.formatter = MCQFormatterWithSynonyms(
                "Which direction does the {0} {1} move in the video?",
                **mc_formatter_kwargs)

    def setup_on_dataset(self, dataset: MultiQuestionLinMotDataset):
        self.dataset = dataset
    
    def form_question(self, question_index: int) -> Question:
        assert question_index >= 0
        sample = self.dataset.moving_samples[question_index]

        if(not self.exclude_still_not_present):
            distractors = sorted(list(
                self.dataset.possible_directions.difference(
                    [sample.move_direction]))
                + [self.dataset.get_random_synonym(self.dataset.NO_MOVEMENT), 
                    self.dataset.get_random_synonym(self.dataset.NOT_PRESENT)])
        else:
            distractors = sorted(list(self.dataset.possible_directions
                                      .difference([sample.move_direction])))
        
        question, ans, candidates = self.formatter.format_to_question_answer(
            [sample.obj_color, sample.obj_shape], sample.move_direction,
            distractors, self.get_random_num_distr())
        
        return Question(sample.video_path, question, ans, candidates)
    
    def get_length(self) -> int:
        return len(self.dataset.moving_samples)
    
class ObservationQGen(QuestionGenerator, DistractorRandomizerMixin):
    video_objects: List[Tuple[str, Set[MotionSample]]]
    all_obj_colors: List[str]
    all_obj_shapes: List[str]

    def __init__(self, min_distr: int, max_distr: int, **mc_formatter_kwargs):
        DistractorRandomizerMixin.__init__(self, min_distr, max_distr)
        self.formatter = MCQuestionFormatter(
            "What kind of motion do you observe in the video?", 
            **mc_formatter_kwargs)
        self.ans_frmt_str = "a {0} {1} moving {2}"

        self.video_objects = []
        self.all_obj_colors = []
        self.all_obj_shapes = []

    def setup_on_dataset(self, dataset: MultiQuestionLinMotDataset):
        self.dataset = dataset

        tmp_mapper: Dict[str, int] = {}
        for s in dataset.moving_samples:
            if(s.video_path not in tmp_mapper):
                tmp_mapper[s.video_path] = len(self.video_objects)
                self.video_objects.append((s.video_path, set()))
            self.video_objects[tmp_mapper[s.video_path]][1].add(s)
            if(s.obj_color not in self.all_obj_colors):
                self.all_obj_colors.append(s.obj_color)
            if(s.obj_shape not in self.all_obj_shapes):
                self.all_obj_shapes.append(s.obj_shape)
        for s in dataset.still_samples:
            if(s.obj_color not in self.all_obj_colors):
                self.all_obj_colors.append(s.obj_color)
            if(s.obj_shape not in self.all_obj_shapes):
                self.all_obj_shapes.append(s.obj_shape)        


    def get_length(self) -> int:
        return len(self.video_objects)
    
    def form_question(self, question_index: int) -> Question:
        assert question_index >= 0

        video_file, video_objs = self.video_objects[question_index]
        gt_ans = []
        for o in video_objs:
            gt_ans.append(self.ans_frmt_str.format(o.obj_color, o.obj_shape,
                                                o.move_direction))
        gt_ans = ", ".join(gt_ans)

        distractors = []
        corresponding_objs = []

        for d in range(self.get_random_num_distr()):
            all_objs_correct = True
            answer_exists = True
            while(all_objs_correct or answer_exists):
                all_objs_correct = True
                answer_exists = True                
                num_objects = random.randint(1, len(video_objs) + 1)
                new_ans = []
                distractor_objs = []
                for _ in range(num_objects):
                    already_included = True
                    while(already_included):
                        c = random.choice(self.all_obj_colors)
                        s = random.choice(sorted(list(self.all_obj_shapes)))
                        mov_dir = random.choice(sorted(
                            list(self.dataset.possible_directions)))
                        proposed = MotionSample(video_file, c, s, mov_dir)
                        if(proposed not in distractor_objs):
                            already_included = False
                    distractor_objs.append(proposed)
                    new_ans.append(self.ans_frmt_str.format(c, s, mov_dir))

                as_set = set(distractor_objs)    
                if(video_objs != as_set):
                    all_objs_correct = False
                if(all([as_set != _ for _ in corresponding_objs])):
                    answer_exists = False
            corresponding_objs.append(as_set)
            new_ans = ", ".join(new_ans)
            assert new_ans != gt_ans
            distractors.append(new_ans)
        
        question, ans, candidates = self.formatter.format_to_question_answer(
            [], gt_ans, distractors, None)
        return Question(video_file, question, ans, candidates)

class StillObjectQGen(QuestionGenerator, DistractorRandomizerMixin):
    video_question_objs: List[MotionSample]

    def __init__(self, min_distr: int, max_distr: int, 
                 num_questions_per_video: int, **mc_formatter_kwargs):
        DistractorRandomizerMixin.__init__(self, min_distr, max_distr)
        self.formatter = MCQuestionFormatter(
            "Which direction does the {0} {1} move in the video?",
            **mc_formatter_kwargs)
        self.num_questions_per_video = num_questions_per_video

        self.video_question_objs = []
        
    def setup_on_dataset(self, dataset: MultiQuestionLinMotDataset):
        self.dataset = dataset
        tmp_mapper = {}
        for still_obj in dataset.still_samples:
            if(still_obj.video_path not in tmp_mapper):
                tmp_mapper[still_obj.video_path] = []
            tmp_mapper[still_obj.video_path].append(still_obj)
        for video_file, still_objs in tmp_mapper.items():
            chosen_objs = random.sample(still_objs, 
                min(self.num_questions_per_video, len(still_objs)))
            self.video_question_objs += [_ for _ in chosen_objs]
        assert (
            len(self.video_question_objs) <= self.num_questions_per_video*
            len(list(tmp_mapper.keys())))        
        
    def form_question(self, question_index: int) -> Question:
        assert question_index >= 0
        sample = self.video_question_objs[question_index]

        distractors = sorted(list(self.dataset.possible_directions))
        
        gt_ans = self.dataset.get_random_synonym(self.dataset.NO_MOVEMENT)
        question, ans, candidates = self.formatter.format_to_question_answer(
            [sample.obj_color, sample.obj_shape], gt_ans, distractors,
            self.get_random_num_distr())
        
        return Question(sample.video_path, question, ans, candidates)        
    
    def get_length(self) -> int:
        return len(self.video_question_objs)
    
class NotPresentObjectQGen(QuestionGenerator, DistractorRandomizerMixin):
    video_objects: List[Tuple[str, Set[MotionSample]]]
    video_question_objs: List[Tuple[str, MotionSample]]
    all_obj_colors: List[str]
    all_obj_shapes: List[str]

    def __init__(self, min_distr: int, max_distr: int, 
                 num_questions_per_video: int, **mc_formatter_kwargs):
        DistractorRandomizerMixin.__init__(self, min_distr, max_distr)
        self.formatter = MCQuestionFormatter(
            "Which direction does the {0} {1} move in the video?",
            **mc_formatter_kwargs)
        self.num_questions_per_video = num_questions_per_video

        self.video_objects = []
        self.all_obj_colors = []
        self.all_obj_shapes = []
        
    def setup_on_dataset(self, dataset: MultiQuestionLinMotDataset):
        tmp_mapper: Dict[str, int] = {}
        self.dataset = dataset
        for s in dataset.moving_samples + dataset.still_samples:
            if(s.video_path not in tmp_mapper):
                tmp_mapper[s.video_path] = len(self.video_objects)
                self.video_objects.append((s.video_path, set()))
            self.video_objects[tmp_mapper[s.video_path]][1].add(s)
            if(s.obj_color not in self.all_obj_colors):
                self.all_obj_colors.append(s.obj_color)
            if(s.obj_shape not in self.all_obj_shapes):
                self.all_obj_shapes.append(s.obj_shape)
        #check if some videos feature all possible objects
        marked_inds = []
        prod = set(itertools.product(self.all_obj_colors, self.all_obj_shapes))
        for video_file, index in tmp_mapper.items():
            video_obj = self.video_objects[index][1]
            assert video_file == self.video_objects[index][0]
            all_video_properties = set([(_.obj_color, _.obj_shape) 
                                        for _ in video_obj])
            if(prod == all_video_properties):
                marked_inds.append(index)
        marked_inds.sort(reverse=True)
        for i in marked_inds:
            self.video_objects.pop(i)
        #form some questions for each video
        self.video_question_objs = []
        for video_file, video_objs in self.video_objects:
            diff = prod.difference([
                (_.obj_color, _.obj_shape) for _ in video_objs])
            assert len(diff) > 0
            for q in range(self.num_questions_per_video):
                color, shape = random.choice(sorted(list(diff)))
                self.video_question_objs.append([video_file, MotionSample(
                    video_file, color, shape, dataset.NO_MOVEMENT)])
        assert (len(self.video_question_objs) 
                == len(self.video_objects)*self.num_questions_per_video)
        
    def get_length(self) -> int:
        return len(self.video_objects)*self.num_questions_per_video
    
    def form_question(self, question_index: int) -> Question:
        video_file, question_obj = self.video_question_objs[question_index]

        distractors = sorted(
            list(self.dataset.possible_directions) 
            + [self.dataset.get_random_synonym(self.dataset.NO_MOVEMENT)])
        question, ans, candidates = self.formatter.format_to_question_answer(
            [question_obj.obj_color, question_obj.obj_shape], 
            self.dataset.get_random_synonym(self.dataset.NOT_PRESENT),
            distractors, self.get_random_num_distr())
        
        return Question(video_file, question, ans, candidates)

class MovementInReverseQGen(QuestionGenerator, DistractorRandomizerMixin):

    def __init__(self, min_distr: int, max_distr: int, 
                 exclude_still_not_present: bool=False, **mc_formatter_kwargs):
        DistractorRandomizerMixin.__init__(self, min_distr, max_distr)
        self.exclude_still_not_present = exclude_still_not_present
        self.formatter = MCQuestionFormatter(
            "If the video were in reverse, which direction would the {0} {1} "
            "move in the video?", **mc_formatter_kwargs)
        
        self.inverse_mapper = {
            "right": "left",
            "left": "right",
            "up and to the right": "down and to the left",
            "down and to the left": "up and to the right",
            "up": "down",
            "down": "up",
            "up and to the left": "down and to the right",
            "down and to the right": "up and to the left"
        }

    def setup_on_dataset(self, dataset: MultiQuestionLinMotDataset):
        self.dataset = dataset

    def form_question(self, question_index: int) -> Question:
        assert question_index >= 0
        sample = self.dataset.moving_samples[question_index]

        if(not self.exclude_still_not_present):
            inv_dir = self.inverse_mapper[sample.move_direction]
            distractors = sorted(
                list(self.dataset.possible_directions.difference([inv_dir]))
                + [self.dataset.get_random_synonym(self.dataset.NO_MOVEMENT), 
                   self.dataset.get_random_synonym(self.dataset.NOT_PRESENT)])
        else:
            distractors = sorted(list(
                self.dataset.possible_directions.difference([inv_dir])))
        
        question, ans, candidates = self.formatter.format_to_question_answer(
            [sample.obj_color, sample.obj_shape], inv_dir,
            distractors, self.get_random_num_distr())
        
        return Question(sample.video_path, question, ans, candidates)
    
    def get_length(self):
        return len(self.dataset.moving_samples)