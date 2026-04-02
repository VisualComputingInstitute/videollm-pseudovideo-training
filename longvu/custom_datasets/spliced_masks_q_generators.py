import random
from typing import Dict, List, Set, Any, Tuple
from dataclasses import dataclass
from os import PathLike
from numpy.typing import NDArray

from .img_splicing_dataset import ImageMask, ImgSplicingDatasetWithMasks
from .qa_dataset import Question
from .question_formatting import MCQuestionFormatter, FreeFormQuestionFormatter
from .spliced_imgs_q_gens import SplicedScenesQGen
from .mask_utils import create_random_relative_path

@dataclass
class MovingObjInfo:
    mask: ImageMask
    corresponding_img: PathLike
    appearance_time: float
    rel_start_time: float
    rel_end_time: float
    rel_path: NDArray

@dataclass
class MovingObjSceneFormation:
    moving_objs_info: List[MovingObjInfo]
    scene_formation_from_imgs: List[Tuple[PathLike, int]]

class IdentifyMovingObjsQGen(SplicedScenesQGen):
    all_obj_classes: Set[str]
    obj_class_to_imgs: Dict[str, List[int]]
    
    def __init__(self, min_num_mov_objs: int, max_num_mov_objs: int, 
                 max_start_as_frac: float, min_end_as_frac: float, 
                 max_path_steps: int, num_distractors: int,
                 mc_formatter_kwargs: Dict[str, Any]={},
                 **parent_kwargs):
        super().__init__(**parent_kwargs)

        self.min_num_mov_objs = min_num_mov_objs
        self.max_num_mov_objs = max_num_mov_objs
        self.max_start_as_frac = max_start_as_frac
        self.min_end_as_frac = min_end_as_frac
        self.max_path_steps = max_path_steps
        self.num_distractors = num_distractors

        self.mcq_formatter = MCQuestionFormatter(
            "Which of the options below best describes the objects seen "
            "moving in the video?", **mc_formatter_kwargs)
        self.free_formatter = FreeFormQuestionFormatter(
            "What objects can be seen moving in the video?")

    def setup_on_dataset(self, dataset: ImgSplicingDatasetWithMasks):
        self.dataset = dataset
        self.all_obj_classes = set()
        self.obj_class_to_imgs = {}
        for img_ind, obj_classes_quants in \
            dataset.img_ind_to_obj_classes_quants.items():
            for obj_class, _ in obj_classes_quants:
                self.all_obj_classes.add(obj_class)
                if(obj_class not in self.obj_class_to_imgs):
                    self.obj_class_to_imgs[obj_class] = []
                self.obj_class_to_imgs[obj_class].append(img_ind)
        self.all_obj_classes = sorted(list(self.all_obj_classes))
        self._randomly_choose_scenes()

    def _format_to_answer(self, 
                          moving_objs_class_counts: Dict[str, int]) -> str:
        return ", ".join("{0} {1}".format(count, cl) for cl, count in
                         moving_objs_class_counts.items())
    
    def form_free_form_question(self, question_index):
        return None

    def form_mc_question(self, question_index):
        q_formation_imgs = self.scene_formations_from_imgs[question_index]
        all_img_inds = set([_[0] for _ in q_formation_imgs])

        num_mov_objs = random.randint(
            self.min_num_mov_objs, self.max_num_mov_objs)
        total_frames = sum([_[1] for _ in q_formation_imgs])
        moving_objs_class_counts = {}
        moving_objs_info = []
        for o in range(num_mov_objs):
            obj_class = random.choice(self.all_obj_classes)
            if(obj_class not in moving_objs_class_counts):
                moving_objs_class_counts[obj_class] = 0
            moving_objs_class_counts[obj_class] += 1
            candidate_inds = sorted(list(set(self.obj_class_to_imgs[
                obj_class]).difference(all_img_inds)))
            chosen_ind = random.choice(candidate_inds)
            masks = self.dataset.get_masks_for_sample(chosen_ind)
            feasible_masks = [
                _ for _ in masks if _.mask_class == obj_class]
            assert len(feasible_masks) > 0

            chosen_mask = random.choice(feasible_masks)
            chosen_path = create_random_relative_path(
                min(self.max_path_steps, total_frames))

            chosen_start_time = random.uniform(0, self.max_start_as_frac)
            chosen_appearance_time = random.uniform(0, chosen_start_time)
            chosen_end_time = random.uniform(self.min_end_as_frac, 1)
            moving_objs_info.append(MovingObjInfo(
                chosen_mask, self.dataset.img_paths[chosen_ind],
                chosen_appearance_time, chosen_start_time, chosen_end_time, 
                chosen_path))
            
        gt_answer = self._format_to_answer(moving_objs_class_counts)
        already_included = [gt_answer]
        distractors = []
        while(len(distractors)< self.num_distractors):
            distractor_count = random.randint(1, self.max_num_mov_objs)
            class_counts = {}
            for _ in range(distractor_count):
                obj_class = random.choice(self.all_obj_classes)
                if(obj_class not in class_counts):
                    class_counts[obj_class] = 0
                class_counts[obj_class] += 1
            distractor_answer = self._format_to_answer(class_counts)
            if(distractor_answer not in already_included):
                distractors.append(distractor_answer)
                already_included.append(distractor_answer)

        imgs_repetitions = [
            (self.dataset.img_paths[_[0]], _[1]) for _ in q_formation_imgs]
        q, ans, candidates = self.mcq_formatter.format_to_question_answer(
            [], gt_answer, distractors)
        return Question(
            MovingObjSceneFormation(moving_objs_info, imgs_repetitions),
            q, ans, candidates)