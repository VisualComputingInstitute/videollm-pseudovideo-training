from typing import List, Tuple, Any, Dict, Set
import random
import numpy as np
from math import ceil

from .qa_dataset import Question, MCAndFreeFormQGen
from .img_splicing_dataset import ImgSplicingDatasetWithObjInfo
from .question_formatting import MCQuestionFormatter, FreeFormQuestionFormatter


class HowManyInstOfClassQGen(MCAndFreeFormQGen):
    scene_formations_from_imgs: List[Tuple[int, int]]
    all_obj_classes: Set[str]
    obj_class_to_img_and_quantity: Dict[str, List[Tuple[int, int]]]
    scene_formations_from_imgs: List[Tuple[int, int]]
    corresponding_gt_class_quantity: List[Tuple[str, int]]
    
    def __init__(self, 
                 max_queried_objs_of_class: int,
                 min_num_distractor_scenes: int, 
                 max_num_distractor_scenes: int, 
                 min_frames_per_scene: int, 
                 max_frames_per_scene: int,
                 enforced_len: int,
                 num_distractors: int,
                 mc_formatter_kwargs: Dict[str, Any]={},
                 **parent_kwargs):
        super().__init__(**parent_kwargs)

        self.max_queried_objs_of_class = max_queried_objs_of_class
        self.min_num_distractor_scenes = min_num_distractor_scenes
        self.max_num_distractor_scenes = max_num_distractor_scenes
        self.min_frames_per_scene = min_frames_per_scene
        self.max_frames_per_scene = max_frames_per_scene
        self.enforced_len = enforced_len
        self.num_distractors = num_distractors
        
        self.scene_formations_from_imgs = []
        self.all_obj_classes = set()
        self.obj_class_to_img_and_quantity = {}
        self.corresponding_gt_class_quantity = []

        self.mcq_formatter = MCQuestionFormatter(
            "How many instances that fit the description \"{0}\" are present "
            "in the video?", **mc_formatter_kwargs)
        self.free_formatter = FreeFormQuestionFormatter(
            "How many instances that fit the description \"{0}\" are present "
            "in the video?")

    def _prepare_object_info(self):

        for img_ind, scene_obj_info in (
            self.dataset.img_ind_to_obj_classes_quants.items()):
            for class_present_in_scene, num_objs in scene_obj_info:
                self.all_obj_classes.add(class_present_in_scene)
                if(class_present_in_scene not in 
                   self.obj_class_to_img_and_quantity):
                    self.obj_class_to_img_and_quantity[
                        class_present_in_scene] = []
                self.obj_class_to_img_and_quantity[
                    class_present_in_scene].append((img_ind, num_objs))
                
        self.all_annotated_inds = set(
            self.dataset.img_ind_to_obj_classes_quants.keys())
        self.all_obj_classes = sorted(list(self.all_obj_classes))

    def _randomly_choose_scenes(self):

        shuffle_every_n = ceil(len(self.all_annotated_inds)/10)
        for i in range(self.enforced_len):
            print(i)
            if(i % shuffle_every_n == 0):
                candidate_inds = np.array(list(self.all_annotated_inds))
                candidate_inds.sort()
                np.random.shuffle(candidate_inds)
            chosen_class = random.choice(self.all_obj_classes)
            num_appearances = random.randint(0, self.max_queried_objs_of_class)
            num_distractor_scenes = random.randint(
                self.min_num_distractor_scenes, self.max_num_distractor_scenes)

            relevant_imgs_quants = self.obj_class_to_img_and_quantity[
                chosen_class]
            relevant_img_inds = [_[0] for _ in relevant_imgs_quants]
            img_inds_as_set = set(relevant_img_inds)
            img_ind_to_num_objs = dict(relevant_imgs_quants)
            distractor_scene_inds = []
            starting_ind = random.randint(
                0, len(candidate_inds) - num_distractor_scenes)
            for candidate_distr_ind in candidate_inds[starting_ind:]:
                if(len(distractor_scene_inds) >= num_distractor_scenes):
                    break
                if(candidate_distr_ind not in img_inds_as_set):
                    distractor_scene_inds.append(candidate_distr_ind)
            
            chosen_img_inds = []
            if(num_appearances != 0):
                actual_appearances = 0
                num_retries = 0
                shuffled_inds = relevant_img_inds.copy()
                random.shuffle(shuffled_inds)
                while(actual_appearances < num_appearances and 
                      len(chosen_img_inds) != len(relevant_img_inds)):
                    img_ind = shuffled_inds.pop()
                    num_objs = img_ind_to_num_objs[img_ind]
                    if(actual_appearances + num_objs> num_appearances + 1
                       and num_retries < 10):
                        shuffled_inds = relevant_img_inds.copy()
                        random.shuffle(shuffled_inds)                        
                        num_retries += 1
                        continue
                    chosen_img_inds.append(img_ind)
                    actual_appearances += num_objs
                #assert actual_appearances >= num_appearances
            else:
                actual_appearances = 0
            all_scenes = chosen_img_inds + distractor_scene_inds
            random.shuffle(all_scenes)

            scene_lengths = random.choices(
                range(self.min_frames_per_scene, 
                      self.max_frames_per_scene + 1), k=len(all_scenes))
            
            self.scene_formations_from_imgs.append(
                list(zip(all_scenes, scene_lengths)))
            self.corresponding_gt_class_quantity.append((
                chosen_class, actual_appearances))
            
    def setup_on_dataset(self, dataset: ImgSplicingDatasetWithObjInfo):
        self.dataset = dataset
        self._prepare_object_info()
        self._randomly_choose_scenes()

    def get_length(self):
        return self.enforced_len
    
    def form_free_form_question(self, question_index: int) -> Question:
        q_formation_imgs = self.scene_formations_from_imgs[question_index] 
        gt_class, gt_num = self.corresponding_gt_class_quantity[question_index]
        
        q = self.free_formatter.format_question([gt_class])
        imgs_repetitions = [
            (self.dataset.img_paths[_[0]], _[1]) for _ in q_formation_imgs]
        return Question(imgs_repetitions, q, str(gt_num), [])

    def form_mc_question(self, question_index):
        q_formation_imgs = self.scene_formations_from_imgs[question_index] 
        gt_class, gt_num = self.corresponding_gt_class_quantity[question_index]
        distractors = list(range(0, gt_num)) + list(range(gt_num + 1, max(
            gt_num, self.max_queried_objs_of_class) + 1))
        
        q, ans, candidates = self.mcq_formatter.format_to_question_answer(
            [gt_class], str(gt_num), [str(_) for _ in distractors], 
            self.num_distractors)
        imgs_repetitions = [
            (self.dataset.img_paths[_[0]], _[1]) for _ in q_formation_imgs]
        return Question(imgs_repetitions, q, ans, candidates)        
