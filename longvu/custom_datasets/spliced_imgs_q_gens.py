from typing import List, Tuple, Any, Dict
import random
from math import factorial
import numpy as np

from .qa_dataset import Question, MCAndFreeFormQGen
from .question_formatting import MCQuestionFormatter, FreeFormQuestionFormatter
from .img_splicing_dataset import ImgSplicingDataset


class SplicedScenesQGen(MCAndFreeFormQGen):
    scene_formations_from_imgs: List[List[Tuple[int, int]]]
    
    def __init__(self, min_scenes_per_question: int, 
                 max_scenes_per_question: int, min_frames_per_scene: int,
                 max_frames_per_scene: int, enforced_len: int,
                 **parent_kwargs):
        super().__init__(**parent_kwargs)
        
        self.min_scenes_per_question = min_scenes_per_question
        self.max_scenes_per_question = max_scenes_per_question
        self.min_frames_per_scene = min_frames_per_scene
        self.max_frames_per_scene = max_frames_per_scene
        self.enforced_len = enforced_len

        self.scene_formations_from_imgs = []

    def _randomly_choose_scenes(self):
        all_inds = range(len(self.dataset.img_paths))
        for i in range(self.enforced_len):
            num_scenes_for_question = random.randint(
                self.min_scenes_per_question, self.max_scenes_per_question)
            scene_lengths = random.choices(
                range(self.min_frames_per_scene, 
                      self.max_frames_per_scene + 1), 
                k=num_scenes_for_question)
            img_indices = random.sample(all_inds, k=num_scenes_for_question)
            self.scene_formations_from_imgs.append(
                list(zip(img_indices, scene_lengths)))
            
            assert (len(self.scene_formations_from_imgs[-1]) 
                    == num_scenes_for_question)

    def setup_on_dataset(self, dataset: ImgSplicingDataset):
        self.dataset = dataset
        self._randomly_choose_scenes()
 
    def get_length(self) -> int:
        return self.enforced_len


class ScenesDescrQGen(SplicedScenesQGen):

    def __init__(self, num_distractors: int, 
                 mc_formatter_kwargs: Dict[str, Any]={}, 
                 use_hard_distractors: bool=False, **parent_kwargs):
        super().__init__(**parent_kwargs)

        self.num_distractors = num_distractors

        self.mcq_formatter = MCQuestionFormatter(
            "Which of the following options best describes the order of "
            "scenes in the video?", **mc_formatter_kwargs)
        self.free_formatter = FreeFormQuestionFormatter(
            "Describe the scenes of the video in the order they appear, i.e., "
            "\"Scene 1: [a description of scene 1]. "
            "Scene 2: [a description of scene 2] ...\"")
        self.use_hard_distractors = use_hard_distractors
    
    def _join_caption_list(self, caption_list: List[str]) -> str:
        formatted = "".join(["Scene {0}: ".format(i + 1) + _ + ". "
                        for i, _ in enumerate(caption_list)])
        return formatted[:-1]
    
    def form_free_form_question(self, question_index: int) -> Question:
        q_formation_imgs = self.scene_formations_from_imgs[question_index]

        captions = [random.choice(
            self.dataset.img_captions[_[0]]) for _ in q_formation_imgs]

        gt_ans = self._join_caption_list(captions)
        q = self.free_formatter.format_question([])

        imgs_repetitions = [
            (self.dataset.img_paths[_[0]], _[1]) for _ in q_formation_imgs]
        return Question(imgs_repetitions, q, gt_ans, None)
    
    def _sample_easy_distractor(self, captions: List[str]) -> str:
        return self._join_caption_list(random.sample(
            captions, k=len(captions)))
    
    def _sample_hard_distractor(self, captions: List[str]) -> str:
        copied_caps = captions.copy()
        indices_to_flip = random.sample(range(len(captions)), 2)
        tmp = copied_caps[indices_to_flip[0]]
        copied_caps[indices_to_flip[0]] = copied_caps[indices_to_flip[1]]
        copied_caps[indices_to_flip[1]] = tmp
        return self._join_caption_list(copied_caps)

    def form_mc_question(self, question_index: int) -> Question:
        q_formation_imgs = self.scene_formations_from_imgs[question_index]

        # captions = [random.choice(
        #     self.dataset.img_captions[_[0]]) for _ in q_formation_imgs]
        captions = []
        for _ in q_formation_imgs:
            possible_caps = self.dataset.img_captions[_[0]].copy()
            random.shuffle(possible_caps)
            i = 0
            while(possible_caps[i] in captions):
                i += 1
            captions.append(possible_caps[i])
        assert len(captions) == len(set(captions))

        gt_ans = self._join_caption_list(captions)
        already_included = [gt_ans]
        distractors = []
        if(not self.use_hard_distractors):
            num_possible_distractors = factorial(len(captions))
        else:
            num_possible_distractors = int(len(captions)*(len(captions) - 1)/2)
        for _ in range(self.num_distractors):
            if(len(already_included) == num_possible_distractors):
                break
            if(not self.use_hard_distractors):
                candidate = self._sample_easy_distractor(captions)
                while(candidate in already_included):
                    candidate = self._sample_easy_distractor(captions)
            else:
                candidate = self._sample_hard_distractor(captions)
                while(candidate in already_included):
                    candidate = self._sample_hard_distractor(captions)

            already_included.append(candidate)
            distractors.append(candidate)
        
        q, ans, candidates = self.mcq_formatter.format_to_question_answer(
            [], gt_ans, distractors, self.num_distractors)
        imgs_repetitions = [
            (self.dataset.img_paths[_[0]], _[1]) for _ in q_formation_imgs]
        return Question(imgs_repetitions, q, ans, candidates)
    
class DescribeNthSceneQGen(SplicedScenesQGen):

    def __init__(self, num_distractors: int, 
                 mc_formatter_kwargs: Dict[str, Any]={}, **parent_kwargs):
        super().__init__(**parent_kwargs)

        self.num_distractors = num_distractors

        self.mcq_formatter = MCQuestionFormatter(
            "There are {0} scenes in the video. What does scene {1} depict?",
            **mc_formatter_kwargs)
        self.free_formatter = FreeFormQuestionFormatter(
            "There are {0} scenes in the video. What does scene {1} depict?")
        
    def form_free_form_question(self, question_index):
        q_formation_imgs = self.scene_formations_from_imgs[question_index]
        num_scenes = len(q_formation_imgs)
        chosen_scene_ind = random.randint(0, num_scenes - 1)
        chosen_scene = q_formation_imgs[chosen_scene_ind][0]

        gt_caption = random.choice(self.dataset.img_captions[chosen_scene])
        gt_caption = gt_caption.capitalize() + "."

        q = self.free_formatter.format_question(
            [num_scenes, chosen_scene_ind + 1])
        imgs_repetitions = [
            (self.dataset.img_paths[_[0]], _[1]) for _ in q_formation_imgs]
        return Question(imgs_repetitions, q, gt_caption, None)        
        
    def form_mc_question(self, question_index: int) -> Question:
        q_formation_imgs = self.scene_formations_from_imgs[question_index]

        num_scenes = len(q_formation_imgs)
        chosen_scene_ind = random.randint(0, num_scenes - 1)
        other_scenes = [_[0] for i, _ in enumerate(q_formation_imgs) 
                        if i != chosen_scene_ind]
        chosen_scene = q_formation_imgs[chosen_scene_ind][0]

        gt_caption = random.choice(self.dataset.img_captions[chosen_scene])
        distractors = []
        all_included_caps = set([gt_caption])
        for s in other_scenes:
            caps = sorted(
                list(set(self.dataset.img_captions[s]).difference(
                    all_included_caps)))
            if(len(caps) > 0):
                distractors.append(random.choice(caps))
                all_included_caps.add(distractors[-1])
        if(len(distractors) < self.num_distractors):
            video_inds = set(other_scenes + [chosen_scene])
            all_inds_shuffled = np.arange(len(self.dataset.img_captions))
            np.random.shuffle(all_inds_shuffled)
            remaining = self.num_distractors - len(distractors)
            non_video_caps = []
            for candidate_ind in all_inds_shuffled:
                if(candidate_ind in video_inds):
                    continue
                if(len(non_video_caps) >= remaining):
                    break
                not_incl = [
                    _  for _ in self.dataset.img_captions[candidate_ind]
                    if _ not in all_included_caps]
                if(len(not_incl) > 0):
                    non_video_caps.append(random.choice(not_incl))
                    all_included_caps.add(non_video_caps[-1])
            distractors += non_video_caps
        
        assert len(distractors) >= self.num_distractors

        q, ans, candidates = self.mcq_formatter.format_to_question_answer(
            [num_scenes, chosen_scene_ind + 1], gt_caption, 
            distractors, self.num_distractors)
        imgs_repetitions = [
            (self.dataset.img_paths[_[0]], _[1]) for _ in q_formation_imgs]
        return Question(imgs_repetitions, q, ans, candidates)
    
class CountScenesQGen(SplicedScenesQGen):

    def __init__(self, num_distractors: int, 
                 mc_formatter_kwargs: Dict[str, Any]={}, **parent_kwargs):
        super().__init__(**parent_kwargs)

        self.num_distractors = num_distractors

        self.mcq_formatter = MCQuestionFormatter(
            "How many different scenes appear in the video?", 
            **mc_formatter_kwargs)
        self.free_formatter = FreeFormQuestionFormatter(
            "How many different scenes appear in the video?")
   
    def form_free_form_question(self, question_index: int) -> Question:
        q_formation_imgs = self.scene_formations_from_imgs[question_index]

        num_scenes = len(q_formation_imgs)
        q = self.free_formatter.format_question([])

        imgs_repetitions = [
            (self.dataset.img_paths[_[0]], _[1]) for _ in q_formation_imgs]
        return Question(imgs_repetitions, q, str(num_scenes), None)                

    def form_mc_question(self, question_index: int) -> Question:
        q_formation_imgs = self.scene_formations_from_imgs[question_index]

        num_scenes = len(q_formation_imgs)
        gt_ans = str(num_scenes)
        distractors = list(range(1, num_scenes + 4))
        distractors.pop(num_scenes - 1)
        
        q, ans, candidates = self.mcq_formatter.format_to_question_answer(
            [], gt_ans, distractors, self.num_distractors)
        imgs_repetitions = [
            (self.dataset.img_paths[_[0]], _[1]) for _ in q_formation_imgs]
        return Question(imgs_repetitions, q, ans, candidates)
    
class DescrAdjacentSceneQGen(SplicedScenesQGen):

    def __init__(self, num_distractors: int, 
                 mc_formatter_kwargs: Dict[str, Any]={}, **parent_kwargs):
        super().__init__(**parent_kwargs)

        self.num_distractors = num_distractors

        self.mcq_formatter = MCQuestionFormatter(
            "One of the scenes in the video can be described as \"{0}\". What "
            "description best fits the scene immediately {1} it?",
            **mc_formatter_kwargs)
        
        self.free_formatter = FreeFormQuestionFormatter(
            "One of the scenes in the video can be described as \"{0}\". "
            "Describe the scene immediately {1} it, if such a scene exists.")
    
    def form_free_form_question(self, question_index: int) -> Question:
        q_formation_imgs = self.scene_formations_from_imgs[question_index]

        reference_scene_ind = random.randint(0, len(q_formation_imgs) - 1)
        ref_caption = random.choice(self.dataset.img_captions[
            q_formation_imgs[reference_scene_ind][0]])
        
        ask_for_scene_ind = random.choice(
            [reference_scene_ind - 1, reference_scene_ind + 1])
        first_sc_cap = "This is the first scene; there is no scene before it"
        last_sc_cap = "This is the last scene; there is no scene after it"
        if(ask_for_scene_ind < 0):
            gt_caption = first_sc_cap
        elif(ask_for_scene_ind >= len(q_formation_imgs)):
            gt_caption = last_sc_cap
        else:
            gt_caption = random.choice(self.dataset.img_captions[
                q_formation_imgs[ask_for_scene_ind][0]])
        gt_caption = gt_caption.capitalize() + "."
            
        temporal_relation = (
            "before" if ask_for_scene_ind < reference_scene_ind else "after")
        q = self.free_formatter.format_question(
            [ref_caption, temporal_relation])
        imgs_repetitions = [
            (self.dataset.img_paths[_[0]], _[1]) for _ in q_formation_imgs]
        return Question(imgs_repetitions, q, gt_caption, [])

    def form_mc_question(self, question_index: int) -> Question:
        q_formation_imgs = self.scene_formations_from_imgs[question_index]

        reference_scene_ind = random.randint(0, len(q_formation_imgs) - 1)
        ref_caption = random.choice(self.dataset.img_captions[
            q_formation_imgs[reference_scene_ind][0]])
        
        ask_for_scene_ind = random.choice(
            [reference_scene_ind - 1, reference_scene_ind + 1])
        first_sc_cap = "This is the first scene; there is no scene before it."
        last_sc_cap = "This is the last scene; there is no scene after it."
        other_scenes = [_[0] for i, _ in enumerate(q_formation_imgs)
                        if i != reference_scene_ind and i != ask_for_scene_ind]  
        video_inds = set(
            other_scenes + [q_formation_imgs[reference_scene_ind][0]])
        if(ask_for_scene_ind < 0):
            gt_caption = first_sc_cap
        elif(ask_for_scene_ind >= len(q_formation_imgs)):
            gt_caption = last_sc_cap
        else:
            video_inds.add(q_formation_imgs[ask_for_scene_ind][0])
            candidates = self.dataset.img_captions[q_formation_imgs[
                ask_for_scene_ind][0]]
            if(ref_caption in candidates):
                candidates.remove(ref_caption)
            gt_caption = random.choice(candidates)

        distractors = []
        if(gt_caption != first_sc_cap):
            distractors.append(first_sc_cap)
        if(gt_caption != last_sc_cap):  
            distractors.append(last_sc_cap)

        distractors = []
        all_included_caps = set([gt_caption, ref_caption])
        for s in other_scenes:
            caps = sorted(
                list(set(self.dataset.img_captions[s]).difference(
                    all_included_caps)))
            if(len(caps) > 0):
                distractors.append(random.choice(caps))
                all_included_caps.add(distractors[-1])
        if(len(distractors) < self.num_distractors):
            all_inds_shuffled = np.arange(len(self.dataset.img_captions))
            np.random.shuffle(all_inds_shuffled)
            remaining = self.num_distractors - len(distractors)
            non_video_caps = []
            for candidate_ind in all_inds_shuffled:
                if(candidate_ind in video_inds):
                    continue
                if(len(non_video_caps) >= remaining):
                    break
                not_incl = [
                    _  for _ in self.dataset.img_captions[candidate_ind]
                    if _ not in all_included_caps]
                if(len(not_incl) > 0):
                    non_video_caps.append(random.choice(not_incl))
                    all_included_caps.add(non_video_caps[-1])
            distractors += non_video_caps

        temporal_relation = (
            "before" if ask_for_scene_ind < reference_scene_ind else "after")
        q, ans, candidates = self.mcq_formatter.format_to_question_answer(
            [ref_caption, temporal_relation], gt_caption, 
            distractors, self.num_distractors)
        imgs_repetitions = [
            (self.dataset.img_paths[_[0]], _[1]) for _ in q_formation_imgs]
        return Question(imgs_repetitions, q, ans, candidates)

class FrLvlAnnotationQGen(SplicedScenesQGen):

    def __init__(self, num_distractors: int, 
                 mc_formatter_kwargs: Dict[str, Any]={}, **parent_kwargs):
        super().__init__(**parent_kwargs)

        self.num_distractors = num_distractors

        self.mcq_formatter = MCQuestionFormatter(
            "Which of the following options is the most accurate frame-level "
            "description of the video?", **mc_formatter_kwargs)
        self.free_formatter = FreeFormQuestionFormatter(
            "Provide a frame-level description of the video. The description "
            "should be in the format: <frame: 1-N_1> [scene 1 description] "
            "<frame: N_1+1:N_2> [scene 2 description] ...")
    
    def _join_caption_list(self, caption_list: List[str]) -> str:
        formatted = "".join(["Scene {0}: ".format(i + 1) + _ + ". "
                        for i, _ in enumerate(caption_list)])
        return formatted[:-1]
    
    def form_free_form_question(self, question_index: int) -> Question:
        q_formation_imgs = self.scene_formations_from_imgs[question_index]

        captions = [random.choice(
            self.dataset.img_captions[_[0]]) for _ in q_formation_imgs]
        frame_intervals = [1]
        for _ in q_formation_imgs:
            frame_intervals[-1] = (
                frame_intervals[-1], frame_intervals[-1] + _[1] - 1)
            frame_intervals.append(frame_intervals[-1][1] + 1)
        frame_intervals.pop()

        gt_ans = self._join_captions_and_frames(captions, frame_intervals)
        q = self.free_formatter.format_question([])

        imgs_repetitions = [
            (self.dataset.img_paths[_[0]], _[1]) for _ in q_formation_imgs]
        return Question(imgs_repetitions, q, gt_ans, None)
    
    def _join_captions_and_frames(self, caption_list: List[str],
                                  frames_list: List[Tuple[int, int]]) -> str:
        formatted = "".join(
            "<frame: {0}-{1}> {2} ".format(start, end, cap)
            for (start, end), cap in zip(frames_list, caption_list))
        return formatted[:-1]
    
    def _sample_distractor(
            self, captions: List[str], possible_lens: List[int],
            num_intervals: int) -> str:
        
        sampled_lens = random.choices(possible_lens, k=num_intervals)
        sampled_intervals = [1]
        for _ in sampled_lens:
            sampled_intervals[-1] = (
                sampled_intervals[-1], sampled_intervals[-1] + _ - 1)
            sampled_intervals.append(sampled_intervals[-1][1] + 1)
        shuffled_caps = random.sample(captions, k=len(captions))
        distr = self._join_captions_and_frames(
            shuffled_caps, sampled_intervals)
        return distr
        

    def form_mc_question(self, question_index: int) -> Question:
        q_formation_imgs = self.scene_formations_from_imgs[question_index]

        captions = [random.choice(
            self.dataset.img_captions[_[0]]) for _ in q_formation_imgs]
        frame_intervals = [1]
        for _ in q_formation_imgs:
            frame_intervals[-1] = (
                frame_intervals[-1], frame_intervals[-1] + _[1] - 1)
            frame_intervals.append(frame_intervals[-1][1] + 1)
        frame_intervals.pop()
        last_fr = frame_intervals[-1][1]

        gt_ans = self._join_captions_and_frames(captions, frame_intervals)
        already_included = [gt_ans]
        distractors = []
        possible_lens = list(range(1, max(
            self.num_distractors + 2, int(1.5*last_fr/len(frame_intervals)))))
        for _ in range(self.num_distractors):
            candidate = self._sample_distractor(captions, possible_lens,
                                                len(frame_intervals))
            while(candidate in already_included):
                candidate = self._sample_distractor(captions, possible_lens,
                                                    len(frame_intervals))
            already_included.append(candidate)
            distractors.append(candidate)
        
        q, ans, candidates = self.mcq_formatter.format_to_question_answer(
            [], gt_ans, distractors, self.num_distractors)
        imgs_repetitions = [
            (self.dataset.img_paths[_[0]], _[1]) for _ in q_formation_imgs]
        return Question(imgs_repetitions, q, ans, candidates)

class BeforeOrAfterQGen(SplicedScenesQGen):

    def __init__(self, min_scenes_per_question: int,
                 mc_formatter_kwargs: Dict[str, Any]={}, **parent_kwargs):
        if(min_scenes_per_question < 2):
            raise ValueError("At least two scenes must be used for each "
                             "question")
        super().__init__(min_scenes_per_question=min_scenes_per_question, 
                         **parent_kwargs)

        self.mcq_formatter = MCQuestionFormatter(
            "In the given video, does the scene that can be captioned as "
            "\"{0}\" happen before or after the scene that can be captioned "
            "as \"{1}\"?", **mc_formatter_kwargs)
        self.free_formatter = FreeFormQuestionFormatter(
            "In the given video, does the scene that can be captioned as "
            "\"{0}\" happen before or after the scene that can be captioned "
            "as \"{1}\"?")

    def form_free_form_question(self, question_index: int) -> Question:
        q_formation_imgs = self.scene_formations_from_imgs[question_index]

        reference_scene_ind = random.randint(0, len(q_formation_imgs) - 1)
        if(reference_scene_ind == 0):
            temporal_relation = "after"
        elif(reference_scene_ind == len(q_formation_imgs) - 1):
            temporal_relation = "before"
        else:
            temporal_relation = random.choice(["before", "after"])

        if(temporal_relation == "after"):
            requested_scene_ind = reference_scene_ind + 1
        else:
            requested_scene_ind = reference_scene_ind - 1
        
        ref_caption = random.choice(self.dataset.img_captions[
            q_formation_imgs[reference_scene_ind][0]])
        requested_cap = random.choice(self.dataset.img_captions[
            q_formation_imgs[requested_scene_ind][0]])

        q = self.free_formatter.format_question(
            [requested_cap, ref_caption])
        imgs_repetitions = [
            (self.dataset.img_paths[_[0]], _[1]) for _ in q_formation_imgs]
        return Question(
            imgs_repetitions, q, temporal_relation.capitalize() + ".", None)

    def form_mc_question(self, question_index: int) -> Question:
        q_formation_imgs = self.scene_formations_from_imgs[question_index]

        reference_scene_ind = random.randint(0, len(q_formation_imgs) - 1)
        if(reference_scene_ind == 0):
            temporal_relation = "after"
        elif(reference_scene_ind == len(q_formation_imgs) - 1):
            temporal_relation = "before"
        else:
            temporal_relation = random.choice(["before", "after"])

        if(temporal_relation == "after"):
            requested_scene_ind = random.randint(reference_scene_ind + 1,
                                                len(q_formation_imgs) - 1)
        else:
            requested_scene_ind = random.randint(0, reference_scene_ind - 1)
        
        ref_caption = random.choice(self.dataset.img_captions[
            q_formation_imgs[reference_scene_ind][0]])
        requested_cap = random.choice(self.dataset.img_captions[
            q_formation_imgs[requested_scene_ind][0]])
        
        q, ans, candidates = self.mcq_formatter.format_to_question_answer(
            [requested_cap, ref_caption], temporal_relation, 
            ["after" if temporal_relation == "before" else "before"])
        imgs_repetitions = [
            (self.dataset.img_paths[_[0]], _[1]) for _ in q_formation_imgs]
        return Question(imgs_repetitions, q, ans, candidates)
    
class WhichHappensFirstLastQGen(SplicedScenesQGen):

    def __init__(self, min_scenes_per_question: int, num_distractors: int,
                 min_choices: int, max_choices: int,
                 mc_formatter_kwargs: Dict[str, Any]={}, **parent_kwargs):
        if(min_scenes_per_question < 2):
            raise ValueError("At least two scenes must be used for each "
                             "question")
        super().__init__(min_scenes_per_question=min_scenes_per_question, 
                         **parent_kwargs)

        self.mcq_formatter = MCQuestionFormatter(
            "The following scenes appear in the video, not necessarily in "
            "this order: {0}. Of those scenes, which occurs {1}?", 
            **mc_formatter_kwargs)
        self.free_formatter = FreeFormQuestionFormatter(
            "The following scenes appear in the video, not necessarily in "
            "this order: {0}. Of those scenes, which occurs {1}?")
        self.num_distractors = num_distractors
        self.min_choices = min_choices
        self.max_choices = max_choices

    def _join_caption_list(self, captions: List[str]) -> str:
        return ", ".join(["\"" + _ + "\"" for _ in captions])
    
    def form_free_form_question(self, question_index: int) -> Question:
        q_formation_imgs = self.scene_formations_from_imgs[question_index]

        selected_inds = random.sample(range(len(q_formation_imgs)), k=min(
            random.randint(self.min_choices, self.max_choices),
            len(q_formation_imgs)))
        
        mode = random.choice(["first", "last"])
        if(mode == "first"):
            gt_ind = min(selected_inds)
        else:
            gt_ind = max(selected_inds)

        captions = []
        for _ in q_formation_imgs:
            possible_caps = self.dataset.img_captions[_[0]].copy()
            random.shuffle(possible_caps)
            i = 0
            while(possible_caps[i] in captions):
                i += 1
            captions.append(possible_caps[i])
        assert len(captions) == len(set(captions))

        selected_caps = [captions[_] for _ in selected_inds]
        gt_cap = selected_caps[gt_ind]
        joined_list = self._join_caption_list(selected_caps)
        
        q = self.free_formatter.format_question([joined_list, mode])
        imgs_repetitions = [
            (self.dataset.img_paths[_[0]], _[1]) for _ in q_formation_imgs]
        return Question(imgs_repetitions, q, gt_cap.capitalize() + ".", None)            

    def form_mc_question(self, question_index: int) -> Question:
        q_formation_imgs = self.scene_formations_from_imgs[question_index]

        selected_inds = random.sample(
            range(len(q_formation_imgs)), k=min(random.randint(
                self.min_choices, self.max_choices), len(q_formation_imgs)))
        
        mode = random.choice(["first", "last"])
        if(mode == "first"):
            gt_ind = min(selected_inds)
        else:
            gt_ind = max(selected_inds)

        captions = []
        for _ in q_formation_imgs:
            possible_caps = self.dataset.img_captions[_[0]].copy()
            random.shuffle(possible_caps)
            i = 0
            while(possible_caps[i] in captions):
                i += 1
            captions.append(possible_caps[i])
        assert len(captions) == len(set(captions))

        selected_caps = [captions[_] for _ in selected_inds]
        distractors = selected_caps.copy()
        gt_cap = captions[gt_ind]
        distractors.remove(gt_cap)
        joined_list = self._join_caption_list(selected_caps)
        
        q, ans, candidates = self.mcq_formatter.format_to_question_answer(
            [joined_list, mode], gt_cap, distractors, self.num_distractors)
        imgs_repetitions = [
            (self.dataset.img_paths[_[0]], _[1]) for _ in q_formation_imgs]
        return Question(imgs_repetitions, q, ans, candidates)

class WhichIsOrIsntInVideoQGen(SplicedScenesQGen):

    def __init__(self, num_distractors: int, 
                 mc_formatter_kwargs: Dict[str, Any]={}, **parent_kwargs):
        super().__init__(**parent_kwargs)

        self.mcq_formatter = MCQuestionFormatter(
            "Exactly one of the following scenes {0} in the video: {1}. "
            "Indicate which.", **mc_formatter_kwargs)
        self.free_formatter = FreeFormQuestionFormatter(
            "Exactly one of the following scenes {0} in the video: {1}. "
            "Indicate which.")
        self.num_distractors = num_distractors

    def _join_caption_list(self, captions: List[str]) -> str:
        return ", ".join(["\"" + _ + "\"" for _ in captions])
    
    def form_free_form_question(self, question_index: int) -> Question:
        q_formation_imgs = self.scene_formations_from_imgs[question_index]

        captions = []
        for _ in q_formation_imgs:
            possible_caps = self.dataset.img_captions[_[0]].copy()
            random.shuffle(possible_caps)
            i = 0
            while(possible_caps[i] in captions):
                i += 1
            captions.append(possible_caps[i])
        assert len(captions) == len(set(captions))
        random.shuffle(captions)

        mode = random.choice(["occurs", "doesn't occur"])
        video_inds = [_[0] for _ in q_formation_imgs]        
        if(mode == "occurs"):
            gt_cap = random.choice(captions)
            included_captions = [gt_cap]
            chosen_inds = video_inds
            for _ in range(self.num_distractors):
                i = random.randint(0, len(self.dataset.img_paths) - 1)
                while(i in chosen_inds):
                    i = random.randint(0, len(self.dataset.img_paths) - 1)
                chosen_inds.append(i)
                possible_caps = self.dataset.img_captions[i].copy()
                random.shuffle(possible_caps)
                cap_added = False
                for c in possible_caps:
                    if(c not in included_captions):
                        cap_added = True
                        included_captions.append(c)
                        break
                assert (cap_added)
        else:
            i = random.randint(0, len(self.dataset.img_paths) - 1)
            while(i in video_inds):
                i = random.randint(0, len(self.dataset.img_paths) - 1)
            possible_caps = self.dataset.img_captions[i].copy()
            random.shuffle(possible_caps)
            cap_added = False
            included_captions = []
            gt_cap = None
            for c in possible_caps:
                if(c not in included_captions):
                    gt_cap = c
                    included_captions.append(c)
                    break
            assert (gt_cap is not None)
            for _ in range(min(self.num_distractors), len(video_inds)):
                included_captions.append(captions[_])
        joined_list = self._join_caption_list(included_captions)

        imgs_repetitions = [
            (self.dataset.img_paths[_[0]], _[1]) for _ in q_formation_imgs]
        q = self.free_formatter.format_question([mode, joined_list])
        return Question(imgs_repetitions, q, gt_cap.capitalize() + ".", None)

    def form_mc_question(self, question_index: int) -> Question:
        q_formation_imgs = self.scene_formations_from_imgs[question_index]

        captions = []
        for _ in q_formation_imgs:
            possible_caps = self.dataset.img_captions[_[0]].copy()
            random.shuffle(possible_caps)
            i = 0
            while(possible_caps[i] in captions):
                i += 1
            captions.append(possible_caps[i])
        assert len(captions) == len(set(captions))
        random.shuffle(captions)

        mode = random.choice(["occurs", "doesn't occur"])
        video_inds = [_[0] for _ in q_formation_imgs]        
        if(mode == "occurs"):
            gt_caption = random.choice(captions)
            included_captions = [gt_caption]
            chosen_inds = video_inds
            for _ in range(self.num_distractors):
                i = random.randint(0, len(self.dataset.img_paths) - 1)
                while(i in chosen_inds):
                    i = random.randint(0, len(self.dataset.img_paths) - 1)
                chosen_inds.append(i)
                possible_caps = self.dataset.img_captions[i].copy()
                random.shuffle(possible_caps)
                cap_added = False
                for c in possible_caps:
                    if(c not in included_captions):
                        cap_added = True
                        included_captions.append(c)
                        break
                assert (cap_added)
        else:
            i = random.randint(0, len(self.dataset.img_paths) - 1)
            while(i in video_inds):
                i = random.randint(0, len(self.dataset.img_paths) - 1)
            possible_caps = self.dataset.img_captions[i].copy()
            random.shuffle(possible_caps)
            cap_added = False
            included_captions = []
            gt_caption = None
            for c in possible_caps:
                if(c not in included_captions):
                    gt_caption = c
                    included_captions.append(c)
                    break
            assert (gt_caption is not None)
            for _ in range(min((self.num_distractors), len(video_inds))):
                included_captions.append(captions[_])
        joined_list = self._join_caption_list(included_captions)
        distractors = included_captions.copy()
        distractors.remove(gt_caption)

        imgs_repetitions = [
            (self.dataset.img_paths[_[0]], _[1]) for _ in q_formation_imgs]
        q, ans, candidates = self.mcq_formatter.format_to_question_answer(
            [mode, joined_list], gt_caption, distractors, self.num_distractors)        
        return Question(imgs_repetitions, q, ans, candidates)
