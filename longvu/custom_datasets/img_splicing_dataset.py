from __future__ import annotations
from abc import abstractmethod
from typing import List, Any, Tuple, Optional, Dict, Type, TypeVar
from os import PathLike
import os
import json
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from .qa_dataset import QADataset
from .mask_utils import (read_BGR_image,
                         convert_png_mask_to_id_map, 
                         convert_id_map_to_binary_masks)

T = TypeVar('T', bound='ImgSplicingDatasetWithMasks')

class ImgSplicingDataset(QADataset):
    img_captions: List[List[str]]
    img_paths: List[PathLike]

    def __init__(self, img_paths: List[PathLike], 
                 img_captions: List[List[str]]):
        super().__init__(False)
        self.img_paths = img_paths
        #sanitize captions 
        self.img_captions = []
        seen_captions = {}
        for img_path, caption_list in zip(img_paths, img_captions):
            self.img_captions.append([])
            for c in caption_list:
                processed = c.rstrip().lstrip().replace("\n", " ")
                if(processed[-1] == "."):
                    processed = processed[:-1]
                if(processed[0].isupper() and processed[1].islower()):
                    processed = processed[0].lower() + processed[1:]
                self.img_captions[-1].append(processed)
                
            as_tuple = tuple(caption_list)
            if(as_tuple in seen_captions):
                raise Exception("Two images have identical caption lists: " + 
                                img_path, seen_captions[as_tuple])
            seen_captions[as_tuple] = img_path
         

    def __getitem__(self, i) -> Tuple[List[Tuple[PathLike, int]], str, str, 
                                      Optional[List[str]]]:
        return super().__getitem__(i)
    
class ImgSplicingDatasetWithObjInfo(ImgSplicingDataset):

    def __init__(self, img_paths: List[PathLike], 
                 img_captions: List[List[str]], 
                 img_ind_to_obj_classes_quants: Dict[
                     int, List[Tuple[str, int]]]):
        super().__init__(img_paths, img_captions)
        self.img_ind_to_obj_classes_quants = img_ind_to_obj_classes_quants

@dataclass
class ImageMask:
    bool_mask: NDArray
    mask_class: str
    bbox_upper_left: Tuple[int, int]
    bbox_h_w: Tuple[int, int]

@dataclass
class SampleWithNamedType:
    data_type: str
    sample_data: Any

class ImgSplicingDatasetWithMasks(ImgSplicingDatasetWithObjInfo):
    data_type_per_qgen: Dict[int, str]

    def __init__(self, *parent_args, **parent_kwargs):
        super().__init__(*parent_args, **parent_kwargs)
        self.data_type_per_qgen = {}
        
    @abstractmethod
    def get_masks_for_sample(self, i: int) -> List[ImageMask]:
        pass

    def set_data_type_for_qgen(self, qgen_idx: int, data_type: str):
        self.data_type_per_qgen[qgen_idx] = data_type    

    @classmethod
    def from_config_KVs(
        cls: Type[ImgSplicingDatasetWithMasks], 
        q_generators_args: Dict[str, Dict[str, Any]],
        data_type_per_generator: Dict[str, str],        
        sampling_prob_per_generator: Optional[Dict[str, float]]=None,
        **constructor_kwargs) -> QADataset:
        unpacked = [
            (gen, gen_args, data_type_per_generator[gen]) 
            for gen, gen_args in q_generators_args.items()]
        q_gens = [cls.known_question_generators[gen](**gen_args)
                  for gen, gen_args, _ in unpacked]
        if(sampling_prob_per_generator is not None):
            gen_probs = [
                sampling_prob_per_generator[gen] for gen, _ in unpacked]
        else:
            gen_probs = None
        obj = cls(**constructor_kwargs)
        obj.setup_with_question_generators(q_gens, gen_probs)
        for i, (gen, gen_args, data_type) in enumerate(unpacked):
            obj.data_type_per_qgen[i] = data_type

        return obj

    def __getitem__(
        self, i: int) -> Tuple[
            SampleWithNamedType, str, str, Optional[List[str]]]:
        """Returns a tuple consisting of information about the data
        sample (type is "Any" since this depends on the subclass), the 
        question, the answer, and a list of all candidate answers that 
        are part of the question text if `also_return_answers` was 
        `True` in `__init__`, in which case it is assumed that the 
        question is in multiple-choice format.
        """
        if(self.total_length is None):
            raise ValueError("The dataset has not been equipped with any "
                             "question generators")        
        if(i < self.total_length):
            gen_ind = next(
                filter(lambda _: self.cumulative_lens[_] > i, 
                       range(len(self.cumulative_lens)))) - 1
            sample_data_type = self.data_type_per_qgen[gen_ind]
            q = self.question_generators[gen_ind].form_question(
                i - self.cumulative_lens[gen_ind])
            return (
                (SampleWithNamedType(sample_data_type, q.corresponding_data), 
                 q.question, q.answer) + 
                 ((q.candidate_answers,) if self.also_return_answers else ()))
        else:
            raise IndexError    


class COCOSplicingDataset(ImgSplicingDatasetWithMasks):
    img_ind_to_mask_pngs: Dict[int, PathLike]

    def __init__(self, coco_base_path: PathLike, 
                 mask_pngs_base_path: PathLike,
                 coco_captions_json: PathLike,
                 coco_objs_json: PathLike):
        
        with open(coco_captions_json, "r") as f:
            coco_caption_data = json.load(f)

        self.coco_base_path = coco_base_path
        self.mask_pngs_base_path = mask_pngs_base_path
        img_id_to_filename = {
            _["id"]: _["file_name"] for _ in coco_caption_data["images"]}
        captions_ids = [(_["caption"], _["image_id"])
                        for _ in coco_caption_data["annotations"]]
        
        img_paths = []
        img_captions = []
        img_ids_to_indices = {}
        for caption, img_id in captions_ids:
            if(img_id not in img_ids_to_indices):
                img_ids_to_indices[img_id] = len(img_paths)
                img_paths.append(os.path.join(coco_base_path, 
                                              img_id_to_filename[img_id]))
                img_captions.append([caption])
            else:
                img_captions[img_ids_to_indices[img_id]].append(caption)

        with open(coco_objs_json, "r") as f:
            coco_objs_data = json.load(f)

        thing_categories = {cat["id"]: cat["name"] for cat in 
                            coco_objs_data["categories"] if cat["isthing"]}
        obj_classes_and_quantities = [{} for _ in img_paths]
        img_ind_to_mask_pngs = {}
        segment_id_to_bbox_info = {}
        segment_categories = {}
        for annotation in coco_objs_data["annotations"]:
            img_id = annotation["image_id"]

            mask_png_path = os.path.join(
                mask_pngs_base_path, annotation["file_name"])
            
            img_index = img_ids_to_indices[img_id]
            for segment in annotation["segments_info"]:
                if(segment["category_id"] not in thing_categories
                   or segment["iscrowd"] == 1):
                    continue
                obj_class = thing_categories[segment["category_id"]]
                segment_id = (img_index, segment["id"])
                assert segment_id not in segment_categories
                segment_categories[segment_id] = obj_class
                bbox_upper_left = (segment["bbox"][0], segment["bbox"][1])
                bbox_h_w = (segment["bbox"][3], segment["bbox"][2])
                segment_id_to_bbox_info[segment_id] = (
                    bbox_upper_left, bbox_h_w)
                if(img_index not in img_ind_to_mask_pngs):
                    img_ind_to_mask_pngs[img_index] = mask_png_path
                if(obj_class not in obj_classes_and_quantities[img_index]):
                    obj_classes_and_quantities[img_index][obj_class] = 1
                else:
                    obj_classes_and_quantities[img_index][obj_class] += 1
        
        obj_classes_and_quantities = {
            i: sorted(d.items(), key=lambda _: _[0]) 
            for i, d in enumerate(obj_classes_and_quantities) if len(d) != 0}
        self.img_ind_to_mask_pngs = img_ind_to_mask_pngs
        self.segment_categories = segment_categories
        self.segment_id_to_bbox_info = segment_id_to_bbox_info

        super().__init__(img_paths, img_captions, obj_classes_and_quantities)

    def get_masks_for_sample(self, i: int) -> List[ImageMask]:

        mask_png_path = self.img_ind_to_mask_pngs[i]
        loaded_mask = read_BGR_image(mask_png_path)
        mask_id_map = convert_png_mask_to_id_map(loaded_mask)
        segment_ids, masks = convert_id_map_to_binary_masks(mask_id_map)
        segment_map = {s_id: i for i, s_id in enumerate(segment_ids)}

        return [ImageMask(
            masks[:, :, segment_map[s_id]], self.segment_categories[(i, s_id)],
            *self.segment_id_to_bbox_info[(i, s_id)])
            for s_id in segment_ids[1:] 
            if (i, s_id) in self.segment_categories]