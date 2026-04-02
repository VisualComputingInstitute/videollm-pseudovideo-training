import transformers
from torch.utils.data import Dataset
from os import PathLike
import os
from decord import cpu, VideoReader
from PIL import Image
import torch
from typing import Dict, List, Any, Tuple
import copy
import numpy as np
import random
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from imgaug import augmenters as aug
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from einops import repeat

from .multiq_linear_motion_dataset import MultiQuestionLinMotDataset
from .qa_dataset import QADataset
from .img_splicing_dataset import SampleWithNamedType
from .spliced_masks_q_generators import MovingObjSceneFormation
from .mask_utils import (relative_path_to_absolute, 
                         relative_time_to_frame,
                         interpolate_path,
                         relative_bbox_to_absolute,
                         draw_mask_on_path)
from ..mm_datautils import preprocess_multimodal, preprocess
from ..constants import IGNORE_INDEX

def format_qa_as_conversation(question: str, answer: str) -> Dict:
    """On-the-fly creation of a human-GPT conversation in the format 
    expected by LongVU."""
    return {"conversations": [{
        "from": "human", "value": "<image>\n" + question},
        {"from": "gpt", "value": answer}]}

class ABCLongVUDatasetWrapper(ABC, Dataset):

    def __init__(
        self,
        wrapped_dataset: QADataset,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
    ):
        super(Dataset, self).__init__()
        self.wrapped_dataset = wrapped_dataset

        self.tokenizer = tokenizer
        self.data_args = data_args
        #no idea if this is necessary anywhere, but it was in the original
        #dataset class' code soooo
        self.length = len(self)
        if(data_args is not None):
            self.debug_print_question = data_args.debug_print_question
        else:
            self.debug_print_question = False

    def __len__(self) -> int:
        return len(self.wrapped_dataset)
    
    def _compute_lengths(self):
        """Compute and cache dummy lengths (data samples are created on
        the fly and so these are unknown) conversations in the dataset.
        """
        if hasattr(self, "length_list") and hasattr(
            self, "modality_length_list"):
            # Return cached values if already computed
            return self.length_list, self.modality_length_list  # pyre-fixme

        self.length_list = [100 for _ in range(len(self))]
        self.modality_length_list = [100 for _ in range(len(self))]
        return self.length_list, self.modality_length_list 

    @property
    def lengths(self):
        length_list, _ = self._compute_lengths()
        return length_list

    @property
    def modality_lengths(self):
        _, modality_length_list = self._compute_lengths()
        return modality_length_list    

    #kept mostly because the original dataset had it, likely unnecessary
    def _has_image(self, sample: dict) -> bool:  # pyre-fixme
        return True    
    
    @abstractmethod
    def load_images_from_question_data(self, data: Any) -> List[NDArray]:
        pass

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        question_data, question, answer = self.wrapped_dataset[i]
        conv = format_qa_as_conversation(question, answer)
        if(self.debug_print_question):
            print(question)
        try:
            image = self.load_images_from_question_data(question_data)
        except:
            return self.__getitem__(0)
        processor_aux_list = self.data_args.image_processor_aux_list

        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), 
                                   background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                # result.paste(pil_img, (0, 0))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), 
                                   background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                # result.paste(pil_img, (0, 0))
                return result

        if self.data_args.image_aspect_ratio != "pad":
            raise NotImplementedError("Only pad is supported for now.")

        image_aux_list = []
        for processor_aux in processor_aux_list:
            image_aux = image
            try:
                target_resolution = processor_aux.crop_size["height"]
            except:
                target_resolution = processor_aux.size["height"]
            if not isinstance(image_aux, Image.Image):
                frame_list = []
                for frame in image_aux:
                    if not isinstance(frame, Image.Image):
                        frame = Image.fromarray(frame)
                    frame_aux = expand2square(
                        frame, tuple(int(x * 255) for x in 
                                     processor_aux.image_mean)
                    ).resize((target_resolution, target_resolution))
                    frame_aux = processor_aux.preprocess(
                        frame_aux, return_tensors="pt"
                    )["pixel_values"][0]
                    frame_list.append(frame_aux)
                image_aux = torch.stack(frame_list)
            else:
                image_aux = expand2square(
                    image_aux, tuple(int(x * 255) for x in 
                                     processor_aux.image_mean)
                ).resize((target_resolution, target_resolution))
                image_aux = processor_aux.preprocess(
                    image_aux, return_tensors="pt"
                )["pixel_values"][0]
            image_aux_list.append(image_aux)

        sources = preprocess_multimodal(
            copy.deepcopy([conv["conversations"]]), self.data_args)

        data_dict = preprocess(sources, self.tokenizer, has_image=True)
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], 
                labels=data_dict["labels"][0]
            )
        if (data_dict["labels"] != IGNORE_INDEX).sum() == 0:
            return self.__getitem__(0)
        data_dict["image_aux_list"] = image_aux_list
        data_dict["image_size"] = image[0].shape[:2]
        return data_dict

class WrappedMotionDatasetLazy(ABCLongVUDatasetWrapper):

    def __init__(
        self,
        wrapped_dataset: MultiQuestionLinMotDataset,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
    ):
        super().__init__(wrapped_dataset, tokenizer, data_args)

        self.repeat_first_frame_only = data_args.repeat_first_frame_only

    def load_images_from_question_data(self, data: PathLike) -> List[NDArray]:
        video_file = data

        if os.path.exists(video_file):
            try:
                vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
                sample_fps = round(
                    vr.get_avg_fps() / self.data_args.video_fps
                )
                frame_idx = [i for i in range(0, len(vr), sample_fps)]
                if(not self.repeat_first_frame_only):
                    image = vr.get_batch(frame_idx).asnumpy()
                else:
                    fr_0 = vr[0].asnumpy()
                    image = [np.copy(fr_0) for _ in frame_idx]
                if self.data_args.uniform_sample:
                    num_sample = 100
                    if len(image) > num_sample:
                        interval = len(image) / float(num_sample)
                        indices = [int(interval * i) 
                                   for i in range(num_sample)]
                        image = [image[idx] for idx in indices]
            except Exception as e:
                print("Failed to load video: ", video_file, flush=True)
                raise e
        else:
            print("Video file does not exist: ", video_file, flush=True)
            raise FileNotFoundError
        
        return image

class WrappedSceneDescrDataset(ABCLongVUDatasetWrapper):

    def load_images_from_question_data(
            self, data: Tuple[PathLike, Tuple[int, int]]) -> List[NDArray]:
        video_file, frame_range = data    

        frame_index = random.choice(np.arange(frame_range[0], frame_range[1]))

        if os.path.exists(video_file):
            try:
                vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
                sample_fps = round(
                    vr.get_avg_fps() / self.data_args.video_fps)
                dummy_frame_idx = [i for i in range(0, len(vr), sample_fps)]
                fr_0 = vr[frame_index].asnumpy() #only actually load 1 image
                image = [np.copy(fr_0) for _ in dummy_frame_idx]
                if self.data_args.uniform_sample:
                    num_sample = 100
                    if len(image) > num_sample:
                        interval = len(image) / float(num_sample)
                        indices = [int(interval * i) 
                                   for i in range(num_sample)]
                        image = [image[idx] for idx in indices]
            except Exception as e:
                print("Failed to load video: ", video_file, flush=True)
                raise e
        else:
            print("Video file does not exist: ", video_file, flush=True)
            raise FileNotFoundError
        
        return image
        
class WrappedSplicedImgsDataset(ABCLongVUDatasetWrapper):

    def __init__(
        self,
        wrapped_dataset: QADataset,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
    ):
        super().__init__(wrapped_dataset, tokenizer, data_args)

        if(data_args is None or not data_args.repeat_first_frame_only):
            self.transform = aug.Sequential([
                aug.Affine(rotate=(-15, 15), scale=(0.9, 1.1),
                        translate_percent=(-0.1, 0.1))])
        else:
            self.transform = None
        
    def load_images_from_question_data(
            self, data: List[Tuple[PathLike, int]]) -> List[NDArray]:
        
        all_images = []
        expected_num_imgs = 0
        largest_h = 0
        largest_w = 0
        for img_path, repetitions in data:
            expected_num_imgs += repetitions
            if(os.path.exists(img_path)):
                try:
                    image = Image.open(img_path).convert("RGB")
                    if(image.height > largest_h):
                        largest_h = image.height
                    if(image.width > largest_w):
                        largest_w = image.width
                    images_repeated = repeat(
                        np.array(image), "H W RGB -> R H W RGB", R=repetitions)
                    all_images.append(images_repeated)
                except Exception as e:
                    print("Failed to load image: ", img_path, flush=True)
                    raise e
            else:
                print("Image file does not exist: ", img_path, flush=True)
                raise FileNotFoundError(img_path)
        resizer = aug.Resize({"height": largest_h, "width": largest_w})
        all_images = [resizer(images=_) for _ in all_images]
        all_images = np.concatenate(all_images, axis=0)
        assert all_images.shape[0] == expected_num_imgs
        if(self.transform is not None):
            all_images = self.transform(images=all_images)

        return all_images
    
class WrappedSplicedImgsDatasetWithMasks(WrappedSplicedImgsDataset):

    def __init__(
        self,
        wrapped_dataset: QADataset,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args):

        super().__init__(wrapped_dataset, tokenizer, data_args)
        if(data_args is None or not data_args.repeat_first_frame_only):
            self.transform = aug.Sequential([
                aug.Affine(rotate=(-2.5, 2.5), scale=(0.98, 1.02),
                        translate_percent=(-0.02, 0.02))])
        else:
            self.transform = None

        self.data_handlers = {
            "no_masks": super().load_images_from_question_data,
            "with_masks": self.load_images_with_masks_from_question_data
        }

    def load_images_with_masks_from_question_data(
            self, data: MovingObjSceneFormation) -> List[NDArray]:
        just_the_images = super().load_images_from_question_data(
            data.scene_formation_from_imgs)
        used_h, used_w = just_the_images.shape[1:3]
        #just_the_images = np.zeros_like(just_the_images)
        num_fr = just_the_images.shape[0]
        for mov_obj in data.moving_objs_info:
            img_for_mask = np.array(
                Image.open(mov_obj.corresponding_img).convert("RGB"))
            assert img_for_mask.shape[:2] == mov_obj.mask.bool_mask.shape

            top_left = mov_obj.mask.bbox_upper_left
            bbox_h_w = mov_obj.mask.bbox_h_w
            
            appear_fr = relative_time_to_frame(mov_obj.appearance_time, num_fr)
            start_fr = relative_time_to_frame(mov_obj.rel_start_time, num_fr)
            end_fr = relative_time_to_frame(mov_obj.rel_end_time, num_fr)
            assert appear_fr <= start_fr and start_fr < end_fr

            abs_path = relative_path_to_absolute(
                mov_obj.rel_path, (used_h, used_w))
            interp_path = interpolate_path(
                abs_path, mov_obj.rel_start_time, mov_obj.rel_end_time, 
                end_fr - start_fr + 1)
            interp_path = np.array(interp_path, dtype=np.int32)
            bbox_contents =  img_for_mask[
                top_left[1]:top_left[1] + bbox_h_w[0],
                top_left[0]:top_left[0] + bbox_h_w[1], :]
            bbox_mask = mov_obj.mask.bool_mask[
                top_left[1]:top_left[1] + bbox_h_w[0],
                top_left[0]:top_left[0] + bbox_h_w[1]]
            #bbox_contents =  img_for_mask[0:bbox_h_w[0], 0:bbox_h_w[1], :]
            just_the_images[start_fr:end_fr + 1, ...] = draw_mask_on_path(
                just_the_images[start_fr:end_fr + 1, ...], bbox_h_w, 
                interp_path, bbox_contents, bbox_mask)
            # if(appear_fr < start_fr):
            #     dummy_path = repeat(
            #         interp_path[0, :], "XY -> FR XY", FR=start_fr - appear_fr)
            #     just_the_images[appear_fr:start_fr, ...] = draw_mask_on_path(
            #         just_the_images[appear_fr:start_fr, ...], bbox_h_w,
            #         dummy_path, bbox_contents, bbox_mask)
            # if(end_fr < len(just_the_images) - 1):
            #     dummy_path = repeat(interp_path[-1, :], "XY -> FR XY", 
            #                         FR=len(just_the_images) - 1 - end_fr)            
            #     just_the_images[end_fr + 1:, ...] = draw_mask_on_path(
            #         just_the_images[end_fr + 1:, ...], bbox_h_w,
            #         dummy_path, bbox_contents, bbox_mask)
            
        return just_the_images
        
    def load_images_from_question_data(
            self, data: SampleWithNamedType) -> List[NDArray]:
        #print(data.data_type)
        return self.data_handlers[data.data_type](data.sample_data)
