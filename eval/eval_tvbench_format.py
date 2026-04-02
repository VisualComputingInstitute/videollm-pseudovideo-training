# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# pyre-strict

# Need to call this before importing transformers.

import datetime
import json
import os
import re
import shutil
import uuid
from itertools import chain
import argparse
from typing import Dict

import sys
sys.path.append('./')
import numpy as np
from PIL import Image
from enum import Enum

import torch

from longvu.builder import load_pretrained_model
from longvu.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)

from decord import cpu, VideoReader  
from torch import distributed as dist
from tqdm import tqdm

from transformers.trainer_pt_utils import IterableDatasetShard

from known_motion_datasets import (
    COMPLETE_EVAL_SUBTASK_JSON, 
    CompleteEvalSubtasks,
    TVBenchStyleEvalMode,
    KnownDatasets)

tasks = {
    "Action Count": (
        "action_count.json", 
        "action_count", 
        "video", False),
    "Object Count": (
        "object_count.json", 
        "object_count", 
        "video", False),
    "Action Sequence": (
        "action_sequence.json", 
        "action_sequence", 
        "video", True),  # has start & end
    "Object Shuffle": (
        "object_shuffle.json",
        "object_shuffle", 
        "video", False),
    "Scene Transition": (
        "scene_transition.json", 
        "scene_transition", 
        "video", False),
    "Action Localization": 
    ("action_localization.json", 
        "action_localization"
        , "video", True),  # has start & end
    "Action Antonym": (
        "action_antonym.json", 
        "action_antonym", 
        "video", False),
    "Unexpected Action": (
        "unexpected_action.json", 
        "unexpected_action",
        "video", False),
    "Egocentric Sequence": (
        "egocentric_sequence.json", 
        "egocentric_sequence",
        "video", False),
    "Moving Direction": (
        "moving_direction.json", 
        "moving_direction", 
        "video", False),
    }

class EvalDataset(torch.utils.data.IterableDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tasks_dict: Dict
    ) -> None:
        super(EvalDataset, self).__init__()

        self.data_path = data_path

        list_data_dict = []
        for task_name, task in sorted(
            list(tasks_dict.items()), key= lambda _: _[0]):
            json_file = os.path.join(data_path, "json", task[0])
            vis_folder = os.path.join(data_path, "video", task[1])
            with open(json_file, "r") as f:
                json_data = json.load(f)
            for data in json_data:
                video_path = os.path.join(vis_folder, data["video"])
                answer = data["answer"]
                question = data["question"]
                answer_idx = -1
                letters = []
                options = data["candidates"]
                options_string = ""
                for option_idx, c in enumerate(options):
                    letters.append(f"{chr(ord('A') + option_idx)}")
                    options_string += f"({chr(ord('A') + option_idx)}) {c}\n"
                    if c == answer:
                        answer_idx = option_idx
                prompt = (f"Question: {question}\nOptions:\n{options_string}"
                          "Answer with the option's letter from the given "
                          "choices directly and only give the best option.")
                list_data_dict.append(
                    {
                        "task_type": task_name,
                        "bound": (data["start"], data["end"]) if task[3] else task[3],
                        "question": question,
                        "prompt": prompt,
                        "answer": answer_idx,
                        "answer_word": data["answer"],
                        "video_name": data["video"].split(".")[0],
                        "video": video_path,
                        "data_type": task[2],
                        "letters": ",".join(letters),
                        **({"question_id": data["question_id"]}
                        if "question_id" in data else {})
                    }
                )

        self.data = list_data_dict

    def __len__(self) -> int:
        return len(self.data)

    # pyre-fixme[3]: Return type must be annotated.
    def __iter__(self):
        return iter(enumerate(self.data))

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __getitem__(self, i):
        return self.data[i]


def train(args, tasks_dict: Dict) -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    version = args.version
    model_name = args.model_name
    model_path = args.model_path

    # torch.distributed.barrier()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,  
        None,
        model_name,
        device_map=None,
    )
    model.get_model().config.drop_threshold = 0.8
    model.config.use_cache = True
    model.cuda()
    dataset = EvalDataset(data_path=args.data_path, tasks_dict=tasks_dict)
    world_size = torch.distributed.get_world_size()
    world_rank = torch.distributed.get_rank()
    shard_dataset = IterableDatasetShard(
        dataset,
        batch_size=1,
        num_processes=world_size,
        process_index=world_rank,
    )
    torch.distributed.barrier()
    output = []
    final_output = [None] * world_size

    for sample_idx, line in tqdm(shard_dataset):
        video_name = line["video_name"]
        answer = line["answer"]
        qs = line["prompt"]
        task_type = line["task_type"]
        video_path = line["video"]
        bound = line["bound"]
        data_type = line["data_type"]
        letters = line["letters"].split(",")

        if(os.path.exists(video_path) 
           and task_type != CompleteEvalSubtasks.SINGLE_BLACK_FR):
            if data_type == "video":
                vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
                max_frame = len(vr) - 1
                fps = float(vr.get_avg_fps())
                if bound:
                    start, end = bound[0], bound[1]
                    start_idx = max(0, round(start * fps))
                    end_idx = min(round(end * fps), max_frame)
                    frame_indices = np.array(
                        [
                            i
                            for i in range(
                                start_idx,
                                end_idx,
                                round(fps / 2),
                            )
                        ]
                    )
                else:
                    frame_indices = np.array(
                        [
                            i
                            for i in range(
                                0,
                                len(vr),
                                round(fps / 2),
                            )
                        ]
                    )
                if(task_type == CompleteEvalSubtasks.REPEATED_FIRST_FR):
                    print("For this task the model will only have access to "
                          "the first video frame (repeated in time).")               
                    video = np.repeat(vr[0].asnumpy()[None, :, :, :], 
                                      len(frame_indices), axis=0)
                else:
                    video = []
                    for frame_index in frame_indices:
                        img = vr[frame_index].asnumpy()
                        video.append(img)
                    video = np.stack(video)
                    if(args.shuffle_frames):
                        np.random.shuffle(video)
            else:
                max_frame = len(os.listdir(video_path))
                images_group = list()
                fps = 3
                if bound:
                    start, end = bound[0], bound[1]
                else:
                    start, end = -100000, 100000
                start_idx = max(1, round(start * fps))
                end_idx = min(round(end * fps), max_frame)
                frame_indices = [
                    i
                    for i in range(
                        start_idx,
                        end_idx,
                        round(fps / 2),
                    )
                ]
                for frame_index in frame_indices:
                    img = Image.open(
                        os.path.join(video_path, f"{frame_index:05d}.jpg")
                    ).convert("RGB")
                    images_group.append(np.array(img))
                video = np.stack(images_group)
            
            image_sizes = [video[0].shape[:2]]
            video = process_images(video, image_processor, model.config)
            video = [item.unsqueeze(0) for item in video]
        else:
            if(not os.path.exists(video_path)):
                print("Video {0} does not exist; a single black frame "
                      "will be used".format(video_path))
            elif(task_type == CompleteEvalSubtasks.SINGLE_BLACK_FR):
                print("Video will intentionally be hidden from the model "
                      "for this task; a single black frame will be used.")
                
            video = np.zeros((1, 1024, 1024, 3)).astype(np.uint8)
            image_sizes = [(1024, 1024)]
            video = process_images(video, image_processor, model.config)

        if getattr(model.config, "mm_use_im_start_end", False):
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[version].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        if "llama3" in version:
            input_ids = input_ids[0][1:].unsqueeze(0)  # remove bos

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=video,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=5,  
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        if isinstance(output_ids, tuple):
            output_ids = output_ids[0]
        pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        tmp = pred
        if pred.endswith(stop_str):
            pred = pred[: -len(stop_str)]
            pred = pred.strip()
        print(pred)
        pred = pred.replace("Answer", "")

        pred_answer = re.findall(
            f"[\(,\ ]*[{letters[0]}-{letters[-1]}][\),\ ]*", pred
        )
        try:
            pred_answer = pred_answer[0].strip()
        except IndexError:
            print("Index error caused by model output " + tmp)
            raise IndexError
        pred_answer = pred_answer.strip("()")
        if pred_answer in letters:
            pred_idx = letters.index(pred_answer)
            pred = letters[pred_idx]
        else:
            print("pred_answer: ", pred_answer, " pred: ", pred, flush=True)
            pred_idx = 2
            pred = letters[pred_idx]

        ans_id = uuid.uuid4()
        output.append(
            (sample_idx, {
                "question": line["question"],
                "prompt": qs,
                "answer": answer,
                "pred": pred_idx,
                "task_type": task_type,
                "answer_id": str(ans_id),
                "model_id": model_name,
                "video_name": video_name,
                "metadata": {},
                **({"question_id": line["question_id"]}
                if "question_id" in line else {})
            })
        )
    print("reached the barrier")
    dist.barrier()
    print("got past the barrier")
    dist.all_gather_object(
        final_output,
        output,
    )
    print("got past the gather")
    
    all_res_output_file = args.all_res_output_file
    acc_output_file = args.acc_output_file
    all_output = list(chain(*final_output))
    global_rank = dist.get_rank()
    if global_rank == 0:
        deduped_output = []
        seen_indices = set()
        for sample_idx, sample_output in all_output:
            if(sample_idx in seen_indices):
                continue
            deduped_output.append(sample_output)
            seen_indices.add(sample_idx)
        assert (len(deduped_output) == len(dataset))
        with open(all_res_output_file, "w") as f:
            json.dump(deduped_output, f)

        task_types = tasks_dict.keys()
        task_acc = {x: [] for x in task_types}
        acc = []

        for i, x in enumerate(deduped_output):
            value = 1
            if x["pred"] != x["answer"]:
                value = 0
            acc.append(value)
            task_acc[x["task_type"]].append(value)

        acc = sum(acc) * 100 / len(acc)
        task_acc = {x: sum(task_acc[x]) * 100 / len(task_acc[x]) 
                    for x in task_acc}
        print(f"Accuracy: ", acc)
        print("Task accuracy", task_acc)

        task_acc["avg"] = acc

        with open(acc_output_file, "w") as f:
            json.dump(task_acc, f)
    print("done?")
    torch.distributed.destroy_process_group()

if(__name__ == "__main__"):

    tasks_dict_excl_tvbench = {
        TVBenchStyleEvalMode.MOTION_EVAL_COMPLETE: {
            _: (COMPLETE_EVAL_SUBTASK_JSON[_], "", "video", False)
            for _ in CompleteEvalSubtasks
        },
        TVBenchStyleEvalMode.MOTION_EVAL_SIMPLE: {
            CompleteEvalSubtasks.MOV_DIRECTION: (
                COMPLETE_EVAL_SUBTASK_JSON[
                    CompleteEvalSubtasks.MOV_DIRECTION], "", "video", False)
        },
        TVBenchStyleEvalMode.COUNTING: {
            "counting": ("counting.json", "", "video", False)
        }
    }
    tasks_dict_tvbench = {
        TVBenchStyleEvalMode.TVBENCH_ORIGINAL: tasks,
        TVBenchStyleEvalMode.MOTION_EVAL_COMPLETE: {
            _: (COMPLETE_EVAL_SUBTASK_JSON[_], 
                "moving_direction", "video", False)
            for _ in CompleteEvalSubtasks
        },
        TVBenchStyleEvalMode.MOTION_EVAL_SIMPLE: {
            CompleteEvalSubtasks.MOV_DIRECTION: (
                COMPLETE_EVAL_SUBTASK_JSON[
                    CompleteEvalSubtasks.MOV_DIRECTION], "moving_direction", 
                    "video", False)
        }
    }
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default="./checkpoints/longvu_qwen")
    parser.add_argument('--model_name', default="cambrian_qwen")
    parser.add_argument('--version', default="qwen")
    parser.add_argument('--local-rank', default=0)
    parser.add_argument('--data_path', required=True)
    parser.add_argument("--eval_mode", type=TVBenchStyleEvalMode, 
                        default=TVBenchStyleEvalMode.TVBENCH_ORIGINAL)
    parser.add_argument("--dataset_name", type=KnownDatasets,
                        default=KnownDatasets.TVBENCH)
    parser.add_argument("--all_res_output_file", type=str)
    parser.add_argument("--acc_output_file", type=str)
    parser.add_argument("--shuffle_frames", action="store_true")
    args = parser.parse_args()
    if(args.shuffle_frames):
        tmp = os.path.split(args.acc_output_file)
        os.makedirs(tmp[0] + "_shuffled", exist_ok=True)
        args.acc_output_file = os.path.join(tmp[0] + "_shuffled", tmp[1])
        tmp = os.path.split(args.all_res_output_file)
        os.makedirs(tmp[0] + "_shuffled", exist_ok=True)
        args.all_res_output_file = os.path.join(tmp[0] + "_shuffled", tmp[1])
    if "llama3" in args.version:
        args.model_name = "cambrian_llama3"
    if(args.dataset_name != KnownDatasets.TVBENCH):
        train(args, tasks_dict_excl_tvbench[args.eval_mode])
    else:
        train(args, tasks_dict_tvbench[args.eval_mode])
