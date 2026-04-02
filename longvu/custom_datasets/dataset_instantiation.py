import yaml
from os import PathLike

from .multiq_linear_motion_dataset import MultiQuestionLinMotDataset
from .scene_descr_dataset import SceneDescrDataset
from .img_splicing_dataset import COCOSplicingDataset

def motion_dataset_from_yaml(
        video_yaml_path: PathLike,
        question_yaml_path: PathLike) -> MultiQuestionLinMotDataset:

    with(open(video_yaml_path, "r") as f):
        config1 = yaml.safe_load(f)
    with(open(question_yaml_path, "r") as f):
        config2 = yaml.safe_load(f)        
    return MultiQuestionLinMotDataset.from_config_KVs(**config1, **config2)

def spliced_coco_imgs_dataset_from_yaml(
        coco_yaml_path: PathLike,
        question_yaml_path) -> COCOSplicingDataset:

    with(open(coco_yaml_path, "r") as f):
        config1 = yaml.safe_load(f)
    with(open(question_yaml_path, "r") as f):
        config2 = yaml.safe_load(f)    
    return COCOSplicingDataset.from_config_KVs(**config1, **config2)

def scene_descr_dataset_from_yaml(
        video_yaml_path: PathLike) -> SceneDescrDataset:

    with(open(video_yaml_path, "r") as f):
        config = yaml.safe_load(f)    
    return SceneDescrDataset.from_config_KVs(**config)