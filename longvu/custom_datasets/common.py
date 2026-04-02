

import os
import json
from os import PathLike
from typing import Dict, List


def parse_jsonl(jsonl_path: PathLike) -> List[Dict]:
    contents = []
    with(open(jsonl_path, "r") as f):
        for line in f.readlines():
            contents.append(json.loads(line))
    return contents


def merge_ground_truths_for_dirs(
        jsonl_files: List[PathLike],
        corresponding_dirs: List[PathLike]) -> List[Dict]:
    all_contents = []
    for jsonl, containing_dir in zip(jsonl_files, corresponding_dirs):
        contents = parse_jsonl(jsonl)
        for c in contents:
            c["video_name"] = os.path.join(containing_dir, c["video_name"])
            all_contents.append(c)
    return all_contents