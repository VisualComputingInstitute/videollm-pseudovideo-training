import argparse
import os
from os import PathLike
import re
import shutil

def main(base_experiments_dir: PathLike, experiment_naming_regex: str, 
         dry_run: bool=True):
    all_subdirs = os.listdir(base_experiments_dir)
    candidate_experiments = [
        _ for _ in all_subdirs if re.fullmatch(experiment_naming_regex, _)
        and os.path.exists(os.path.join(base_experiments_dir, _, "slurm", 
                                        "training_complete.flag"))]
    ckpt_naming_regex = r"checkpoint-\d+"
    for exp in candidate_experiments:
        exp_path = os.path.join(base_experiments_dir, exp)
        experiment_name, postfix = re.fullmatch(
            experiment_naming_regex, exp).groups()

        renamed = experiment_name
        c = -1
        while(renamed in all_subdirs):
            c += 1
            renamed = renamed + "_" + str(c)
        try:
            checkpoints = os.listdir(os.path.join(exp_path, "checkpoints"))
        except FileNotFoundError:
            checkpoints = []
        if(len(checkpoints) > 0):
            checkpoints = [
                _ for _ in checkpoints if re.fullmatch(ckpt_naming_regex, _)]
    
            checkpoints.sort(reverse=True, key=lambda _ : re.fullmatch(
                ckpt_naming_regex, _).group(0))
            ckpt_to_delete = [os.path.join(exp_path, "checkpoints", _) for _ in 
                            checkpoints[1:]]
            ckpt_to_keep = os.path.join(exp_path, "checkpoints", 
                                        checkpoints[0])

            for ckpt_dir in ckpt_to_delete:
                if(dry_run):
                    print("Directory \"{0}\" will be deleted".format(ckpt_dir))
                else:
                    shutil.rmtree(ckpt_dir)
            if(dry_run):
                print("Directory \"{0}\" will be kept".format(ckpt_to_keep))
        
        assert renamed not in all_subdirs
        if(dry_run):
            print("Experiment \"{0}\" will be renamed to \"{1}\"".format(
                exp, renamed))
            i = all_subdirs.index(exp)
            all_subdirs[i] = renamed
        else:
            os.rename(os.path.join(base_experiments_dir, exp), 
                      os.path.join(base_experiments_dir, renamed))
            all_subdirs = os.listdir(base_experiments_dir)
            

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("base_exp_dir", type=str)
    parser.add_argument("--no_dry", action="store_true", default=False)
    args = parser.parse_args()
    pattern = r"(.*?)_(\d{4}_\d{2}_\d{2}_\d{2}h_\d{2}m_\d{2}s)"
    
    main(args.base_exp_dir, pattern, not args.no_dry)