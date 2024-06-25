'''
create_mot_output_from_dump.py

This script converts the trajDump files into MOTChallenge format 
for MOT evaluation, and creates the corresponding seqmap file
'''
import os
import yaml
import subprocess


home_dir = os.getenv("HOME")
workspace_dir = home_dir + "/Documents/deepstream/workspace"
experiment_dir = workspace_dir + "/experiments/exp100cam360s"
mot_gt_ordered_dir = "ground_truth"
mot_gt_4eval_dir = "mot_gt"
mot_out_eval_dir = "mot_out"

def handle_paths():
    os.makedirs(experiment_dir + "/" + mot_out_eval_dir, exist_ok=True)
    os.makedirs(experiment_dir + "/" + mot_gt_4eval_dir, exist_ok=True)
    with open(experiment_dir + "/experiment.yml", "r") as f:
        experiment = yaml.load(f, Loader=yaml.FullLoader)
    seqmap = open(experiment_dir + "/seqmap.txt", "w")
    for seq, cam in experiment["seq2cam"].get("seq2orig", {}).items():
        print(int(seq), int(cam))
        seqmap.write("seq_%03d\n" % int(cam))
        os.symlink("%s/%s/%03d" % (experiment_dir, mot_gt_ordered_dir, int(seq)),
                   "%s/%s/%03d" % (experiment_dir, mot_gt_4eval_dir, int(cam)))
    # convert trajDump files to MOTChallenge format
    seqmap.close()


def run_evaluation():
    # Run the evaluation script
    evaluate_cmd = workspace_dir + "/py-motmetrics/motmetrics/apps/evaluateTracking.py"
    ground_truth_path = experiment_dir + "/ground_truth"
    traj_dumps_path = experiment_dir + "/" + dump_dir
    seqmap_path = experiment_dir + "/seqmap.txt"
    log_path = experiment_dir + "/evalLogs"
    subprocess.run(["python", evaluate_cmd, ground_truth_path, traj_dumps_path, seqmap_path, 
                                "--loglevel", "debug", "--fmt", "mot16", "--log", log_path])

def main():
    handle_paths()

if __name__ == "__main__":
    main()