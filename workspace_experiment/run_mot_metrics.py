import os
import sys
import cv2
import tqdm
import time
import shutil
import subprocess

occlusion_bbox = [1/3, 1/3, 2/3, 2/3] # [x1, y1, x2, y2] in scale (0, 1)

# define paths to directories
curr_path = os.getcwd()
root_path = "/home/bhernandez/Documents/deepstream/workspace_experiment"

dump_dir = "trajDumps"
# dump_dir = "triesMultipleStrings/conf2"

dataset_videos_path = root_path + "/dataset/videos"
orig_dump_path = root_path + "/" + dump_dir
out_dataset_path = root_path + "/py-motmetrics/experiment/dataset"

sys.path.append(root_path)
from run_containers import run_app

# define paths to files
dataset_gt_path = root_path + "/dataset/ground_truth.txt"

# Define the camera set
camset = [1, 38, 48, 74] # [1, 74] # [1, 38, 48, 74]
# Define the camera set for occlusion. It must be a subset of camset
camset_occ = [] # [1, 48]


def handle_paths(camset, camset_occ):
    # make it smarter by checking existence of the directories
    os.makedirs(out_dataset_path, exist_ok=True)
    os.makedirs(out_dataset_path + "/videos", exist_ok=True)
    os.makedirs(out_dataset_path + "/" + dump_dir, exist_ok=True)
    os.makedirs(out_dataset_path + "/ground_truth", exist_ok=True)
    # For evalLog files
    os.makedirs(out_dataset_path + "/evalLogs", exist_ok=True)
    # Create seqmap.txt file
    with open(out_dataset_path + "/seqmap.txt", "w") as seqmap:
        for cam in camset:
            os.makedirs(out_dataset_path + "/ground_truth/seq_%03d/gt" % cam, exist_ok=True)
            seqmap.write("seq_%03d\n" % cam)
            if cam in camset_occ:
                os.makedirs(out_dataset_path + "/ground_truth/seq_%03d_occ/gt" % cam, exist_ok=True)
                seqmap.write("seq_%03d_occ\n" % cam)

# copied
# For ground-truth result, the format is (9)  <frame>, <id>, <bb_left>,
# <bb_top>, <bb_width>, <bb_height>, <conf>, <obj_class>, <visibility>
def convert_mtmc_to_mot(line):
    fields = line[:-1].split(' ')
    cam = int(fields[0])
    idp = int(fields[1])
    frame = int(fields[2])
    x = int(fields[3])
    y = int(fields[4])
    w = int(fields[5])
    h = int(fields[6])
    # x_feet = float(fields[7])
    # y_feet = float(fields[8])
    line_out = "%d,%d,%d,%d,%d,%d,%d,%d,%.1f" % (frame, idp, x, y, w, h, 1, 1, 1.0)
    return cam, frame, line_out

# copied
def write_gt_files(cam, seq_name, max_frame, gt_by_cam):
    with open(out_dataset_path + "/ground_truth/%s/gt/gt.txt" % seq_name, "w") as gt:
            for line in gt_by_cam[cam]:
                gt.write(line + "\n")
                    # Create the seqinfo.ini file
    with open(out_dataset_path + "/ground_truth/%s/seqinfo.ini" % seq_name, "w") as seqinfo:
        seqinfo.write("[Sequence]\nseqLength=%d\n" % max_frame)
        seqinfo.close()
# copied
def process_gt_by_cam(camset, camset_occ):
    t1 = time.time()
    camset = set(camset)
    gt_by_cam = {cam:[] for cam in camset} # {cam: [lines]}
    max_frame = 0
    with open(dataset_gt_path, "r") as gt:
        for line in gt.readlines():
            if int(line[:-1].split(' ')[0]) not in camset: continue
            cam, frame, line_out = convert_mtmc_to_mot(line)
            gt_by_cam[cam].append(line_out)
            if frame > max_frame:
                max_frame = frame
    for cam in camset:
        # Write ground truth for each camera
        write_gt_files(cam, 'seq_%03d' % cam, max_frame, gt_by_cam)
        # Write ground truth for occlusion cameras
        if cam in camset_occ:
            write_gt_files(cam, 'seq_%03d_occ' % cam, max_frame, gt_by_cam)
    t2 = time.time()
    print("GT processing took", "%.3f" % (t2 - t1), "seconds")



# For tracking result, the format is (10)     <frame>, <id>, <bb_left>,
# <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
#
# For extended tracking result (with SV3DT), the format is (15) <frame>, <id>, <bb_left>,
# <bb_top>, <bb_width>, <bb_height>, <conf>, <foot_world_x>, <foot_world_y>, <foot_world_z>,
# -<obj_class>, <visibility>,<foot_img_x>,<foot_img_y>,<convex hull>

def process_trajDump_for_cam(cam, occ=False):
    comp = "_occ" if occ else ""
    print("Processing trajDump for camera %d%s..." % (cam, comp))
    dump = open(out_dataset_path + "/%s/trajDump_Stream_%d%s.txt" % (dump_dir, cam, comp), "r")
    dump_out = open(out_dataset_path + "/%s/seq_%03d%s.txt" % (dump_dir, cam, comp), "w")
    for line in dump.readlines():
        fields = line[:-1].split(',')
        frame = int(fields[0]) - 1
        idp = int(fields[1])
        x = float(fields[2])
        y = float(fields[3])
        w = float(fields[4])
        h = float(fields[5])
        conf = float(fields[6])
        # obj_class = int(fields[10])
        # visibility = float(fields[11])
        # line_out = "%d,%d,%d,%d,%d,%d,%.1f,%d,%.1f,-1" % (frame, idp, x, y, w, h, conf, obj_class, visibility)
        line_out = "%d,%d,%d,%d,%d,%d,%.1f,-1,-1,-1" % (frame, idp, x, y, w, h, conf)
        dump_out.write(line_out + "\n")
    dump.close()
    dump_out.close()

# copied
def occlude_video(inp_file_name, out_file_name):
    vid = cv2.VideoCapture(inp_file_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Frames: ", frames, "FPS: ", fps, "res: ", (width, height))
    out = cv2.VideoWriter(out_file_name, fourcc, fps, (width, height))
    x1, y1 = int(occlusion_bbox[0] * width), int(occlusion_bbox[1] * height)
    x2, y2 = int(occlusion_bbox[2] * width), int(occlusion_bbox[3] * height)
    for i in tqdm.tqdm(range(frames)):
        fr = vid.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = vid.read()
        if ret:
            frame[y1:y2, x1:x2, :] = 0
            out.write(frame)
    vid.release()
    out.release()

def create_aug_subset(camset, camset_occ):
    handle_paths(camset, camset_occ)
    process_gt_by_cam(camset, camset_occ)
    # create a subset of the original dataset
    for cam in camset:
        print("Copying video for camera %d..." % cam)
        vid_name = "Warehouse_Synthetic_Cam%03d.mp4" % cam
        dump_name = "trajDump_Stream_%d.txt" % cam
        shutil.copy(dataset_videos_path + "/" + vid_name, out_dataset_path + "/videos")
        shutil.copy(orig_dump_path + "/" + dump_name, out_dataset_path + "/" + dump_dir)
        # Create output file for the sequence from the trajDump file
        process_trajDump_for_cam(cam)
        if cam in camset_occ:
            print("Creating video for occlusion camera %d..." % cam)
            inp_video = dataset_videos_path + "/" + vid_name
            out_video = out_dataset_path + "/videos/" + "Warehouse_Synthetic_Cam%03d_occ.mp4" % cam
            occlude_video(inp_video, out_video)
            print("Running pipeline for occluded camera %d..." % cam)
            os.chdir(root_path)
            inp_vid = "file://" + out_video
            out_vid = "outVideos/out_Warehouse_Synthetic_Cam%03d_occ.mp4" % cam
            dmp_file = out_dataset_path + "/%s/trajDump_Stream_%d_occ.txt" % (dump_dir, cam)
            log_file = "ds_logs/logs_cam_%03d_occ.log" % cam
            run_app(cam, inp_vid, out_vid, dmp_file, log_file)
            os.chdir(curr_path)
            process_trajDump_for_cam(cam, occ=True)

def run_evaluation():
    # Run the evaluation script
    evaluate_cmd = root_path + "/py-motmetrics/motmetrics/apps/evaluateTracking.py"
    ground_truth_path = out_dataset_path + "/ground_truth"
    traj_dumps_path = out_dataset_path + "/" + dump_dir
    seqmap_path = out_dataset_path + "/seqmap.txt"
    log_path = out_dataset_path + "/evalLogs"
    subprocess.run(["python", evaluate_cmd, ground_truth_path, traj_dumps_path, seqmap_path, 
                                "--loglevel", "debug", "--fmt", "mot16", "--log", log_path])

if __name__ == "__main__":
    # Create the subset of the original dataset
    print("Creating the subset", camset, "from the original dataset",
          "with cameras", camset_occ, "to be occluded")
    create_aug_subset(camset, camset_occ)
    
    # Run the evaluation script
    run_evaluation()
