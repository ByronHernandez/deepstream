'''
This file is used to create the MOT ground truth from MTMC annotations.
'''
import os
import tqdm
import time

mtmc_dir = os.getenv("HOME") + "/Documents/deepstream/workspace/datasets/orig"
mot_gt_dir = mtmc_dir + "/ground_truth"

cams_to_process = list(range(1, 101))

# For MOT ground-truth result, the format is (9)  
# <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <obj_class>, <visibility>
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

def write_gt_files(cam, seq_name, max_frame, gt_by_cam):
    with open(mot_gt_dir + "/%s/gt/gt.txt" % seq_name, "w") as gt:
            for line in gt_by_cam[cam]:
                gt.write(line + "\n")
                    # Create the seqinfo.ini file
    with open(mot_gt_dir + "/%s/seqinfo.ini" % seq_name, "w") as seqinfo:
        seqinfo.write("[Sequence]\nseqLength=%d\n" % max_frame)
        seqinfo.close()

def process_gt_by_cam(camset):
    t1 = time.time()
    camset = set(camset)
    gt_by_cam = {cam:[] for cam in camset} # {cam: [lines]}
    max_frame = 0
    with open(mtmc_dir + "/ground_truth.txt", "r") as gt:
        for line in tqdm.tqdm(gt.readlines(), desc="Processing MTMC GT"):
            if int(line[:-1].split(' ')[0]) not in camset: continue
            cam, frame, line_out = convert_mtmc_to_mot(line)
            gt_by_cam[cam].append(line_out)
            if frame > max_frame:
                max_frame = frame
    for cam in tqdm.tqdm(camset, desc="Writing MOT GT"):
        # Write ground truth for each camera
        write_gt_files(cam, '%03d' % cam, max_frame, gt_by_cam)
    t2 = time.time()
    print("GT processing took", "%.3f" % (t2 - t1), "seconds")

def main():
    for cam in cams_to_process:
        os.makedirs(mot_gt_dir + "/%03d/gt" % cam, exist_ok=True)
    process_gt_by_cam(cams_to_process)

if __name__ == "__main__":
    main()
