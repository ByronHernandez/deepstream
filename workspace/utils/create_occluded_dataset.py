'''
This script creates occluded videos from the original videos in the dataset.
'''
import os
import cv2
import tqdm
import shutil
import subprocess
import multiprocessing
from multiprocessing import Pool

cam_file = "Warehouse_Synthetic_Cam%03d.yml"
vid_file = "Warehouse_Synthetic_Cam%03d.mp4"

home_dir = "/home/bhernandez" # os.getenv("HOME")
mtmc_dir = home_dir + "/Documents/deepstream/workspace/datasets/orig"
mtmc_occ_out_dir = home_dir + "/Documents/deepstream/workspace/datasets/occ"

occlusion_bbox = [1/3, 1/3, 2/3, 2/3] # [x1, y1, w, h] in scale (0, 1)
cams_to_process = list(range(1, 101)) # [1, 2, 3, 4] # list(range(1, 101))
nproc = 10

def occlude_video(intuple):
    inp_file_name, out_file_name, log_file = intuple
    print("Creating occluded video:", inp_file_name.split('/')[-1])
    vid = cv2.VideoCapture(inp_file_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x1, y1 = int(occlusion_bbox[0] * width), int(occlusion_bbox[1] * height)
    x2, y2 = int(occlusion_bbox[2] * width), int(occlusion_bbox[3] * height)
    # print("Frames: ", frames, "FPS: ", fps, "res: ", (width, height))
    # out = cv2.VideoWriter(out_file_name, fourcc, fps, (width, height))
    # for _ in tqdm.tqdm(range(frames), desc="Processing video: ", file=log):
    #     pass
    #     fr = vid.get(cv2.CAP_PROP_POS_FRAMES)
    #     ret, frame = vid.read()
    #     if ret:
    #         frame[y1:y2, x1:x2, :] = 0
    #         out.write(frame)
    # out.release()
    vid.release()
    with open(log_file, "w") as log:
        subprocess.run(["ffmpeg", "-y", "-i", inp_file_name,
                        # "-t", "30",         # time limit
                        "-vf", "drawbox=x=%d:y=%d:w=%d:h=%d:color=black:t=fill" % (x1, y1, x2-x1, y2-y1), # video filter
                        "-c:v", "libx264",  # video codec
                        # "-crf", "20",       # constant rate factor to 22
                        "-b:v", "2250k",    # bitrate
                        out_file_name], stdout=log, stderr=log)

def main():
    os.makedirs(mtmc_occ_out_dir + '/logs', exist_ok=True)
    os.makedirs(mtmc_occ_out_dir + '/videos', exist_ok=True)
    try:
        os.remove(mtmc_occ_out_dir + "/camInfo")
        os.remove(mtmc_occ_out_dir + "/ground_truth")
    except FileNotFoundError:
        pass
    os.symlink(mtmc_dir + "/camInfo", mtmc_occ_out_dir + "/camInfo")
    os.symlink(mtmc_dir + "/ground_truth", mtmc_occ_out_dir + "/ground_truth")

    arg_collector = []
    for cam in cams_to_process:
        inp_video = mtmc_dir + "/videos/" + vid_file % cam
        out_log = mtmc_occ_out_dir + "/logs/%03d.log" % cam
        out_video = mtmc_occ_out_dir + "/videos/" + vid_file % cam
        arg_collector.append((inp_video, out_video, out_log))

    cpus = multiprocessing.cpu_count()
    ncams = len(cams_to_process)
    print("Available processors:", cpus)
    print("Number of processors to use:", min(nproc, cpus, ncams))
    with Pool(min(nproc, ncams, cpus)) as p:
        p.map(occlude_video, arg_collector)

if __name__ == "__main__":
    main()
