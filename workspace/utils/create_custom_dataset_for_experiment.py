import os
import json
import shutil
import subprocess

cam_file = "Warehouse_Synthetic_Cam%03d.yml"
vid_file = "Warehouse_Synthetic_Cam%03d.mp4"

workspace_dir = os.getenv("HOME") + "/Documents/deepstream/workspace"
mtmc_dir = workspace_dir + "/datasets/orig"
mtmc_occ_dir = workspace_dir + "/datasets/occ"
mot_gt_dir = "/ground_truth"

def get_cams_from_filtered_calib(version=1):
    with open(mtmc_dir + "/calibration_filtered_v%d.json" % version, "r") as f:
        data = json.load(f)
    ids = set()
    for sensor in data['sensors']:
        ids.add(int(sensor['id'][-3:].lstrip("0")))
    print(sorted(list(ids)))
    return sorted(list(ids))

version = 1 # 1: 40cam # 2: 80cam
cams_to_process = [38, 74] # get_cams_from_filtered_calib(version) # list(range(1, 101)) # [1, 38, 48, 74] # [1, 74] # [38, 74]
cams_to_process_occ = [] # [] # list(range(2, 101, 2)) # [1, 48]
max_frames = -1 # -1 # seconds * fps
experiment = '/experiments/exp002cam060s'

out_dataset_path = workspace_dir + experiment

def handle_cams_paths(cams, cams_dir, cont, seqmap, exp):
    for cam in cams:
        os.symlink(cams_dir + "/camInfo/" + cam_file % cam, out_dataset_path + "/camInfo/" + cam_file % cont)
        if max_frames == -1:
            os.symlink(cams_dir + "/videos/" + vid_file % cam, out_dataset_path + "/videos/" + vid_file % cont)
            os.symlink(cams_dir + mot_gt_dir + "/%03d" % cam, out_dataset_path + mot_gt_dir + "/%03d" % cont)
        else:
            subprocess.run(["ffmpeg", "-i", 
                            cams_dir + "/videos/" + vid_file % cam,
                            "-t", f'{max_frames / 30}', "-c", "copy", 
                            out_dataset_path + "/videos/" + vid_file % cont
                            ])
            os.makedirs(out_dataset_path + mot_gt_dir + "/%03d/gt" % cont, exist_ok=True)
            file_inp = open(cams_dir + mot_gt_dir + "/%03d/gt/gt.txt" % cam, "r")
            file_out = open(out_dataset_path + mot_gt_dir + "/%03d/gt/gt.txt" % cont, "w")
            file_seq = open(out_dataset_path + mot_gt_dir + "/%03d/gt/seqinfo.ini" % cont, "w")
            for line in file_inp.readlines():
                frame = int(line.split(",")[0])
                if frame > max_frames: break
                file_out.write(line)
            file_seq.write("[Sequence]\nseqLength=%d\n" % max_frames)
            file_inp.close()
            file_out.close()
            file_seq.close()
        seqmap.write("%03d\n" % cont)
        exp.write("    %03d: %03d\n" % (cont, cam))
        cont += 1
    return cont

def handle_paths():
    shutil.rmtree(out_dataset_path, ignore_errors=True)
    os.makedirs(out_dataset_path + "/videos", exist_ok=True)
    os.makedirs(out_dataset_path + "/camInfo", exist_ok=True)
    os.makedirs(out_dataset_path + mot_gt_dir, exist_ok=True)
    os.makedirs(out_dataset_path + "/trajDumps", exist_ok=True) # to run the tracker
    os.makedirs(out_dataset_path + "/kitti_detector", exist_ok=True)
    os.makedirs(out_dataset_path + "/kitti_tracker", exist_ok=True)
    seqmap = open(out_dataset_path + "/seqmap.txt", "w")
    exp = open(out_dataset_path + "/experiment.yml", "w")
    exp.write("seq2cam:\n")
    if len(cams_to_process) > 0: exp.write("  seq2orig:\n")
    cont = handle_cams_paths(cams_to_process, mtmc_dir, cont=0, seqmap=seqmap, exp=exp)
    if len(cams_to_process_occ) > 0:  exp.write("  seq2occ:\n")
    handle_cams_paths(cams_to_process_occ, mtmc_occ_dir, cont, seqmap, exp)
    seqmap.close()
    exp.close()

def config_deepstream():
    lines_in = open(workspace_dir + "/configs/DS_configs/config_deepstream.txt", "r").readlines()
    lines_out = []
    for line in lines_in:
        if "num-sources=0" in line:
            line = line.replace("num-sources=0", "num-sources=%d" % (len(cams_to_process) + len(cams_to_process_occ)))
        if "batch-size=0" in line:
            line = line.replace("batch-size=0", "batch-size=%d" % (len(cams_to_process) + len(cams_to_process_occ)))
        if "model-engine-file=models" in line:
            line = line.replace("model-engine-file=models", "model-engine-file=" + workspace_dir + "/models")
            line = line.replace("b000", "b%d" % (len(cams_to_process) + len(cams_to_process_occ)))
        if "label-dir=ground_truth":
            line = line.replace("label-dir=ground_truth", "label-dir=" + out_dataset_path + mot_gt_dir + "/")
        if "file-names=000.txt" in line:
            names = ";" .join(["%03d/gt/gt.txt" % i for i in range(len(cams_to_process) + len(cams_to_process_occ))])
            line = line.replace("file-names=000.txt", "file-names=" + names)
        lines_out.append(line)

    with open(out_dataset_path + "/config_deepstream.txt", "w") as f:
        f.writelines(lines_out)

def config_pgie():
    lines_in = open(workspace_dir + "/configs/DS_configs/config_pgie.txt", "r").readlines()
    lines_out = []
    for line in lines_in:
        if "int8-calib-file=models" in line:
            line = line.replace("int8-calib-file=models", "int8-calib-file=" + workspace_dir + "/models")
        if "labelfile-path=models" in line:
            line = line.replace("labelfile-path=models", "labelfile-path=" + workspace_dir + "/models")
        if "tlt-encoded-model" in line:
            line = line.replace("tlt-encoded-model=models", "tlt-encoded-model=" + workspace_dir + "/models")
        lines_out.append(line)

    with open(out_dataset_path + "/config_pgie.txt", "w") as f:
        f.writelines(lines_out)

def config_tracker():
    lines_in = open(workspace_dir + "/configs/DS_configs/config_tracker.yml", "r").readlines()
    lines_out = []
    for line in lines_in:
        if "terminatedTrackFilename: ":
            line = line.replace("trackDump_Stream_", "trajDumps/trackDump_Stream_")
        if "trajectoryDumpFileName: ": 
            line = line.replace("trajDump_Stream_", "trajDumps/trajDump_Stream_")
        if "modelEngineFile: models" in line:
            line = line.replace("modelEngineFile: models", "modelEngineFile: " + workspace_dir + "/models")
            line = line.replace("b000", "b100")
        if "tltEncodedModel: models" in line:
            line = line.replace("tltEncodedModel: models", "tltEncodedModel: " + workspace_dir + "/models")
        if "camInfo/Warehouse_Synthetic_Cam000.yml" in line:
            linei = line.replace("camInfo/" + cam_file % 0, out_dataset_path + "/camInfo/" + cam_file % 0)
            for i in range(len(cams_to_process) + len(cams_to_process_occ) - 1):
                lines_out.append(linei)
                linei = linei.replace(cam_file % (i), cam_file % (i + 1))
            line = linei
        lines_out.append(line)

    with open(out_dataset_path + "/config_tracker.yml", "w") as f:
        f.writelines(lines_out)    

def main():
    handle_paths()
    config_deepstream()
    config_pgie()
    config_tracker()

if __name__ == "__main__":
    main() # test1