import os
import yaml
import shutil
import subprocess

import numpy as np

FPS = 30

class StringLoader(yaml.SafeLoader):
    pass

def construct_yaml_str(loader, node):
    return loader.construct_scalar(node)

for scalar_type in ['float', 'int', 'bool', 'null']:
    StringLoader.add_constructor(f'tag:yaml.org,2002:{scalar_type}', construct_yaml_str)

class MTMC_Analyzer():
    def __init__(self, home_dir, cams_orig, cams_occ, max_frames, pgie, mv):
        cams_orig, cams_occ = list(cams_orig), list(cams_occ)
        self.home_dir = home_dir
        self.cams_orig = list(cams_orig)
        self.cams_occ = list(cams_occ)
        self.max_frames = 360 * FPS if max_frames == -1 else max_frames
        self.pgie = "tpgie" if pgie else "fpgie"
        self.occ = "orig" if len(cams_occ) == 0 else "occ"
        self.mv = "mv" if mv else "sv"
        self.workspace_dir = home_dir + "/Documents/deepstream/workspace"
        self.docker_image_version = "ds-24-06-11:7.1-mv-assoc" if mv else "ds-24-06-11:7.1-fake-pgie"
        self.experiment_name = f"exp{len(cams_orig + cams_occ):03d}cam{self.max_frames // FPS}s_{self.occ}_{self.pgie}_{self.mv}"  
        self.experiment_dir = self.workspace_dir + "/experiments/" + self.experiment_name   # out_dataset_path
        self.mtmc_orig_dir = self.workspace_dir + "/datasets/orig/"
        self.mtmc_occ_dir = self.workspace_dir + "/datasets/occ/"
        self.mot_gt_dir = "ground_truth"                                                    # gt_ordered_dir
        self.mot_gt_eval_dir = "mot_gt"                                                     # mot_gt_4eval_dir
        self.mot_out_eval_dir = "mot_out"
        self.cam_file = "Warehouse_Synthetic_Cam%03d.yml"
        self.vid_file = "Warehouse_Synthetic_Cam%03d.mp4"

    @staticmethod
    def replace_CamInfo_yaml(cam_list, cam_file, height=1.6, radius=0.2):
        for cam in cam_list:
            with open(cam_file % cam, "r") as file:
                cam_yaml_data = yaml.load(file, Loader=yaml.FullLoader)
            cam_yaml_data["modelInfo"]["height"] = height
            cam_yaml_data["modelInfo"]["radius"] = radius
            elements = ["projectionMatrix_3x4_w2p", "modelInfo"]
            with open(cam_file % cam, "w") as file:
                for element in elements:
                    yaml.dump({element: cam_yaml_data[element]}, file, default_flow_style=False)
                    if element != elements[-1]: file.write("\n")

    def get_docker_command(self):
        return ['docker', 'run', '-t', '--rm', '--privileged', '--network', 'host', '--gpus', 'all', '--ipc=host', 
                '--cap-add=SYS_PTRACE', '--shm-size=8g', '--security-opt', 'seccomp=unconfined',
                '-e', 'HOME=/home/bhernandez',
                '-e', 'DISPLAY=$DISPLAY',
                '-e', 'P4ROOT=$P4ROOT',
                '-v', '/tmp/.X11-unix/:/tmp/.X11-unix',
                '-v', '/home/bhernandez:/home/bhernandez',
                '-w', self.experiment_dir, self.docker_image_version]
    
    # def handle_cams_paths(self, cams, cams_dir, cont, exp, seqmap=0):
    #     for cam in cams:
    #         os.symlink(cams_dir + "/camInfo/" + self.cam_file % cam, self.experiment_dir + "/camInfo/" + self.cam_file % cont)
    #         if self.max_frames == 360 * FPS:
    #             os.symlink(cams_dir + "/videos/" + self.vid_file % cam, self.experiment_dir + "/videos/" + self.vid_file % cont)
    #             os.symlink(cams_dir + self.mot_gt_dir + "/%03d" % cam, self.experiment_dir + "/" + self.mot_gt_dir + "/%03d" % cont) # TODO: replace
    #         else:
    #             subprocess.run(["ffmpeg", "-i", 
    #                             cams_dir + "/videos/" + vid_file % cam,
    #                             "-t", f'{max_frames // FPS}', "-c", "copy", 
    #                             self.experiment_dir + "/videos/" + vid_file % cont
    #                             ])
    #             os.makedirs(self.experiment_dir + self.mot_gt_dir + "/%03d/gt" % cont, exist_ok=True)
    #             file_inp = open(cams_dir + self.mot_gt_dir + "/%03d/gt/gt.txt" % cam, "r")
    #             file_out = open(self.experiment_dir + self.mot_gt_dir + "/%03d/gt/gt.txt" % cont, "w")
    #             file_seq = open(self.experiment_dir + self.mot_gt_dir + "/%03d/seqinfo.ini" % cont, "w")
    #             for line in file_inp.readlines():
    #                 frame = int(line.split(",")[0])
    #                 if frame > max_frames: break
    #                 file_out.write(line)
    #             file_seq.write("[Sequence]\nseqLength=%d\n" % max_frames)
    #             file_inp.close()
    #             file_out.close()
    #             file_seq.close()
    #         exp.write("    %03d: %03d\n" % (cont, cam))
    #         cont += 1
    #     return cont

    def handle_cams_paths(self, cams, cams_dir, cont, exp, seqmap=0):
        for cam in cams:
            os.symlink(cams_dir + "/camInfo/" + self.cam_file % cam, self.experiment_dir + "/camInfo/" + self.cam_file % cont)
            if self.max_frames >= 360 * FPS:
                os.symlink(cams_dir + "/videos/" + self.vid_file % cam, self.experiment_dir + "/videos/" + self.vid_file % cont)
            else:
                subprocess.run(["ffmpeg", "-i", 
                                cams_dir + "/videos/" + self.vid_file % cam,
                                "-t", f'{self.max_frames // FPS}', "-c", "copy", 
                                self.experiment_dir + "/videos/" + self.vid_file % cont
                                ])
            exp.write("    %03d: %03d\n" % (cont, cam))
            
            os.makedirs(self.experiment_dir + '/' + self.mot_gt_dir + "/%03d/gt" % cont, exist_ok=True)
            file_inp = open(cams_dir + '/' + self.mot_gt_dir + "/%03d/gt/gt.txt" % cam, "r")
            file_out = open(self.experiment_dir + '/' + self.mot_gt_dir + "/%03d/gt/gt.txt" % cont, "w")
            file_seq = open(self.experiment_dir + '/' + self.mot_gt_dir + "/%03d/seqinfo.ini" % cont, "w")
            for line in file_inp.readlines():
                frame, idp, x, y, w, h, x_feet, y_feet, conf = line.split(',') 
                if int(frame) > self.max_frames: break
                x, y, w, h = map(float, [x, y, w, h])
                if w < 20 and h < 20: continue # Filter bounding boxes by size
                if cams_dir == self.mtmc_orig_dir:
                    file_out.write(line)
                else: # it is an occlusion camera
                    # trim the bounding box to the occlusion area
                    x1, x2, y1, y2 = x, x + w, y, y + h
                    # TODO: more level of automation here
                    if x1 >= 1920 / 3 and x2 <= 1920 * 2 / 3 and y1 >= 1080 / 3 and y2 <= 1080 * 2 / 3: continue
                    # Adjust the bounding box
                    if x1 < 1920 * 1 / 3 < x2: x2 = int(np.round(1920 * 1 / 3))
                    if x1 < 1920 * 2 / 3 < x2: x1 = int(np.round(1920 * 2 / 3))
                    if y1 < 1080 * 1 / 3 < y2: y2 = int(np.round(1080 * 1 / 3))
                    if y1 < 1080 * 2 / 3 < y2: y1 = int(np.round(1080 * 2 / 3))
                    w, h = x2 - x1, y2 - y1
                    if w < 20 and h < 20: continue # Filter bounding boxes by size
                    file_out.write("%s,%s,%d,%d,%d,%d,%s,%s,%s" % (frame, idp, x1, y1, w, h, x_feet, y_feet, conf))

            file_seq.write("[Sequence]\nseqLength=%d\n" % self.max_frames)
            file_inp.close()
            file_out.close()
            file_seq.close()

            cont += 1
        return cont

    def handle_dataset_paths(self):
        shutil.rmtree(self.experiment_dir, ignore_errors=True)
        os.makedirs(self.experiment_dir + "/videos", exist_ok=True)
        os.makedirs(self.experiment_dir + "/camInfo", exist_ok=True)
        os.makedirs(self.experiment_dir + "/trajDumps", exist_ok=True)
        os.makedirs(self.experiment_dir + "/outVideos", exist_ok=True)
        os.makedirs(self.experiment_dir + "/outMVReAssoc", exist_ok=True)
        os.makedirs(self.experiment_dir + "/outCommunicator", exist_ok=True)
        os.makedirs(self.experiment_dir + "/kitti_detector", exist_ok=True)
        os.makedirs(self.experiment_dir + "/kitti_tracker", exist_ok=True)
        os.makedirs(self.experiment_dir + "/evalLogs", exist_ok=True)
        os.makedirs(self.experiment_dir + "/" + self.mot_gt_dir, exist_ok=True)
        os.makedirs(self.experiment_dir + "/" + self.mot_gt_eval_dir, exist_ok=True)
        os.makedirs(self.experiment_dir + "/" + self.mot_out_eval_dir, exist_ok=True)
        exp = open(self.experiment_dir + "/experiment.yml", "w")
        exp.write("seq2cam:\n")
        if len(self.cams_orig) > 0: exp.write("  seq2orig:\n")
        cont = self.handle_cams_paths(self.cams_orig, self.mtmc_orig_dir, cont=0, exp=exp)
        if len(self.cams_occ) > 0:  exp.write("  seq2occ:\n")
        cont = self.handle_cams_paths(self.cams_occ, self.mtmc_occ_dir, cont=cont, exp=exp)
        exp.close()

    def config_deepstream(self):
        lines_in = open(self.workspace_dir + "/configs/DS_configs/config_deepstream.txt", "r").readlines()
        if self.pgie == "tpgie":
            lines_in[lines_in.index("[primary-gie]\n") + 1] = "enable=1\n"
        elif self.pgie == "fpgie":
            lines_in[lines_in.index("[loadlabels]\n") + 1] = "enable=1\n"
        lines_out = []
        for line in lines_in:
            if "num-sources=0" in line:
                line = line.replace("num-sources=0", "num-sources=%d" % (len(self.cams_orig) + len(self.cams_occ)))
            if "batch-size=0" in line:
                line = line.replace("batch-size=0", "batch-size=%d" % (len(self.cams_orig) + len(self.cams_occ)))
            if "model-engine-file=models" in line:
                line = line.replace("model-engine-file=models", "model-engine-file=" + self.workspace_dir + "/models")
                line = line.replace("b000", "b%d" % (len(self.cams_orig) + len(self.cams_occ)))
            if "label-dir=ground_truth":
                line = line.replace("label-dir=ground_truth", "label-dir=%s/%s/" % (self.experiment_dir, self.mot_gt_dir))
            if "file-names=000.txt" in line:
                names = ";" .join(["%03d/gt/gt.txt" % i for i in range(len(self.cams_orig) + len(self.cams_occ))])
                line = line.replace("file-names=000.txt", "file-names=" + names)
            lines_out.append(line)
        with open(self.experiment_dir + "/config_deepstream.txt", "w") as f:
            f.writelines(lines_out)

    def config_pgie(self):
        lines_in = open(self.workspace_dir + "/configs/DS_configs/config_pgie.txt", "r").readlines()
        lines_out = []
        for line in lines_in:
            if "int8-calib-file=models" in line:
                line = line.replace("int8-calib-file=models", "int8-calib-file=" + self.workspace_dir + "/models")
            if "labelfile-path=models" in line:
                line = line.replace("labelfile-path=models", "labelfile-path=" + self.workspace_dir + "/models")
            if "tlt-encoded-model" in line:
                line = line.replace("tlt-encoded-model=models", "tlt-encoded-model=" + self.workspace_dir + "/models")
            lines_out.append(line)
        with open(self.experiment_dir + "/config_pgie.txt", "w") as f:
            f.writelines(lines_out)

    def config_tracker(self):
        lines_in = open(self.workspace_dir + "/configs/DS_configs/config_tracker.yml", "r").readlines()
        lines_out = []
        for line in lines_in:
            if "multiViewAssociatorType" in line:
                line = line.replace("multiViewAssociatorType: 0", "multiViewAssociatorType: %d" % (1 if self.mv == "mv" else 0))
            if "communicatorType" in line:
                line = line.replace("communicatorType: 0", "communicatorType: %d" % (1 if self.mv == "mv" else 0))
            if "terminatedTrackFilename" in line:
                line = line.replace("trackDump_Stream_", self.experiment_dir + "/trajDumps/trackDump_Stream_")
            if "trajectoryDumpFileName": 
                line = line.replace("trajDump_Stream_", self.experiment_dir + "/trajDumps/trajDump_Stream_")
            if "modelEngineFile: models" in line:
                line = line.replace("modelEngineFile: models", "modelEngineFile: " + self.workspace_dir + "/models")
                line = line.replace("b000", "b100")
            if "tltEncodedModel: models" in line:
                line = line.replace("tltEncodedModel: models", "tltEncodedModel: " + self.workspace_dir + "/models")
            if "camInfo/Warehouse_Synthetic_Cam000.yml" in line:
                linei = line.replace("camInfo/" + self.cam_file % 0, self.experiment_dir + "/camInfo/" + self.cam_file % 0)
                for i in range(len(self.cams_orig) + len(self.cams_occ) - 1):
                    lines_out.append(linei)
                    linei = linei.replace(self.cam_file % (i), self.cam_file % (i + 1))
                line = linei
            if "cameraIDs: " in line:
                clist = ", ".join([str(cam) for cam in self.cams_orig])
                clist += ", " + ", ".join([str(cam + 1000) for cam in self.cams_occ])
                line = line.replace("cameraIDs: [0]", "cameraIDs: [" + clist.strip(", ") + "]")
            lines_out.append(line)
        with open(self.experiment_dir + "/config_tracker.yml", "w") as f:
            f.writelines(lines_out) 

    def handle_seq_paths(self, seq, cam, comp):
        try:
            os.symlink( "%s/%s/%03d" % (self.experiment_dir, self.mot_gt_dir, int(seq)),
                        "%s/%s/%03d%s" % (self.experiment_dir, self.mot_gt_eval_dir, int(cam), comp))
        except FileExistsError: print("File '%03d%s' already exists" % (int(cam), comp))
        # convert trajDump files to MOTChallenge format
        traj_dump = "%s/trajDumps/trajDump_Stream_%d.txt" % (self.experiment_dir, int(seq))
        mot_out = "%s/%s/%03d%s.txt" % (self.experiment_dir, self.mot_out_eval_dir, int(cam), comp)
        with open(traj_dump, "r") as f:
            lines = f.readlines()
        with open(mot_out, "w") as f:
            for line in lines:
                frame, id, x, y, w, h, conf = line.split(',')[:7]
                f.write("%s,%s,%s,%s,%s,%s,%s,-1,-1,-1\n" % (frame, id, x, y, w, h, conf))
    
    def handle_eval_paths(self):
        with open(self.experiment_dir + "/experiment.yml", "r") as f:
            experiment = yaml.load(f.read(), Loader=StringLoader)
        seqmap = open(self.experiment_dir + "/seqmap.txt", "w")
        for seq, cam in experiment["seq2cam"].get("seq2orig", {}).items():
            seqmap.write("%03d\n" % int(cam))
            self.handle_seq_paths(seq, cam, "")
            
        for seq, cam in experiment["seq2cam"].get("seq2occ", {}).items():
            seqmap.write("%03d_occ\n" % int(cam))
            self.handle_seq_paths(seq, cam, "_occ")
        seqmap.close()

    def create_dataset(self, hard=False):
        if os.path.exists(self.experiment_dir + "/experiment.yml"):
            if hard:
                print("The dataset %s already exists, overwriting the dataset" % self.experiment_name)
            else: 
                print("The dataset %s already exists, use hard=True to overwrite" % self.experiment_name)
                return
        self.handle_dataset_paths()
        self.config_deepstream()
        self.config_tracker()
        self.config_pgie()

    def run_app(self, hard=False):
        os.chdir(self.experiment_dir)
        with open(self.experiment_dir + "/experiment.yml", "r") as file:
            yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        if "run_with" in yaml_data:
            if hard:
                print("Overwriting the experiment")
            else:
                print("The experiment %s has already been run with the docker image version: %s, use hard=True to re-run" % 
                        (self.experiment_name, yaml_data["run_with"]))
                return
        yaml_data["run_with"] = self.docker_image_version
        with open(self.experiment_dir + "/experiment.yml", "a") as file:
            file.write("run_with: " + self.docker_image_version + "\n")
        run_ds = self.get_docker_command()
        with open("run_deepstream_app.log", "w") as log:
            subprocess.run(run_ds + ["deepstream-app", "-c", "config_deepstream.txt"], stdout=log, stderr=log)
            subprocess.run(run_ds + ["chmod", "-R", "777", self.experiment_dir], stdout=log, stderr=log)

    def run_evaluation(self, hard=False):
        os.chdir(self.experiment_dir)
        with open(self.experiment_dir + "/experiment.yml", "r") as file:
            yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        if "evaluated" in yaml_data:
            if hard:
                print("Overwriting the evaluation")
            else:
                print("The evaluation for %s has already been run, use hard=True to re-evaluate" % self.experiment_name) 
                return
        yaml_data["evaluated"] = True
        with open(self.experiment_dir + "/experiment.yml", "a") as file:
            file.write("evaluated: True\n")
        self.handle_eval_paths()
        # Run the evaluation script
        evaluate_cmd = self.workspace_dir + "/py-motmetrics/motmetrics/apps/evaluateTracking.py"
        seqmap_path = self.experiment_dir + "/seqmap.txt"
        log_path = self.experiment_dir + "/evalLogs"
        log = open("run_eval.log", "w")
        subprocess.run(["python", evaluate_cmd, self.mot_gt_eval_dir, self.mot_out_eval_dir, seqmap_path, 
                                    "--loglevel", "debug", "--fmt", "mot16", "--log", log_path], stdout=log, stderr=log)

if __name__ == '__main__':
    home_dir = os.getenv("HOME") # "/home/bhernandez" # os.getenv("HOME")

    # cam_file = home_dir + "/Documents/deepstream/workspace/datasets/orig/camInfo/Warehouse_Synthetic_Cam%03d.yml"
    # MTMC_Analyzer.replace_CamInfo_yaml(cam_list=list(range(1, 101)), cam_file=cam_file, height=1.6, radius=0.2)

    analyzers = [    
        MTMC_Analyzer(home_dir=home_dir, cams_orig=[1, 48], cams_occ=[38, 74], max_frames=-1, pgie=True, mv=False),
        MTMC_Analyzer(home_dir=home_dir, cams_orig=[1, 48], cams_occ=[38, 74], max_frames=-1, pgie=True, mv=True),
        MTMC_Analyzer(home_dir=home_dir, cams_orig=[1, 48], cams_occ=[38, 74], max_frames=-1, pgie=False, mv=False),
        MTMC_Analyzer(home_dir=home_dir, cams_orig=[1, 48], cams_occ=[38, 74], max_frames=-1, pgie=False, mv=True),
        MTMC_Analyzer(home_dir=home_dir, cams_orig=[1, 38, 48, 74], cams_occ=[], max_frames=-1, pgie=True, mv=False),
        MTMC_Analyzer(home_dir=home_dir, cams_orig=[1, 38, 48, 74], cams_occ=[], max_frames=-1, pgie=True, mv=True),
        MTMC_Analyzer(home_dir=home_dir, cams_orig=[1, 38, 48, 74], cams_occ=[], max_frames=-1, pgie=False, mv=False),
        MTMC_Analyzer(home_dir=home_dir, cams_orig=[1, 38, 48, 74], cams_occ=[], max_frames=-1, pgie=False, mv=True),
        # MTMC_Analyzer(home_dir=home_dir, cams_orig=range(1, 101), cams_occ=[], max_frames=-1, pgie=False, mv=False),
        # MTMC_Analyzer(home_dir=home_dir, cams_orig=range(1, 101), cams_occ=[], max_frames=-1, pgie=False, mv=True),
        # MTMC_Analyzer(home_dir=home_dir, cams_orig=range(1, 101), cams_occ=[], max_frames=-1, pgie=True, mv=False),
        # MTMC_Analyzer(home_dir=home_dir, cams_orig=range(1, 101), cams_occ=[], max_frames=-1, pgie=True, mv=True)
    ]

    for analyzer in analyzers:
        print("Creating dataset %s ..." % analyzer.experiment_name)
        analyzer.create_dataset(hard=True)
        print("Running deepstream app %s ..." % analyzer.experiment_name)
        analyzer.run_app(hard=False)
        print("Running evaluation %s ..." % analyzer.experiment_name)
        analyzer.run_evaluation(hard=False)
