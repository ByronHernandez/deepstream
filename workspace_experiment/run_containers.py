import os
import yaml
import subprocess
from multiprocessing import Pool

# define names
root = "/home/bhernandez/Documents/deepstream/workspace_experiment"
app_config = "config_deepstream%s.txt"
trk_config = "config_tracker%s.yml"
cam_file = "dataset/camInfo/Warehouse_Synthetic_Cam%03d.yml"
out_dmp_file = "trajDump_Stream%s_%s"
inp_vid_file = "file://dataset/videos/Warehouse_Synthetic_Cam%03d.mp4"
out_vid_file = "outVideos/out_Warehouse_Synthetic_Cam%03d.mp4"

docker_image_version = "ds-24-06-07:7.0.0-build-tot"
cams_to_process = [1] #[1, 38, 48, 74]

# define model parameters in CamInfo yaml files
height = 1.6
radius = 0.2

def replace_CamInfo_yaml(cam_yaml, height, radius):
    with open(cam_yaml, "r") as file:
        cam_yaml_data = yaml.load(file, Loader=yaml.FullLoader)
    cam_yaml_data["modelInfo"]["height"] = height
    cam_yaml_data["modelInfo"]["radius"] = radius
    elements = ["projectionMatrix_3x4_w2p", "modelInfo"]
    # write the yaml file
    with open(cam_yaml, "w") as file:
        for element in elements:
            yaml.dump({element: cam_yaml_data[element]}, file, default_flow_style=False)
            if element != elements[-1]: file.write("\n")

def get_docker_command():
    return ['docker', 'run', '-t', '--rm', '--privileged', '--network', 'host', '--gpus', 'all', '--ipc=host', 
            '--cap-add=SYS_PTRACE', '--shm-size=8g', '--security-opt', 'seccomp=unconfined',
            '-e', 'HOME=/home/bhernandez',
            '-e', 'DISPLAY=$DISPLAY',
            '-e', 'P4ROOT=$P4ROOT',
            '-v', '/tmp/.X11-unix/:/tmp/.X11-unix',
            '-v', '/home/bhernandez:/home/bhernandez',
            '-w', root, docker_image_version]

def run_app(cam, inp_vid, out_vid, dmp_file, log_file):
    replace_CamInfo_yaml(cam_file % cam, height=height, radius=radius)
    # make a copy of the config files contents
    with open(app_config % "", "r") as file: app = file.read() 
    with open(trk_config % "", "r") as file: trk = file.read()
    # Create names for the temporal config files
    app_file = app_config % ("_cam%d" % cam)
    trk_file = trk_config % ("_cam%d" % cam)
    # Update the tracker config file contents
    trk = trk.replace(cam_file % 0, cam_file % cam)
    trk = trk.replace(out_dmp_file % ("", ""), out_dmp_file % ("_cam%d" % cam, ""))
    # Update the app config file contents
    app = app.replace(trk_config % "", trk_file)
    app = app.replace(inp_vid_file % 0, inp_vid)
    app = app.replace(out_vid_file % 0, out_vid)
    # create the temporal config files
    with open(app_file, "w") as file: file.write(app + "\n")
    with open(trk_file, "w") as file: file.write(trk + "\n")

    run_ds = get_docker_command()
    # print('deepstream command:', run_ds)
    with open(log_file, "w") as log:
        subprocess.run(run_ds + ["deepstream-app", "-c", app_file], stdout=log, stderr=log)
        curr_dmp = out_dmp_file % ("_cam%d" % cam, "0.txt")
        subprocess.run(run_ds + ["chmod", "666", curr_dmp], stdout=log, stderr=log)
    os.replace(curr_dmp, dmp_file)
    os.remove(app_file), os.remove(trk_file)

def process_video(cam):
    print("Running pipeline for camera %d..." % cam)
    dmp_file = "trajDumps/" + out_dmp_file % ("", "%d.txt" % cam)
    inp_vid = inp_vid_file % cam
    out_vid = out_vid_file % cam
    log_file = "ds_logs/logs_cam_%03d.txt" % cam
    run_app(cam, inp_vid, out_vid, dmp_file, log_file)
    return "Done cam %d" % cam

def run_app_for_set(cam_set):
    os.makedirs("ds_logs", exist_ok=True)
    os.makedirs("trajDumps", exist_ok=True)
    os.makedirs("outVideos", exist_ok=True)
    # for cam in cam_set: print(process_video(cam))
    with Pool(4) as p:
        print(p.map(process_video, cam_set))
    yaml.dump({"height": height, "radius": radius}, 
              open("trajDumps/parameters.yml", "w"), default_flow_style=False)
    yaml.dump({"height": height, "radius": radius}, 
              open("outVideos/parameters.yml", "w"), default_flow_style=False)

if __name__ == "__main__":
    run_app_for_set(cams_to_process)
