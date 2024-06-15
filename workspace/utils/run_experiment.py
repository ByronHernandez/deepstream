import os
import yaml
import subprocess

workspace_dir = "/home/bhernandez/Documents/deepstream/workspace"
experiment_dir = workspace_dir + "/experiments/exp002cam060s"

docker_image_version = "ds-24-06-11:7.1-mv-assoc"

def get_docker_command():
    return ['docker', 'run', '-t', '--rm', '--privileged', '--network', 'host', '--gpus', 'all', '--ipc=host', 
            '--cap-add=SYS_PTRACE', '--shm-size=8g', '--security-opt', 'seccomp=unconfined',
            '-e', 'HOME=/home/bhernandez',
            '-e', 'DISPLAY=$DISPLAY',
            '-e', 'P4ROOT=$P4ROOT',
            '-v', '/tmp/.X11-unix/:/tmp/.X11-unix',
            '-v', '/home/bhernandez:/home/bhernandez',
            '-w', experiment_dir, docker_image_version]

def main():
    os.makedirs("outVideos", exist_ok=True)
    os.makedirs("trajDumps", exist_ok=True)
    os.chdir(experiment_dir)
    with open(experiment_dir + "/experiment.yml", "r") as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
    if "run_with" in yaml_data:
        print("The experiment has already been run with the docker image version: " + yaml_data["run_with"])
        return
    yaml_data["run_with"] = docker_image_version
    with open(experiment_dir + "/experiment.yml", "a") as file:
        file.write("run_with: " + docker_image_version + "\n")
    run_ds = get_docker_command()
    with open("run_deepstream_app.log", "w") as log:
        subprocess.run(run_ds + ["deepstream-app", "-c", "config_deepstream.txt"], stdout=log, stderr=log)
        subprocess.run(run_ds + ["chmod", "-R", "777", "trajDumps", "outVideos"], stdout=log, stderr=log)

if __name__ == "__main__":
    main()
