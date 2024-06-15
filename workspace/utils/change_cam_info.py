import os
import tqdm
import yaml

cams_to_process = list(range(1, 101))
cam_file = os.getenv("HOME") + "/Documents/deepstream/workspace/datasets/orig/camInfo/Warehouse_Synthetic_Cam%03d.yml"

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

if __name__ == "__main__":
    for cam in tqdm.tqdm(cams_to_process, desc="Processing cameras: "):
        replace_CamInfo_yaml(cam_file % cam, height=height, radius=radius)
    print("All cameras processed")
