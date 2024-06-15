import os
import cv2
import tqdm
import yaml
import numpy as np
import draw_fovs_for_camera_set as draw_fovs

np.set_printoptions(precision=3, suppress=True)

workspace_path = os.getenv("HOME") + "/Documents/deepstream/workspace/"
proj_path = workspace_path + "projections/"
map_path = proj_path + "map.png"

roi_ov = [-50, 0, -100, -100] # [x1, y1, x2, y2]

# Independent scales
sx = (1080 - 4) / 110  # Compensating for additional pixels
sy = (1920 - 2) / 196  # Compensating for additional pixels

# Transformation matrix OV to PX
T_ov2px = np.array([[-sx, 0, 48 * sx + 1],
                    [0, sy, 105 * sy + 2],
                    [0, 0, 1]])

T_px2ov = np.linalg.inv(T_ov2px)


def get_roi_mask(mask):
    pts = np.array(roi_ov, dtype=float).reshape(1, -1, 2)
    pts_px = cv2.perspectiveTransform(pts, T_ov2px).reshape(-1, 2)
    pts_px = np.round(pts_px).astype(int)
    cv2.rectangle(mask, tuple(pts_px[0]), tuple(pts_px[1]), (1, 1, 1), -1)
    cv2.imshow("mask", (mask * 255).astype(np.uint8))
    return mask

def main():
    map = cv2.imread(map_path)
    roi_mask = get_roi_mask(np.zeros_like(map, dtype=np.uint8))
    print("Saving ROI mask to %s" % proj_path + "output/roi_mask.png")
    cv2.imwrite(proj_path + "output/roi_mask.png", (roi_mask * 255).astype(np.uint8))
    cam_matrices = draw_fovs.load_and_process_camera_matrices()
    outcams = []
    for cam in tqdm.tqdm(range(1, 101), desc="Checking cameras"):
        fov_mask = draw_fovs.get_camera_fov_mask(map, cam_matrices[cam], num_pix=100).astype(np.uint8)
        # cv2.imshow("mask", (roi_mask * fov_mask * 255).astype(np.uint8))
        # key = cv2.waitKey(2000)
        # if key == ord('q'): break
        if (fov_mask * roi_mask).sum() / fov_mask.sum() > 0.5:
            outcams.append(cam)
    draw_fovs.camsets = [outcams]
    draw_fovs.main()


if __name__ == "__main__":
    main()