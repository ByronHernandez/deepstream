import os
import cv2
import yaml
import numpy as np

np.set_printoptions(precision=3, suppress=True)

# Independent scales
sx = (1080 - 4) / 110  # Compensating for additional pixels
sy = (1920 - 2) / 196  # Compensating for additional pixels

# Transformation matrix OV to PX
T_ov2px = np.array([[-sx, 0, 48 * sx + 1],
                    [0, sy, 105 * sy + 2],
                    [0, 0, 1]])

T_px2ov = np.linalg.inv(T_ov2px)

cams_path = os.getenv("HOME") + "/Documents/MTMC_Warehouse_Synthetic_012424/camInfo/"
vids_path = os.getenv("HOME") + "/Documents/MTMC_Warehouse_Synthetic_012424/videos/"
map_path = os.getenv("HOME") + "/Documents/deepstream/DS_Perf/workspace_experiment/projections/map.png"

def load_and_process_camera_matrices():
    name_by_cam = cams_path + "Warehouse_Synthetic_Cam%03d.yml"
    cam_matrices = {}
    for cam in range(1, 101):
        with open(name_by_cam % cam, 'r') as file:
            yaml_data = yaml.safe_load(file)
        height = yaml_data['modelInfo']["height"] * 3.28084 # meters to feet (OV)
        P = np.array(yaml_data["projectionMatrix_3x4_w2p"]).reshape(3, 4)
        Q = np.linalg.pinv(P)
        K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
        K = K / K[2, 2]
        # Translation vector in OV (-R.t @ t)
        pos = t[:2] / t[-1]
        # point at 3 OV units (feet) in front of the camera
        end = pos + R[-1:, :2].T * (3 / np.linalg.norm(R[-1, :2]))
        pos_px = cv2.perspectiveTransform(pos.reshape(1, 1, 2), T_ov2px).reshape(2)
        end_px = cv2.perspectiveTransform(end.reshape(1, 1, 2), T_ov2px).reshape(2)
        cam_matrices[cam] = P, Q, K, R, t, pos, end, pos_px, end_px, height
    return cam_matrices

def plot_camera_position(map, cam, cam_matrices):
    _, _, _, R, _, _, _, pos_px, end_px, _ = cam_matrices[cam]
    x, y = pos_px
    dx, dy = end_px
    cv2.circle(map, (int(x), int(y)), 7, (0, 255, 0), -1)
    cv2.putText(map, str(cam), (int(x + 5), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.arrowedLine(map, (int(x), int(y)), (int(dx), int(dy)), (0, 0, 255), 2)
    return map

def create_2x2_mosaic(images):
    if len(images) != 4:
        raise ValueError("Input list must contain exactly four images.") 
    mosaic = np.zeros((2 * images[0].shape[0], 2 * images[0].shape[1], 3), dtype=np.uint8)   
    mosaic[:images[0].shape[0], :images[0].shape[1]] = images[0]
    mosaic[:images[1].shape[0], images[0].shape[1]:] = images[1]
    mosaic[images[0].shape[0]:, :images[2].shape[1]] = images[2]
    mosaic[images[0].shape[0]:, images[2].shape[1]:] = images[3]
    return mosaic

def get_cam_frame_from_video(video_path, frame_num):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not cap.isOpened() or not ret:
        raise ValueError("Error: Could not read first frame.")
    cap.release()
    return frame

def create_camset_mosaic(camset, name):
    frames = []
    for cam in camset:
        name_vid = vids_path + "Warehouse_Synthetic_Cam%03d.mp4" % cam
        frame = get_cam_frame_from_video(name_vid, 0)
        h, w = frame.shape[:2]
        text = "CAM%03d" % cam
        putCenteredText(frame, text, (w // 2, h - 50), (0, 0, 0), 2, 3)
        frames.append(frame)
    mosaic = create_2x2_mosaic(frames)
    mosaic_resized = cv2.resize(mosaic, (mosaic.shape[1] // 2, mosaic.shape[0] // 2))
    print("Saving camset mosaic to output/cam_sets/" + name + ".png")
    cv2.imwrite("output/cam_sets/" + name + ".png", mosaic_resized)

def back_project_z_plane(pos, cam, cam_matrices, z = 0):
    _, Q, _, _, _, Rt, _, _, _ ,_ = cam_matrices[cam]
    Rt = Rt.reshape(3, 1)
    pos = np.array(pos).T
    Qx = Q @ pos
    lam = (z * Qx[3, :] - Qx[2, :]) / (Rt[2, :] - z)
    # Output only x and y
    X = (Qx[:2, :] + lam * Rt[:2, :]) / (lam + Qx[3, :]) 
    # Check: make it until 3 to include z
    # X = (Qx[:3, :] + lam * Rt[:3, :]) / (lam + Qx[3, :]) 
    return X.T

def get_camera_fov_mask(map, cam_calib, num_pix):
    P, _, _, R, _, pos, end, _, _, height = cam_calib
    h, w = map.shape[:2]
    # Create array of coordinates inside the warehouse in pixels (using T_gt2px)
    limits_ov = np.array([[[0, -100]],[[-100, 0]]], dtype=float)
    limits_px = cv2.perspectiveTransform(limits_ov, T_ov2px)
    # Create a grid of points in the image plane
    x_px = np.arange(np.round(limits_px[0, 0, 0]), np.round(limits_px[1, 0, 0]))
    y_px = np.arange(np.round(limits_px[0, 0, 1]), np.round(limits_px[1, 0, 1]))
    xy_px = np.array(np.meshgrid(x_px, y_px), dtype=float).reshape(2, -1)
    # Apply T_px2ov to get the corresponding coordinates in the OV system
    xy_ov = cv2.perspectiveTransform(xy_px.T.reshape(1, -1, 2), T_px2ov).reshape(-1, 2)
    # plot the grid in the map
    xy_px = xy_px.astype(int)
    map2 = map.copy()
    map2[xy_px[1], xy_px[0]] = [100, 150, 50]
    cv2.imwrite("output/grid.png", (0.5 * map + 0.5 * map2).astype(np.uint8))
    # Create 3D points in z=0, z=h/2, z=h
    xyz_0 = cv2.convertPointsToHomogeneous(xy_ov)
    xyz_m = cv2.convertPointsToHomogeneous(xy_ov)
    xyz_h = cv2.convertPointsToHomogeneous(xy_ov)
    xyz_0[:, :, 2], xyz_m[:, :, 2], xyz_h[:, :, 2] = 0, height / 2, height
    # Project to the image plane
    xy0_cam = cv2.perspectiveTransform(xyz_0, P).reshape(-1, 2).T
    xym_cam = cv2.perspectiveTransform(xyz_m, P).reshape(-1, 2).T
    xyh_cam = cv2.perspectiveTransform(xyz_h, P).reshape(-1, 2).T
    # Create the mask
    mask = np.zeros((h, w), dtype=bool)
    # mask1 = (xy0_cam[0] > 0) & (xy0_cam[0] < w)
    # mask2 = (xy0_cam[1] > 0) & (xy0_cam[1] < h)
    mask1 = (xym_cam[0] > 0) & (xym_cam[0] < w) # OK, but... not sure if the height scale is correct
    mask2 = (xym_cam[1] > 0) & (xym_cam[1] < h) # OK, but... not sure if the height scale is correct 
    mask3 = np.linalg.norm(xyh_cam - xy0_cam, ord=np.inf, axis=0) > num_pix # OK
    mask4 = ((xy_ov - pos.T) @ (end - pos)).flatten() > 0                   # OK
    # Update the mask
    mask[xy_px[1], xy_px[0]] = mask1 & mask2 & mask3 & mask4
    cv2.imwrite('mask.png', (mask * 255).astype(np.uint8))
    return np.stack((mask, mask, mask), axis=-1)

def putCenteredText(img, text, coords, color, fontScale=0.5, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, fontScale, thickness)
    text_x = coords[0] - text_size[0] // 2
    text_y = coords[1] + text_size[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, fontScale, color, thickness)

def random_color():
    r = np.random.randint(0, 255)
    g = np.random.randint(0, 255)
    b = np.random.randint(0, 255)
    return r, g, b

def draw_camset_fovs(map, camset, cam_matrices):
    colores = [random_color() for _ in camset]
    masks = []
    maps = []
    for i, cam in enumerate(camset):
        mask = get_camera_fov_mask(map, cam_matrices[cam], num_pix=150)
        masked = mask * np.array(colores[i]).reshape(1, 1, 3)
        maps.append(map - map * mask + masked)
        masks.append(mask)
    map = np.mean(maps + [map], axis=0).astype(np.uint8)
    for i, cam in enumerate(camset):
        text = "CAM%03d" % cam
        mask_indices = np.where(masks[i] > 0)
        centroid_x = int(np.mean(mask_indices[1]))
        centroid_y = int(np.mean(mask_indices[0]))
        putCenteredText(map, text, (centroid_x, centroid_y), colores[i], 0.5, 2)
        map = plot_camera_position(map, cam, cam_matrices)
    return map

if __name__ == "__main__":
    map = cv2.imread(map_path)
    np.random.seed(100)
    cam_matrices = load_and_process_camera_matrices()
    for cam in range(1, 101):
        map = plot_camera_position(map, cam, cam_matrices)
    cv2.imwrite("output/map_cameras.png", map)

    # camsets = [
    #     [1, 38, 48, 74],
    #     [2, 3, 4, 96],
    #     [10, 30, 43, 90],
    #     [9, 40, 49, 51],
    #     [11, 27, 59, 69],
    #     [12, 16, 20, 84]
    # ]
    # camsets = [[16, 20, 49, 49]]
    camsets = [[9, 9, 49, 49]]
    for camset in camsets:
        name = "_".join([str(cam) for cam in camset])
        create_camset_mosaic(camset, name)
        map = cv2.imread(map_path)
        map = draw_camset_fovs(map, camset, cam_matrices)
        cv2.imwrite("output/cam_sets/map_" + name + ".png", map)
