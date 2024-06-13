import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

np.set_printoptions(precision=6, suppress=True)

# Points are on the ground plane in OV for Warehouse scene
ov_list = [(-9.07319, -31.2628), (-31.5262, -37.8624), (-9.24796, -91.2252), 
           (-10.965, -90.8196), (-90.8524, -70.7123), (-90.9901, -89.2896), 
           (-90.7814, -70.8201), (-90.9873, -49.2912), (-91.1306, -51.2141), 
           (-48.9115, -11.244), (-50.7874, -10.9257), (-30.3571, -34.8974), 
           (-28.9662, -40.399), (-29.8498, -32.3434), (-19.3901, -24.8779), 
           (-39.5281, -77.375), (-30.0726, -74.5224), (-49.0567, -51.5252), 
           (-9.37159, -48.9889), (-70.9398, -50.8573), (-71.03, -49.0856)]

# Points are in CAD-like 2D Warehouse global map in the pixel coordinate system (origin is at top-left)
floor_list = [(558, 725), (782,658), (558, 139), (574, 143), (1359, 336), (1361,156), (1362, 338), 
              (1362, 548), (1362, 530), (950,923), (969, 923), (765, 693), (757, 633), (765, 717), 
              (659,790), (852, 274), (767, 300), (951, 526), (559, 548), (1164, 532), (1165, 548)]

# independent scales
sx = (1080 - 4) / 110 # compensating for additional pixels
sy = (1920 - 2) / 196 # compensating for additional pixels
# sy = sx
print("Scale = (%.6f, %.6f)" % (sx, sy))

s = (sx + sy) / 2
# s = 9.871
print("Scale = %.6f" % s)

def get_transform(pts_source, pts_target, transform_type='homography'):
    '''
    Types:  homography: 8 dof (full projective)
            affine:     6 dof (full scale + translation)
            similar:    4 dof (sRt)
            st:         3 dof (st: no rotation)
    '''
    pts_source = np.array(pts_source)
    pts_target = np.array(pts_target)
    if transform_type == 'homography': # Works fine
        T = cv2.findHomography(pts_source, pts_target)[0]
    elif transform_type == 'affine':  # Works fine
        T = np.eye(3)
        T[:2, :]  = cv2.estimateAffine2D(pts_source, pts_target)[0]
    elif transform_type == 'similar': # Does not work well,
        T = np.eye(3)                 # since no rotation can mirror only one axis
        T[:2, :] = cv2.estimateAffinePartial2D(pts_source, pts_target)[0]
    elif transform_type == 'st':
        M, b = [], []
        for x, y in zip(pts_source, pts_target):
            M.extend([[x[0], 0, 1, 0], [0, x[1], 0, 1]])
            b.extend(y.tolist())
        st = np.linalg.pinv(M) @ b
        T = np.array([[st[0], 0, st[2]], [0, st[1], st[3]], [0, 0, 1]])
    return T

def perspective_transform(pts, T):
    pts = np.float32(pts).reshape(-1, 1, 2)
    pts_new = cv2.perspectiveTransform(pts, T)
    pts_new = pts_new.squeeze(1)
    return pts_new.astype(int).tolist()

# For ground-truth result, the format is (9)  <frame>, <id>, <bb_left>,
# <bb_top>, <bb_width>, <bb_height>, <conf>, <obj_class>, <visibility>

def read_dump(file_path):
    f = open(file_path)
    trajs = {}
    for line in f.readlines():
        fields = line[:-1].split(',')
        t = int(fields[0]) - 1
        id = int(fields[1])
        x = float(fields[7])
        y = float(fields[8])
        if id not in trajs: trajs[id] = [[t, x, y]]
        else: trajs[id].append([t, x, y])
    return trajs

# For tracking result, the format is (10)     <frame>, <id>, <bb_left>,
# <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
#
# For extended tracking result (with SV3DT), the format is (15) <frame>, <id>, <bb_left>,
# <bb_top>, <bb_width>, <bb_height>, <conf>, <foot_world_x>, <foot_world_y>, <foot_world_z>,
# -<obj_class>, <visibility>,<foot_img_x>,<foot_img_y>,<convex hull>

def read_gt(file_path):
    f = open(file_path)
    xmin, ymin, xmax, ymax = np.inf, np.inf, 0, 0
    trajs = {}
    maxmins = {'xmin':"", 'xmax':"", 'ymin':"", 'ymax':""}
    for line in f.readlines():
        fields = line[:-1].split(' ')
        id = int(fields[1])
        t = int(fields[2]) - 1
        x = float(fields[7])
        y = float(fields[8])
        if id not in trajs: trajs[id] = {}
        else: trajs[id][t] = [t, x, y]
        if x > xmax: 
            xmax = x
            maxmins['xmax'] = line
        if x < xmin: 
            xmin = x 
            maxmins['xmin'] = line
        if y > ymax: 
            ymax = y
            maxmins['ymax'] = line
        if y < ymin: 
            ymin = y
            maxmins['ymin'] = line
    trajs_out = {}
    for val in maxmins.values(): print(val, end="")
    print("x:", (xmin, xmax), "y: ", (ymin, ymax))
    for id in trajs.keys():
        trajs_out[id] = []
        for frame in sorted(list(trajs[id].keys())):
            trajs_out[id].append(trajs[id][frame])
    return trajs_out

def random_color():
    r = np.random.randint(0, 255)
    g = np.random.randint(0, 255)
    b = np.random.randint(0, 255)
    return r, g, b

def draw_coords(coords, names, map, img_name, do_round=False, render=False):
    if do_round:
        names = [(round(x), round(y)) for x, y in names]

    points = np.array(coords) + np.array([10, -10])
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    for i in range(len(points)):
        for j in range(len(points)):
            if np.abs(diff[i, j, 0]) < 50 and np.abs(diff[i, j, 1]) < 15 and i > j:
                if diff[i, j, 1] >= 0: points[i, 1] += 20 - diff[i, j, 1]
    points = points.tolist()

    for coord, point, name in zip(coords, points, names):
        name = str(name)
        # cv2.circle(map, coord, 5, (0, 255, 0), -1)
        cv2.drawMarker(map, coord,  (0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2, line_type=16)
        cv2.putText(map, name, point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imwrite("output/coord_tests/%s.png" % img_name, map)
    if render:
        cv2.imshow('Anns', map)
        cv2.waitKey(0)

def draw_trajs(trajs, map, T=np.eye(3)):
    ids = sorted(list(trajs.keys()))
    for idi in ids:
        traji = trajs[idi]
        x, y = map.shape[:2]
        color = random_color()
        for pt in traji:
            _, x, y = pt
            pt = perspective_transform([[x, y]], T)
            x, y = pt[0][0], pt[0][1]
            cv2.circle(map, (x, y), 2, color, -1)
        cv2.putText(map, str(idi), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('map', map)
        key = cv2.waitKey(0)
        if key == ord('q'): break
    cv2.destroyWindow('map')
        

if __name__ == "__main__":
    map = cv2.imread('map.png')
    h, w, _ = map.shape # 1080, 1920, 3

    # To observe the corners
    # Make plt interactive
    # plt.ion()
    # img = plt.imread("map.png")
    # input("Press Enter to continue...")
    # plt.imshow(img, interpolation="none", extent=(0, w, 0, h))
    # plt.show()
    
    # First experiment
    print("Drawing coordinates...\n")
    draw_coords(coords=floor_list, names=ov_list, map=map.copy(), img_name="projs_anns")

    T_ov2px = np.array([[-sx, 0,  48*sx + 1], 
                        [ 0, sy, 105*sy + 2], 
                        [ 0, 0,   1       ]])
    print("T_ov2px:\n", T_ov2px) # This is a good transform
    # T_ov2px = get_transform(ov_list, floor_list, transform_type="st")
    # print("T_ov2px:\n", T_ov2px) # The obtained transform shows points slightly shifted
    
    T_gt2ov = np.array([[-1, 0,  48  + 1/sx],
                        [ 0, 1, -105 - 2/sy],
                        [ 0, 0,  1        ]])
    print("\nT_gt2ov:\n", T_gt2ov)  # This is a good transform

    T_gt2px = np.array([[sx, 0,  0],
                        [ 0, sy, 0],
                        [ 0, 0,  1]])
    print("\nT_gt2px:\n", T_gt2px) # This is a good transform
    # T_gt2px[:2, :2] = np.abs(T_ov2px[:2, :2])
    # print("T_gt2px:\n", T_gt2px) # The obtained transform shows points slightly shifted

    pts1 = np.meshgrid([-48, 148] + list(range(-40, 141, 10)), [-5, 105] + list(range(0, 111, 10)))
    pts1 = list(zip(pts1[0].flatten(), pts1[1].flatten())) 
    pts = perspective_transform([[p1 + 48 + 1/sx, p2 + 5 + 2/sy] for p1, p2 in pts1], T_gt2px)
    draw_coords(pts, pts1, map.copy(), "projs_gt", render=True)

    pts1 = np.meshgrid([48, -148] + list(range(-140, 41, 10)), [5, -105] + list(range(-100, 1, 10)))
    pts1 = list(zip(pts1[0].flatten(), pts1[1].flatten()))
    pts = perspective_transform(pts1, T_ov2px)
    draw_coords(pts, pts1, map.copy(), "projs_ov", render=True)

    def make_matrix_comment(arr, name):
        arr = arr.__str__().replace("[[", "")
        arr = arr.replace(" [", "")
        arr = arr.replace("]", "")
        lines = arr.__str__().split("\n")
        N = len(name) + 4
        lines[0] = "#" + " " * N + "|" + lines[0] + "|"
        lines[1] = "# %s = |%s" % (name, lines[1]) + "|"
        lines[2] = "#" + " " * N + "|" + lines[2] + "|"
        return '\n'.join(lines)

    with open('transforms.yml', "w") as f:
        f.write("# Include the transforms as a list of elements in row-major order\n")
        f.write("# For example for a transform with elements:\n#\n")
        f.write("#     | a  b  c |\n")
        f.write("# T = | d  e  f |\n")
        f.write("#     | g  h  i |\n#\n")
        f.write("# The yaml config node will be:\n#\n# T: \n#   - ")
        f.write("\n#   - ".join(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']) + "\n\n")

        f.write("# The transform from wold coordinates to pixel coordinates\n#\n")
        f.write(make_matrix_comment(T_ov2px, "T_ov2px") + "\n#\n")
        f.write('# to be entered as below:\nT_ov2px:\n')
        for val in T_ov2px.flatten(): f.write("  - %.6f\n" % val)

        f.write("\n\n# The transform from ground truth coordinates to pixel coordinates\n#\n")
        f.write(make_matrix_comment(T_gt2px, "T_gt2px") + "\n#\n")
        f.write('# to be entered as below:\nT_gt2px:\n')
        for val in T_gt2px.flatten(): f.write("  - %.6f\n" % val)
    
    # trajDump = "/home/bhernandez/Documents/deepstream/DS_Perf/workspace_experiment/trajDumps/trajDump_Stream_1.txt"
    # print("Reading", trajDump, "\n")
    # trajs = read_dump(trajDump)
    # draw_trajs(trajs, map.copy(), T_ov2px)  

    # gt_file = "/home/bhernandez/Documents/MTMC_Warehouse_Synthetic_012424/ground_truth.txt"
    # print("Reading", gt_file, "\n")
    # trajs_gt = read_gt(gt_file)
    # draw_trajs(trajs_gt, map.copy(), T_gt2px)
