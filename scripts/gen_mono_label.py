import os
import math
import cv2
import csv
import argparse
import numpy as np
from tqdm import tqdm

from utils import KITTIDataset, compute_box_3d_camera, compute_box_3d_image, bbox_lidar2cam

def parse_option():
    parser = argparse.ArgumentParser('Split Multi-View Labels into Mono3D labels for V2X-Radar-I-KITTI', add_help=False)
    parser.add_argument('--data_root', type=str, default='data/V2X-Radar-I-kitti', help='root path to KITTI Dataset')
    parser.add_argument('--camera_view', type=str, default='CAM_FRONT', help='')    
    args = parser.parse_args()
    return args

mono_label_dict = {"CAM_LEFT": "label_mono_1", "CAM_FRONT": "label_mono_2", "CAM_RIGHT": "label_mono_3"}
mono_calib_dict = {"CAM_LEFT": "calib_mono_1", "CAM_FRONT": "calib_mono_2", "CAM_RIGHT": "calib_mono_3"}

def label_camera_view(ann, camera_view="CAM_FRONT"):
    dim, loc_lidar, yaw, calib = ann["dim"], ann["loc"], ann["yaw"], ann[camera_view]
    l, w, h = dim
    P, rmat, tmat = calib
    Rlidar_camera = np.hstack((rmat, tmat.reshape(-1, 1)))

    Tr_velo2cam = np.eye(4)
    Tr_velo2cam[:3, :] = Rlidar_camera
    alpha, rot_y, dim, loc_cam = bbox_lidar2cam(dim, loc_lidar, yaw, Tr_velo2cam)
    alpha, rot_y = alpha - 0.03, rot_y - 0.03  # small trick
    
    loc_lidar[2] += h / 2
    corners_image = compute_box_3d_image(dim, rot_y, loc_lidar, Rlidar_camera, P[:3,:3])
    min_value, max_value = np.amin(corners_image, axis=0), np.amax(corners_image, axis=0)
    xmin, ymin, xmax, ymax = min_value[0,0], min_value[0,1], max_value[0,0], max_value[0,1]
    bbox = [xmin, ymin, xmax, ymax]
    center_image = np.mean(corners_image, axis=0)
    return center_image.reshape([3,]), bbox, dim, rot_y, alpha, loc_cam    

def write_label_txt(infos, label_path):
    out_file = open(label_path, 'w')
    
    for info in infos:
        bbox, dim, rot_y, alpha, loc_cam = info["bbox"], info["dim"], info["rot_y"], info["alpha"], info["loc_cam"]
        l, w, h = dim
        out_file.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(info["class"].capitalize(), info["truncated"], info["occluded"], alpha, bbox[0] , bbox[1], bbox[2], bbox[3], h, w, l, loc_cam[0], loc_cam[1], loc_cam[2], rot_y) + '\n')
    
def write_calib_txt(calib, calib_path):
    P_array, r_velo2cam, t_velo2cam = calib
    Tr_velo2cam = np.hstack((r_velo2cam, t_velo2cam.reshape(-1, 1)))
    R0_rect_array = np.eye(3)
    P0 = 'P0: ' + " ".join([" ".join(map(str, row)) for row in np.eye(4)[:3, :]])
    P1 = 'P1: ' + " ".join([" ".join(map(str, row)) for row in np.eye(4)[:3, :]])
    P2 = 'P2: ' + " ".join([" ".join(map(str, row)) for row in P_array])
    P3 = 'P3: ' + " ".join([" ".join(map(str, row)) for row in np.eye(4)[:3, :]])
    R0_rect = 'R0_rect: ' + " ".join([" ".join(map(str, row)) for row in R0_rect_array])
    Tr_velo_to_cam = 'Tr_velo_to_cam: ' + " ".join([" ".join(map(str, row)) for row in Tr_velo2cam])
    Tr_imu_to_velo = 'Tr_imu_to_velo: ' + " ".join([" ".join(map(str, row)) for row in np.eye(4)[:3, :]])
    
    calib_file = P0 + '\n' + P1 + '\n' + P2 + '\n' + P3 + '\n' + R0_rect + '\n' + Tr_velo_to_cam + '\n' + Tr_imu_to_velo
    with open(calib_path, "w") as f:
        f.write(calib_file)
    
if __name__ == "__main__":
    args = parse_option()
    dataset = KITTIDataset(args.data_root)
    os.makedirs(mono_label_dict[args.camera_view], exist_ok=True)
    
    for i in tqdm(range(len(dataset))):
        annos, sensor_params, data_path = dataset[i]
        infos = []
        for ann in annos:
            center_image, bbox, dim, rot_y, alpha, loc_cam = label_camera_view(ann, camera_view=args.camera_view)
            if center_image[0,0] > 0 and center_image[0,0] < 1536 and center_image[0,1] > 0 and center_image[0,1] < 864:
              infos.append(
                  {
                      "bbox": bbox,
                      "dim": dim,
                      "rot_y": rot_y,
                      "alpha": alpha,
                      "loc_cam": loc_cam,
                      "class": ann["class"],
                      "truncated": ann["truncated"],
                      "occluded": ann["occluded"]
                  }
              )
        calib = annos[0][args.camera_view]
        label_path = os.path.join(args.data_root, "training", mono_label_dict[args.camera_view], data_path["original_idx"] + ".txt")
        calib_path = os.path.join(args.data_root, "training", mono_calib_dict[args.camera_view], data_path["original_idx"] + ".txt")
        
        os.makedirs(os.path.join(args.data_root, "training", mono_label_dict[args.camera_view]), exist_ok=True)
        os.makedirs(os.path.join(args.data_root, "training", mono_calib_dict[args.camera_view]), exist_ok=True)
        write_label_txt(infos, label_path)
        write_calib_txt(calib, calib_path)
        
        
        