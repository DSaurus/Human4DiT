import os.path as osp
import argparse

import numpy as np
import torch
import cv2
import os
import argparse
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
import trimesh
from imageio_ffmpeg import write_frames
from renderer import render_smpl_vcano_map

def render_smpl(smpl_path, camera_path, smpl_name, output_dir):
    if not os.path.exists(camera_path):
        return
    if not os.path.exists(smpl_path):
        return
    if os.path.exists(os.path.join(output_dir, smpl_name + ".mp4")):
        return
    smpl = trimesh.load(smpl_path)

    video_writer = write_frames(os.path.join(output_dir, smpl_name + ".mp4"), (1024, 1024), fps=30.0, macro_block_size=2)
    video_writer.send(None)
    
    cam_file = open(camera_path, "r")
    for cam_idx in range(90):
        _ = cam_file.readline()
        intrinsic = torch.eye(3)
        for i in range(3):
            a, b, c = cam_file.readline().split(' ')
            a, b, c = float(a), float(b), float(c)
            intrinsic[i, 0] = a
            intrinsic[i, 1] = b
            intrinsic[i, 2] = c
        extr = torch.eye(4)
        for i in range(3):
            a, b, c, d = cam_file.readline().split(' ')
            a, b, c, d = float(a), float(b), float(c), float(d)
            extr[i, 0] = a
            extr[i, 1] = b
            extr[i, 2] = c
            extr[i, 3] = d
        render_map = render_smpl_vcano_map(smpl.vertices, smpl.faces, cam_R=extr[:3, :3], cam_t=extr[:3, 3:], cam_K=intrinsic, img_w=1024, img_h=1024, device=torch.device("cuda:0"))
        render_map = ((render_map.detach().cpu().numpy()[0, :, :, :3])*255).astype(np.uint8)
        render_map = np.ascontiguousarray(render_map)
        video_writer.send(render_map)
    video_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj-dir", type=str, default="", required=True)
    parser.add_argument("--camera-dir", type=str, default="", required=True)
    parser.add_argument("--output-dir", type=str, default="", required=True)
    args = parser.parse_args()

    obj_dir = args.obj_dir
    output_dir = args.output_dir
    camera_dir = args.camera_dir
    os.makedirs(output_dir, exist_ok=True)

    for obj_name in tqdm(sorted(os.listdir(obj_dir))):
        render_smpl(os.path.join(obj_dir, obj_name, "norm_smpl.obj"), os.path.join(camera_dir, obj_name + ".txt"), obj_name, output_dir)
