import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import math
from tqdm import tqdm
import argparse
from imageio_ffmpeg import write_frames
os.environ['PYOPENGL_PLATFORM'] = 'egl'

def normalize(v):
    return v/np.linalg.norm(v)

def convertgl2cv(c2w):
    # c2w = np.linalg.inv(w2c)
    flip_yz = np.eye(4)
    flip_yz[2, 2] = -1
    flip_yz[1, 1] = -1
    c2w = c2w @ flip_yz
    return c2w
    # return np.linalg.inv(c2w)


def samples2matrix(
    camera_positions,
    camera_centers
):
    up = np.array([0, 1, 0])
    lookat = normalize(camera_centers - camera_positions)
    right = normalize(np.cross(lookat, up))
    up = normalize(np.cross(right, lookat))
    c2w3x4 = np.zeros((3, 4))
    c2w3x4[:, 0] = right
    c2w3x4[:, 1] = up
    c2w3x4[:, 2] = -lookat
    c2w3x4[:, 3] = camera_positions
    c2w = np.eye(4)
    c2w[:3] = c2w3x4
    w2c = np.linalg.inv(c2w)
    return w2c, c2w

def get_intrinsic(fovx, fovy, width, height):
    intrinsic = np.eye(3)
    intrinsic[0, 0] = width / (2 * math.tan(fovx / 2))
    intrinsic[1, 1] = height / (2 * math.tan(fovy / 2))
    intrinsic[0, 2] = width / 2
    intrinsic[1, 2] = height / 2
    intrinsic[2, 2] = 1.0
    return intrinsic

def render_obj(obj_path, obj_name, output_dir):
    obj_base_name = obj_name.split(".")[-1]
    if os.path.exists(os.path.join(output_dir, obj_base_name + ".txt")):
        return
    if not os.path.exists(obj_path):
        print("Not found", obj_path)
        return
    scene = pyrender.Scene(ambient_light = [1.0,1.0,1.0,1.0])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    width = 1024
    height = 1024
    intrinsic = get_intrinsic(np.pi / 3.0, np.pi / 3.0, width, height)
    _, camera_pos = samples2matrix(np.array([0, 2, 5]), np.array([0, 2, 0]))
    print(camera_pos)
    camera_node = scene.add(camera, pose=camera_pos)

    fuze_trimesh = trimesh.load(obj_path)
    y_max = np.max(fuze_trimesh.vertices[:, 1])
    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
    scene.add(mesh)

    r = pyrender.OffscreenRenderer(height, width)

    ci = 0
    radius = 6 / 2.5 * y_max
    y_middle = 0
    camera_pos_list = []
    stride = 4

    video_writer = write_frames(os.path.join(output_dir, obj_base_name + ".mp4"), (width, height), fps=30, macro_block_size=2)
    video_writer.send(None)
    os.makedirs(output_dir, exist_ok=True)   
    
    for ci in tqdm(range(0, 360, stride)):
        rad = ci / 180 * math.pi
        _, camera_pos = samples2matrix(np.array([math.sin(rad)*radius, y_middle, math.cos(rad)*radius]), np.array([0, y_middle, 0]))
        scene.set_pose(camera_node, camera_pos)
        color, depth = r.render(scene)
        color = np.ascontiguousarray(color).astype(np.uint8)
        video_writer.send(color)
        
        camera_pos = convertgl2cv(camera_pos)
        camera_pos = np.linalg.inv(camera_pos)
        camera_pos_list.append(camera_pos)
    video_writer.close()

    camera_file = open(os.path.join(output_dir, obj_base_name + ".txt"), "w")
    for i in range(len(camera_pos_list)):
        camera_file.write("%d\n" % i)
        camera_pos = camera_pos_list[i]
        for r in range(3):
            camera_file.write("%f %f %f\n" % (intrinsic[r][0], intrinsic[r][1], intrinsic[r][2]))
        for r in range(3):
            camera_file.write("%f %f %f %f\n" % (camera_pos[r][0], camera_pos[r][1], camera_pos[r][2], camera_pos[r][3]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj-dir", type=str, default="", required=True)
    parser.add_argument("--output-dir", type=str, default="", required=True)
    args = parser.parse_args()

    obj_dir = args.obj_dir
    output_dir = args.output_dir
    for obj_name in tqdm(sorted(os.listdir(obj_dir))):
        print(obj_name)
        render_obj(os.path.join(obj_dir, obj_name, "norm_mtl.obj"), obj_name, output_dir)