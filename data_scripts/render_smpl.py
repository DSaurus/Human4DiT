import cv2
import numpy as np
from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.structures import Meshes
from torch.utils.data._utils.collate import default_collate
import torch
import joblib
from smplx import SMPL
from tqdm import tqdm
import os
import decord
import argparse
from renderer import render_smpl_vcano_map
import multiprocessing
from imageio_ffmpeg import write_frames

def pix_faces_to_samples(pix_to_face, verts, faces, img):
    focal_length = 5000 / 256 * max(img.shape[0], img.shape[1])

    intrinsic = np.eye(3)
    intrinsic[0, 0] = -focal_length
    intrinsic[1, 1] = focal_length
    intrinsic[0, 2] = img.shape[1] / 2
    intrinsic[1, 2] = img.shape[0] / 2

    print("faces", faces.shape)
    N_F, _ = faces.shape
    N_V, _ = verts.shape
    faces = faces.reshape(-1)
    faces_verts = verts[faces, :]
    faces_verts = faces_verts.reshape(N_F, 3, 3)
    faces_verts_center = torch.mean(faces_verts, dim=1)

    print(pix_to_face.shape)
    pix_to_face = pix_to_face.detach().cpu()
    H, W, _ = pix_to_face.shape
    valid_face = pix_to_face >= 0
    pix_to_face = pix_to_face.reshape(-1)
    pix_to_face[pix_to_face < 0] = 0
    
    pix_verts = faces_verts_center[pix_to_face, :] # N 3

    pix_verts = pix_verts[valid_face.reshape(-1), :]
    
    proj_pix_verts = pix_verts[:, :3] / pix_verts[:, 2:] # N 3
    proj_pix_verts[:, :2] *= -1

    proj_pix_verts = proj_pix_verts.detach().cpu().numpy()
    proj_pix_verts = (intrinsic @ proj_pix_verts.T).T

    for i in tqdm(range(proj_pix_verts.shape[0])):
        cv2.circle(img, (int(proj_pix_verts[i, 0]), int(proj_pix_verts[i, 1])), 1, (0, 0, 255), -1)
    cv2.imwrite("demo_out/test_proj.jpg", img)

def torch_render_smpl_obj(img, verts, faces):
    width = img.shape[1]
    height = img.shape[0]
    focal_length = 5000 / 256 * max(width, height)

    intrinsic = torch.eye(4)
    intrinsic[0, 0] = -focal_length / min(width, height) * 2
    intrinsic[1, 1] = focal_length / min(width, height) * 2
    intrinsic[0, 3] = img.shape[1] / 2 / width
    intrinsic[1, 3] = img.shape[0] / 2 / height

    device = torch.device("cuda:0")
    
    mesh = Meshes(
        verts=[verts.to(device)],   
        faces=[faces.to(device)]
    )

    torch_focal_length = torch.zeros(2)
    torch_focal_length[0] = intrinsic[0, 0]
    torch_focal_length[1] = intrinsic[1, 1]
    torch_focal_length = torch_focal_length.unsqueeze(0)

    principal_point = torch.ones(2) * 0
    camera = PerspectiveCameras(
        R=torch.eye(3).unsqueeze(0),
        T=torch.zeros(3).unsqueeze(0),
        focal_length=torch_focal_length,
        principal_point=principal_point.unsqueeze(0),
        device=device,
        in_ndc=True,
    )

    raster_settings = RasterizationSettings(
        image_size=(img.shape[0], img.shape[1]), 
        blur_radius=0, 
        faces_per_pixel=1, 
    )
    rasterizer = MeshRasterizer(
        cameras=camera,
        raster_settings=raster_settings
    )
    raster_results = rasterizer(mesh)
    print(raster_results.pix_to_face.shape)
    print(raster_results.zbuf.shape)
    zbuf = raster_results.zbuf.detach().cpu().numpy()[0][:, :, 0]
    zbuf[zbuf < 0] = 0
    zbuf[zbuf > 0] = 255
    print(np.sum(zbuf > 0))
    zbuf = zbuf.astype(np.uint8)
    # img[:, :, 2] = zbuf
    # cv2.imwrite("demo_out/zbuf.jpg", img)
    return raster_results.pix_to_face[0]

def smpl_pose_to_obj(smpl, pose_pkl, camera_pkl, device):
    camera_t = torch.Tensor(camera_pkl)
    smpl_params = default_collate(pose_pkl)
    for key in smpl_params.keys():
        if isinstance(smpl_params[key], torch.Tensor):
            smpl_params[key] = smpl_params[key].to(device).float()
    # print(smpl_params)
    smpl_output = smpl(**smpl_params, pose2rot=False)
    pred_vertices = smpl_output.vertices.cpu().detach()
    # print(pred_vertices.shape)
    pred_vertices = pred_vertices + camera_t.unsqueeze(1)
    # rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])[:3, :3]
    # rot = torch.Tensor(rot).unsqueeze(0)
    # pred_vertices = (rot @ pred_vertices.transpose(1, 2)).transpose(1, 2)
    B, N, C = pred_vertices.shape
    pred_vertices = pred_vertices.reshape(B * N, C)
    pred_vertices = pred_vertices.detach().numpy()
    # pred_vertices[:, 2] = -pred_vertices[:, 2]
    return pred_vertices


def save_obj(path, verts):
    with open(path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")


def process_video(path, video_dir, smpl_dir, output_dir, smpl, cuda_list, visit_list, lock):
    with lock:
        cuda_id = cuda_list.pop(0)
    device = torch.device(f"cuda:{cuda_id}")
    smpl = smpl.to(device)
    try:
        for _ in range(1):
            video_path = path
            video_name = os.path.basename(video_path)
            pkl_file_name = "demo_" + video_name[:-4] + ".pkl"
            if not os.path.exists(os.path.join(smpl_dir, pkl_file_name)):
                continue
            
            output_name = os.path.join(output_dir, video_name)
            if os.path.exists(output_name):
                print(f"skip {video_name}")
                break
            try:
                video_reader = decord.VideoReader(video_path)
                img = video_reader[0].asnumpy()
                width = img.shape[1]
                height = img.shape[0]
                fps = video_reader.get_avg_fps()
                video_length = len(video_reader)
                del video_reader
            except Exception as e:
                print(e)
                break
            
            extrinsic_list = []
            intrinsic_list = []
            render_video = write_frames(output_name, (width, height), fps=fps, macro_block_size=2)
            render_video.send(None)
            pkl_file = joblib.load(os.path.join(smpl_dir, pkl_file_name))
            
            for image_key in pkl_file:
                pkl_image_key = image_key
                break
            pkl_image_key_id = pkl_image_key.split("/")[-1]
            pkl_image_key = pkl_image_key[:-len(pkl_image_key_id)] + "%06d.jpg"
            for video_idx in range(video_length):
                image_key = pkl_image_key % (video_idx + 1)
                if image_key in pkl_file:
                    smpl_params = pkl_file[image_key]["smpl"]
                    camera_params = pkl_file[image_key]["camera"]
                    verts = smpl_pose_to_obj(smpl, smpl_params, camera_params, device)
                    faces = smpl.faces
                    focal_length = 5000 / 256 * max(height, width)
                    intrinsic = np.eye(3)
                    intrinsic[0, 0] = focal_length
                    intrinsic[1, 1] = focal_length
                    intrinsic[0, 2] = width / 2
                    intrinsic[1, 2] = height / 2

                    render_img = render_smpl_vcano_map(verts, faces, cam_K=torch.FloatTensor(intrinsic), img_h=height, img_w=width, device=device)
                    render_img = (render_img.detach().cpu().numpy()[0, :, :, :3] * 255).astype(np.uint8)
                    intrinsic_list.append(intrinsic)
                else:
                    render_img = np.zeros((height, width, 3), dtype=np.uint8)
                extrinsic_list.append(np.eye(4))
                render_video.send(render_img)
            render_video.close()
    except Exception as e:
        print(e)
    with lock:
        cuda_list.append(cuda_id)
    visit_list.append(0)
    pbar.n = len(visit_list)
    pbar.refresh()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--process-nums", type=int, default=1)
    parser.add_argument("--video-dir", type=str, default="", required=True)
    parser.add_argument("--smpl-dir", type=str, default="", required=True)
    parser.add_argument("--output-dir", type=str, default="", required=True)
    parser.add_argument("--smpl-model-path", type=str, default="", required=True)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    smpl_model_path = args.smpl_model_path
    smpl = SMPL(model_path=smpl_model_path)

    video_dir = args.video_dir
    smpl_dir = args.smpl_dir
    video_list = os.listdir(video_dir)

    video_path_list = [os.path.join(video_dir, video_name) for video_name in video_list]

    with multiprocessing.Manager() as manager:
        cuda_list = manager.list()
        visit_list = manager.list()
        for i in range(args.process_nums):
            cuda_list.append(i % args.num_devices)
        lock = manager.Lock()
        pbar = tqdm(total=len(video_path_list))
        with multiprocessing.Pool(args.process_nums) as pool:
            pool.starmap(process_video, [(video_path, video_dir, smpl_dir, output_dir, smpl, cuda_list, visit_list, lock) for video_path in video_path_list])
        