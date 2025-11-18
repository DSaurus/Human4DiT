import csv
import os
import cv2

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
import random 
from . import video_transforms
from .utils import center_crop_arr
from decord import VideoReader

def get_transforms_video(resolution=256):
    transform_video = transforms.Compose(
        [
            # video_transforms.ToTensorVideo(),  # TCHW
            # video_transforms.RandomHorizontalFlipVideo(),
            # video_transforms.UCFCenterCropVideo(resolution),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    return transform_video


def get_transforms_image(image_size=256):
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    return transform

# data_name = ['0529']
class DatasetIMGFromCSV(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        csv_path,
        num_frames=16,
        frame_interval=1,
        transform=None,
        root=None,
        resolution=512,
        multiview=1,
    ):

        self.samples = []
        # for al in data_name:
        pa = f'./test0614/vid'
        self.samples1 = os.listdir(pa)
        # self.samples1 = sorted(self.samples1)
        self.samples1 = [os.path.join(pa,i) for i in self.samples1]
        self.samples = self.samples + self.samples1
        print(self.samples)
        print("LEN DATA NOW")
        print(len(self.samples))


        self.is_video = True

        self.transform = transform

        self.multiview = multiview
        self.num_frames = num_frames // multiview
        self.frame_interval = frame_interval
        self.temporal_sample = video_transforms.TemporalRandomCrop(self.num_frames * frame_interval)
        self.root = root
        self.resolution = resolution
        self.ref_dict = {}
        self.getitem(0)
    
    def get_ref_table(self):
        for sample in self.samples:
            path = sample[0]
            pose_path = sample[1]
            normal_path = sample[2]
            file_name = os.path.basename(path)
            ref_prefix_length = int(sample[4])
            ref_video_length = int(sample[3])
            if file_name[:ref_prefix_length] not in self.ref_dict:
                self.ref_dict[file_name[:ref_prefix_length]] = [
                    { 
                        "path": path, 
                        "length": ref_video_length,
                        "pose_path": pose_path,
                        "normal_path": normal_path
                    }
                ]
            else:
                self.ref_dict[file_name[:ref_prefix_length]].append(
                    { 
                        "path": path, 
                        "length": ref_video_length,
                        "pose_path": pose_path,
                        "normal_path": normal_path
                    }
                )
    
    def random_get_ref_path(self, path, ref_prefix_length):
        file_name = os.path.basename(path)
        ref_prefix = file_name[:ref_prefix_length]
        ref_paths = self.ref_dict[ref_prefix]
        return random.choice(ref_paths)

    def getitem(self, index, start_frame_ind=None, end_frame_ind=None, ref_img_index=None, random_ref=True):
        index = index % len(self.samples)
        folder_name = self.samples[index]

        # pppp = folder_name.split('/')[-2][-4:]
        # naa = folder_name.split('/')[-1]
        naa = '1.mp4'
        vid_na = os.path.join(f'./test0614/vid', naa)
        normal_path = os.path.join(f'./test0614/vid_normal', naa)
        pose_path = os.path.join(f'./test0614/vid_pose', naa)

        video_reader = VideoReader(vid_na)
        normal_reader = VideoReader(normal_path)
        pose_reader = VideoReader(pose_path)

        length = max(len(normal_reader), 0)

        ref_video_length = length
        
        def process_video(video):
            T, C, H, W = video.shape
            if H > W:
                pad_video = torch.zeros(T, C, H, H)
                pad_video[:, :, :, (H - W) // 2 : (H - W) // 2 + W] = video
            else:
                pad_video = torch.zeros(T, C, W, W)
                pad_video[:, :, (W - H) // 2 : (W - H) // 2 + H, :] = video
            resize = transforms.Resize(self.resolution)
            video = resize(pad_video)
            video = self.transform(video)  # T C H W
            video = video.permute(1, 0, 2, 3)
            return video

        def read_video(path, frame_indice):
            img_list = []
            for i in frame_indice:
                if os.path.exists(path % i):
                    img = cv2.imread(path % i)[:, :, ::-1]
                    img = torch.FloatTensor(img.copy()).permute(2, 0, 1) / 255.0
                else:
                    img = torch.zeros(3, 512, 512).float()
                img_list.append(img)
            return torch.stack(img_list, dim=0)
        
        total_frames = length

        ref_img_index = None
        if ref_img_index is None:
            ref_img_index = random.randint(30, ref_video_length-1)
        # img_index = random.randint(0, total_frames-1)

        video_list = []
        cond_video_list = []
        for mv_id in range(self.multiview):
            # if mv_id != 0:
            #     ref = self.random_get_ref_path(sample[0], int(sample[4]))
            #     total_frames = ref["length"]
            #     path = ref["path"]
            #     pose_path = ref["pose_path"]
            #     normal_path = ref["normal_path"]

            if start_frame_ind is None:
                sample_start_frame_ind, sample_end_frame_ind = self.temporal_sample(total_frames)
                # assert (
                #     sample_end_frame_ind - sample_start_frame_ind >= self.num_frames
                # ), f"{path} with index {index} has not enough frames."
            else:
                sample_start_frame_ind = start_frame_ind
                sample_end_frame_ind = end_frame_ind
            # if mv_id != 0:
            #     print(sample_start_frame_ind, sample_end_frame_ind)
            frame_indice = np.linspace(sample_start_frame_ind, sample_end_frame_ind - 1, self.num_frames, dtype=int)


            ### load dwpose 
            # pose_video = read_video(pose_path, frame_indice)
            pose_video = []
            for index in frame_indice:
                img = pose_reader[index]
                img = img.asnumpy()
                # print(img.shape,img.max(),img.min())    # (1520, 1728, 3) 255 0
                pose_video.append(torch.FloatTensor(img.copy()).permute(2, 0, 1) / 255.0)
            pose_video = torch.stack(pose_video, dim=0)
            # torch.Size([24, 3, 1520, 1728]) tensor(1.) tensor(0.)
            cond_pose_video = process_video(pose_video)
            # torch.Size([3, 24, 512, 512]) tensor(1.0000) tensor(-1.)



            ### normal normal
            # normal_video = read_video(normal_path, frame_indice)
            normal_video = []
            for index in frame_indice:
                img = normal_reader[index]
                img = img.asnumpy()
                normal_video.append(torch.FloatTensor(img.copy()).permute(2, 0, 1) / 255.0)
            normal_video = torch.stack(normal_video, dim=0)
            cond_normal_video = process_video(normal_video)

            
            # cat pose and normal
            cond_video = torch.cat([cond_pose_video, cond_normal_video], dim=0)

            #  load  rgb video
            # video = read_video(path, frame_indice)
            video = []
            for index in frame_indice:
                img = video_reader[index]
                img = img.asnumpy()
                video.append(torch.FloatTensor(img.copy()).permute(2, 0, 1) / 255.0)
            video = torch.stack(video, dim=0)
            video = process_video(video)

            video_list.append(video)
            cond_video_list.append(cond_video)

        # load reference
        # ref_video = read_video(ref_path, [ref_img_index])
        ref_video = []
        img = video_reader[ref_img_index]
        img = img.asnumpy()
        ref_video.append(torch.FloatTensor(img.copy()).permute(2, 0, 1) / 255.0)
        ref_video = torch.stack(ref_video, dim=0)
        ref_video = process_video(ref_video)


        
        item = {"video": torch.stack(video_list, dim=1), "ref_video":ref_video.unsqueeze(1), "text": vid_na}
        if cond_video is not None:
            item.update({"cond_video": torch.stack(cond_video_list, dim=1)})

        return item

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples) * 10000
