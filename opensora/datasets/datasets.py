import csv
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

from . import video_transforms
from .utils import center_crop_arr


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


class DatasetFromCSV(torch.utils.data.Dataset):
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
        resolution=512
    ):
        self.csv_path = csv_path
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            self.samples = list(reader)

            delete_list = ["/data/human_data_pyx/video_clip/00016_002.mp4", "/data/human_data_pyx/video_clip/00021_004.mp4"]
            filter_list = []
            for i in range(len(self.samples)):
                if self.samples[i][0] in delete_list:
                    continue
                filter_list.append(self.samples[i])
            self.samples = filter_list

        ext = self.samples[0][0].split(".")[-1]
        if ext.lower() in ("mp4", "avi", "mov", "mkv"):
            self.is_video = True
        else:
            assert f".{ext.lower()}" in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            self.is_video = False

        self.transform = transform

        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
        self.root = root
        self.resolution = resolution

    def getitem(self, index):
        index = index % len(self.samples)
        sample = self.samples[index]
        path = sample[0]
        if self.root:
            path = os.path.join(self.root, path)
        text = sample[1]
        frame_indice = None
        def process_video(vframes, frame_indice):
            video = vframes[frame_indice]
            trans = video_transforms.ToTensorVideo()
            video = trans(video)
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

        if len(sample) > 3:
            cond_path = sample[3]
            vframes, aframes, info = torchvision.io.read_video(filename=cond_path, pts_unit="sec", output_format="TCHW")
            total_frames = len(vframes)

            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert (
                end_frame_ind - start_frame_ind >= self.num_frames
            ), f"{path} with index {index} has not enough frames."
            frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)
            cond_video = process_video(vframes, frame_indice)

            cond_normal_path = sample[4]
            vframes, aframes, info = torchvision.io.read_video(filename=cond_normal_path, pts_unit="sec", output_format="TCHW")
            cond_normal_video = process_video(vframes, frame_indice)

            cond_video = torch.cat([cond_video, cond_normal_video], dim=0)
        else:
            cond_video = None

        if self.is_video:
            vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
            total_frames = len(vframes)

            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert (
                end_frame_ind - start_frame_ind >= self.num_frames
            ), f"{path} with index {index} has not enough frames."
            if frame_indice is None:
                print("frame_indice!")
                frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)

            video = vframes[frame_indice]
            
            trans = video_transforms.ToTensorVideo()
            video = trans(video)
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
        else:
            image = pil_loader(path)
            image = self.transform(image)
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        
        item = {"video": video, "text": text}
        if cond_video is not None:
            item.update({"cond_video": cond_video})

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
        return len(self.samples) * 10
