import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset


class VideoAnomalyDataset_C3D(Dataset):
    def __init__(self, data_dir, frame_num=7):
        self.data_dir = data_dir
        self.frame_num = frame_num
        self.frames_list = self._load_frames(data_dir)

    def _load_frames(self, data_dir):
        frames_list = []
        videos = sorted(os.listdir(data_dir))
        for video in videos:
            frames = sorted(os.listdir(os.path.join(data_dir, video)))
            for i in range(self.frame_num // 2, len(frames) - self.frame_num // 2):
                frames_list.append({"video_name": video, "frame": i})
        return frames_list

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx):
        record = self.frames_list[idx]
        clip = self._load_clip(record["video_name"], record["frame"])
        label = np.arange(self.frame_num)  # Sequential label
        spatial_perm = np.random.permutation(9)
        return {
            "clip": torch.tensor(clip, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "trans_label": torch.tensor(spatial_perm, dtype=torch.long),
        }

    def _load_clip(self, video_name, frame):
        video_dir = os.path.join(self.data_dir, video_name)
        frames = sorted(os.listdir(video_dir))
        clip = []
        for i in range(frame - self.frame_num // 2, frame + self.frame_num // 2 + 1):
            img_path = os.path.join(video_dir, frames[i])
            img = Image.open(img_path).convert("L").resize((64, 64))
            clip.append(np.array(img) / 255.0)
        return np.stack(clip, axis=0)


class VideoAnomalyDataset_C3D_for_Clip(Dataset):
    def __init__(self, data_dir, clip_num=5, frame_num=7):
        self.data_dir = data_dir
        self.clip_num = clip_num
        self.frame_num = frame_num
        self.frames_list = self._load_frames(data_dir)

    def _load_frames(self, data_dir):
        frames_list = []
        videos = sorted(os.listdir(data_dir))
        for video in videos:
            frames = sorted(os.listdir(os.path.join(data_dir, video)))
            for i in range(self.clip_num * self.frame_num // 2, len(frames) - self.clip_num * self.frame_num // 2):
                frames_list.append({"video_name": video, "frame": i})
        return frames_list

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx):
        record = self.frames_list[idx]
        stacked_clips = self._load_stacked_clips(record["video_name"], record["frame"])
        return {"stacked_clips": torch.tensor(stacked_clips, dtype=torch.float32)}

    def _load_stacked_clips(self, video_name, frame):
        video_dir = os.path.join(self.data_dir, video_name)
        frames = sorted(os.listdir(video_dir))
        stacked_clips = []
        for i in range(frame - self.clip_num * self.frame_num // 2, frame + self.clip_num * self.frame_num // 2 + 1, self.frame_num):
            clip = []
            for j in range(i - self.frame_num // 2, i + self.frame_num // 2 + 1):
                img_path = os.path.join(video_dir, frames[j])
                img = Image.open(img_path).convert("L").resize((64, 64))
                clip.append(np.array(img) / 255.0)
            stacked_clips.append(np.stack(clip, axis=0))
        return np.stack(stacked_clips, axis=0)
