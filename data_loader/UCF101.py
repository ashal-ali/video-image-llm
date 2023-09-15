import json
import os
import random

import numpy as np
import pandas as pd
import glob
import torch
from torch.utils.data import Dataset
from data_loader import transforms
from base.base_dataset import sample_frames
import decord
from PIL import Image
decord.bridge.set_bridge('torch')
#from base.base_dataset import TextImageDataset, VideoImageDataset

# Wrapper for UCF101 dataset
class UCF101Dataset(Dataset):
    def __init__(self, n_frames=4, kind="video", data_dir="/mnt/datasets_mnt/ucf101/videos/UCF-101", split="test", tfms=None):
        self.n_frames = n_frames
        self.data_dir = data_dir
        self.split = split
        self.kind = kind
        if self.kind == "image":
            self.n_frames = 1
        if tfms is not None:
            self.tfms = tfms
        else:
            self.tfms = transforms.init_transform_dict(use_clip_norm=True)["test"]
        self._load_metadata(kind=kind) # Loads paths and labels
    def _load_metadata(self, kind="video"):
        if self.split != 'test':
            raise NotImplementedError("Assumes inference, no text, hence cant be used for training...")
        if kind == "video":
            TARGET_EXT = "*.avi"
        else:
            TARGET_EXT = "*.jpg"
        paths = os.path.join(self.data_dir, "**", TARGET_EXT)
        self.paths = glob.glob(paths, recursive=True)
        self.text_labels = [x.split('/')[-2] for x in self.paths]
        self.len = len(self.paths)
        self.text_labels = [x.replace('_', ' ') for x in self.text_labels]
        self.text2cat = {x:i for i, x in enumerate(sorted(set(self.text_labels)))}
        self.cat_labels = [self.text2cat[x] for x in self.text_labels]
    
    def __getitem__(self, idx):
        path, label = self.paths[idx], self.cat_labels[idx]
        if self.kind == "video":
            video_reader = decord.VideoReader(path, num_threads=1)
            vlen = len(video_reader)
            frame_idxs = sample_frames(self.n_frames, vlen, sample='uniform', fix_start=None)
            frames = video_reader.get_batch(frame_idxs)
            frames = frames.float() / 255
            frames = frames.permute(0, 3, 1, 2)
            vis_input = self.tfms(frames)
        elif self.kind == "image":
            vis_input = Image.open(path).convert("RGB")
            vis_input = self.tfms(vis_input)
        return vis_input, label
        
        #img_li = glob.glob(img_glob, recursive=True)
        #img_li = [x.replace(self.data_dir, '').strip('/') for x in img_li]
        #img_li = sorted(img_li)
        #self.metadata = pd.Series(img_li)
    def __len__(self):
        return self.len

if __name__ == "__main__":
    #from data_loader import transforms
    #tsfms = transforms.init_transform_dict(use_clip_norm=True)
    from base.transforms import transforms
    ds = UCF101Dataset()

    for x in range(100):
        print(ds.__getitem__(x))