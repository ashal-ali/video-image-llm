import os
import zlib

import pandas as pd

from base.base_dataset import TextImageDataset


class ConceptualCaptions3M(TextImageDataset):
    """
    Conceptual Captions dataset. Split files are specific to my download regime.
    """
    def _load_metadata(self):
        # download specific
        split_files = {
            'train': 'train.csv',
            'train_1k': 'train_1k.csv',
            'train_rewrites': 'train_rewrites.csv',
            'train_rewrites_equiv': 'train_rewrites_equiv.csv',
            'val': 'val.csv',
            # there is no test
            'test': 'val.csv'
        }

        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(self.metadata_dir, target_split_fp))
        print("===========================================")
        print("Metadata path: ", os.path.join(self.metadata_dir, target_split_fp))
        print("===========================================")


        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        elif self.split == 'val':
            metadata = metadata.sample(1000, random_state=0)  # 15k val is unnecessarily large, downsample.

        self.metadata = metadata

    def _get_rel_fp(self, path):
        """
        Get the relative filepath from the absolute filepath.
        """
        # Data is in format /path/to/data_dir/final_split/0000xx/x_xxxx.jpg
        # images (6 + 1 = 7) characters after
        # final_split (12 + 1 = 13) characters after 
        idx = path.find('final_split') 

        return path[idx+ 13:]

    def _get_video_path(self, sample):
        path = sample['img_path']
        return os.path.join(path), self._get_rel_fp(path)
        # return os.path.join(self.data_dir, rel_fp), rel_fp


    def _get_caption(self, sample):
        return sample['text']
