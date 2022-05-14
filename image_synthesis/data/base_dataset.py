import bisect
import os
import numpy as np
import albumentations
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        # return self.datasets[dataset_idx][sample_idx], dataset_idx
        data = self.datasets[dataset_idx][sample_idx]
        data['dataset_idx'] = dataset_idx
        return data


class ImagePaths(Dataset):
    def __init__(self, paths, labels=None):
        self.labels = dict() if labels is None else labels
        self.labels["abs_path"] = paths
        self._length = len(paths)

    def __len__(self):
        return self._length

    def _read_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        return image

    def preprocess_image(self, image_path):
        image = self._read_image(image_path)
        image = image.astype(np.float32) # (image/127.5 - 1.0).astype(np.float32)
        images = {"image": image}
        # images = {'image': np.transpose(image, (2, 0, 1))}

        noise_image = False # True
        if noise_image:
            import skimage
            from skimage import io as skimageio
            image_noise = skimage.util.random_noise(image.astype(np.uint8), mode='gaussian') #, clip=True)
            image_noise = np.array(image_noise * 255).astype(np.uint8)
            image_noise = image_noise.astype(np.float32) # (image_noise/127.5 - 1.0).astype(np.float32)
            images['image_noise'] = image_noise # np.transpose(image_noise, (2, 0, 1))
        return images

    def __getitem__(self, i):
        example = self.preprocess_image(self.labels["abs_path"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

    
