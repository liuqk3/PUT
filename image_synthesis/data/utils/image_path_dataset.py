import bisect
import os
import cv2
import numpy as np
import albumentations
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


class ImagePaths(Dataset):
    def __init__(self, paths, labels=None):

        self.labels = dict() if labels is None else labels
        self.labels["abs_path"] = paths
        # self._length = len(paths)
        self.valid_index = list(range(len(paths)))

    def __len__(self):
        # return self._length
        return len(self.valid_index)

    def _read_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        # image = cv2.imread(image_path)
        # image = image[:, :, ::-1] # change to RGB

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

    def __getitem__(self, idx):
        i = self.valid_index[idx]
        example = self.preprocess_image(self.labels["abs_path"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

    
    def remove_files(self, file_path_set):
        if not isinstance(file_path_set, set):
            file_path_set = set(file_path_set)
        valid_index = []
        for i in range(len(self.labels['abs_path'])):
            # import pdb; pdb.set_trace()
            p = self.labels['abs_path'][i]
            if p not in file_path_set:
                valid_index.append(i)
        print('remove {} files, current {} files'.format(len(self.valid_index)-len(valid_index), len(valid_index)))
        self.valid_index = valid_index
            