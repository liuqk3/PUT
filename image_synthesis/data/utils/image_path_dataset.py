import bisect
import os
import cv2
import numpy as np
import albumentations
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


class ImagePaths(Dataset):
    def __init__(self, paths, sketch_paths=None, segmentation_paths=None, labels=None, mode="RGB"):

        self.labels = dict() if labels is None else labels
        self.labels["abs_path"] = paths
        if segmentation_paths is not None:
            self.labels["segmentation_path"] = segmentation_paths
        if sketch_paths is not None:
            self.labels["sketch_path"] = sketch_paths
        # self._length = len(paths)
        self.valid_index = list(range(len(paths)))
        self.mode = mode

    def __len__(self):
        # return self._length
        return len(self.valid_index)

    def _read_image(self, image_path, type='image'):
        image = Image.open(image_path)
        if type == 'image':
            # if not image.mode == "RGB":
            #     image = image.convert("RGB")
            if not image.mode == self.mode:
                image = image.convert(self.mode)
            image = np.array(image).astype(np.float32) # H x W x 3
        elif type in ['segmentation_map', 'sketch_map']:
            image = np.array(image).astype(np.float32) # H x W
        else:
            raise NotImplementedError

        return image
    

    def __getitem__(self, idx):
        i = self.valid_index[idx]
        image = self._read_image(self.labels["abs_path"][i], type='image')
        example = {
            'image': image,
        }
        if 'segmentation_path' in self.labels:
            seg_map = self._read_image(self.labels["segmentation_path"][i], type='segmentation_map')
            example['segmentation_map'] = seg_map
        if 'sketch_path' in self.labels:
            ske_map = self._read_image(self.labels["sketch_path"][i], type='sketch_map')  
            example['sketch_map'] = ske_map  
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
            