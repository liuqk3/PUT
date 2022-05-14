import albumentations
import random
import numpy as np
from PIL import Image
import cv2
from io import BytesIO

class SimplePreprocessor(object):
    def __init__(self,
                 size=None, 
                 random_crop=False, 
                 horizon_flip=False,
                 change_brightness=False,
                 add_noise=False,
                 random_rotate=False,
                 smallest_max_size=None,
                 additional_targets=None,
                 max_spatial_ratio=2, # the max spatial ratio, the min spatial ratio is 1
                 random_spatial_ratio=-1, # the probability to sample a random spatial ratio
                 keep_origin_spatial_ratio=-1 # the probability to keep spatial ratio
                 ):
        """
        This image preprocessor is implemented based on `albumentations`
        """
        if isinstance(size, int): 
            size = (size, size) # height, width
        self.size = size
        
        identity = True
        if size is not None:
            if min(size) > 0:
                identity = False
        if not identity:
            if smallest_max_size is None:
                smallest_max_size = min(self.size)
            self.smallest_max_size = smallest_max_size
            self.max_spatial_ratio = max_spatial_ratio
            self.random_spatial_ratio = random_spatial_ratio
            self.keep_origin_spatial_ratio = keep_origin_spatial_ratio
            
            transforms = list()
            rescaler = albumentations.SmallestMaxSize(max_size=smallest_max_size)
            transforms.append(rescaler)
            if not random_crop:
                cropper = albumentations.CenterCrop(height=size[0], width=size[1])
                transforms.append(cropper)
            else:
                cropper = albumentations.RandomCrop(height=size[0], width=size[1])
                transforms.append(cropper)
            if horizon_flip:
                flipper = albumentations.HorizontalFlip()
                transforms.append(flipper)
            if change_brightness:
                raise RuntimeError('There is a bug in this augmentation, please do not use it before fix it!')
                brightness = albumentations.RandomBrightnessContrast(p=0.2)
                transforms.append(brightness)
            if add_noise:
                raise RuntimeError('There is a bug in this augmentation, please do not use it before fix it!')
                noise = albumentations.OneOf([
                            albumentations.IAAAdditiveGaussianNoise(),
                            albumentations.GaussNoise(),
                        ], p=0.2)
                transforms.append(noise)
            if random_rotate:
                rotate = albumentations.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=20, p=0.2)
                transforms.append(rotate)

            preprocessor = albumentations.Compose(transforms,
                                                additional_targets=additional_targets)
        else:
            preprocessor = lambda **kwargs: kwargs            

        self.preprocessor = preprocessor
    
    def __call__(self, **input):

        if self.size is not None:
            if random.random() <= self.random_spatial_ratio:
                cache = {}
                for k in input.keys():
                    if k in ['image', 'mask']:
                        image = input[k]
                        if 'area' in cache:
                            x1, y1, x2, y2 = cache['area']
                        else:
                            h, w = image.shape[0], image.shape[1]
                            if random.random() <= self.keep_origin_spatial_ratio:
                                x1, y1, x2, y2 = 0, 0, w, h
                            else:
                                if h > w:
                                    max_ratio = h / float(w)
                                    ratio = min(random.random() * (self.max_spatial_ratio - 1) + 1, max_ratio)
                                    h_ = min(int(w * ratio), h)
                                    w_ = w
                                else:# w > h:
                                    max_ratio = w / float(h) 
                                    ratio = min(random.random() * (self.max_spatial_ratio - 1) + 1, max_ratio)
                                    w_ = min(int(h * ratio), w)
                                    h_ = h
                                x1 = random.randint(0, w-w_)
                                y1 = random.randint(0, h-h_)
                                x2 = x1 + w_ 
                                y2 = y1 + h_
                            cache['area'] = (x1, y1, x2, y2)
                        
                        # crop and resize
                        if len(image.shape) == 2:
                            image = image[y1:y2, x1:x2]
                        else:
                            image = image[y1:y2, x1:x2, :]
                        if k in ['image']:
                            inter = cv2.INTER_LINEAR
                        elif k in ['mask']:
                            inter = cv2.INTER_NEAREST
                        else:
                            raise NotImplementedError
                        if self.smallest_max_size > min(self.size):
                            size = (self.smallest_max_size, self.smallest_max_size) # [w, h]
                        else:
                            size = self.size[::-1] # [w, h]
                        image = cv2.resize(image, size, interpolation=inter)
                        input[k] = image
                    else:
                        raise NotImplementedError('Random crop {} is not implemented!'.format(k)) 
        # print(input['image'].shape)
        return self.preprocessor(**input)

