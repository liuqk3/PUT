import os
import numpy as np
import yaml
import copy
import cv2
import random
from torch.utils.data import Dataset
from PIL import Image
from image_synthesis.data.utils.image_path_dataset import ImagePaths
import image_synthesis.data.utils.imagenet_utils as imagenet_utils
from image_synthesis.utils.misc import instantiate_from_config, get_all_file
from image_synthesis.data.utils.util import generate_mask_based_on_landmark, generate_stroke_mask, rgba_to_depth, visualize_depth
from image_synthesis.utils.io import load_dict_from_json

MASK_RATIO_INDEX = {
    '0.01': 0,
    '0.1': 2000,
    '0.2': 4000,
    '0.3': 6000,
    '0.4': 8000,
    '0.5': 10000,
    '0.6': 12000
}

def my_print(info, logger=None):
    if logger is None:
        print(info)
    else:
        logger.log_info(info)


class ImageListDataset(Dataset):
    """
    This class can be used to load images when given a file contain the list of image paths
    """
    def __init__(self, 
                 name,
                 image_list_file='',
                 data_root='',
                 coord=False,
                 im_preprocessor_config={
                     'target': 'image_synthesis.data.utils.image_preprocessor.SimplePreprocessor',
                     'params':{
                        'size': 256,
                        'random_crop': True,
                        'horizon_flip': True
                        }
                 },
                 image_end_with='',
                 mask=-1.0, # mask probility to get a mask for image
                 mask_low_to_high=0.0,
                 mask_low_size=None, # height, width
                 zero_mask=0.0, # the probability to set the mask all zeros
                 provided_mask_name='',
                 provided_mask_list_file='',
                 use_provided_mask_ratio=[0.0, 1.0],
                 use_provided_mask=1.0, # other with random generated mask
                 image_mask_paired=False, # if true, one image should match with one mask in the given order
                 stroken_mask_params=None,
                 multi_image_mask=False, # if true, the returned image will be multiplied by the mask
                 erase_image_with_mask=-1.0,
                 return_data_keys=None,
                 ):
        super().__init__()
        self.name = name
        self.image_list_file = image_list_file
        self.erase_image_with_mask = erase_image_with_mask
        self.provided_mask_name = provided_mask_name
        self.provided_mask_list_file = provided_mask_list_file
        self.image_mask_paired = image_mask_paired
        self.use_provided_mask = use_provided_mask
        self.use_provided_mask_ratio = use_provided_mask_ratio
        
        if data_root == '':
            data_root = 'data'
        self.data_root = data_root
        root = os.path.join(data_root, self.name)

        if os.path.isfile(self.image_list_file):
            with open(self.image_list_file, "r") as f:
                relpaths = f.read().splitlines()
        elif self.image_list_file == '':
            assert image_end_with != ''
            image_end_with = image_end_with.split(',')
            if self.name == 'naturalscene':
                # onlu a sub set of natural scene
                relpaths = []
                sub_root = ['m/mountain','m/mountain_path','m/mountain_snowy','b/butte','c/canyon','f/field/cultivated','t/tundra','v/valley']
                # count_ = [5000, ]
                for sb in sub_root:
                    relpaths_ = get_all_file(os.path.join(root, sb), end_with=image_end_with, path_type='relative')
                    relpaths_ = [os.path.join(sb, rp) for rp in relpaths_]
                    relpaths = relpaths + relpaths_
            else:
                relpaths = get_all_file(root, end_with=image_end_with, path_type='relative')
        else:
            raise NotImplementedError
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, labels={'relative_path': relpaths})

        # get preprocessor
        self.preprocessor = instantiate_from_config(im_preprocessor_config)
        self.coord = coord
        self.mask = mask
        self.mask_low_to_high = mask_low_to_high
        self.mask_low_size = mask_low_size
        self.zero_mask = zero_mask
        self.stroken_mask_params = stroken_mask_params
        # import pdb; pdb.set_trace()
        self.set_provided_mask_ratio()
        self.multi_image_mask = multi_image_mask
        self.return_data_keys = return_data_keys

        self.debug = False # for debug

    def set_provided_mask_ratio(self, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.use_provided_mask_ratio
        else:
            self.use_provided_mask_ratio = mask_ratio
        if self.provided_mask_name != '' and self.use_provided_mask > 0:
            if self.provided_mask_list_file == '':
                mask_dir = os.path.join(self.data_root, self.provided_mask_name)
                mask_paths = get_all_file(mask_dir, end_with='.png', path_type='abs')
                mask_paths = sorted(mask_paths)
            else:
                with open(self.provided_mask_list_file, "r") as f:
                    relpaths = f.read().splitlines()
                mask_paths = [os.path.join(self.data_root, self.provided_mask_name, p) for p in relpaths]
            if mask_ratio is None:
                start = 0
                end = -1
            elif isinstance(mask_ratio[0], float):
                start = int(len(mask_paths) * mask_ratio[0])
                end = int(len(mask_paths) * mask_ratio[1])
            elif isinstance(mask_ratio[0], str):
                start = MASK_RATIO_INDEX[mask_ratio[0]]
                end = MASK_RATIO_INDEX[mask_ratio[1]]
            mask_paths = mask_paths[start:end]
            self.masks = ImagePaths(paths=mask_paths)
            my_print('Number of masks: {}, start: {}, end: {}'.format(end-start, start, end))
            if len(self.masks) == 0:
                self.masks = None
                my_print('Found masks length 0, set to None')
        else:
            self.masks = None
        
        if self.use_provided_mask <= 0:
            if self.stroken_mask_params is None:
                self.stroken_mask_params = {
                    'max_parts': 15,
                    'maxVertex': 25,
                    'maxLength': 100, 
                    'maxBrushWidth': 24
                }
            self.stroken_mask_params['keep_ratio'] = [1-float(mask_ratio[1]), 1-float(mask_ratio[0])]


    def __len__(self):
        if self.debug:
            return min(1000, len(self.data))
        return len(self.data)

    def get_mask(self, im_size, erase_mask=False, index=None):
        if self.masks is not None and random.random() < self.use_provided_mask:
            if self.image_mask_paired:
                assert len(self.masks) == len(self.data), 'If image and mask are paired with each other, the number of them should be the same!'
            else:
                index = random.randint(0, len(self.masks)-1)
            mask = self.masks[index]['image']
            mask = cv2.resize(mask, im_size[::-1], interpolation=cv2.INTER_NEAREST) # size [w, h]
            mask = 1 - mask / 255.0
            mask = mask[:, :, 0:1]
            # import pdb; pdb.set_trace()
        else:
            if self.stroken_mask_params is None:
                stroken_mask_params = {
                    'max_parts': 15,
                    'maxVertex': 25,
                    'maxLength': 100, 
                    'maxBrushWidth': 24
                }
            else:
                stroken_mask_params = self.stroken_mask_params
            stroken_mask_params['im_size'] = im_size 
            mask = generate_stroke_mask(**stroken_mask_params)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0:1]
        elif len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        else:
            raise ValueError('Invalide shape of mask:', mask.shape)
        
        if not erase_mask:
            if random.random() < self.zero_mask:
                mask = mask * 0
            else:
                if random.random() < self.mask_low_to_high:
                    assert isinstance(self.mask_low_size, (tuple, list))
                    mask = mask[:, :, 0].astype(np.uint8) # h, w
                    ori_size = mask.shape
                    mask = cv2.resize(mask, tuple(self.mask_low_size), interpolation=cv2.INTER_NEAREST)
                    mask = cv2.resize(mask, tuple(ori_size), interpolation=cv2.INTER_NEAREST)
                    mask = mask[:, :, np.newaxis]
        return mask # H W 1

    def __getitem__(self, index):
        data = self.data[index]
        
        if not self.coord:
            image = self.preprocessor(image=data['image'])['image']
            data['image'] = np.transpose(image.astype(np.float32), (2, 0, 1)) # 3 x H x W
        else:
            h, w, _ = data['image'].shape
            coord = (np.arange(h*w).reshape(h,w,1)/(h*w)).astype(np.float32)
            # import pdb; pdb.set_trace()
            out = self.preprocessor(image=data["image"], coord=coord)
            data['image'] = np.transpose(out["image"].astype(np.float32), (2, 0, 1))
            data["coord"] = np.transpose(out["coord"].astype(np.float32), (2, 0, 1))
        
        if random.random() < self.erase_image_with_mask:
            mask_ = self.get_mask(im_size=(data['image'].shape[1], data['image'].shape[2]), erase_mask=True) # H W 1
            mask_ = np.transpose(mask_.astype(data['image'].dtype), (2, 0, 1)) # 1 x H x W
            data['image'] = mask_ * data['image']
            data['erase_mask'] = mask_
        else:
            if self.return_data_keys is not None and 'erase_mask' in self.return_data_keys:
                data['erase_mask'] = np.ones((1, data['image'].shape[-2], data['image'].shape[-1]), dtype=np.float32)

        if random.random() < self.mask:
            mask = self.get_mask(im_size=(data['image'].shape[1], data['image'].shape[2]), index=index)
            data['mask'] = np.transpose(mask.astype(np.bool), (2, 0, 1)) # 1 x H x W

            if self.multi_image_mask:
                data['image'] = data['image'] * data['mask'].astype(np.float32)
        
        # data['image_hr'] = data['image'].copy()
        # data['mask_hr'] = data['mask'].copy()
        # image_lr = Image.fromarray(np.transpose(data['image'].astype(np.uint8), (1,2,0))).resize((256,256), resample=Image.BILINEAR)
        # data['image'] = np.transpose(np.array(image_lr).astype(np.float32), (2, 0, 1)) # 3 x H x W
        # mask_lr = Image.fromarray(data['mask'][0].astype(np.uint8)).resize((256,256), resample=Image.NEAREST)
        # data['mask'] = np.array(mask_lr).astype(np.bool)[np.newaxis, :,:] # 1 x H x W

        if self.return_data_keys is not None:
            data_out = {}
            for k in self.return_data_keys:
                data_out[k] = data[k]

            return data_out
        else:
            return data

    #######################################
    ##     functions used for inference   
    #######################################
    def get_mask_info(self, type='str'):
        if type == 'str':
            if self.use_provided_mask <= 0:
                keep_ratio = self.stroken_mask_params['keep_ratio']
                mask_ratio = [1-keep_ratio[1], 1-keep_ratio[0]]
                info = 'usm{}_mr{}_{}'.format(1-self.use_provided_mask, mask_ratio[0], mask_ratio[1])
            else:
                info = 'upm{}_mr{}_{}'.format(self.use_provided_mask, self.use_provided_mask_ratio[0], self.use_provided_mask_ratio[1])
        else:
            raise NotImplementedError('{}'.format(type))
        return info

    def remove_files(self, file_path_set, path_type='abs_path'):
        if path_type == 'abs_path':
            self.data.remove_files(file_path_set)
        elif path_type == 'relative_path':
            file_path_set_ = []
            for rp in list(file_path_set):
                p = os.path.join(self.data_root, self.name, rp)
                file_path_set_.append(p)
            self.data.remove_files(set(file_path_set_))
        else:
            raise NotImplementedError('path type: {}'.format(path_type))


class ImageListDeepFashion(Dataset):
    """
    This class can be used to load images when given a file contain the list of image paths
    """
    def __init__(self, 
                 name,
                 data_root='',
                 im_preprocessor_config={
                     'target': 'image_synthesis.data.utils.image_preprocessor.SimplePreprocessor',
                     'params':{
                        'size': 256,
                        'random_crop': True,
                        'horizon_flip': True
                        }
                 },
                 image_end_with='',
                 mask=False, # return mask
                 mask_path_replace=None,
                 return_data_keys=None
                 ):
        super().__init__()
        self.name = name
        
        if data_root != '':
            root = os.path.join(data_root, self.name)
        else:
            root = os.path.join("data", self.name)

        image_end_with = image_end_with.split(',')
        paths = get_all_file(root, end_with=image_end_with)
        self.data = ImagePaths(paths=paths)

        self.mask = mask
        self.mask_path_replace = mask_path_replace

        # get preprocessor
        self.preprocessor = instantiate_from_config(im_preprocessor_config)
        self.return_data_keys = return_data_keys

    def __len__(self):
        # return min(1200, len(self.data))#TODO
        return len(self.data)

    def load_mask(self, path):
        assert self.mask_path_replace is not None 
        path_mask = path
        for rep in self.mask_path_replace:
            path_mask = path_mask.replace(rep[0], rep[1])
        mask = Image.open(path_mask).convert('RGB')
        mask = np.array(mask).astype(np.float32)
        return mask

    def __getitem__(self, index):
        data = self.data[index]
        
        if not self.mask:
            image = self.preprocessor(image=data['image'])['image']
            data['image'] = np.transpose(image.astype(np.float32), (2, 0, 1)) # 3 x H x W
        else:

            mask = self.load_mask(data['abs_path']) # .astype(np.uint8)
            # import pdb; pdb.set_trace()
            out = self.preprocessor(image=data["image"], mask=mask)
            data['image'] = np.transpose(out["image"].astype(np.float32), (2, 0, 1))
            data["mask"] = np.transpose(out["mask"].astype(np.float32), (2, 0, 1))

        if self.return_data_keys is not None:
            data_out = {}
            for k in self.return_data_keys:
                data_out[k] = data[k]

            return data_out
        else:
            return data   



class ImageListImageNet(Dataset):
    """
    This class can be used to load images when given image_list_file
    """
    def __init__(self, 
                 name,
                 image_list_file='',
                 data_root='',
                 im_preprocessor_config={
                     'target': 'image_synthesis.data.utils.image_preprocessor.SimplePreprocessor',
                     'params':{
                        'size': 256,
                        'random_crop': True,
                        'horizon_flip': True
                        }
                 },
                 depth_name=None, # imagenet depth name
                 depth_list_file='',
                 file_end_with='.JPEG',
                 return_depth=False,
                 return_data_keys=None
                 ):
        super().__init__()
        self.name = name
        self.depth_name = depth_name
        assert self.name in ['imagenet/train', 'imagenet/val']
        self.image_list_file = image_list_file
        self.depth_list_file = depth_list_file
        self.data_root = data_root if data_root != '' else 'data'
        self.file_end_with = file_end_with
        self.return_depth = return_depth

        # load dataset
        self._load()

        # get preprocessor
        self.preprocessor = instantiate_from_config(im_preprocessor_config)

        self.return_data_keys = return_data_keys

    def _get_class_info(self, path_to_yaml="data/imagenet_class_to_idx.yaml"):
        with open(path_to_yaml) as f:
            class2id = yaml.full_load(f)
        id2class = {}
        for classn, idx in id2class.items():
            id2class[idx] = classn
        return class2id, id2class

    def _load(self):
        self.class2id, self.id2class = self._get_class_info()
        
        if os.path.isfile(self.image_list_file):
            with open(self.image_list_file, "r") as f:
                relative_path = f.read().splitlines()
            self.abspaths = [os.path.join(self.data_root, self.name, p) for p in relative_path]
            my_print('Found {} files with the given {}'.format(len(self.abspaths), self.image_list_file))
        else: # read 
            file_end_with = self.file_end_with.split(',')
            self.abspaths = get_all_file(os.path.join(self.data_root, self.name), end_with=file_end_with)
            my_print('Found {} files by searching {} with extensions {}'.format(len(self.abspaths), os.path.join(self.data_root, self.name), str(self.file_end_with)))
        
        # check if there is correspoing depth file
        if self.return_depth:
            if self.depth_list_file == '':
                depth_abspaths = None 
            else:
                with open(self.depth_list_file, "r") as f:
                    relative_depth_path = f.read().splitlines()
                depth_abspaths = [os.path.join(self.data_root, self.depth_name, p) for p in relative_depth_path]
                depth_abspaths = set(depth_abspaths)
            abspaths = []
            for im_path in self.abspaths:
                depth_path = self.get_depth_path(im_path)
                if depth_abspaths is None:
                    if os.path.exists(depth_path):
                        abspaths.append(im_path)
                else:
                    if depth_path in depth_abspaths:
                        abspaths.append(im_path)
            my_print('Filter {} files since they have no depth file'.format(len(self.abspaths)-len(abspaths)))
            self.abspaths = abspaths
        
        self.class_labels = [self.class2id[s.split(os.sep)[-2]] for s in self.abspaths]
        labels = {
            "abs_path": self.abspaths,
            "class_id": np.array(self.class_labels).astype(np.int64),
        }
        self.data = ImagePaths(paths=self.abspaths,
                               labels=labels)

    def __len__(self):
        # return min(600, len(self.data)) #TODO
        return len(self.data)

    def get_depth_path(self, im_path):
        depth = im_path.replace(self.name, self.depth_name).replace('.JPEG', '.png')
        return depth

    def load_depth(self, path):
        depth = np.array(Image.open(path))
        depth = rgba_to_depth(depth)
        depth = visualize_depth(depth)
        return depth

    def __getitem__(self, index):
        data = self.data[index]
        if self.return_depth:
            depth_path = self.get_depth_path(data['abs_path'])
            depth = self.load_depth(depth_path)
            transformed = self.preprocessor(image=data['image'], depth=depth)
            data['image'] = np.transpose(transformed['image'].astype(np.float32), (2, 0, 1))
            data['depth'] = transformed['depth'].astype(np.float32)[np.newaxis, :, :] # 1 X H x W
        else:
            transformed = self.preprocessor(image=data['image'])
            data['image'] = np.transpose(transformed['image'].astype(np.float32), (2, 0, 1))
        
        if self.return_data_keys is not None and 'class_name' in self.return_data_keys:
            data['class_name'] = imagenet_utils.IMAGENET_CLASSES[int(data['class_id'])]
            data['text'] = imagenet_utils.get_random_text_label(class_names=[data['class_name']])[0]

        if self.return_data_keys is None:
            return data
        else:
            data_ = {}
            for k in self.return_data_keys:
                data_[k] = data[k]
            return data_

class ImageListPriorInpainting(Dataset):
    """
    This class can be used to load images when given a file contain the list of image paths.
    It is implemented for image inpainting. It is similar with the dataset in https://github.com/knazeri/edge-connect/blob/master/src/dataset.py.
    But we replace the edge with other inferior content, such as low resolution image, which is also degraded by quantization.
    """
    def __init__(self, 
                 name,
                 image_list_file='',
                 mask_list_file=None,
                 stroken_mask_params=None,
                 prior_size=32, # h, w
                 prior_random_degree=2,
                 mask_low_to_high=-1.0,
                 data_root='',
                 im_preprocessor_config={
                     'target': 'image_synthesis.data.utils.image_preprocessor.SimplePreprocessor',
                     'params':{
                        'size': 256,
                        'random_crop': True,
                        'horizon_flip': True
                        }
                 },
                 pixel_kmens_center_path='data/kmeans_centers.npy',
                 image_end_with=''
        ):
        super().__init__()
        self.name = name
        self.image_list_file = image_list_file
        self.mask_list_file = mask_list_file
        self.stroken_mask_params = stroken_mask_params
        self.prior_size = [prior_size, prior_size] if isinstance(prior_size, int) else prior_size
        self.prior_random_degree = prior_random_degree
        self.mask_low_to_high = mask_low_to_high

        if data_root != '':
            root = os.path.join(data_root, self.name)
        else:
            root = os.path.join("data", self.name)

        # for images
        if os.path.isfile(self.image_list_file):
            with open(self.image_list_file, "r") as f:
                relpaths = f.read().splitlines()

            paths = [os.path.join(root, relpath) for relpath in relpaths]
        elif self.image_list_file == '': # directory
            assert image_end_with != ''
            image_end_with = image_end_with.split(',')
            paths = get_all_file(root, end_with=image_end_with)
            # import pdb; pdb.set_trace()
        else:
            raise NotImplementedError
        self.images = ImagePaths(paths=paths)

        # for masks
        if self.mask_list_file is not None:
            with open(self.mask_list_file, "r") as f:
                relpaths = f.read().splitlines()
            paths = [os.path.join(root, relpath) for relpath in relpaths]
            self.masks = ImagePaths(paths=paths)      
        else:
            self.masks = None 

        # for priors
        self.pixel_centers = np.load(pixel_kmens_center_path)
        self.pixel_centers = np.rint(127.5 * (1 + self.pixel_centers)) # map to origin [0-255]

        # get preprocessor
        self.preprocessor = instantiate_from_config(im_preprocessor_config)


    def __len__(self):
        # return 400
        return len(self.images)


    def get_mask(self, image, index=None):
        if self.masks is not None:
            if index is None or index >= len(self.masks):
                index = random.randint(0, len(self.masks)-1)
            mask = self.masks[index]
        else:
            im_size = image.shape[0:2]
            if self.stroken_mask_params is None:
                stroken_mask_params = {
                    'max_parts': 15,
                    'maxVertex': 25,
                    'maxLength': 100, 
                    'maxBrushWidth': 24
                }
            else:
                stroken_mask_params = self.stroken_mask_params
            stroken_mask_params['im_size'] = im_size 
            mask = generate_stroke_mask(**stroken_mask_params)
            # mask = generate_stroke_mask(im_size=im_size,
            #                             maxVertex=40,
            #                             maxLength=100, 
            #                             maxBrushWidth=50,
            #                             minVertex=5,
            #                             minBrushWidth=20)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0:1]
        elif len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        else:
            raise ValueError('Invalide shape of mask:', mask.shape)
        return mask

    def get_prior(self, image):
        """
        The inferior is infact the low resolution image, which is also
        be degraded by quantization.
        """
        def squared_euclidean_distance_np(a,b):
            b = b.T
            a2 = np.sum(np.square(a),axis=1)
            b2 = np.sum(np.square(b),axis=0)
            ab = np.matmul(a,b)
            d = a2[:,None] - 2*ab + b2[None,:]
            return d

        def color_quantize_np_topK(x, clusters,K):
            x = x.reshape(-1, 3)
            d = squared_euclidean_distance_np(x, clusters)
            # print(np.argmin(d,axis=1))
            top_K=np.argpartition(d, K, axis=1)[:,:K] 

            h,w=top_K.shape
            select_index=np.random.randint(w,size=(h))
            return top_K[range(h),select_index]

        def prior_degradation(img,clusters,prior_size,K=1): ## Downsample and random change

            LR_img_cv2=img.resize((prior_size[1], prior_size[0]), resample=Image.BILINEAR)
            LR_img_cv2=np.array(LR_img_cv2)

            token_id=color_quantize_np_topK(LR_img_cv2.astype(clusters.dtype),clusters,K)
            primers = token_id.reshape(-1,prior_size[0]*prior_size[1])
            primers_img = [np.reshape(clusters[s], [prior_size[0],prior_size[1], 3]).astype(np.uint8) for s in primers]

            degraded=Image.fromarray(primers_img[0])

            return degraded ## degraded by inferior cluster 

        h, w = image.shape[0:2]

        inferior = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        inferior = prior_degradation(inferior, self.pixel_centers, self.prior_size, K=self.prior_random_degree)
        # inferior = inferior.resize((w, h),resample=Image.BICUBIC)
        inferior = inferior.resize((w, h),resample=Image.BILINEAR)
        inferior = np.array(inferior).astype(np.uint8)

        # import torch
        # inferior = np.array(inferior)
        # inferior = torch.tensor(inferior).to(torch.float32) # H, W, C
        # inferior = torch.nn.functional.upsample(inferior.permute(2, 0, 1).unsqueeze(0), size=(h, w), mode='bilinear').squeeze(dim=0).permute(1, 2, 0).to(torch.uint8)
        # inferior = inferior.numpy()

        return inferior


    def __getitem__(self, index):
        data = self.images[index]
        
        # generator stroken mask
        mask = self.get_mask(image=data['image'])

        # augment image and mask
        
        im_mask = self.preprocessor(image=data['image'].astype(np.uint8), mask=mask.astype(np.uint8))
        image = im_mask['image']
        mask = im_mask['mask']
        
        if random.random() < self.mask_low_to_high:
            h, w = self.prior_size[0], self.prior_size[1]
            mask = Image.fromarray(mask[:, :, 0]).resize((w, h), resample=Image.NEAREST) # H , W
            h, w = image.shape[0:2]
            mask = mask.resize((w, h), resample=Image.NEAREST)
            mask = np.array(mask)[:, :, np.newaxis] # H x W x 1
            
        # get inferior
        # the inferior should be generated from the input image
        inferior = self.get_prior(image=image)

        data['image'] = np.transpose(image.astype(np.float32), (2, 0, 1)) # 3 x H x W
        data['mask'] = np.transpose(mask.astype(np.bool), (2, 0, 1)) # 1 x H x W
        data['inferior'] = np.transpose(inferior.astype(np.float32), (2, 0, 1)) # 3 x H x W

        return data


class ImageListCelebaAlign(Dataset):
    """
    This class can be used to load images when given image_list_file,  and it is implemented
    for celeba attributed, aligned faces
    """
    def __init__(self, 
                 name,
                 image_list_file,
                 data_root='',
                 load_landmark=False,
                 load_random_mask=False,
                 mask_based_on_landmark=True,
                 all_masked=-1.0,
                 im_preprocessor_config={
                     'target': 'image_synthesis.data.utils.image_preprocessor.SimplePreprocessor',
                     'params':{
                        'size': [208, 176],
                        'smallest_max_size': 178,
                        'random_crop': True,
                        'horizon_flip': True
                        }
                 },
                 im_preprocessor_config_hr=None, # It is useful for image completion
                 mask_low_to_high=True
                 
        ):
        super().__init__()
        self.name = name
        self.image_list_file = image_list_file
        self.data_root = data_root if data_root != '' else 'data'
        self.load_landmark = load_landmark
        self.load_random_mask = load_random_mask
        self.mask_based_on_landmark = float(mask_based_on_landmark)
        self.all_masked = all_masked

        self.im_preprocessor_config_hr = im_preprocessor_config_hr
        self.mask_low_to_high = mask_low_to_high
        # import pdb; pdb.set_trace()

        # Index of the attributes from celebA that will be ignored, start from 1. This is copied from
        # https://github.com/Guim3/IcGAN/blob/master/data/donkey_celebA.lua
        # ignored_attr_index = [1,2,3,4,7,8,11,14,15,17,20,24,25,26,28,30,31,35,37,38,39,40] # start from 1
        # ignored_attr_index = [0,1,2,3,6,7,10,13,14,16,19,23,24,25,27,29,30,34,36,37,38,39] # start from 0
        self.used_attr_index = [4,5,8,9,11,12,15,17,18,20,21,22,26,28,31,32,33,35] # start from 0
        self.num_attrs = len(self.used_attr_index)

        self.all_attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                         'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
                         'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
                         'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 
                         'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

        self.used_attr = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Gray_Hair', 'Heavy_Makeup',
                          'Male', 'Mouth_Slightly_Open', 'Mustache', 'Pale_Skin', 'Receding_Hairline', 'Smiling', 'Straight_Hair', 'Wavy_Hair','Wearing_Hat']
        
        self.attr_to_token = {self.used_attr[idx]: idx for idx in range(len(self.used_attr))}

        self.conflict_attr = {
            'Bald': ['Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Receding_Hairline', 'Straight_Hair', 'Wavy_Hair'], 
            'Bangs': ['Bald', 'Receding_Hairline'],
            'Black_Hair': ['Bald', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'],
            'Blond_Hair': ['Bald', 'Blck_Hair', 'Brown_Hair', 'Gray_Hair'],
            'Brown_hair': ['Bald', 'Black_Hair', 'Blond_Hair', 'Gray_Hair'],
            'Gray_hair': ['Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair'],
            'Heavy_Makeup': ['Male', 'Mustache'],
            'Male': ['Heavy_Makeup'],
            'Mustache': ['Heavy_Makeup'],
            'Straight_Hair': ['Wavy_Hair'],
            'Wavy_Hair': ['Straight_Hair']
        }

        # load dataset
        self._load()

        # get preprocessor
        self.preprocessor = instantiate_from_config(im_preprocessor_config)
        self.preprocessor_hr = instantiate_from_config(im_preprocessor_config_hr)
        # import pdb; pdb.set_trace()

    def _load_attribute(self, im_names):
        list_attr_path = os.path.join(self.data_root, self.name, 'Anno', 'list_attr_celeba.txt')
        with open(list_attr_path, 'r') as attr_f:
            attr_ = attr_f.readlines()
            attr_ = attr_[2:] # remove the first line: 202599 and attribute name
            attr_ = [a.replace('\n', '') for a in attr_]

            img_name_to_attr = {}
            attr_token = np.array([v for k, v in self.attr_to_token.items()]).astype(np.int64)
            for a in attr_:
                a = a.split(' ')
                a = [a_ for a_ in a if len(a_) > 0]
                im_name = a[0].replace('.jpg', '.png')
                attr_idx = np.array(a[1:])[self.used_attr_index].astype(np.int64)
                mask = attr_idx > 0
                attr_idx[mask] = attr_idx[mask] + attr_token[mask] - 1 # minus 1 so that the attribute can start from 0
                img_name_to_attr[im_name] = attr_idx

        attr_idx_list = []
        attr_name_list = []
        for im_name in im_names:
            attr_idx_list.append(img_name_to_attr[im_name])
            attr_name = np.array(self.used_attr)[img_name_to_attr[im_name] > -1].tolist()
            attr_name = ','.join(attr_name)
            attr_name_list.append(attr_name)
        
        return attr_idx_list, attr_name_list

    def _load_landmark(self, im_names):
        list_landmark_path = os.path.join(self.data_root, self.name, 'Anno', 'list_landmarks_align_celeba.txt')
        with open(list_landmark_path, 'r') as ld_f:
            landmark = ld_f.readlines()
            landmark = landmark[2:] # remove the first line: 202599 and landmark name
            landmark = [a.replace('\n', '') for a in landmark]

            img_name_to_landmark = {}
            margin = 0.4
            for ld in landmark:
                ld = ld.split(' ')
                ld = [ld_ for ld_ in ld if len(ld_) > 0]
                im_name = ld[0].replace('.jpg', '.png')
                ld = ld[1:11]
                ld = np.array([[int(ld[2*i]), int(ld[2*i+1])] for i in range(5)]).astype(np.int)
                area = [ld[:, 0].min(), ld[:, 1].min(), ld[:, 0].max(), ld[:, 1].max()] # x1, y1, x2, y2
                landmark_ = {
                    'coord': ld.tolist(),
                    'area': area
                }
                img_name_to_landmark[im_name] = landmark_

        landmark_list = []
        for im_name in im_names:
            landmark_list.append(img_name_to_landmark[im_name])
        
        return landmark_list



    def _load(self):
        with open(self.image_list_file, "r") as f:
            self.relative_path = f.read().splitlines()
        self.abspaths = [os.path.join(self.data_root, self.name, 'img_align_celeba', p ) for p in self.relative_path]

        # load annotation
        attr_idx_list, attr_name_list = self._load_attribute(self.relative_path)

        labels = {
            'relative_path': self.relative_path,
            'attribute_index': attr_idx_list,
            'attribute_name': attr_name_list,
        }
        # load landmark
        if self.load_landmark:
            landmark_list = self._load_landmark(self.relative_path)
            labels['landmark'] = landmark_list

        self.data = ImagePaths(paths=self.abspaths,
                               labels=labels)

    def __len__(self):
        # return 50
        return len(self.data)


    def get_data_for_ui_demo(self, index):
        data = self.data[index]
        data['file_name'] = os.path.basename(data['relative_path'])
        data['index'] = index
        return data

    def __getitem__(self, index):
        data = self.data[index]
        if self.load_random_mask:
            im_size = data['image'].shape[0:2]
            if random.random() < self.mask_based_on_landmark:
                landmark_coord = data['landmark']['coord']
                landmark_area = data['landmark']['area']
                mask = generate_mask_based_on_landmark(im_size=im_size,
                                                        landmark_coord=landmark_coord,
                                                        landmark_area=landmark_area,
                                                        maxBrushWidth=30,
                                                        area_margin=0.5)
            else:
                mask = generate_stroke_mask(im_size=im_size,
                                            max_parts=15,
                                            maxVertex=25,
                                            maxLength=100, 
                                            maxBrushWidth=24)
            
            if random.random() < self.all_masked:
                mask = mask * 0

            processed_data = self.preprocessor(image=data['image'].astype(np.uint8), mask=mask.astype(np.uint8))
            # import pdb; pdb.set_trace()
            if self.preprocessor_hr is not None:

                # get the mask from low resolution
                if self.mask_low_to_high:
                    processed_data_hr = self.preprocessor_hr(image=data['image'].astype(np.uint8))
                    data['image_hr'] = np.transpose(processed_data_hr['image'].astype(np.float32), (2, 0, 1)) # 3 x H x W
                    h, w = data['image_hr'].shape[-2:]

                    mask_hr = Image.fromarray(processed_data['mask'][:, :, 0]).resize((w, h), resample=Image.NEAREST) # H , W
                    mask_hr = np.array(mask_hr)[np.newaxis, :, :] # 1 x H x W
                    data['mask_hr'] = mask_hr.astype(np.bool)
                else:
                    processed_data_hr = self.preprocessor_hr(image=data['image'].astype(np.uint8), mask=mask.astype(np.uint8))
                    data['image_hr'] = np.transpose(processed_data_hr['image'].astype(np.float32), (2, 0, 1)) # 3 x H x W
                    data['mask_hr'] = np.transpose(processed_data_hr['mask'].astype(np.bool), (2, 0, 1))
                    
                    # resize mask from hr to lr
                    h, w = processed_data['image'].shape[0:2]
                    mask =  Image.fromarray(processed_data_hr['mask'][:, :, 0]).resize((w, h), resample=Image.NEAREST) # H , W
                    mask = np.array(mask)[:, :, np.newaxis] # H x W x 1
                    processed_data['mask'] = mask              

                # resize the high resolution image_hr to low resolution image
                image = Image.fromarray(processed_data_hr['image']).convert('RGB')
                h, w = processed_data['image'].shape[0:2]
                image = image.resize((w, h), resample=Image.BILINEAR)
                image = np.array(image)
                processed_data['image'] = image

            data['mask'] = np.transpose(processed_data['mask'].astype(np.bool), (2, 0, 1))
            data['image'] = np.transpose(processed_data['image'].astype(np.float32), (2, 0, 1))
   
        else:
            if self.preprocessor_hr is not None:
                image_hr = self.preprocessor_hr(image=data['image'].astype(np.uint8))['image']
                data['image_hr'] = np.transpose(image_hr.astype(np.float32), (2, 0, 1))
                
                # resize the high resolution image_hr to low resolution image
                image = Image.fromarray(image_hr).convert('RGB')
                h, w = self.preprocessor.size
                image = image.resize((w, h), resample=Image.BILINEAR)
                image = np.array(image)
            else:
                image = self.preprocessor(image=data['image'].astype(np.uint8))['image']
            data['image'] = np.transpose(image.astype(np.float32), (2, 0, 1))

        return data


class ImageListImageText(Dataset):
    def __init__(    
        self,             
        name,
        image_list_file,
        data_root='',
        load_random_mask=False,
        all_masked=-1.0,
        image_load_size=None, # height, width
        image_to_caption_dir_replace=['images/', 'captions/'],
        image_to_caption_ext_replace=['.jpg', '.txt'],
        im_preprocessor_config={
            'target': 'image_synthesis.data.utils.image_preprocessor.SimplePreprocessor',
            'params':{
            'size': [256, 256],
            'smallest_max_size': 256,
            'random_crop': True,
            'horizon_flip': True
            }
        },
        im_preprocessor_config_hr=None, # It is useful for image completion
        text_tokenizer_config=None,
        # args for image inpainting
        inferior_size=None, # h, w
        inferior_random_degree=2,
        mask_low_to_high=-1.0,
        mask_type=1,
        pixel_kmens_center_path='data/kmeans_centers.npy'
    ):
        self.name = name
        assert self.name in ['cub-200-2011', 'flowers102', 'multi-modal-celeba-hq']

        self.data_root = 'data' if data_root == '' else data_root
        self.image_list_file = image_list_file
        self.load_random_mask = load_random_mask
        self.mask_type = mask_type
        self.all_masked = all_masked
        self.image_to_caption_dir_replace = image_to_caption_dir_replace
        self.image_to_caption_ext_replace = image_to_caption_ext_replace
        self.image_load_size = image_load_size


        self.preprocessor = instantiate_from_config(im_preprocessor_config)
        if im_preprocessor_config_hr is not None:
            raise NotImplementedError
        self.preprocessor_hr = instantiate_from_config(im_preprocessor_config_hr)
        self.text_tokenizer = instantiate_from_config(text_tokenizer_config)


        # for priors
        self.inferior_size = inferior_size
        self.inferior_random_degree = inferior_random_degree
        self.mask_low_to_high = mask_low_to_high
        self.pixel_centers = np.load(pixel_kmens_center_path)
        self.pixel_centers = np.rint(127.5 * (1 + self.pixel_centers)) # map to origin [0-255]

        self._load()
        self._filter_based_on_text()

    def _load(self):
        # get image list
        with open(self.image_list_file, 'r') as f:
            relative_image_path = f.readlines()
            relative_image_path = [p.strip() for p in relative_image_path]
            f.close()
        
        # load captions
        abs_image_path = []
        image_path_to_captions = {}
        for p in relative_image_path:
            image_p = os.path.join(self.data_root, self.name, p)
            abs_image_path.append(image_p)
            p = p.replace(self.image_to_caption_dir_replace[0], self.image_to_caption_dir_replace[1])
            p = p.replace(self.image_to_caption_ext_replace[0], self.image_to_caption_ext_replace[1])
            text_p = os.path.join(self.data_root, self.name, p)
            with open(text_p, 'r') as txt_f:
                caps = txt_f.readlines()
                caps = [c.strip() for c in caps]
                caps = [c for c in caps if len(c) > 5] # for caption, we set it should be more than 5 charaters
                image_path_to_captions[image_p] = caps
                txt_f.close()
        self.abs_image_path = abs_image_path
        self.image_path_to_captions = image_path_to_captions

        # laad box if needed
        if self.name == 'cub-200-2011':
            image_path_to_box = {}
            image_to_box = load_dict_from_json(os.path.join(self.data_root, self.name, 'image_to_box.json'))
            for k in image_to_box:
                image_path_to_box[os.path.join(self.data_root, self.name, k)] = image_to_box[k]
            self.image_path_to_box = image_path_to_box


    def _filter_based_on_text(self):
        if self.text_tokenizer is not None:
            abs_im_path = []
            for im_path in self.abs_image_path:
                captions_ = copy.deepcopy(self.image_path_to_captions[im_path])
                captions = []
                for txt in captions_:
                    txt_token = self.text_tokenizer.get_tokens([txt])[0]
                    valid = self.text_tokenizer.check_length(txt_token)
                    if valid:
                        captions.append(txt)
                if len(captions) > 0:
                    self.image_path_to_captions[im_path] = captions
                    abs_im_path.append(im_path)
            my_print('Filter data done: from {} to {}'.format(len(self.abs_image_path), len(abs_im_path)))
            self.abs_image_path = abs_im_path

    def load_image(self, im_path, resize=True):
        im = Image.open(im_path)
        if not im.mode == "RGB":
            im = im.convert("RGB")
        
        if hasattr(self, 'image_path_to_box'):
            box = copy.deepcopy(self.image_path_to_box[im_path])
            
            width, height = im.size

            r = int(np.maximum(box[2], box[3]) * 0.75)
            center_x = int((2 * box[0] + box[2]) / 2)
            center_y = int((2 * box[1] + box[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            box = [x1, y1, x2, y2]
            im = im.crop(box)
        
        if self.image_load_size is not None and resize:
            im = im.resize((self.image_load_size[1], self.image_load_size[0]), Image.BILINEAR)
        im = np.array(im).astype(np.uint8)
        return im

    def load_caption(self, im_path):
        # import pdb; pdb.set_trace()
        captions = copy.deepcopy(self.image_path_to_captions[im_path])
        idx = random.randint(0, len(captions)-1)
        caption = captions[idx]
        return caption
    
    def load_mask(self, im):
        if self.mask_type == 1:
            mask = generate_stroke_mask(im_size=[256, 256],
                                        max_parts=15,
                                        maxVertex=25,
                                        maxLength=100, 
                                        maxBrushWidth=24) # H x W x 1
        elif self.mask_type == 2:
            mask = generate_stroke_mask(im_size=[256, 256], #[256, 256],
                                        max_parts=15,
                                        maxVertex=50,
                                        maxLength=100, 
                                        maxBrushWidth=40) # H x W x 1
        else:
            raise NotImplementedError
        if random.random() < self.all_masked:
            mask = mask * 0
        
        if self.inferior_size is not None and random.random() < self.mask_low_to_high:
            h, w = self.inferior_size[0], self.inferior_size[1]
            mask = Image.fromarray(mask[:, :, 0].astype(np.uint8))
            mask = mask.resize((w, h), resample=Image.NEAREST)
            mask = np.array(mask)[:, :, np.newaxis]

        return mask.astype(np.uint8)


    def load_inferior(self, im):
        """
        The inferior is infact the low resolution image, which is also
        be degraded by quantization.
        """
        def squared_euclidean_distance_np(a,b):
            b = b.T
            a2 = np.sum(np.square(a),axis=1)
            b2 = np.sum(np.square(b),axis=0)
            ab = np.matmul(a,b)
            d = a2[:,None] - 2*ab + b2[None,:]
            return d

        def color_quantize_np_topK(x, clusters,K):
            x = x.reshape(-1, 3)
            d = squared_euclidean_distance_np(x, clusters)
            # print(np.argmin(d,axis=1))
            top_K=np.argpartition(d, K, axis=1)[:,:K] 

            h,w=top_K.shape
            select_index=np.random.randint(w,size=(h))
            return top_K[range(h),select_index]

        def inferior_degradation(img,clusters,prior_size,K=1): ## Downsample and random change

            LR_img_cv2=img.resize((prior_size[1], prior_size[0]), resample=Image.BILINEAR)
            LR_img_cv2=np.array(LR_img_cv2)

            token_id=color_quantize_np_topK(LR_img_cv2.astype(clusters.dtype),clusters,K)
            primers = token_id.reshape(-1,prior_size[0]*prior_size[1])
            primers_img = [np.reshape(clusters[s], [prior_size[0],prior_size[1], 3]).astype(np.uint8) for s in primers]

            degraded=Image.fromarray(primers_img[0])

            return degraded ## degraded by inferior cluster 

        h, w = im.shape[0:2]

        inferior = Image.fromarray(im.astype(np.uint8)).convert("RGB")
        inferior = inferior_degradation(inferior, self.pixel_centers, self.inferior_size, K=self.inferior_random_degree)
        # inferior = inferior.resize((w, h),resample=Image.BICUBIC)
        inferior = inferior.resize((w, h),resample=Image.BILINEAR)
        inferior = np.array(inferior).astype(np.uint8)

        return inferior


    def __len__(self):
        return len(self.abs_image_path)

    def get_data_for_ui_demo(self, index):
        # data = self.data[index]
        # data['file_name'] = os.path.basename(data['relative_path'])
        # data['index'] = index
        # return data
        image_path = self.abs_image_path[index]
        im = self.load_image(image_path, resize=False)

        data = {}
        data['file_name'] = os.path.basename(image_path)
        data['image'] = im
        data['captions'] = self.image_path_to_captions[image_path]
        return data

    def __getitem__(self, index):
        image_path = self.abs_image_path[index]
        im = self.load_image(image_path)
        caption = self.load_caption(image_path)
        if self.load_mask:
            mask = self.load_mask(im)
        
        # preprocess image and mask
        if self.preprocessor is not None:
            im = self.preprocessor(image=im)['image']

        data = {
            'image': np.transpose(im.astype(np.float32), (2, 0, 1)), # 3 x H x W
            'text': caption
        }

        if self.load_mask:
            h, w = im.shape[0:2]
            mask = Image.fromarray(mask[:, :, 0]).resize((w, h), resample=Image.NEAREST)
            mask = np.array(mask)[:, :, np.newaxis]
            data['mask'] = np.transpose(mask.astype(np.bool), (2, 0, 1)) # 1 x H x W

        if self.inferior_size is not None:
            inferior = self.load_inferior(im)
            data['inferior'] = np.transpose(inferior.astype(np.float32), (2, 0, 1)) # 3 x H x W

        return data




if __name__ == '__main__':
    import cv2 
    from PIL import Image
    import albumentations
    from image_synthesis.data.utils.util import np_coord_form_mask

    dataset = ImageListCelebaAlign(name='celeba', 
                                image_list_file='data/celebaaligntrain.txt',
                                load_landmark=True,
                                load_random_mask=True,
                                mask_based_on_landmark=1.0,
                                all_masked=0.,
                                im_preprocessor_config={
                                    'target': 'image_synthesis.data.utils.image_preprocessor.SimplePreprocessor',
                                    'params':{
                                        'size': [52, 44],
                                        'smallest_max_size': 46,
                                        'random_crop': True,
                                        'horizon_flip': True,
                                        'add_noise': False,
                                        'change_brightness': True,
                                        }})

    for i in range(len(dataset)):
        batch = dataset.__getitem__(i)
        print(i)
        im = batch['image'].astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1]
        cv2.imshow('im', im)
        
        if 'mask' in batch:
            mask = batch['mask'].astype(np.float32).transpose(1, 2, 0)
            cv2.imshow('im_mask', (im * mask).astype(np.uint8))
        
        # inferior = batch['inferior'].astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1]
        # cv2.imshow('inferior', inferior)

        while cv2.waitKey(0) == 27:
            import sys
            sys.exit(0)


