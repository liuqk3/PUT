from dataclasses import replace
import os
import numpy as np
import yaml
import copy
import cv2
import random
from torch.utils.data import Dataset
from PIL import Image
from image_synthesis.data.utils.image_path_dataset import ImagePaths
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


def seg2unknown(seg_map, unknow_class_param):

    d_type = seg_map.dtype
    seg_map = seg_map.astype(np.uint8)
    # cnt = 0
    replace_cnt = np.random.randint(low=1, high=unknow_class_param['max_number'])

    # probability to add unknown token
    p = np.random.uniform(low=0, high=1)

    num_unknow_class = unknow_class_param['num_unknow_classes']
    unknown_class_start = unknow_class_param['unknow_start']

    if p < unknow_class_param['prob_unknow']:

        lb_known = list(np.unique(seg_map))
        replace_cnt = min(replace_cnt, len(lb_known))
        lb_unknown = [i for i in range(unknown_class_start, num_unknow_class+unknown_class_start)]

        for i in range(replace_cnt):
            replace_class = np.random.choice(lb_known)
            lb_uk = np.random.choice(lb_unknown)

            mark = (seg_map == replace_class).astype(seg_map.dtype)
            seg_map = (1 - mark) * seg_map + mark * lb_uk
            lb_unknown.remove(lb_uk)
            lb_known.remove(replace_class)

    seg_map = seg_map.astype(np.float32)
    return seg_map


class ImageListDataset(Dataset):
    """
    This class can be used to load images when given a file contain the list of image paths
    """
    def __init__(self, 
                 name, # the name of dataset
                 image_list_file='', # the path to the file that contains relative paths
                 load_sketch_map=False,
                 load_segmentation_map=False,
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
                 mask=-1.0, #  probility to get a inpainting mask for image
                 zero_mask=0.0, # the probability to set the mask all zeros
                 mask_low_to_high=0.0, # the probability to download and upload masks
                 mask_low_size=None, # height, width
                 provided_mask_name='',
                 provided_mask_list_file='',
                 use_provided_mask_ratio=[0.0, 1.0],
                 use_provided_mask=1.0, # other with random generated mask
                 image_mask_paired=False, # if true, one image should match with one mask in the given order
                 stroken_mask_params=None,
                 multi_image_mask=False, # if true, the returned image will be multiplied by the mask
                 erase_image_with_mask=-1.0, # the probability to erase some pixels with erase mask
                 return_data_keys=None,
                 mode="RGB",
                 unknow_class_param=None,
                 ):
        super().__init__()
        self.name = name
        self.erase_image_with_mask = erase_image_with_mask
        self.provided_mask_name = provided_mask_name
        self.provided_mask_list_file = provided_mask_list_file
        self.image_mask_paired = image_mask_paired
        self.use_provided_mask = use_provided_mask
        self.use_provided_mask_ratio = use_provided_mask_ratio
        
        if data_root == '':
            data_root = 'data'
        self.data_root = data_root

        self.unknown_class_param = unknow_class_param

        self.mode = mode

        # get image file
        image_relative_paths = self.get_file_list(root=os.path.join(data_root, name), name=name, file_list=image_list_file, file_end_with=image_end_with)
        image_paths = [os.path.join(data_root, name, relpath) for relpath in image_relative_paths]

        if load_segmentation_map:
            # import pdb; pdb.set_trace()
            semgmentation_paths = self.get_file_list(root=os.path.join(self.data_root, 'segmentation', name), name=name, file_list=image_list_file, file_end_with='.png', path_type='abs')
            # semgmentation_paths = self.get_file_list(root=os.path.join(self.data_root, 'sketch', name), name=name, file_list=image_list_file, file_end_with='.png', path_type='abs')
            if image_list_file != '':
                semgmentation_paths = [os.path.join(self.data_root, 'segmentation', name, relpath) for relpath in semgmentation_paths]
            
            assert len(semgmentation_paths) == len(image_paths), 'the number of segmentation maps should be the same with images, image: {}, maps: {}'.format(len(image_paths), len(semgmentation_paths))
        else:
            semgmentation_paths = None
        
        if load_sketch_map:
            sketch_paths = self.get_file_list(root=os.path.join(self.data_root, 'sketch', name), name=self.name, file_list=image_list_file, file_end_with='.png', path_type='abs')
            if image_list_file != '':
                sketch_paths = [os.path.join(self.data_root, 'sketch', name, relpath) for relpath in sketch_paths]
            assert len(sketch_paths) == len(image_paths), 'the number of sketch maps should be the same with images, image: {}, maps: {}'.format(len(image_paths), len(sketch_paths))
        else:
            sketch_paths = None
            
        self.data = ImagePaths(paths=image_paths, sketch_paths=sketch_paths, segmentation_paths=semgmentation_paths, labels={'relative_path': image_relative_paths}, mode=mode)

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

    def get_file_list(self, root, name, file_list, file_end_with='.png', path_type='relative'):
        # import pdb; pdb.set_trace()
        if os.path.isfile(file_list):
            with open(file_list, "r") as f:
                relpaths = f.read().splitlines()
        elif file_list == '':
            assert file_end_with != ''
            file_end_with = file_end_with.split(',')
            if name == 'naturalscene':
                # only a sub set of natural scene
                relpaths = []
                sub_root = ['m/mountain','m/mountain_path','m/mountain_snowy','b/butte','c/canyon','f/field/cultivated','t/tundra','v/valley']
                # count_ = [5000, ]
                for sb in sub_root:
                    if path_type == 'relative':
                        relpaths_ = get_all_file(os.path.join(root, sb), end_with=file_end_with, path_type='relative')
                        relpaths_ = [os.path.join(sb, rp) for rp in relpaths_]
                    elif path_type == 'abs':
                        relpaths_ = get_all_file(os.path.join(root, sb), end_with=file_end_with, path_type='abs')
                    else:
                        raise NotImplementedError
                    relpaths = relpaths + relpaths_
            else:
                relpaths = get_all_file(root, end_with=file_end_with, path_type=path_type)
        else:
            raise NotImplementedError
        relpaths = sorted(relpaths)
        return relpaths

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
                start = int(20000 * mask_ratio[0])
                end = int(20000 * mask_ratio[1])
            elif isinstance(mask_ratio[0], str):
                start = MASK_RATIO_INDEX[mask_ratio[0]]
                end = MASK_RATIO_INDEX[mask_ratio[1]]
            mask_paths = mask_paths[start:end]
            self.erase_masks = ImagePaths(paths=mask_paths)
            my_print('Number of masks: {}, start: {}, end: {}'.format(end-start, start, end))
            if len(self.erase_masks) == 0:
                self.erase_masks = None
                my_print('Found masks length 0, set to None')
        else:
            self.erase_masks = None
        
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

    def get_erase_mask(self, im_size, erase_mask=False, index=None):
        if self.erase_masks is not None and random.random() < self.use_provided_mask:
            if self.image_mask_paired:
                assert len(self.erase_masks) == len(self.data), 'If image and mask are paired with each other, the number of them should be the same!'
            else:
                index = random.randint(0, len(self.erase_masks)-1)
            mask = self.erase_masks[index]['image']
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
            image = data['image']
            maps = []
            map_keys = []
            if 'segmentation_map' in data:
                maps.append(data['segmentation_map'])
                map_keys.append('segmentation_map')
            if 'sketch_map' in data:
                maps.append(data['sketch_map'])
                map_keys.append('sketch_map')
            if len(maps) > 0:
                augmented_data = self.preprocessor(image=image, masks=maps)
            else:
                augmented_data = self.preprocessor(image=image)

            if self.mode == 'L':
                data['image'] = augmented_data['image'].astype(np.float32)[np.newaxis, :, :]
                if self.unknown_class_param is not None:
                    data['image'] = seg2unknown(data['image'], self.unknown_class_param)
            elif self.mode == 'RGB':
                data['image'] = np.transpose(augmented_data['image'].astype(np.float32), (2, 0, 1)) # 3 x H x W
            else:
                raise NotImplementedError
            # data['image'] = np.transpose(augmented_data['image'].astype(np.float32), (2, 0, 1)) # 3 x H x W
            for i in range(len(map_keys)):
                # import pdb; pdb.set_trace()
                data[map_keys[i]] = augmented_data['masks'][i].astype(np.float32)[None, :, :] # 1 x H x W

        else: # the following code is not used
            h, w, _ = data['image'].shape
            coord = (np.arange(h*w).reshape(h,w,1)/(h*w)).astype(np.float32)
            # import pdb; pdb.set_trace()
            out = self.preprocessor(image=data["image"], coord=coord)
            data['image'] = np.transpose(out["image"].astype(np.float32), (2, 0, 1))
            data["coord"] = np.transpose(out["coord"].astype(np.float32), (2, 0, 1))
        
        if random.random() < self.erase_image_with_mask:
            mask_ = self.get_erase_mask(im_size=(data['image'].shape[1], data['image'].shape[2]), erase_mask=True) # H W 1
            mask_ = np.transpose(mask_.astype(data['image'].dtype), (2, 0, 1)) # 1 x H x W
            data['image'] = mask_ * data['image']
            data['erase_mask'] = mask_
        else:
            if self.return_data_keys is not None and 'erase_mask' in self.return_data_keys:
                data['erase_mask'] = np.ones((1, data['image'].shape[-2], data['image'].shape[-1]), dtype=np.float32)

        if random.random() < self.mask:
            mask = self.get_erase_mask(im_size=(data['image'].shape[1], data['image'].shape[2]), index=index)
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


if __name__ == '__main__':
    import cv2 
    from image_synthesis.utils.misc import instantiate_from_config
    from image_synthesis.utils.io import load_yaml_config

    # config_path = 'configs/test_image_segmentation_sketch.yaml'
    config_path = 'configs/test_seg_only.yaml'
    dataset_config = load_yaml_config(config_path)
    dataset_config = dataset_config['dataloader']['train_datasets'][0]
    dataset = instantiate_from_config(dataset_config)

    for i in range(len(dataset)):
        batch = dataset.__getitem__(i)
        print(i)
        im = batch['image'].astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1]
        cv2.imshow('im', im)
        
        if 'mask' in batch:
            mask = batch['mask'].astype(np.float32).transpose(1, 2, 0)
            cv2.imshow('im_mask', (im * mask).astype(np.uint8))
        
        if 'segmentation_map' in batch:
            seg_map = batch['segmentation_map'].astype(np.float32).transpose(1, 2, 0)
            seg_map = np.tile(seg_map, (1,1,3))
            cv2.imshow('seg_map', seg_map.astype(np.uint8))
        if 'sketch_map' in batch:
            ske_map = batch['sketch_map'].astype(np.float32).transpose(1, 2, 0)
            ske_map = np.tile(ske_map, (1,1,3))
            cv2.imshow('ske_map', ske_map.astype(np.uint8))

        while cv2.waitKey(0) == 27:
            import sys
            sys.exit(0)

