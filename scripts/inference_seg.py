import os
import sys
from numpy.lib.arraysetops import isin
import torch
import json
import cv2
import random
import time
import argparse
import numpy as np
import warnings
import copy
from torch.nn.functional import l1_loss, mse_loss
from torch.serialization import save
import torchvision
import glob
from PIL import Image
from collections import OrderedDict, defaultdict
from torch.utils.data import ConcatDataset, dataset, DataLoader, DistributedSampler


from tqdm import tqdm

from image_synthesis.utils.io import load_yaml_config, load_dict_from_json, save_dict_to_json
from image_synthesis.utils.misc import get_all_file, get_all_subdir, instantiate_from_config
from image_synthesis.utils.cal_metrics import get_PSNR, get_mse_loss, get_l1_loss, get_SSIM, get_mae
from image_synthesis.modeling.build import build_model
from image_synthesis.utils.misc import format_seconds, merge_opts_to_config
from image_synthesis.distributed.launch import launch
from image_synthesis.distributed.distributed import get_rank, reduce_dict, synchronize, all_gather
from image_synthesis.utils.misc import get_model_parameters_info, get_model_buffer

from image_synthesis.modeling.modules.mask2former.inference_mask2former_infonly import Mask2Former, setup_cfg


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        if os.path.isfile(image_dir):
            image_paths = [image_dir]
            image_dir = ''
        else:
            image_paths = sorted(get_all_file(image_dir, path_type='relative', end_with=['.jpg', '.png', '.JPEG']))
        

        # import pdb; pdb.set_trace()
        self.image_dir = image_dir 
        # import pdb; pdb.set_trace()
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)


    def read_mask(self, mask_path):
        mask = Image.open(mask_path)
        # if not mask.mode == "RGB":
        #     mask = mask.convert("RGB")
        mask = np.array(mask).astype(np.float32)
        h, w = mask.shape[0], mask.shape[1]
        if (h,w) != tuple(self.size):
            mask = cv2.resize(mask, tuple(self.size), interpolation=cv2.INTER_NEAREST)
        mask = mask / 255.0
        if len(mask.shape) == 3:
            mask = mask[:, :, 0:1] # h, w, 1
        else:
            mask = mask[:, :, np.newaxis]
        mask = torch.tensor(mask).permute(2, 0, 1).bool() # 1, h, w
        return mask
    
    def read_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = np.array(image).astype(np.float32)
        return image


    def __getitem__(self, i):
        image_path = self.image_paths[i]
        image = self.read_image(os.path.join(self.image_dir, image_path))
        data = {
            'relative_path': image_path,
            'image': image,
            # 'mask': mask
        }

        return data

    def remove_files(self, relative_paths):
        source_ext = self.image_paths[0].split('.')[-1]
        process_ext = relative_paths[0].split('.')[-1]
        if source_ext != process_ext:
            relative_paths = [p.replace('.'+process_ext, '.'+source_ext) for p in relative_paths]
        image_paths = set(self.image_paths) - set(relative_paths)

        print('Remove files done! Before {}, after {}'.format(len(self.image_paths), len(image_paths)))
        self.image_paths = sorted(list(image_paths))


def seg_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    new_batch = {}
    for k in batch[0].keys():
        batch_k = []
        for b in batch:
            batch_k.append(b[k])
        new_batch[k] = batch_k
    return new_batch



def inference_seg(local_rank=0, args=None):
    config_path = 'image_synthesis/modeling/modules/mask2former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_50ep.yaml'
    config = setup_cfg(config_path)

    model = Mask2Former(
        config_file=config,
        model_weight='OUTPUT/Mask2Former/checkpoint/model_final_54b88a.pkl'
    )
    model = model.cuda()

    accumulate_time = None

    # save_root = 'RESULT/places/gt'
    save_root = args.save_root

    save_root_tmp = save_root
    os.makedirs(save_root_tmp, exist_ok=True)
    print('results will be saved in {}'.format(save_root_tmp))
    
    data = ImagePathDataset(image_dir=args.image_dir)

    if args.world_size > 1:
        sampler = DistributedSampler(data, num_replicas=args.world_size, rank=get_rank())
    else:
        sampler = None

    dataloader = DataLoader(data, batch_size=4, sampler=sampler, collate_fn=seg_collate, num_workers=4)

    for data_i in tqdm(dataloader):
        save_root_ = save_root_tmp
       
        with torch.no_grad():
            content_dict = model(
                copy.deepcopy(data_i['image']),
            ) # B x C x H x W

        ext = data_i['relative_path'][0].split('.')[-1]

        for i, p in enumerate(data_i['relative_path']):
            save_path = os.path.join(save_root_, p.replace(ext, 'png'))

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            Image.fromarray(content_dict[i].detach().cpu().numpy().astype(np.uint8)).save(save_path)


def get_args():
    parser = argparse.ArgumentParser(description='Inference script for segmentation')

    parser.add_argument('--save_dir', type=str, default='', 
                        help='directory to save results') 

    parser.add_argument('--func', type=str, default='inference_seg', 
                        help='the name of inference function')

    parser.add_argument('--image_dir', type=str, default='data/inpainting-image-mask/naturalscene/gt',
                        help='gt images need to be loaded') 
    parser.add_argument('--save_root', type=str, default='RESULT/naturalscene/gt',
                        help='gt images need to be loaded') 
    # args for ddp
    parser.add_argument('--num_node', type=int, default=1,
                        help='number of nodes for distributed training')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', type=str, default='auto', 
                        help='url used to set up distributed training')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')

    args = parser.parse_args()
    args.cwd = os.path.abspath(os.path.dirname(__file__))

    return args





inference_func_map = {
    'inference_seg': inference_seg,
}

if __name__ == '__main__':

    args = get_args()

    # args.gpu = 0

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable ddp.')
        torch.cuda.set_device(args.gpu)
        args.ngpus_per_node = 1
        args.world_size = 1
    else:
        if args.num_node == 1:
            args.dist_url == "auto"
        else:
            assert args.num_node > 1
        args.ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.ngpus_per_node * args.num_node

    args.distributed = args.world_size > 1

    launch(inference_func_map[args.func], args.ngpus_per_node, args.num_node, args.node_rank, args.dist_url, args=(args,))

