"""
This file is used for calculating the metrics when given gt directory and result directory.
Each gt has a corresponding result with the same name.

"""

import os
import json
import argparse
import tqdm
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
from image_synthesis.data.utils.image_path_dataset import ImagePaths
from image_synthesis.utils.misc import instantiate_from_config, get_all_file
from image_synthesis.utils.cal_metrics import get_l1_loss, get_mae, get_SSIM, get_PSNR, get_mse_loss


IMAGE_EXTENSIONS = ['bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp', 'JPEG']

class SimpleDataset(Dataset):
    def __init__(self, gt_file_names, res_file_names, gt_dir, result_dir, preprocessor=None):
        gt_paths = [os.path.join(gt_dir, f) for f in gt_file_names]
        res_paths = [os.path.join(result_dir, f) for f in res_file_names]

        self.gt_data = ImagePaths(gt_paths, labels={'relative_path':gt_file_names})
        self.res_data = ImagePaths(res_paths, labels={'relative_path':res_file_names})
        self.preprocessor = preprocessor
    
    def __getitem__(self, idx):

        gt = self.gt_data[idx]
        res = self.res_data[idx]

        if tuple(gt['image'].shape[:2]) != self.preprocessor.size:
            gt_image = self.preprocessor(image=gt['image'])['image'] # H x W x 3
        else:
            gt_image = gt['image']
        if tuple(res['image'].shape[:2]) != self.preprocessor.size:
            res_image = self.preprocessor(image=res['image'])['image'] # H x W x 3
        else:
            res_image = res['image']

        # convert to rgb image
        # import pdb; pdb.set_trace()
        gt_image = cv2.cvtColor(gt_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gt_image = gt_image[..., np.newaxis].astype(np.float32)
        res_image = cv2.cvtColor(res_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        res_image = res_image[..., np.newaxis].astype(np.float32)
        # print(gt['relative_path'], res['relative_path'])
        out = {
            'gt': np.transpose(gt_image, (2, 0, 1)),
            'result': np.transpose(res_image, (2, 0, 1)),
            # 'gt_name': [gt['relative_path']],

        }

        return out

    def __len__(self):
        return len(self.gt_data)


def get_im_preprocess(args):
    
    if args.im_size is None:
        cfg = {
            'target': 'image_synthesis.data.utils.image_preprocessor.SimplePreprocessor',
            'params': {
                'size': None
            }
        }
    else:
        cfg = {
            'target': 'image_synthesis.data.utils.image_preprocessor.SimplePreprocessor',
            'params': {
                'size': args.im_size,
                'smallest_max_size': args.im_size
            }
        }
    preprocessor = instantiate_from_config(cfg)

    return preprocessor


def get_dataset(args):
    if args.file_list == '':
        gt_file_names = get_all_file(args.gt_dir, IMAGE_EXTENSIONS, path_type='relative')
        res_file_names = get_all_file(args.result_dir, IMAGE_EXTENSIONS, path_type='relative')
        # import pdb; pdb.set_trace()
        gt_ext = '.' + gt_file_names[0].split('.')[-1]
        res_ext = '.' + res_file_names[0].split('.')[-1]
        if gt_ext != res_ext:
            gt_file_names = set([f.replace(gt_ext, '') for f in gt_file_names])
            res_file_names = set([f.replace(res_ext, '') for f in res_file_names])
            
            files = sorted(list(gt_file_names & res_file_names))
            gt_file_names = [f+gt_ext for f in files]
            res_file_names = [f+res_ext for f in files]
        else:
            # gt_file_names = sorted(list(set(gt_file_names) & set(res_file_names)))
            # res_file_names = gt_file_names

            gt_file_names = sorted(gt_file_names) 
            res_file_names = sorted(res_file_names)     
        # import pdb; pdb.set_trace()
    else:
        with open(args.file_list) as f:
            file_names = f.readlines()
            file_names = sorted([l.strip() for l in file_names])
        gt_file_names = file_names
        res_file_names = file_names      
    preprocessor = get_im_preprocess(args)

    # len_ = min(len(gt_file_names), len(res_file_names))
    # gt_file_names = gt_file_names[:len_]
    # res_file_names = res_file_names[:len_]

    if args.same_file:
        gt_file_names = set(gt_file_names)
        res_file_names = set(res_file_names)
        overlap = gt_file_names & res_file_names
        gt_file_names = list(overlap)
        res_file_names = list(overlap)


    dataset = SimpleDataset(
        gt_file_names=gt_file_names,
        res_file_names=res_file_names,
        gt_dir=args.gt_dir,
        result_dir=args.result_dir,
        preprocessor=preprocessor,
    )

    return dataset
            



def main(args):
    if args.result_dir.endswith(os.path.sep):
        args.result_dir = args.result_dir[:-len(os.path.sep)]
    if args.gt_dir.endswith(os.path.sep):
        args.gt_dir = args.gt_dir[:-len(os.path.sep)]
    
    dataset = get_dataset(args)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False
    )

    metrics = {
        'l1': 0.0,
        'mse': 0.0,
        'mae': 0.0,
        'psnr': 0.0,
        'ssim': 0.0
    }
    count = 0
    for batch in tqdm.tqdm(dataloader):
        gt = batch['gt']
        res = batch['result']

        bs = gt.shape[0]
        count += bs 

        # if count > 100:
        #     break

        l1 = get_l1_loss(gt, res)
        mse = get_mse_loss(gt, res)
        mae = get_mae(gt, res)
        psnr = get_PSNR(gt, res, tool='skimage')
        ssim = get_SSIM(gt, res, full=False, win_size=51)

        metrics['l1'] += l1 * bs
        metrics['mse'] += mse * bs 
        metrics['mae'] += mae * bs
        metrics['psnr'] += psnr * bs
        metrics['ssim'] += ssim * bs

    
    for k in metrics.keys():
        metrics[k] = float(metrics[k]/count)
    print(metrics)
    save_path = os.path.join(args.result_dir, '..', 'metrics_{}_{}.json'.format(args.result_dir.split(os.sep)[-1], len(dataset)))
    json.dump(metrics, open(save_path, 'w'), indent=4)
    print('saved to {}'.format(save_path))



def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--gt_dir', type=str, default='', 
                        help='directory to of ground truth') 
    parser.add_argument('--result_dir', type=str, default='', 
                        help='directory to of generated results') 
    parser.add_argument('--file_list', type=str, default='', 
                        help='relative path of file name. It is optional') 
    
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='batch size for dataloader')
    parser.add_argument('--num_workers', type=int, default=1, 
                        help='numberf of workers for dataloader')   

    parser.add_argument('--im_size', type=int, default=None, 
                        help='the resolution to be tested')   

    parser.add_argument('--same_file', action='store_true', default=False, 
                        help='numberf of workers for dataloader')                  
    args = parser.parse_args()
    args.cwd = os.path.abspath(os.path.dirname(__file__))

    return args


if __name__ == '__main__':
    args = get_args()
    main(args)