import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import numpy as np
import torch
import random
import lpips
import glob
import torchvision.transforms as TF
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from image_synthesis.utils.misc import get_all_file, get_all_subdir
from image_synthesis.utils.io import save_dict_to_json

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--path1', type=str, default='RESULT/taming_facehq_coordCond_learnRandomCardinal16Softmax_orthoEquipartionLoss_normEmb_normFeat_KlLossAll_transformerTemp50_val_e99/cond1_cont0_fr0_image_k16_t1',
                    help='Paths to the images')
parser.add_argument('--count', type=int, default=None,
                    help='count of images in path to be coumputed') 
parser.add_argument('--im_size', type=str, default='',
                    help='count of images in path1 to be coumputed') 
parser.add_argument('--type', type=str, default='diversity',
                    help='which type of LPIPS to be computed, i.e. for Diversity or Perceptron?') 

# args for diversity   
parser.add_argument('--loops', type=int, default=5,
                    help='count of images in path1 to be coumputed')  

# args for perceptron
parser.add_argument('--path2', type=str, default='',
                    help='Paths to the images')



parser.add_argument('--net', type=str, default='vgg', 
                    choices=['vgg', 'alex'],
                    help='which type of network to use?')  


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp', 'JPEG'}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None, count=None):
        self.files = files
        self.transforms = transforms
        self.count = count

    def __len__(self):
        if self.count is not None:
            return min(self.count, len(self.files))
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        try:
            img = Image.open(path).convert('RGB')
        except:
            raise RuntimeError('File invalid: {}'.format(path))
        if self.transforms is not None:
            img = self.transforms(img)
        img = (img-0.5) / 0.5
        return img


    def get_path(self, index):
        return self.files[index]


def get_dataset(path, count=None, im_size=None):
    # files = get_all_file(path, end_with=IMAGE_EXTENSIONS)
    if isinstance(path, str):
        files = glob.glob(os.path.join(path, '*completed*.png'))
        if len(files) == 0: # TODO for ICT 
            files = glob.glob(os.path.join(path, '*_image_*.png'))
            if len(files) == 0:
                files = glob.glob(os.path.join(path, '*.png'))
            assert len(files) == 10
        random.shuffle(files)
    elif isinstance(path, list):
        files = path
    else:
        raise NotImplementedError
    if count is not None and count > len(files):
        files = files[:count]
    
    # print('Evaluate {} images'.format(len(files)))
    if im_size is not None:
        transforms = TF.Compose([
            TF.Resize(im_size),
            TF.CenterCrop(im_size),
            TF.ToTensor()
        ])
    else:
        transforms = TF.ToTensor()

    dataset = ImagePathDataset(files, transforms=transforms)
    return dataset


def calculate_lpips_value_diversity(path, device, net, loops, count=None, im_size=None):

    sub_dirs = sorted(get_all_subdir(path, max_depth=2, min_depth=2, path_type='abs'))
    if len(sub_dirs) == 0:
        sub_dirs = sorted(get_all_subdir(path, max_depth=1, min_depth=1, path_type='abs'))
    # import pdb; pdb.set_trace()
    loss = lpips.LPIPS(net=net, spatial=True).to(device)
    value = []
    for sdi in tqdm(range(len(sub_dirs))):
        # if sdi > 10:
        #     break
        sd = sub_dirs[sdi]
        # print('Processing {}/{}...'.format(sdi, len(sub_dirs)))
        dataset = get_dataset(sd, count, im_size)
    
        processed_pair = set([])
        im_index = list(range(len(dataset)))

        # value = []
        sampled_index = set([])
        with torch.no_grad():
            bar = range(loops)
            if loops > 100:
                bar = tqdm(bar)
            for l in bar:
                # print('{}/{}'.format(l, loops))
                if len(im_index) <= 2:
                    continue
                two_idx = tuple(sorted(random.sample(im_index, 2)))
                # if len(sampled_index) >= len(im_index) // 2:
                #     two_idx = tuple(sorted(random.sample(im_index, 2)))
                # else:
                #     cnt = len(sampled_index)
                #     two_idx = (cnt*2, cnt*2+1)
                sampled_index.add(two_idx)
                while two_idx in processed_pair:
                    two_idx = tuple(sorted(random.sample(im_index, 2)))

                processed_pair.add(two_idx)

                im1 = dataset[two_idx[0]].unsqueeze(dim=0).to(device)
                im2 = dataset[two_idx[1]].unsqueeze(dim=0).to(device)
                
                # path1 = dataset.get_path(two_idx[0])
                # path2 = dataset.get_path(two_idx[1])
                # try:
                #     im1 = lpips.im2tensor(lpips.load_image(path1)).to(device)
                #     im2 = lpips.im2tensor(lpips.load_image(path2)).to(device)
                # except:
                #     print(path1)
                #     print(path2)
                #     raise RuntimeError('Load file error!')

                v = loss.forward(im1, im2).detach().mean().to('cpu')
                value.append(v)
                # import pdb; pdb.set_trace()
        
    if len(value) > 0:
        value = torch.stack(value, dim=0)

    overall_mean = (float(value.sum()/len(value))) if len(value)> 0 else 0
    overall_std = (float(value.std())) if len(value)> 0 else 0
    print('number of pairs: {}'.format(len(value)))
    statics = {
        'overall_mean': overall_mean,
        'overall_std': overall_std
    }

    last_dir = path.split(os.path.sep)[-1]
    save_path = os.path.join(path, '..', 'lpips_{}_subd_{}_count{}_loops{}_diversity.json'.format(last_dir, len(sub_dirs), count, loops))
    save_dict_to_json(statics, save_path, indent=4)
    print('saved to {}'.format(save_path))
    return statics


def calculate_lpips_value_perceptron(path1, path2, device, net, count=None, im_size=None):
    if path1.endswith(os.path.sep):
        path1 = path1[:-len(os.path.sep)]
    if path2.endswith(os.path.sep):
        path2 = path2[:-len(os.path.sep)]

    files1 = sorted(get_all_file(path1, end_with=IMAGE_EXTENSIONS, path_type='abs'))
    files2 = sorted(get_all_file(path2, end_with=IMAGE_EXTENSIONS, path_type='abs'))
    assert len(files1) == len(files2), 'Images in two directories are not the same! {}, {}'.format(len(files1), len(files2))

    dataset1 = get_dataset(files1, count, im_size)
    dataset2 = get_dataset(files2, count, im_size)
    loss = lpips.LPIPS(net=net, spatial=True).to(device)
    value = []
    indices = list(range(len(dataset1)))
    for idx in tqdm(indices):
        im1 = dataset1[idx].unsqueeze(dim=0).to(device)
        im2 = dataset2[idx].unsqueeze(dim=0).to(device)

        # imp1 = dataset1.get_path(idx)
        # imp2 = dataset2.get_path(idx)
        # try:
            # im1 = lpips.im2tensor(lpips.load_image(imp1)).to(device)
        #     im2 = lpips.im2tensor(lpips.load_image(imp2)).to(device)
        # except:
        #     print(imp1)
        #     print(imp2)
        #     raise RuntimeError('Load file error!')

        v = loss.forward(im1, im2).detach().mean().to('cpu')
        value.append(v)

    if len(value) > 0:
        value = torch.stack(value, dim=0)

    overall_mean = (float(value.sum()/len(value))) if len(value)> 0 else 0
    overall_std = (float(value.std())) if len(value)> 0 else 0
    print('number of pairs: {}'.format(len(value)))
    statics = {
        'overall_mean': overall_mean,
        'overall_std': overall_std
    }
    save_path = os.path.join(path2, '..', 'lpips_{}_count{}_perceptron.json'.format(path2.split(os.path.sep)[-1], count))
    save_dict_to_json(statics, save_path, indent=4)
    print('saved to {}'.format(save_path))
    return statics



def main():
    args = parser.parse_args()
    if args.im_size != '':
        args.im_size = [int(s) for s in args.im_size.split(',')]
        if len(args.im_size) == 1:
            args.im_size = args.im_size[0] 
    else:
        args.im_size = None    

    if args.path1.endswith(os.path.sep):
        args.path1 = args.path1[:-len(os.path.sep)]
    if args.path2.endswith(os.path.sep):
        args.path2 = args.path2[:-len(os.path.sep)]

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.type == 'diversity':
        statics = calculate_lpips_value_diversity(path=args.path1,
                                            device=device,
                                            net=args.net,
                                            loops=args.loops,
                                            count=args.count,
                                            im_size=args.im_size)
    elif args.type == 'perceptron':
        statics = calculate_lpips_value_perceptron(path1=args.path1,
                                            path2=args.path2,
                                            device=device,
                                            net=args.net,
                                            count=args.count,
                                            im_size=args.im_size)
    else:
        raise NotImplementedError
    print('statics: mean {} , std {}'.format(statics['overall_mean'], statics['overall_std']))


if __name__ == '__main__':

    main()