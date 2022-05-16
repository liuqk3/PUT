"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from multiprocessing import cpu_count
import sklearn.svm
import numpy as np
import random
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from sklearn.decomposition import PCA
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.transforms.transforms import CenterCrop

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


from image_synthesis.utils.fid.inception import InceptionV3
# from .inception import InceptionV3
from image_synthesis.utils.misc import get_all_file
from image_synthesis.utils.misc import instantiate_from_config


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size to use')
parser.add_argument('--num_workers', type=int, default=1,
                    help='Number of processes to use for data loading')
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--path1', type=str, default='',
                    help='Paths to the generated images or txt file contain the relative path of images')
parser.add_argument('--path_prefix1', type=str, default='',
                    help='prefix to add to relative path to form a abs path of images')   
parser.add_argument('--file_list1', type=str, default='', 
                        help='relative path of file name')                  
parser.add_argument('--path2', type=str, default='',
                    help='Paths to the generated images or txt file contain the relative path of images')
parser.add_argument('--path_prefix2', type=str, default='',
                    help='prefix to add to relative path to form a abs path of images')  
parser.add_argument('--file_list2', type=str, default='', 
                        help='relative path of file name')  
parser.add_argument('--share_same_files', action='store_true', default=False,
                    help='share the same files with the same relative file names') 
parser.add_argument('--max_count1', type=int, default=None,
                    help='count of images in path1 to be coumputed') 
parser.add_argument('--max_count2', type=int, default=None,
                    help='the max count of images in path2 to be coumputed') 
parser.add_argument('--count1', type=int, default=None,
                    help='the count of images to be sampled from path1 to be coumputed') 
parser.add_argument('--count2', type=int, default=None,
                    help='the count of images to be sampled in path2 to be coumputed') 
parser.add_argument('--im_size1', type=str, default=None,
                    help='count of images in path1 to be coumputed') 
parser.add_argument('--im_size2', type=str, default=None,
                    help='count of images in path1 to be coumputed')                     

parser.add_argument('--loops', type=int, default=1,
                    help='how many times to get the fid, so we can get the mean and std of those fid score')  

parser.add_argument('--net', type=str, default='inception',
                    choices=['inception', 'clip'],
                    help='which model to use for getting the activations?')  


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp', 'JPEG'}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, preprocessor=None, transforms=None, count=None):
        self.files = files
        self.transforms = transforms
        self.preprocessor = preprocessor
        self.count = count

    def __len__(self):
        if self.count is not None:
            return min(self.count, len(self.files))
        return len(self.files)

    def __getitem__(self, i):
        flag = False 
        path = self.files[i]
        while not flag:
            try:
                img = Image.open(path).convert('RGB')
                flag = True
            except:
                print('Can not load {}'.format(path))
                i = i + 1
                path = self.files[i+1]

        if self.preprocessor is not None:
            img = np.array(img).astype(np.float32)
            if tuple(img.shape[:2]) != self.preprocessor.size:
                img = self.preprocessor(image=img)['image'] 
            img = np.transpose(img, (2, 0, 1))
            img = img / 255.0
        else:
            if self.transforms is not None:
                img = self.transforms(img)
        # import pdb; pdb.set_trace()
        return img


def get_im_preprocess(size):

    if size is not None:
        if not isinstance(size, (tuple, list)):
            size = [size, size]
        cfg = {
            'target': 'image_synthesis.data.utils.image_preprocessor.SimplePreprocessor',
            'params': {
                'size': size,
                'smallest_max_size': size[0],
            }
        }
    else:
        cfg = {
            'target': 'image_synthesis.data.utils.image_preprocessor.SimplePreprocessor',
            'params': {
                'size': None,
            }
        }
    preprocessor = instantiate_from_config(cfg)

    return preprocessor

def get_activations(files, model, batch_size=50, dims=2048, device='cpu', num_workers=8, im_size=None):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)
    # import pdb; pdb.set_trace()

    
    if model.name == 'clip':
        transforms = model.preprocess
        preprocessor = None 
    else:
        # import pdb; pdb.set_trace()
        if im_size is not None:
            transforms = TF.Compose([
                TF.Resize(im_size),
                TF.CenterCrop(im_size),
                TF.ToTensor()
            ])
        else:
            transforms = TF.ToTensor()
        preprocessor = get_im_preprocess(im_size)

    dataset = ImagePathDataset(files, preprocessor=preprocessor, transforms=transforms)
    # import pdb; pdb.set_trace()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers
                                             )

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            if model.name == 'inception':
                pred = model(batch)[0]
            elif model.name == 'clip':
                pred = model.encode_image(batch) # B x 512
            else:
                raise NotImplementedError
        # import pdb; pdb.set_trace()

        if pred.dim() == 4: # inception
            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2)

        pred = pred.cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    # import pdb; pdb.set_trace()
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)



def calculate_pids_uids(activations1, activations2, reduce_dim=False):
    """Numpy implementation of the  Paired/Unpaired Inception Discriminative Score.
    paper: LARGE SCALE IMAGE COMPLETION VIA CO-MODULATED GENERATIVE ADVERSARIAL NETWORKS
    
    -- activations1: activation features for real images with shape [N, D]
    -- activations2: activation features for fake images with shape [N, D]

    Returns:
    --   : PIDS and UIDS
    """
    def reduce_dimension(features1, features2, output_dim=None):
        features = np.concatenate([features1, features2])
        if output_dim is None:
            output_dim = int(min(max(features.shape[0] / 5, 1), features.shape[1]))
        reducer = PCA(n_components=output_dim, random_state=0)
        features = reducer.fit_transform(features)
        out_feat1 = features[0:features1.shape[0],:]
        out_feat2 = features[-features2.shape[0]:,:]
        return out_feat1, out_feat2
    # calculate pids and uids
    real_activations = activations1
    fake_activations = activations2
    if reduce_dim:
        real_activations, fake_activations = reduce_dimension(real_activations, fake_activations)
    svm_inputs = np.concatenate([real_activations, fake_activations])
    svm_targets = np.array([1] * real_activations.shape[0] + [0] * fake_activations.shape[0])
    print('SVM fitting for PIDS and UIDS...')
    svm = sklearn.svm.LinearSVC(dual=False)
    svm.fit(svm_inputs, svm_targets)
    uids = 1 - svm.score(svm_inputs, svm_targets)
    real_outputs = svm.decision_function(real_activations)
    fake_outputs = svm.decision_function(fake_activations)
    pids = np.mean(fake_outputs > real_outputs)

    return pids, uids


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=8, im_size=None):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers, im_size)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    output = {
        'mu': mu,
        'sigma': sigma,
        'activation': act,
    }
    return output


def compute_statistics_of_path(path, path_prefix, model, batch_size, dims, device, num_workers=8, max_count=None, count=None, im_size=None):
    if isinstance(path, str) and path.endswith('.npz'):
        print('Loaded from ', path)
        with np.load(path) as f:
            m, s, act = f['mu'][:], f['sigma'][:], f['activation'][:]
            output = {
                'mu': m,
                'sigma': s,
                'activation': act
            }
    else:
        # import pdb; pdb.set_trace()
        if isinstance(path, str) and path.endswith('.txt'):
            files = []
            with open(path, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    l = l.replace('\n', '')
                    files.append(os.path.join(path_prefix, l))
                f.close()
        elif isinstance(path, str) and os.path.isdir(path):
            # path = pathlib.Path(path)
            # files = sorted([file for ext in IMAGE_EXTENSIONS
            #             for file in path.glob('*.{}'.format(ext))])
            # import pdb; pdb.set_trace()
            files = sorted(get_all_file(path, end_with=IMAGE_EXTENSIONS))
        elif isinstance(path, list):
            files = path
        else:
            raise NotImplementedError

        if max_count is not None:
            assert max_count > 0
            random.shuffle(files)
            files = files[:max_count]
        if count is not None:
            files = random.sample(files, count)
            # files = files[-count:]
        print('Number of files: ', len(files))
        output = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, num_workers, im_size)
    # save_path = os.path.join(path_prefix, 'fid_statcs.npz')
    # np.savez(save_path, mu=m, sigma=s)
    # print('saved to {}'.format(save_path))
    output['length'] = len(files)
    # import pdb; pdb.set_trace()
    return output


def calculate_fid_given_paths(path1, path_prefix1, file_list1,
                              path2, path_prefix2, file_list2,
                              batch_size,  device, 
                              dims, net='inception',
                              num_workers=8, 
                              max_count1=None, max_count2=None,
                              count1=None, count2=None,
                              im_size1=None, im_size2=None,
                              share_same_files=False):
    """Calculates the FID of two paths"""

    assert os.path.exists(path1), 'Invalid path: %s' % path1
    assert os.path.exists(path2), 'Invalid path: %s' % path2

    if net == 'inception':
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)
    elif net == 'clip':
        from image_synthesis.modeling.modules.clip import clip
        model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
        setattr(model, 'preprocess', preprocess)
        assert dims == 512, 'For clip, the output dimensionality of feature is 512'
    else:
        raise RuntimeError('Unknown type of net {}'.format(net))
    setattr(model, 'name', net)

    if file_list1 == '':
        files1 = get_all_file(path1, end_with=IMAGE_EXTENSIONS, path_type='relative')
    else:
        with open(file_list1) as f:
            files1 = f.readlines()
            files1 = sorted([l.strip() for l in files1])
    # import pdb; pdb.set_trace()
    if file_list2 == '':
        files2 = get_all_file(path2, end_with=IMAGE_EXTENSIONS, path_type='relative')
    else:
        with open(file_list2) as f:
            files2 = f.readlines()
            files2 = sorted([l.strip() for l in files2])
    # import pdb; pdb.set_trace()
    if share_same_files:
        files1_ext = '.' + files1[0].split('.')[-1]
        files2_ext = '.' + files2[0].split('.')[-1]
        if files1_ext != files2_ext:
            files1 = [f.replace(files1_ext, '') for f in files1]
            files2 = [f.replace(files2_ext, '') for f in files2]
            files1 = set(files1)
            files2 = set(files2)
            files = list(files1 & files2)
            if max_count1 is not None or max_count2 is not None:
                if max_count1 is not None and max_count2 is not None:
                    assert max_count1 == max_count2
                random.shuffle(files)
                if max_count1 is not None:
                    files = files[:max_count1]
                else:
                    files = files[:max_count2]
            path1 = [os.path.join(path1, f+files1_ext) for f in files]
            path2 = [os.path.join(path2, f+files2_ext) for f in files]
        else:
            files1 = set(files1)
            files2 = set(files2)
            files = list(files1 & files2)
            if max_count1 is not None or max_count2 is not None:
                if max_count1 is not None and max_count2 is not None:
                    assert max_count1 == max_count2
                random.shuffle(files)
                if max_count1 is not None:
                    files = files[:max_count1]
                else:
                    files = files[:max_count2]
            path1 = [os.path.join(path1, f) for f in files]
            path2 = [os.path.join(path2, f) for f in files]
    else:
        files2_ = []

        # for ours lpips company fid
        for f in files2:
            # if 'gt_' in f or 'mask_' in os.path.basename(f):
            #     continue 
            if 'completed' in os.path.basename(f):
                files2_.append(f) 
        
        if len(files2_) == 0: # #TODO for ict lpips
            files2_ = files2

        files2 = files2_
        path1 = [os.path.join(path1, f) for f in files1]
        path2 = [os.path.join(path2, f) for f in files2]
    path1 = sorted(path1)
    path2 = sorted(path2)
    output1 = compute_statistics_of_path(path1, path_prefix1, model, batch_size,
                                        dims, device, num_workers, max_count1, count1, im_size1)
    output2 = compute_statistics_of_path(path2, path_prefix2, model, batch_size,
                                        dims, device, num_workers, max_count2, count2, im_size2)
    # import pdb; pdb.set_trace()
    fid = calculate_frechet_distance(output1['mu'], output1['sigma'], output2['mu'], output2['sigma'],)
    if len(path1) == len(path2):
        pids, uids = calculate_pids_uids(output1['activation'], output2['activation'])
    else:
        pids, uids = -1, -1

    output = {
        'fid': fid,
        'pids': pids,
        'uids': uids,
        'length1': output1['length'],
        'length2': output2['length'],
        'dims': dims,
    }
    return output

 
def main():
    args = parser.parse_args()
    if args.path1.endswith(os.sep):
        args.path1 = args.path1[:-len(os.sep)]
    if args.path2.endswith(os.sep):
        args.path2 = args.path2[:-len(os.sep)]

    if args.im_size1 is not None:
        args.im_size1 = [int(s) for s in args.im_size1.split(',')]
        if len(args.im_size1) == 1:
            args.im_size1 = args.im_size1[0]
    if args.im_size2 is not None:
        args.im_size2 = [int(s) for s in args.im_size2.split(',')]
        if len(args.im_size2) == 1:
            args.im_size2 = args.im_size2[0]       


    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.net == 'clip':
        args.dims = 512

    fid_values = []
    pids_values = []
    uids_values = []
    for l in range(args.loops):
        output = calculate_fid_given_paths(
                                            path1=args.path1,
                                            path_prefix1=args.path_prefix1,
                                            file_list1=args.file_list1,
                                            path2=args.path2,
                                            path_prefix2=args.path_prefix2,
                                            file_list2=args.file_list2,
                                            batch_size=args.batch_size,
                                            device=device,
                                            dims=args.dims,
                                            net=args.net,
                                            num_workers=args.num_workers,
                                            max_count1=args.max_count1,
                                            max_count2=args.max_count2,
                                            count1=args.count1,
                                            count2=args.count2,
                                            im_size1=args.im_size1,
                                            im_size2=args.im_size2,
                                            share_same_files=args.share_same_files)
        
        # import pdb; pdb.set_trace()
        fid_values.append(output['fid'])
        pids_values.append(output['pids'])
        uids_values.append(output['uids'])

    
    fid_mean, fid_std = np.mean(fid_values), np.std(fid_values)
    pids_mean, pids_std = np.mean(pids_values), np.std(pids_values)
    uids_mean, uids_std = np.mean(uids_values), np.std(uids_values)
    result = {
        'fid':{
            'mean': fid_mean,
            'std': fid_std
        },
        'pids':{
            'mean': pids_mean,
            'std': pids_std
        },
        'uids':{
            'mean': uids_mean,
            'std': uids_std
        }
    }
    # if args.share_same_files:
    last_folder = args.path2.split(os.sep)[-1]
    last_folder = args.path2.split(os.sep)[-1]
    if args.share_same_files:
        save_path = os.path.join(args.path2, '..', 'fid_ids_{}_{}_{}_{}_{}_sameFile.json'.format(last_folder, args.net, args.dims, output['length1'], output['length2']))
    else:
        save_path = os.path.join(args.path2, '..', 'fid_ids_{}_{}_{}_{}_{}.json'.format(last_folder, args.net, args.dims, output['length1'], output['length2']))

    json.dump(result, open(save_path, 'w'), indent=4)
    print('saved to {}'.format(save_path))
    
    print('FID: {}, PIDS: {}, UIDS: {}'.format(fid_mean, pids_mean, uids_mean))


if __name__ == '__main__':

    main()