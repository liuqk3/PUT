import importlib
import random
from sys import path
import numpy as np
import torch
import warnings
import os
from itertools import repeat
import collections.abc

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp', 'JPEG'}



# From PyTorch internals
def to_ntuple(x, n):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, n))


def seed_everything(seed, cudnn_deterministic=False):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    
    Args:
        seed: the integer value seed for global random state
    """
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def merge_opts_to_config(config, opts):
    def modify_dict(c, nl, v):
        if len(nl) == 1:
            c[nl[0]] = type(c[nl[0]])(v)
        else:
            # print(nl)
            c[nl[0]] = modify_dict(c[nl[0]], nl[1:], v)
        return c

    if opts is not None and len(opts) > 0:
        assert len(opts) % 2 == 0, "each opts should be given by the name and values! The length shall be even number!"
        for i in range(len(opts) // 2):
            name = opts[2*i]
            value = opts[2*i+1]
            config = modify_dict(config, name.split('.'), value)
    return config 


def modify_config_for_debug(config):
    config['dataloader']['num_workers'] = 0
    config['dataloader']['batch_size'] = 1
    return config


def get_model_parameters_info(model):
    # for mn, m in model.named_modules():
    parameters = {'overall': {'trainable': 0, 'non_trainable': 0, 'total': 0, 'buffer': 0}}
    for child_name, child_module in model.named_children():
        parameters[child_name] = {'trainable': 0, 'non_trainable': 0, 'buffer': 0}
        for pn, p in child_module.named_parameters():
            if p.requires_grad:
                parameters[child_name]['trainable'] += p.numel()
            else:
                parameters[child_name]['non_trainable'] += p.numel()
        # import pdb; pdb.set_trace()
        for pn, p in child_module.named_buffers():
            parameters[child_name]['buffer'] += p.numel()
        
        parameters[child_name]['total'] = parameters[child_name]['trainable'] + parameters[child_name]['non_trainable'] + parameters[child_name]['buffer']

        parameters['overall']['trainable'] += parameters[child_name]['trainable']
        parameters['overall']['non_trainable'] += parameters[child_name]['non_trainable']
        parameters['overall']['buffer'] += parameters[child_name]['buffer']
        parameters['overall']['total'] += parameters[child_name]['total']
    
    def count_dict(d):
        count_ = 0
        for k,v in d.items():
            if isinstance(v, dict):
                count_ += count_dict(v)
            else:
                count_ += v.numel()
        return count_

    # format the numbers
    def format_number(num, binary=False):
        if binary:
            K = 2**10
            M = 2**20
            G = 2**30
        else:
            K = 1e3
            M = 1e6
            G = 1e9
        if num > G: # K
            uint = 'G'
            num = round(float(num)/G, 3)
        elif num > M:
            uint = 'M'
            num = round(float(num)/M, 3)
        elif num > K:
            uint = 'K'
            num = round(float(num)/K, 3)
        else:
            uint = ''
        res = '{}{}'.format(num, uint)
        if binary:
            res += 'b'
        return res
    
    def format_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                format_dict(v)
            else:
                d[k] = format_number(v, binary=True) + "; " + format_number(v, binary=False)
    

    total = count_dict(model.state_dict())
    parameters['total'] = total
    format_dict(parameters)

    return parameters


def format_seconds(seconds):
    h = int(seconds // 3600)
    m = int(seconds // 60 - h * 60)
    s = int(seconds % 60)

    d = int(h // 24)
    h = h - d * 24

    if d == 0:
        if h == 0:
            if m == 0:
                ft = '{:02d}s'.format(s)
            else:
                ft = '{:02d}m:{:02d}s'.format(m, s)
        else:
           ft = '{:02d}h:{:02d}m:{:02d}s'.format(h, m, s)
 
    else:
        ft = '{:d}d:{:02d}h:{:02d}m:{:02d}s'.format(d, h, m, s)

    return ft


def instantiate_from_config(config):
    if config is None:
        return None
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    for k in config.keys():
        if k not in ['target', 'params']:
            print('warning: Unused key {} while instantiating {}'.format(k, config['target']))
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls(**config.get("params", dict()))

def get_relative_path(path, root):
    if root.endswith(os.path.sep):
        root = root[:-len(os.path.sep)]
    path_  = path.split(os.path.sep)
    root_ = root.split(os.path.sep)
    path_ = os.path.sep.join(path_[len(root_):])
    return path_


def get_all_file(dir, end_with='.h5', path_type='abs'):
    """
    Load all files from the given dir, and return a list of file paths.
    The path can be relative path or abs path. 
    """
    if isinstance(end_with, str):
        end_with = [end_with]
    if dir.endswith(os.sep):
        dir = dir[0:-len(os.sep)]
    filenames = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            for ew in end_with:
                if f.endswith(ew):
                    if path_type == 'abs':
                        filenames.append(os.path.join(root, f))
                    elif path_type == 'relative':
                        path_ = get_relative_path(os.path.join(root, f), root=dir)
                        filenames.append(path_)
                    elif path_type == 'basename':
                        filenames.append(os.path.basename(f))
                    else:
                        raise NotImplementedError('path type not implemented: {}'.format(path_type))
                    break
    return filenames


def get_all_subdir(dir, max_depth=-1, min_depth=-1, path_type='abs'):
    subdirs = []
    for root, dirs, files in os.walk(dir):
        depth_ = get_relative_path(root, dir)
        if depth_ == '':
            depth_ = 1
        else:
            depth_ = len(depth_.split(os.sep)) + 1
        if min_depth > 0 and depth_ < min_depth:
            continue
        if max_depth > 0 and depth_ > max_depth:
            continue
        for d in dirs:
            if path_type == 'abs':
                subdirs.append(os.path.join(root, d))
            elif path_type == 'relative':
                path_ = get_relative_path(os.path.join(root, d), root=dir)
                subdirs.append(path_)
            else:
                raise NotImplementedError('path type not implemented: {}'.format(path_type))
    return subdirs


def get_sub_dirs(dir, abs=True):
    sub_dirs = os.listdir(dir)
    if abs:
        sub_dirs = [os.path.join(dir, s) for s in sub_dirs]
    return sub_dirs


def get_model_buffer(model):
    state_dict = model.state_dict()
    buffers_ = {}
    params_ = {n: p for n, p in model.named_parameters()}

    for k in state_dict:
        if k not in params_:
            buffers_[k] = state_dict[k]
    return buffers_


if __name__ == '__main__':
    dir = 'image_synthesis'
    dirs = get_all_subdir(dir, max_depth=2, min_depth=1, path_type='relative')


    a = 1
