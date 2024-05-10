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
from torch.utils.data import ConcatDataset, dataset

from image_synthesis.utils.io import load_yaml_config, load_dict_from_json, save_dict_to_json
from image_synthesis.utils.misc import get_all_file, get_all_subdir, instantiate_from_config
from image_synthesis.utils.cal_metrics import get_PSNR, get_mse_loss, get_l1_loss, get_SSIM, get_mae
from image_synthesis.modeling.build import build_model
from image_synthesis.utils.misc import format_seconds, merge_opts_to_config
from image_synthesis.distributed.launch import launch
from image_synthesis.distributed.distributed import get_rank, reduce_dict, synchronize, all_gather
from image_synthesis.utils.misc import get_model_parameters_info, get_model_buffer



class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir=None, size=(256,256)):
        if os.path.isfile(image_dir):
            image_paths = [image_dir]
            image_dir = ''
        else:
            # import pdb; pdb.set_trace()
            image_paths = sorted(get_all_file(image_dir, path_type='relative'))
        
        if mask_dir is not None:
            if os.path.isfile(mask_dir):
                mask_paths = [mask_dir]
                mask_dir = ''
            else:
                mask_paths = sorted(get_all_file(mask_dir, path_type='relative'))
        else:
            mask_paths = None

        match_image_with_mask = True
        if mask_paths is not None:
            assert len(image_paths) > 0 and len(mask_paths) > 0
            if len(mask_paths) == 1 and len(mask_paths) != len(image_paths):
                match_image_with_mask = False
                # import pdb; pdb.set_trace()
                print('The number of masks and images are not the same, replicate the mask!')
                while len(mask_paths) < len(image_paths):
                    # import pdb; pdb.set_trace()
                    mask_paths += copy.deepcopy(mask_paths)
                    
                mask_paths = mask_paths[0:len(image_paths)]
            elif len(mask_paths) != len(image_paths):
                raise RuntimeError('number of masks and images are invalid! mask: {}, image : {}'.format(len(mask_paths), len(image_paths)))
        # import pdb; pdb.set_trace()
        self.image_dir = image_dir 
        self.mask_dir = mask_dir
        # import pdb; pdb.set_trace()
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        
        self.size = size # h, w
        self.match_image_with_mask = match_image_with_mask

    def __len__(self):
        return len(self.image_paths)


    def read_mask(self, mask_path):
        mask = Image.open(mask_path)
        # if not mask.mode == "RGB":
        #     mask = mask.convert("RGB")
        mask = np.array(mask).astype(np.float32)
        mask = mask / 255.0
        
        h, w = mask.shape[0], mask.shape[1]
        if (h,w) != tuple(self.size):
            mask_inp = cv2.resize(mask, tuple(self.size), interpolation=cv2.INTER_NEAREST)
        else:
            mask_inp = copy.deepcopy(mask)
        
        if len(mask.shape) == 3:
            mask = mask[:, :, 0:1] # h, w, 1
            mask_inp = mask_inp[:, :, 0:1]
        else:
            mask = mask[:, :, np.newaxis]
            mask_inp = mask_inp[:, :, np.newaxis]

        mask_inp = torch.tensor(mask_inp).permute(2, 0, 1).bool() # 1, h, w
        mask = torch.tensor(mask).permute(2, 0, 1).bool() # 1, h, w
        return mask_inp, mask
    
    def read_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        # if not mask.mode == "RGB":
        #     mask = mask.convert("RGB")
        image = np.array(image).astype(np.float32)
        h, w = image.shape[0], image.shape[1]
        if (h,w) != tuple(self.size):
            image_inp = cv2.resize(image, tuple(self.size), interpolation=cv2.INTER_LINEAR)
        else:
            image_inp = copy.deepcopy(image)
        image_inp = torch.tensor(image_inp).permute(2, 0, 1) # 3, h, w
        image = torch.tensor(image).permute(2, 0, 1) # 3, h, w
        return image_inp, image


    def __getitem__(self, i):
        # print(get_rank(), )
        # import pdb; pdb.set_trace()
        image_path = self.image_paths[i]
        image_inp, image = self.read_image(os.path.join(self.image_dir, image_path))

        data = {
            'relative_path': image_path,
            'image': image_inp,
            'image_ori': image,
        }

        if self.mask_paths is not None:
            mask_path = self.mask_paths[i]
            # if self.match_image_with_mask:
            #     assert os.path.basename(image_path) == os.path.basename(mask_path), 'image: {}, mask: {}'.format(image_path, mask_path)
            mask_inp, mask = self.read_mask(os.path.join(self.mask_dir, mask_path))
            data['mask_ori'] = mask
            data['mask'] = mask_inp

        return data

    def remove_files(self, relative_paths):
        image_paths = set(self.image_paths) - set(relative_paths)
        if self.match_image_with_mask:
            mask_paths = set(self.mask_paths) - set(relative_paths)
        else:
            mask_paths = self.mask_paths[:len(image_paths)]

        print('Remove files done! Before {}, after {}'.format(len(self.image_paths), len(mask_paths)))
        self.image_paths = sorted(list(image_paths))
        self.mask_paths = sorted(list(mask_paths))
         


def get_model(args=None, model_name='2020-11-09T13-33-36_faceshq_vqgan'):
    if os.path.isfile(model_name):
        # import pdb; pdb.set_trace()
        if model_name.endswith(('.pth', '.ckpt')):
            model_path = model_name
            config_path = os.path.join(os.path.dirname(model_name), '..', 'configs', 'config.yaml')
        elif model_name.endswith('.yaml'):
            config_path = model_name
            model_path = os.path.join(os.path.dirname(model_name), '..', 'checkpoint', 'last.pth')
        else:
            raise RuntimeError(model_name)
        
        if 'OUTPUT' in model_name: # pretrained model
            model_name = model_name.split(os.path.sep)[-3]
        else: # just give a config file, such as test_openai_dvae.yaml, which is no need to train, just test
            model_name = os.path.basename(config_path).replace('.yaml', '')
    else:
        model_path = os.path.join('OUTPUT', model_name, 'checkpoint', 'last.pth')
        config_path = os.path.join('OUTPUT', model_name, 'configs', 'config.yaml')

    args.model_path = model_path
    args.config_path = config_path

    config = load_yaml_config(config_path)
    # config = merge_opts_to_config(config, args.opts)
    model = build_model(config)
    model_parameters = get_model_parameters_info(model)
    # import pdb; pdb.set_trace()
    print(model_parameters)

    # if model_path.endswith('.pth') or model_path.endswith('.ckpt'):
    #     save_path = model_path.replace('.pth', '_parameters.json').replace('.ckpt', '_parameters.json')
    #     json.dump(model_parameters, open(save_path, 'w'), indent=4)
    # sys.exit(1)

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location="cpu")
    else:
        ckpt = {}
    if 'last_epoch' in ckpt:
        epoch = ckpt['last_epoch']
    elif 'epoch' in ckpt:
        epoch = ckpt['epoch']
    else:
        epoch = 0

    if 'model' in ckpt:
        # #TODO
        # # import pdb; pdb.set_trace()
        # model_static = OrderedDict()
        # for k in ckpt['model'].keys():
        #     if k.startswith('content_codec.'):
        #         # del ckpt['model'][k]
        #         print('delet: {}'.format(k))
        #     else:
        #         model_static[k] = ckpt['model'][k]
        # ckpt['model'] = model_static

        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    elif 'state_dict' in ckpt:

        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        missing, unexpected = [], []
        print("====> Warning! No pretrained model!")
    print('Missing keys in created model:\n', missing)
    print('Unexpected keys in state dict:\n', unexpected)
    # import pdb; pdb.set_trace()

    model = model.eval()

    return {'model': model, 'epoch': epoch, 'model_name': model_name, 'parameter': model_parameters}





def inference_reconstruction(local_rank=0, args=None):
    # used  for image based auto encoder
    info = get_model(args=args, model_name=args.name)
    model = info['model']
    epoch = info['epoch']
    model_name = info['model_name']

    data = ImagePathDataset(image_dir=args.image_dir, mask_dir=args.mask_dir, size=tuple(int(s) for s in args.input_res.split(',')))
    num_images_per_rank = len(data) // args.world_size
    start_index = get_rank() * num_images_per_rank
    end_index = start_index + num_images_per_rank
    print('number of images:', len(data)//args.world_size)

    total_loss = {"mse_loss": 0.0, "psnr": 0.0, "l1_loss": 0.0, "ssim": 0.0, "mae": 0.0}
    total_batch = 0.0
    # save images
    save_root = os.path.join(args.save_dir, model_name)
    save_root = save_root + '_e{}'.format(epoch)

    # if local_rank == 0:
    #     # save model parameters info
    #     info_path = (os.path.join(save_root, 'model_parameters.json'))
    #     os.makedirs(os.path.dirname(info_path), exist_ok=True)
    #     json.dump(info['parameter'], open(info_path, 'w'), indent=4)
    # import sys
    # sys.exit(1)

    print('results will be saved in {}'.format(save_root))
    save_count = -1 # the number of images to be saved, -1 means save all iamges
    save_token = False

    token_freq = OrderedDict()
    if token_freq is not None:
        num_tokens = model.get_number_of_tokens()
        token_freq['default'] = OrderedDict()
        for i in range(num_tokens): # such as VQVAE1
            token_freq['default'][i] = torch.tensor(0.0).cuda()

    # rec_time = 0
    # count = 0
    start_gen = time.time()

    # Since we aim at reconstructing images, the reference branch is no need
    try:
        model.decoder.requires_image = False
        model.decoder.up_layer_with_image = False
        pass
    except:
        pass

    for i in range(start_index, end_index):
        data_i = data[i]

        print("{}/{}".format(i, end_index-start_index))

        # tic = time.time()
        # count += 1

        with torch.no_grad():
            img = data_i['image'].to(model.device).unsqueeze(dim=0).contiguous()
            mask = data_i.get('mask', None) 
            if mask is not None:
                mask = mask.to(model.device).unsqueeze(dim=0)
            token = model.get_tokens(img, return_token_index=True, cache=False)

            rec = model.decode(token['token'], combine_rec_and_gt=False, token_shape=token.get('token_shape', None)).contiguous()
            # mask = torch.ones_like(img)[:,0:1,:,:].bool()
            # rec = model.decode(token['token'], mask_im=img, mask=mask, combine_rec_and_gt=False, token_shape=token.get('token_shape', None)).contiguous()

            # import pdb; pdb.set_trace()
            # token = model.get_tokens(img, return_token_index=True, cache=False)
            # rec = model.decode(token['token'], mask_im=img*mask, mask=mask, token_shape=token.get('token_shape', None))

            # save tokens
            if (save_count < 0 and i < save_count) and save_token:
                os.makedirs(save_root, exist_ok=True)
                token_save_path = os.path.join(save_root, 'token_rank{}_batch{}.pth'.format(local_rank, i))
                torch.save(token['token'], token_save_path)
                print('token saved in {}'.format(token_save_path))

            # import pdb; pdb.set_trace()
            mse_loss = get_mse_loss(img, rec)
            l1_loss = get_l1_loss(img, rec)
            psnr = get_PSNR(img, rec, tool='skimage')
            ssim = get_SSIM(img, rec, full=False, win_size=51)
            mae = get_mae(img, rec)

        total_loss['mse_loss'] += mse_loss * img.shape[0]
        total_loss['l1_loss'] += l1_loss * img.shape[0]
        total_loss['psnr'] += psnr * img.shape[0]
        total_loss['ssim'] += ssim * img.shape[0]
        total_loss['mae'] += mae * img.shape[0]

        total_batch += img.shape[0]
        if save_count < 0 or i < save_count:
            basename = data_i['relative_path']
            basename = os.path.basename(basename)
            basename = basename.split('.')[:-1] + ['png']
            basename = '.'.join(basename)

            gt_im = img[0].permute(1, 2, 0).to('cpu').numpy()
            gt_im = Image.fromarray(gt_im.astype(np.uint8))
            save_path = os.path.join(save_root, 'gt', basename)
            # import pdb; pdb.set_trace()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            gt_im.save(save_path)

            rec_im = rec[0].permute(1, 2, 0).to('cpu').numpy()
            rec_im = Image.fromarray(rec_im.astype(np.uint8))
            save_path = os.path.join(save_root, 'rec', basename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            rec_im.save(save_path)

            if mask is not None:
                mask_im = (img[0] * mask[0]).permute(1, 2, 0).to('cpu').numpy()
                mask_im = Image.fromarray(mask_im.astype(np.uint8))
                save_path = os.path.join(save_root, 'mask_gt', basename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                mask_im.save(save_path)
        # count token frequency
        if token_freq is not None and 'token_index' in token:
            token_index = token['token_index'].view(-1)
            token_unique = torch.unique(token_index).tolist()
            # import pdb; pdb.set_trace()
            for idx in token_unique:
                token_freq['default'][int(idx)] += int((token_index==idx).sum())


    # print('rec time {} s/im, {} ims/s'.format(rec_time / count, count / rec_time))

    # save token frequency
    if token_freq is not None:
        for k in token_freq:
            token_freq_tmp = reduce_dict(token_freq[k], average=False)
            token_idx = []
            token_count = []
            for i, v in token_freq_tmp.items():
                token_idx.append(int(i))
                token_count.append(int(v))
            token_freq_tmp_ = OrderedDict()
            index = np.argsort(token_count)
            for i in range(len(index)-1, -1, -1):
                i = index[i]
                cnt = token_count[i]
                idx = token_idx[i]

                # import pdb; pdb.set_trace()
                token_freq_tmp_[idx] = int(cnt)
            token_freq[k] = token_freq_tmp_

        # import pdb;pdb.set_trace()
        token_freq_path = os.path.join(save_root, 'token_freqency.json')
        json.dump(token_freq, open(token_freq_path, 'w'), indent=4)

    synchronize()
    total_loss = reduce_dict(total_loss, average=False)
    total_batch = sum(all_gather(total_batch))
    for k in total_loss:
        total_loss[k] = total_loss[k] / total_batch

    if local_rank == 0:
        for k in total_loss.keys():
            total_loss[k] = float(total_loss[k])
        loss_path = os.path.join(save_root, 'total_loss.json')
        os.makedirs(os.path.dirname(loss_path), exist_ok=True)
        json.dump(total_loss, open(loss_path, 'w'), indent=4)
        print(total_loss)

        # save model parameters info
        info_path = (os.path.join(save_root, 'model_parameters.json'))
        json.dump(info['parameter'], open(info_path, 'w'), indent=4)




def inference_inpainting(local_rank=0, args=None):
    """
    This is used for image completion. Each gt image is with a mask,
    and the generate several results. Each gt along with the mask, results are 
    all saved into one folder.
    
    """
    info = get_model(args=args, model_name=args.name)
    model = info['model']
    model = model.cuda()

    epoch = info['epoch']
    model_name = info['model_name']


    filter_ratio = [float(fr) if '.' in fr else int(fr) for fr in args.num_token_for_sampling.split(',')] 
    num_token_per_iter = [int(ntp) if '_' not in ntp else ntp for ntp in args.num_token_per_iter.split(',')]

    accumulate_time = None

    # save images
    if args.save_dir:
        save_root = os.path.join('RESULT', model_name+'_e{}'.format(epoch,args.num_sample), args.save_dir)
    else:
        save_root = os.path.join('RESULT', model_name+'_e{}'.format(epoch,args.num_sample), args.mask_dir.split(os.path.sep)[-1])

    for fr in filter_ratio:
        for ntp in num_token_per_iter:
            save_root_tmp = save_root + '_top{}_nTpi{}_numSample{}'.format(fr, ntp, args.num_sample)
            if args.raster_order:
                save_root_tmp = save_root_tmp + '_raster'
            os.makedirs(save_root_tmp, exist_ok=True)

            print('results will be saved in {}'.format(save_root_tmp))
            
            # read all processed images
            processed_image_path = glob.glob(os.path.join(save_root_tmp, 'processed_image_rank*.txt'))
            if len(processed_image_path) > 0:
                processed_image = set([])
                for path in processed_image_path:
                    pf = open(path, 'r')
                    processed_image_ = pf.readlines()
                    processed_image_ = set([p.strip() for p in processed_image_])
                    processed_image = processed_image | processed_image_
                    pf.close()
            else:
                processed_image = set([])

            # # modify dataset
            data = ImagePathDataset(image_dir=args.image_dir, mask_dir=args.mask_dir, size=tuple(int(s) for s in args.input_res.split(',')))
            data.remove_files(list(processed_image))

            # keep a cache file to record those images that have been processed
            processed_image_path = os.path.join(save_root_tmp, 'processed_image_rank{}.txt'.format(get_rank()))
            if os.path.exists(processed_image_path):
                processed_image_writer = open(processed_image_path, 'a')
            else:
                processed_image_writer = open(processed_image_path, 'w')

            num_images_per_rank = len(data) // args.world_size
            start_index = get_rank() * num_images_per_rank
            end_index = start_index + num_images_per_rank

            start_gen = time.time()
            for i in range(start_index, end_index):
                # if i >= 40:
                #     break

                data_i = data[i]
                if data_i['relative_path'] in processed_image:
                    print('{} exist! pasted!'.format(data_i['relative_path']))
                    continue 

                print("Rank: {} {}/{}".format(get_rank(), i-start_index, num_images_per_rank))

                if args.num_sample > 1:
                    basename = os.path.basename(data_i['relative_path'])
                    basename = '.'.join(basename.split('.')[:-1])
                    save_root_ = os.path.join(save_root_tmp, basename)
                    os.makedirs(save_root_, exist_ok=True)
                else:
                    save_root_ = save_root_tmp
                
                # make a batch
                data_i['relative_path'] = [data_i['relative_path']]
                data_i['image'] = data_i['image'].unsqueeze(dim=0)
                data_i['mask'] = data_i['mask'].unsqueeze(dim=0)

                # import pdb; pdb.set_trace()

                # save masked image
                if args.save_masked_image:
                    mask_im = (data_i['image'][0] * data_i['mask'][0]).permute(1, 2, 0).to('cpu').numpy()
                    gt_im = (data_i['image'][0]).permute(1, 2, 0).to('cpu').numpy()
                    # import pdb; pdb.set_trace()
                    mask_im = mask_im * 0.85 + gt_im * 0.15
                    mask_im_ = Image.fromarray(mask_im.astype(np.uint8))
                    if args.num_sample > 1:
                        save_path = os.path.join(save_root_, 'gt_mask_image.png')
                    else:
                        # import pdb; pdb.set_trace()
                        save_path = os.path.join(save_root_, 'masked_gt', data_i['relative_path'][0])
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    # import pdb; pdb.set_trace()
                    mask_im_.save(save_path)
                
                # generate samples in a batch manner
                if args.num_sample == 1:
                    count_per_cond_ = 0
                else:
                    count_per_cond_ = len(glob.glob(os.path.join(save_root_, '*completed*.png')))
                while count_per_cond_ < args.num_sample:
                    start_batch = time.time()
                    with torch.no_grad():
                        content_dict = model.generate_content(
                            batch=copy.deepcopy(data_i),
                            filter_ratio=fr,
                            filter_type='count',
                            replicate=1 if args.num_sample == 1 else args.num_replicate,
                            with_process_bar=True,
                            mask_low_to_high=False,
                            sample_largest=True,
                            calculate_acc_and_prob=False,
                            num_token_per_iter=ntp,
                            accumulate_time=accumulate_time,
                            raster_order=args.raster_order
                        ) # B x C x H x W
                    accumulate_time = content_dict.get('accumulate_time', None)
                    print('Time consmption: ', accumulate_time)
                    # save results
                    for k in content_dict.keys():
                        # import pdb; pdb.set_trace()
                        if k in ['completed']:
                            content = content_dict[k].permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
                            for b in range(content.shape[0]):
                                if args.num_sample > 1:
                                    cnt = count_per_cond_ + b
                                    save_path = os.path.join(save_root_, '{}_cnt{}_fr{}.png'.format(k, str(cnt).zfill(6), fr))
                                else:
                                    save_path = os.path.join(save_root_, k,  data_i['relative_path'][b])
                                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                im = Image.fromarray(content[b])
                                im.save(save_path)
                                print('Rank {}, Total time {}, batch time {:.2f}s, saved in {}'.format(local_rank, format_seconds(time.time()-start_gen), time.time()-start_batch, save_path))
                    
                    # prepare for next iteration
                    print('==> batch time {}s'.format(round(time.time() - start_batch, 1)))
                    if args.num_sample > 1:
                        count_per_cond_ = len(glob.glob(os.path.join(save_root_, '*completed*.png')))
                    else:
                        count_per_cond_ += 1

                # processed_image_writer.write(''.join([p+'\n' for p in data_i['relative_path']]))
                processed_image_writer.write(data_i['relative_path'][0]+'\n')
                processed_image_writer.flush()

    if accumulate_time is not None and len(list(accumulate_time.keys())) > 0:
        accumulate_time = {k: torch.tensor(v) if not torch.is_tensor(v) else v for k,v in accumulate_time.items()}
        accumulate_time = reduce_dict(accumulate_time, average=True)
        if local_rank == 0:
            time_path = os.path.join(save_root_tmp, 'time_consumption.json')
            os.makedirs(os.path.dirname(time_path), exist_ok=True)
            accumulate_time = {k: float(v) for k,v in accumulate_time.items()}
            json.dump(accumulate_time, open(time_path, 'w'), indent=4)
            print(accumulate_time)


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--save_dir', type=str, default='', 
                        help='directory to save results') 

    parser.add_argument('--name', type=str, default='', 
                        help='the name of this experiment, if not provided, set to'
                             'the name of config file') 
    parser.add_argument('--func', type=str, default='inference_inpainting', 
                        help='the name of inference function')

    parser.add_argument('--input_res', type=str, default='256,256', 
                        help='input resolution (h,w)')
    parser.add_argument('--image_dir', type=str, default='',
                        help='gt images need to be loaded') 
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='mask dirs for image completion, Each gt image should have'
                        'a correspoding mask to be loaded')   

    
    # args for sampling
    parser.add_argument('--num_token_per_iter', type=str, default='1', 
                        help='the number of patches to be inpainted in one iteration')
    parser.add_argument('--num_token_for_sampling', type=str, default='200', 
                        help='the top-k tokens remained for sampling for each patch')
    parser.add_argument('--save_masked_image', action='store_true', default=False,    
                        help='Save the masked image, i.e., the input')
    parser.add_argument('--raster_order', action='store_true', default=False,    
                        help='Get the k1 patches in a raster order')

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
    parser.add_argument('--num_replicate', type=int, default=1,
                        help='replaicate the batch data while forwarding. This may accelerate the sampling speed if num_sample > 1')
    parser.add_argument('--num_sample', type=int, default=1,
                        help='The number of inpatined results to get for each image')

    args = parser.parse_args()
    args.cwd = os.path.abspath(os.path.dirname(__file__))

    return args


inference_func_map = {
    'inference_reconstruction': inference_reconstruction,
    'inference_inpainting': inference_inpainting,
}

if __name__ == '__main__':

    args = get_args()

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

