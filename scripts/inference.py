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


def read_mask(mask_root, relative_path, size=None):
    masks = []
    if isinstance(relative_path, str):
        relative_path = [relative_path]
    
    for p in relative_path:
        path = os.path.join(mask_root, p)
        path = path.replace('.jpg', '.png').replace('.jpeg', '.png').replace('.JPEG', '.png')
        assert os.path.exists(path), '{} file not exists!'.format(path)
        mask = Image.open(path)
        # if not mask.mode == "RGB":
        #     mask = mask.convert("RGB")
        mask = np.array(mask).astype(np.float32)
        h, w = mask.shape[0], mask.shape[1]
        if (h,w) != tuple(size):
            mask = cv2.resize(mask, tuple(size), interpolation=cv2.INTER_NEAREST)
        mask = mask / 255.0
        if len(mask.shape) == 3:
            mask = mask[:, :, 0:1] # h, w, 1
        else:
            mask = mask[:, :, np.newaxis]
        mask = torch.tensor(mask).permute(2, 0, 1).bool() # 1, h, w
        masks.append(mask)
    masks = torch.stack(masks, dim=0)
    return masks
    


def get_model_and_dataset(args=None, model_name='2020-11-09T13-33-36_faceshq_vqgan'):
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

    data_type = 'validation_datasets'
    if args is not None and args.data_type == 'train':
        data_type = 'train_datasets'

    val_dataset = []
    for ds_cfg in config['dataloader'][data_type]:
        ds = instantiate_from_config(ds_cfg)
        val_dataset.append(ds)
    if len(val_dataset) > 1:
        val_dataset = ConcatDataset(val_dataset)
    else:
        val_dataset = val_dataset[0]
    
    return {'model': model, 'data': val_dataset, 'epoch': epoch, 'model_name': model_name, 'parameter': model_parameters}



def inference_reconstruction(local_rank=0, args=None):
    # used  for image based auto encoder
    info = get_model_and_dataset(args=args, model_name=args.name)
    model = info['model']
    data = info['data']
    epoch = info['epoch']
    model_name = info['model_name']

    num_workers = 0 # max( 2, batch_size // 4)
    if args is not None and args.distributed:
        print('DDP')
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        sampler = torch.utils.data.distributed.DistributedSampler(data, shuffle=False)
        dataloader = torch.utils.data.DataLoader(data, 
                                             batch_size=args.batch_size, 
                                             shuffle=False, #(val_sampler is None),
                                             num_workers=num_workers, 
                                             pin_memory=True, 
                                             sampler=sampler, 
                                             drop_last=True)
    else:
        model = model.cuda()
        dataloader = torch.utils.data.DataLoader(data, 
                                                batch_size=args.batch_size, 
                                                num_workers=num_workers, 
                                                shuffle=False, 
                                                drop_last=False)
    
    num_batch = len(data) // (args.batch_size * args.world_size)
    print('images:', len(data)//args.world_size)
    total_loss = {"mse_loss": 0.0, "psnr": 0.0, "l1_loss": 0.0, "ssim": 0.0, "mae": 0.0}
    total_batch = 0.0
    # save images
    save_root = os.path.join(args.save_dir, model_name+'_{}'.format(args.data_type))
    save_root = save_root + '_e{}'.format(epoch)
    print('results will be saved in {}'.format(save_root))
    save_count = -1
    save_token = False

    token_freq = OrderedDict()
    if token_freq is not None:
        num_tokens = model.module.get_number_of_tokens() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.get_number_of_tokens()
        token_freq['default'] = OrderedDict()
        for i in range(num_tokens): # such as VQVAE1
            token_freq['default'][i] = torch.tensor(0.0).cuda()

    for i, data_i in enumerate(dataloader):
        print("{}/{}".format(i, num_batch))

        # tic = time.time()
        # count += 1

        if args.mask_dir != '':
            mask = read_mask(args.mask_dir, relative_path=[os.path.basename(rp) for rp in data_i['relative_path']], size=data_i['image'].shape[-2:])
            data_i['mask'] = mask

        with torch.no_grad():
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                img = data_i['image'].to(model.module.device)
                mask = data_i.get('mask', None)
                if mask is not None:
                    mask = mask.to(model.module.device)
                token = model.module.get_tokens(img, return_token_index=True, cache=False)
                rec = model.module.decode(token['token'], combine_rec_and_gt=False, token_shape=token.get('token_shape', None))

            else:
                img = data_i['image'].to(model.device)
                mask = data_i.get('mask', None) 
                if mask is not None:
                    mask = mask.to(model.device)
                token = model.get_tokens(img, return_token_index=True, cache=False)
                rec = model.decode(token['token'], combine_rec_and_gt=False, token_shape=token.get('token_shape', None))
            # save tokens
            if (save_count < 0 and i < save_count) and save_token:
                os.makedirs(save_root, exist_ok=True)
                token_save_path = os.path.join(save_root, 'token_rank{}_batch{}.pth'.format(local_rank, i))
                torch.save(token['token'], token_save_path)
                print('token saved in {}'.format(token_save_path))

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

            for idx in range(img.shape[0]):
                basename = data_i['relative_path'][idx]
                basename = os.path.basename(basename)
                basename = basename.split('.')[:-1] + ['png']
                basename = '.'.join(basename)

                gt_im = img[idx].permute(1, 2, 0).to('cpu').numpy()
                gt_im = Image.fromarray(gt_im.astype(np.uint8))
                save_path = os.path.join(save_root, 'gt', basename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                gt_im.save(save_path)

                rec_im = rec[idx].permute(1, 2, 0).to('cpu').numpy()
                rec_im = Image.fromarray(rec_im.astype(np.uint8))
                save_path = os.path.join(save_root, 'rec', basename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                rec_im.save(save_path)

                if mask is not None:
                    mask_im = (img[idx] * mask[idx]).permute(1, 2, 0).to('cpu').numpy()
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


def inference_complet_sample_in_feature_for_diversity(local_rank=0, args=None):
    """
    This is used for image completion. Each gt image is with a mask,
    and the generate several results. Each gt along with the mask, results are 
    all saved into one folder.
    
    """
    info = get_model_and_dataset(args=args, model_name=args.name)
    model = info['model']
    data = info['data']
    epoch = info['epoch']
    model_name = info['model_name']

    if args is not None and args.distributed:
        print('DDP')
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model = model.cuda()

    count_cond = -1 #1000 // (args.world_size if args is not None else 1)
    count_per_cond = 10
    filter_ratio = [50]

    filter_type = 'count'
    sample_largest = True 
    save_gt = True 
    save_mask = True
    save_masked_gt = True

    # save images
    save_root = os.path.join(args.save_dir, model_name+'_{}'.format(args.data_type))
    save_root = save_root + '_e{}_completion_in_feature_diversity'.format(epoch)
    if sample_largest:
        save_root = save_root + '_sample_largest'
    else:
        save_root = save_root + '_sample_raster'

    mask_ratio = [
        ['0.5', '0.6'],
        ['0.4', '0.5'],
        ['0.3', '0.4'],
        ['0.2', '0.3'],
        ['0.1', '0.2'],
    ]

    for mr in mask_ratio:
        for fr in filter_ratio:

            if filter_type == 'count':
                save_root_ = save_root + '_top{}'.format(fr)
            elif filter_type == 'prob':
                save_root_ = save_root + '_prob{}'.format(fr) 
            else:
                raise NotImplementedError

            data.set_provided_mask_ratio(mask_ratio=mr)
            save_root_mr = save_root_ + '_' + data.get_mask_info(type='str')
            os.makedirs(save_root_mr, exist_ok=True)
            print('results will be saved in {}'.format(save_root_mr))
            
            # read all processed images
            processed_image_path = glob.glob(os.path.join(save_root_mr, 'processed_image_rank*.txt'))
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

            # modify dataset
            data.remove_files(list(processed_image), path_type='relative_path')
            if len(data) == 0:
                continue
            
            if args is not None and args.distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(data, shuffle=True)
                dataloader = torch.utils.data.DataLoader(data, 
                                                    batch_size=1, # multi results for one image is formed in a batch
                                                    num_workers=1, 
                                                    pin_memory=True, 
                                                    sampler=sampler, 
                                                    drop_last=False)
            else:
                dataloader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=0, shuffle=True, drop_last=False)
            print('images:', len(data)//args.world_size)
            
            # keep a cache file to record those images that have been processed
            processed_image_path = os.path.join(save_root_mr, 'processed_image_rank{}.txt'.format(get_rank()))
            if os.path.exists(processed_image_path):
                processed_image_writer = open(processed_image_path, 'a')
            else:
                processed_image_writer = open(processed_image_path, 'w')
            
            if save_mask:
                saved_mask_path = os.path.join(save_root_mr, 'saved_mask_rank{}.txt'.format(get_rank()))
                if os.path.exists(saved_mask_path):
                    saved_mask_writer = open(saved_mask_path, 'a')
                else:
                    saved_mask_writer = open(saved_mask_path, 'w')
            if save_gt:
                saved_image_path = os.path.join(save_root_mr, 'saved_image_rank{}.txt'.format(get_rank()))
                if os.path.exists(saved_image_path):
                    saved_image_writer = open(saved_image_path, 'a')
                else:
                    saved_image_writer = open(saved_image_path, 'w')


            start_gen = time.time()
            for i, data_i in enumerate(dataloader):
                if len(list(set(data_i['relative_path']) & processed_image)) == len(data_i['relative_path']):
                    print('{} exist! pasted!'.format(data_i['relative_path']))
                    continue
                if count_cond > 0 and i > count_cond:
                    break
                print("{}/{}".format(i, len(data)//(args.world_size if args is not None else 1) if count_cond <= 0 else count_cond))

                if args.mask_dir != '':
                    mask_dir = os.path.join(args.mask_dir, 'mr{}_{}'.format(mr[0], mr[1]))
                    mask = read_mask(mask_dir, relative_path=[os.path.basename(rp) for rp in data_i['relative_path']], size=data_i['image'].shape[-2:])
                    data_i['mask'] = mask

                basename = os.path.basename(data_i['relative_path'][0])
                basename = '.'.join(basename.split('.')[:-1])
                save_root_ = os.path.join(save_root_mr, basename)
                os.makedirs(save_root_, exist_ok=True)

                # save gt image
                gt_im = data_i['image'][0].permute(1, 2, 0).to('cpu').numpy()
                if save_gt:
                    gt_im_ = Image.fromarray(gt_im.astype(np.uint8))
                    basename = os.path.basename(data_i['relative_path'][0])
                    basename = '.'.join(basename.split('.')[:-1]) + '.png'
                    save_gt_path = os.path.join(save_root_, basename)
                    gt_im_.save(save_gt_path)

                # save masked image
                if save_masked_gt:
                    mask_im = (data_i['image'][0] * data_i['mask'][0]).permute(1, 2, 0).to('cpu').numpy()
                    mask_im = mask_im * 0.85 + gt_im * 0.15
                    mask_im_ = Image.fromarray(mask_im.astype(np.uint8))
                    save_path = os.path.join(save_root_, 'gt_mask_image_rank{}.png'.format(local_rank))
                    mask_im_.save(save_path)

                # save mask
                if save_mask:
                    mask = data_i['mask'][0].permute(1, 2, 0).to('cpu').numpy()
                    mask = Image.fromarray((mask[:,:,0]*255).astype(np.uint8))
                    save_mask_path = os.path.join(save_root_, 'mask_rank{}.png'.format(local_rank))
                    mask.save(save_mask_path)
                
                if 'image_hr' in data_i:
                    gt_hr = data_i['image_hr'].permute(0, 2, 3, 1).to('cpu').numpy()
                    save_path = os.path.join(save_root_mr, 'gt_hr', basename)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    im = Image.fromarray(gt_hr[0].astype(np.uint8))
                    im.save(save_path)
                if 'mask_hr' in data_i:
                    mask_hr = data_i['mask_hr'].permute(0, 2, 3, 1).to('cpu').numpy()
                    save_path = os.path.join(save_root_mr, 'mask_hr', basename)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    im = Image.fromarray(mask_hr[0][:,:,-1])
                    im.save(save_path)
                # generate samples in a batch manner
                count_per_cond_ = len(glob.glob(os.path.join(save_root_, '*completed*.png')))
                while count_per_cond_ < count_per_cond:
                    start_batch = time.time()
                    with torch.no_grad():
                        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                            content_dict = model.module.generate_content(
                                batch=data_i,
                                filter_ratio=fr,
                                filter_type=filter_type,
                                replicate=args.batch_size,
                                with_process_bar=True,
                                sample_largest=sample_largest,
                            ) # B x C x H x W
                        else:
                            content_dict = model.generate_content(
                                batch=data_i,
                                filter_ratio=fr,
                                filter_type=filter_type,
                                replicate=args.batch_size,
                                with_process_bar=True,
                                sample_largest=sample_largest
                            ) # B x C x H x W
                    # save results
                    for k in content_dict.keys():
                        content = content_dict[k].permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
                        if k in ['completed']:
                            for b in range(content.shape[0]):
                                cnt = count_per_cond_ + b
                                save_path = os.path.join(save_root_, 'rank{}_{}_cnt{}_fr{}.png'.format(local_rank, k, str(cnt).zfill(6), fr))
                                im = Image.fromarray(content[b])
                                im.save(save_path)
                                print('Rank {}, Total time {}, batch time {:.2f}s, saved in {}'.format(local_rank, format_seconds(time.time()-start_gen), time.time()-start_batch, save_path))




                    print('==> batch time {}s'.format(round(time.time() - start_batch, 1)))
                    count_per_cond_ = len(glob.glob(os.path.join(save_root_, '*completed*.png')))

                if save_mask:
                    saved_mask_writer.write(save_mask_path+'\n')
                    saved_mask_writer.flush()
                if save_gt:
                    saved_image_writer.write(save_gt_path+'\n')
                    saved_image_writer.flush()
                
                # processed_image_writer.write(''.join([p+'\n' for p in data_i['relative_path']]))
                processed_image_writer.write(data_i['relative_path'][0]+'\n')
                processed_image_writer.flush()

def inference_complet_sample_in_feature_for_evaluation(local_rank=0, args=None):
    """
    This is used for image completion. Each gt image is with one mask and one result is 
    generated. The mask, generated result are saved into different folders and named with
    the gt name.
    
    """
    info = get_model_and_dataset(args=args, model_name=args.name)
    model = info['model']
    data = info['data']
    epoch = info['epoch']
    model_name = info['model_name']
    model_param = info['parameter']

    if args is not None and args.distributed:
        print('DDP')
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model = model.cuda()
    
    count_cond = -1
    filter_ratio = [50] 

    filter_type = 'count'

    sample_largest = True
    save_gt = True
    save_mask = True
    save_masked_gt = True 
    save_reconstruction = False # True
    num_token_per_iter = 1

    # fr = random.choice(filter_ratio)
    for fr in filter_ratio:
        # save images
        save_root = os.path.join(args.save_dir, model_name+'_{}'.format(args.data_type))
        save_root = save_root + '_e{}_completion_in_feature_eval'.format(epoch)
        if sample_largest:
            save_root = save_root + '_sample_largest'
        else:
            save_root = save_root + '_sample_raster'
        if filter_type == 'count':
            save_root = save_root + '_top{}'.format(fr)
        elif filter_type == 'prob':
            save_root = save_root + '_prob{}'.format(fr)
        else:
            raise NotImplementedError

        mask_ratio = [
            ['0.4','0.6'],
            ['0.2','0.4'],
            ['0.1', '0.6'],
        ]
        
        for mr in mask_ratio:
            data.set_provided_mask_ratio(mask_ratio=mr)

            save_root_mr = save_root + '_' + data.get_mask_info(type='str')
            if num_token_per_iter != 1:
                save_root_mr = save_root_mr + '_nTpi' + str(num_token_per_iter)
            os.makedirs(save_root_mr, exist_ok=True)
            print('results will be saved in {}'.format(save_root_mr))
            save_dict_to_json(model_param, os.path.join(save_root_mr, 'parameters.json'), indent=4)

            # read all processed images
            processed_image_path = glob.glob(os.path.join(save_root_mr, 'processed_image_rank*.txt'))
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

            # modify dataset
            data.remove_files(list(processed_image), path_type='relative_path')
            
            if len(data) == 0:
                continue

            if args is not None and args.distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(data, shuffle=True)
                dataloader = torch.utils.data.DataLoader(data, 
                                                    batch_size=args.batch_size, 
                                                    num_workers=1, 
                                                    pin_memory=True, 
                                                    sampler=sampler, 
                                                    drop_last=False)
            else:
                # import pdb; pdb.set_trace()
                dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, num_workers=0, shuffle=True, drop_last=False)
            print('images:', len(data)//args.world_size)

            # keep a cache file to record those images that have been processed
            processed_image_path = os.path.join(save_root_mr, 'processed_image_rank{}.txt'.format(get_rank()))
            if os.path.exists(processed_image_path):
                processed_image_writer = open(processed_image_path, 'a')
            else:
                processed_image_writer = open(processed_image_path, 'w')

            metrics = {
                'ssim': 0.0,
                'psnr': 0.0,
                'mse': 0.0,
                'mae': 0.0,
                'l1': 0.0
            }
            start_gen = time.time()
            total_count = 0
            for i, data_i in enumerate(dataloader):
                if count_cond > 0 and i > count_cond:
                    break
                # import pdb; pdb.set_trace()
                if len(list(set(data_i['relative_path']) & processed_image)) == len(data_i['relative_path']):
                    continue
                
                if args.mask_dir != '':
                    mask_dir = os.path.join(args.mask_dir, 'mr{}_{}'.format(mr[0], mr[1]))
                    mask = read_mask(mask_dir, relative_path=[os.path.basename(rp) for rp in data_i['relative_path']], size=data_i['image'].shape[-2:])
                    data_i['mask'] = mask


                total_count += data_i['image'].shape[0]
                print("count {}/{}: batch {}/{}".format(total_count, len(data) , i, len(dataloader) if count_cond <= 0 else count_cond))
                # generate samples in a batch manner

                start_batch = time.time()
                with torch.no_grad():
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        content_dict = model.module.generate_content(
                            batch=data_i,
                            filter_ratio=fr,
                            filter_type=filter_type,
                            replicate=args.batch_size,
                            with_process_bar=True,
                            sample_largest=sample_largest,
                            return_reconstruction=save_reconstruction,
                            num_token_per_iter=num_token_per_iter
                        ) # B x C x H x W
                    else:
                        content_dict = model.generate_content(
                            batch=data_i,
                            filter_ratio=fr,
                            filter_type=filter_type,
                            replicate=args.batch_size,
                            with_process_bar=True,
                            sample_largest=sample_largest,
                            return_reconstruction=save_reconstruction,
                            num_token_per_iter=num_token_per_iter
                        ) # B x C x H x W
                # save results
                for k in ['completed']:
                    # get all losses
                    l1 = get_l1_loss(data_i['image'], content_dict[k])
                    mse = get_mse_loss(data_i['image'], content_dict[k])
                    mae = get_mae(data_i['image'], content_dict[k])
                    psnr = get_PSNR(data_i['image'], content_dict[k])
                    ssim = get_SSIM(data_i['image'], content_dict[k])
                    
                    metrics['mse'] += mse * data_i['image'].shape[0]
                    metrics['l1'] += l1 * data_i['image'].shape[0]
                    metrics['psnr'] += psnr * data_i['image'].shape[0]
                    metrics['ssim'] += ssim * data_i['image'].shape[0]
                    metrics['mae'] += mae * data_i['image'].shape[0]

                    content = content_dict[k].permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
                    gt = data_i['image'].permute(0, 2, 3, 1).to('cpu').numpy()
                    mask = data_i['mask'].permute(0, 2, 3, 1).to('cpu').numpy()
                    masked_gt = ((gt * mask) * 0.7 + gt * 0.3).astype(np.uint8)
                    mask = (mask * 255).astype(np.uint8)

                    if save_reconstruction:
                        reconstruction = content_dict['reconstruction'].permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)

                    if 'mask_hr' in data_i:
                        mask_hr = data_i['mask_hr'].permute(0, 2, 3, 1).to('cpu').numpy()
                    if 'image_hr' in data_i:
                        gt_hr = data_i['image_hr'].permute(0, 2, 3, 1).to('cpu').numpy()
                    
                    for b in range(content.shape[0]):
                        # basename = os.path.basename(data_i['abs_path'][b])
                        # basename = data_i['relative_path'][b]
                        basename = os.path.basename(data_i['relative_path'][0])
                        if not basename.endswith('.png'):
                            basename = basename.split('.')[:-1] + ['png']
                            basename = '.'.join(basename)

                        # save result
                        save_path = os.path.join(save_root_mr, k, basename)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        im = Image.fromarray(content[b])
                        im.save(save_path)

                        if save_masked_gt:
                            save_path = os.path.join(save_root_mr, 'masked_gt', basename)
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            # import pdb; pdb.set_trace()
                            im = Image.fromarray(masked_gt[b])
                            im.save(save_path)

                        if save_gt:
                            save_path = os.path.join(save_root_mr, 'gt', basename)
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            im = Image.fromarray(gt[b].astype(np.uint8))
                            im.save(save_path)
                        if save_mask:
                            save_path = os.path.join(save_root_mr, 'mask', basename)
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            im = Image.fromarray(mask[b][:,:,-1])
                            im.save(save_path)
                        if save_reconstruction:
                            save_path = os.path.join(save_root_mr, 'reconstruction', basename)
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            im = Image.fromarray(reconstruction[b])
                            im.save(save_path)

                        if 'image_hr' in data_i:
                            save_path = os.path.join(save_root_mr, 'gt_hr', basename)
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            im = Image.fromarray(gt_hr[b].astype(np.uint8))
                            im.save(save_path)
                        if 'mask_hr' in data_i:
                            save_path = os.path.join(save_root_mr, 'mask_hr', basename)
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            im = Image.fromarray(mask_hr[b][:,:,-1])
                            im.save(save_path)

                        print('Rank {}, Total time {}, batch time {:.2f}s, saved in {}'.format(local_rank, format_seconds(time.time()-start_gen), time.time()-start_batch, save_path))
                # import pdb; pdb.set_trace()
                # processed_image = processed_image | set(data_i['relative_path'])
                processed_image_writer.write(''.join([p+'\n' for p in data_i['relative_path']]))
                processed_image_writer.flush()
                print('==> batch time {}s'.format(round(time.time() - start_batch, 1)))
            processed_image_writer.close()


            synchronize()
            metrics = reduce_dict(metrics, average=False)
            total_batch = sum(all_gather(total_count))
            for k in metrics:
                metrics[k] = metrics[k] / total_batch

            if local_rank == 0:
                for k in metrics.keys():
                    metrics[k] = float(metrics[k])
                loss_path = os.path.join(save_root_mr, 'metrics.json')
                os.makedirs(os.path.dirname(loss_path), exist_ok=True)
                json.dump(metrics, open(loss_path, 'w'), indent=4)
                print(metrics)

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--save_dir', type=str, default='RESULT', 
                        help='directory to save results') 

    parser.add_argument('--name', type=str, default='', 
                        help='the name of this experiment, if not provided, set to'
                             'the name of config file') 
    parser.add_argument('--func', type=str, default='inference_reconstruction', #TODO'', 
                        help='the name of inference function') 
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
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size while inference')
    parser.add_argument('--data_type', type=str, default='val',
                        choices=['val', 'train'],
                        help='which type of dataset')                       

    parser.add_argument('--debug', action='store_true', # default=True,
                        help='set as debug mode')

    # args for:
    # inference_complet_sample_in_feature_for_evaluation and 
    # inference_complet_sample_in_feature_for_diversity
    parser.add_argument('--mask_dir', type=str, default='',
                        help='mask dirs for image completion, if provided, each gt image should have'
                        'a correspoding mask to be loaded')                       

    # args for modify config
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )  

    args = parser.parse_args()
    args.cwd = os.path.abspath(os.path.dirname(__file__))

    # modify args for debugging
    if args.debug:
        args.name = 'debug'
        if args.gpu is None:
            args.gpu = 0

    return args


inference_func_map = {
    'inference_reconstruction': inference_reconstruction,
    'inference_complet_sample_in_feature_for_diversity': inference_complet_sample_in_feature_for_diversity,
    'inference_complet_sample_in_feature_for_evaluation': inference_complet_sample_in_feature_for_evaluation
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

