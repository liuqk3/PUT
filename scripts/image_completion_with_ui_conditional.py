import os
import sys
import torch
import time
import argparse
import numpy as np

import glob
from PIL import Image, ImageDraw
from collections import OrderedDict

from image_synthesis.utils.io import load_yaml_config
from image_synthesis.utils.misc import get_all_file, get_all_subdir
from image_synthesis.modeling.build import build_model
from image_synthesis.utils.misc import get_model_parameters_info


def get_model(args):
    model_path_dict = {
        'tpami2024_ffhq_256': 'OUTPUT/tpami2024_vit_base_ffhq_seg_sketch_dual_encoder_res256/checkpoint/last.pth',
        'tpami2024_imagenet_256': 'OUTPUT/tpami2024_vit_base_imagenet_seg_sketch_dual_encoder_res256/checkpoint/last.pth',
        'tpami2024_naturalscene_256': 'OUTPUT/tpami2024_vit_base_naturalscene_seg_sketch_dual_encoder_res256/checkpoint/last.pth'
    }
    
    model_name = args.name
    if model_name in model_path_dict:
        model_name = model_path_dict[model_name]

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
            model_name = model_path.split(os.path.sep)[-3]
        else: # just give a config file, such as test_openai_dvae.yaml, which is no need to train, just test
            model_name = os.path.basename(config_path).replace('.yaml', '')
    else:
        model_path = os.path.join('OUTPUT', model_name, 'checkpoint', 'last.pth')
        config_path = os.path.join(os.path.join('OUTPUT', model_name, 'configs', 'config.yaml'))

    config = load_yaml_config(config_path)
    # config = merge_opts_to_config(config, args.opts)
    model = build_model(config)

    model.content_codec.combine_rec_and_gt = not args.not_merge

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

    print('Model missing keys:\n', missing)
    print('Model unexpected keys:\n', unexpected)

    return {'model': model, 'epoch': epoch, 'model_name': model_name, 'parameter': model_parameters}


def preprocess(args, im, type='im'):
    """
    im: PIL.Image 
    """
    if min(im.size) != min(args.input_res):
        w, h = im.size
        if w < h:
            w_ = min(args.input_res)
            h_ = int(w_/w * h)
        else:
            h_ = min(args.input_res)
            w_ = int(h_/h * w)
        if type == 'im':
            im = im.resize((w_, h_), resample=Image.BILINEAR)
        else:
            im = im.resize((w_, h_), resample=Image.NEAREST)

    w, h = im.size
    hh, hw = tuple(args.input_res)

    if w != hw or h != hh:
        assert hh < h and hw < w, "input resolution should be smaller than the origin image size"
        dx = (w - hw) // 2
        dy = (h - hh) // 2
        im = im.crop((dx, dy, dx + hw, dy + hh))

    return im


def prepare_im_and_mask(args, im, mask, seg_result, sketch=None):
    if isinstance(im, str):
        im = Image.open(im).convert('RGB')
    im = preprocess(args, im, type='im')

    if isinstance(mask, str):
        mask = Image.open(mask).convert('L')
    mask = preprocess(args, mask, type='mask')

    im = np.array(im).astype(np.float32)
    im = torch.tensor(im).permute(2, 0, 1)

    mask = np.array(mask).astype(np.float32)
    mask = mask / 255.0
    if len(mask.shape) == 3:
        mask = mask[:, :, 0:1] # h, w, 1
    else:
        mask = mask[:, :, np.newaxis]
    mask = torch.tensor(mask).permute(2, 0, 1).bool() # 1, h, w

    seg_result = np.array(seg_result)
    seg_result = torch.tensor(seg_result)

    data = {
        'image': im.unsqueeze(dim=0),
        'mask': mask.unsqueeze(dim=0),
        'seg_result': seg_result.unsqueeze(dim=0).unsqueeze(dim=0),
    }  

    if sketch is not None:

        if sketch.ndim == 2:
            sketch = sketch[:, :, np.newaxis]

        sketch = torch.tensor(sketch.astype(np.float32)).permute(2,0,1)

        data['sketch_map'] = sketch.unsqueeze(dim=0)
    return data


def inpaint(args, model, data, mode=None):

    count = 0
    result = []
    att_weight = []
    while count < args.num_samples:
        replicate = min(args.num_samples - count, args.batch_size)
        inputs = {
            'batch': {k:v.clone() if k != 'text' else v for k, v in data.items()}, # data,
            'filter_ratio': args.topk,
            'replicate': replicate,
            'with_process_bar': True,
            'mask_low_to_high': args.mask_low_to_high,
            'sample_largest': True,
            'return_att_weight': args.att_weight,
            'num_token_per_iter': args.num_token_per_iter,
        }

        if mode is not None:
            inputs['mode'] = mode

        # import pdb; pdb.set_trace()
        with torch.no_grad():
            output = model.generate_content(**inputs)
        result.append(output['completed'].to('cpu'))
        if args.att_weight:
            att_weight += output['att_weight']
        count += output['completed'].shape[0]
    
    if len(result) > 1:
        result = torch.cat(result, dim=0)
    else:
        result = result[0]
    result_ = {
        'completed': result,
    }
    if args.att_weight:
        result_['att_weight'] = att_weight
    return result_

def prepare_data_paths(args):
    # im_path = args.im_path
    # import pdb; pdb.set_trace()
    if os.path.isfile(args.im_path):
        if args.im_path.endswith('.txt'):
            with open(args.im_path) as f:
                im_paths = f.readlines()
                im_paths = [p.strip() for p in im_paths]
            with open(args.mask_path) as f:
                mask_paths = f.readlines()
                mask_paths = [p.strip() for p in im_paths]
        else:
            im_paths = [args.im_path]
            mask_paths = [args.mask_path]
    elif os.path.isdir(args.im_path):
        if args.im_path == args.mask_path:
            sub_dirs = get_all_subdir(args.im_path, max_depth=1, min_depth=1, path_type='abs')
            im_paths = []
            mask_paths = []
            for sd in sub_dirs:
                paths = get_all_file(sd, end_with=['.png', '.jpg', '.JPEG'], path_type='abs')
                for p in paths:
                    basename = os.path.basename(p)
                    if 'completed' in basename or 'gt_mask_image' in basename:
                        continue
                    elif 'mask' in basename:
                        mask_paths.append(p)
                    else:
                        im_paths.append(p)
                assert len(mask_paths) == len(im_paths), 'mask and image path are not the same'
        else:
            im_paths = sorted(get_all_file(args.im_path, end_with=['.png', '.jpg', '.JPEG']))
            mask_paths = sorted(get_all_file(args.mask_path, end_with='.png'))
    else:
        if not args.ui:
            raise RuntimeError('provided image path not foud!')
        else:
            im_paths = []
            mask_paths = []
    return {'im_paths': im_paths, 'mask_paths': mask_paths}

def main(local_rank, args):
    torch.cuda.set_device(local_rank)
    if not args.ddp:
        local_rank = 0
    paths = prepare_data_paths(args)
    im_paths = paths['im_paths']
    mask_paths = paths['mask_paths']

    # filter processed files
    # import pdb; pdb.set_trace()
    if args.check_exist:
        im_paths_ = []
        mask_paths_ = []
        # import pdb; pdb.set_trace()
        for i in range(len(im_paths)):
            p = im_paths[i]
            basename = os.path.basename(p).replace('.png', '').replace('.JPEG', '').replace('.jpg', '')
            save_path = save_path = os.path.join(args.save_dir, basename, 'completed_{}.png'.format(str(args.num_samples).zfill(2)))
            
            if not os.path.exists(save_path):
                im_paths_.append(im_paths[i])
                mask_paths_.append(mask_paths[i])
        print('Filter {} files, current: {}'.format(len(im_paths)-len(im_paths_), len(im_paths_)))
        im_paths = im_paths_ 
        mask_paths = mask_paths_

    num_per_node = (len(im_paths) + args.world_size - 1) // args.world_size
    start = local_rank * num_per_node
    im_paths = im_paths[start: start + num_per_node]
    mask_paths = mask_paths[start: start + num_per_node]
    print('Rank: {}, number of files: {}'.format(local_rank, len(im_paths)))

    model_info = get_model(args)
    model = model_info['model'].cuda()
    # import pdb; pdb.set_trace()
    for idx in range(len(im_paths)):
        tic = time.time()
        im_path = im_paths[idx]
        mask_path = mask_paths[idx]
        
        data = prepare_im_and_mask(args, im_path, mask_path)
        
        result = inpaint(args=args, model=model, data=data)
        completed = result['completed']
        basename = os.path.basename(im_path).replace('.png', '').replace('.JPEG', '').replace('.jpg', '')
        save_dir = os.path.join(args.save_dir, basename)
        os.makedirs(save_dir, exist_ok=True)
        
        # save_gt and mask
        gt = data['image'].permute(0, 2, 3, 1).to('cpu').numpy()
        mask = data['mask'].int().permute(0, 2, 3, 1).to('cpu').numpy()

        if args.check_exist:
            pass 
        else:
            mask_count = len(glob.glob(os.path.join(save_dir, '*_mask.png')))
        
        if not args.not_save_mask:
            if args.check_exist:
                save_path = os.path.join(save_dir, 'mask.png')
            else:
                save_path = os.path.join(save_dir, '{}_mask.png'.format(str(mask_count).zfill(4)))
            mask = Image.fromarray((mask[0,:,:,0]*255).astype(np.uint8))
            mask.save(save_path)        
        
        if not args.not_save_im:
            save_path = os.path.join(save_dir, basename+'.png')
            gt = Image.fromarray(gt[0].astype(np.uint8))
            gt.save(save_path)

        completed = completed.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for i in range(completed.shape[0]):
            if args.check_exist:
                save_path = os.path.join(save_dir, 'completed_{}.png'.format(str(i).zfill(2)))
            else:
                save_path = os.path.join(save_dir, '{}_completed_{}.png'.format(str(mask_count).zfill(4), str(i).zfill(2)))
            res = Image.fromarray(completed[i])
            res.save(save_path)
            print('result saved to {}'.format(save_path))

            if args.att_weight:
                p_h = 8
                p_w = 8
                att_weight = result['att_weight']
                if args.check_exist:
                    save_dir_att = os.path.join(save_dir, 'attention_weight_{}'.format(str(i).zfill(2))) 
                else:
                    save_dir_att = os.path.join(save_dir, '{}_attention_weight_{}'.format(str(mask_count).zfill(4), str(i).zfill(2)))
                os.makedirs(save_dir_att, exist_ok=True)
                for token_idx in att_weight[i].keys():
                    res_arr = np.array(res).astype(np.float)

                    att_color = np.zeros_like(res_arr)
                    att_color[:,:,1] = 255.0 # green
                    att = att_weight[i][token_idx].to('cpu')  # H x W
                    att = (att - att.min()) / (att.max() - att.min())
                    h, w = att.shape[0], att.shape[1]
                    att = Image.fromarray(att.numpy()).resize((args.input_res[1], args.input_res[0]), resample=Image.BILINEAR)
                    att_arr = np.array(att)[:, :, np.newaxis]
                    # res_att_arr = (att_arr * att_color + res_arr * (1-att_arr)).astype(np.uint8)
                    att_color = att_arr * att_color

                    alpha = 0.7
                    res_att_arr = (alpha * att_color + (1-alpha) * res_arr).astype(np.uint8)
                    
                    res_att = Image.fromarray(res_att_arr)
                    
                    # get the position of this patch
                    rows = token_idx // w 
                    cols = token_idx % w
                    x1 = cols * p_w 
                    y1 = rows * p_h
                    x2 = (cols + 1) * p_w 
                    y2 = (rows + 1) * p_h 
                    draw = ImageDraw.ImageDraw(res_att)
                    draw.rectangle(((x1, y1), (x2, y2)), fill=None, outline='yellow', width=2)
                    save_path = os.path.join(save_dir_att, '{}.png'.format(str(token_idx).zfill(3)))
                    res_att.save(save_path)

        toc = time.time()
        print('{}/{}, {}s: {}'.format(idx, len(im_paths), round(toc-tic, 3), im_path))
        


def ddp_launch(func, args=()):
    import torch.multiprocessing as mp
    num_gpu = torch.cuda.device_count()
    mp.spawn(
        func,
        nprocs=num_gpu,
        args=args,
        daemon=False,
    )




def get_args():
    parser = argparse.ArgumentParser(description='PyTorch image inpainting inference')

    parser.add_argument('--name', type=str, default='naturalscene',#TODO'',
                        help='the name of this experiment, if not provided, set to'
                             'the name of config file') 
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')   
    parser.add_argument('--input_res', type=str, default='256,256',
                        help='input resolution, (H, W)')                        
    parser.add_argument('--num_samples', type=int, default=10,
                        help='number of samples to get') 
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size') 
    parser.add_argument('--topk', type=int, default=200,
                        help='top k tokens for sampling')  
             
    parser.add_argument('--mask_low_to_high', action='store_true', default=False,
                        help='downsample the mask by NEAREST, and then upsample the mask by NEARSET.'
                        'this type of operation is the same with ICT')  
    
    parser.add_argument('--save_dir', type=str, default='RESULT/visualization/naturalscene',#TODO'RESULT/visualization', 
                        help='directory to save results') 
    parser.add_argument('--im_path', type=str, default='RESULT/naturalscene_selected',#TODO'', 
                        help='path to image') 
    parser.add_argument('--mask_path', type=str, default='RESULT/naturalscene_selected', #TODO'', 
                        help='path to mask') 


    parser.add_argument('--att_weight', action='store_true', default=False,
                        help='save attention weight for each patch (token)')  
    parser.add_argument('--not_merge', action='store_true', default=False,
                        help='not merge gt with result while reconstructing the inpainted image')  
    parser.add_argument('--not_save_im', action='store_true', default=False,
                        help='not save gt images')  
    parser.add_argument('--not_save_mask', action='store_true', default=False,
                        help='not save mask image')  
    parser.add_argument('--ddp', action='store_true', default=False,
                        help='ddp testing, this is useful when lots of images and masks are provided. Only effective without ui') 
    parser.add_argument('--check_exist', action='store_true', default=False,
                        help='check existing results if there are already some results in the given save directory')   
    parser.add_argument('--ui', action='store_true', default=False,
                        help='launch with ui')  
    parser.add_argument('--num_token_per_iter', type=int, default=1,
                        help='number of tokens per iter')  


    args = parser.parse_args()
    args.cwd = os.path.abspath(os.path.dirname(__file__))

    args.input_res = tuple([int(r) for r in args.input_res.split(',')])

    return args


if __name__ == '__main__':

    args = get_args()
    
    if args.ui:
        torch.cuda.set_device(args.gpu)

        from PyQt5.QtWidgets import QApplication
        from image_synthesis.ui.image_inpaint.demo_conditional import Ex

        data_paths = prepare_data_paths(args)

        model_info = get_model(args)
        model = model_info['model'].cuda()

        app = QApplication(sys.argv)
        ex = Ex(
            args=args,
            model=model_info['model'],
            data_paths=data_paths,
            prepare_im_and_mask_func=prepare_im_and_mask,
            inpaint_func=inpaint,
        )
        sys.exit(app.exec_())

    else:
        if args.ddp:
            num_node = 1
            args.world_size = torch.cuda.device_count() * num_node
            ddp_launch(main, args=(args,))
        else:
            args.world_size = 1
            main(local_rank=args.gpu, args=args)












