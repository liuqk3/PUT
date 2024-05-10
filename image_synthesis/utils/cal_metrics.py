import os
import cv2
from PIL import Image
import numpy as np
import math
import torch
from skimage.color import rgb2gray


def read_results(dir, name):
    results = cv2.imread(os.path.join(dir, name))
    H, W, _ = results.shape
    nums = int(W/H)
    w = H
    input = results[:,:w,:]
    fake = results[:,(nums-2)*w:(nums-1)*w,:]
    real = results[:,(nums-1)*w:,:]
    input = input[...,::-1] # to rgb
    fake = fake[...,::-1] # to rgb
    real = real[...,::-1] # to rgb
    # read the original bmp image
    idx = name.split('-')[1]
    input_name = name.split('_')[0]+'.bmp'
    root = os.path.join('/home/tzt/dataset/20200317',idx, input_name)
    input = cv2.imread(root)[...,::-1]
    return input, fake, real

def get_entropy(img0):
    if len(img0.shape) == 3:
        img_ = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    else:
        img_ = img0.copy()
    x, y = img_.shape[0:2]
    # img_ = cv2.resize(img_, (100, 100)) # 缩小的目的是加快计算速度
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    img = np.array(img_)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k =  float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res

def get_hight_V(img0):
    if len(img0.shape) == 3:
        img_ = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    else:
        img_ = img0.copy()
    H = np.mean(img_)
    V = np.mean((img_ - H)**2)
    return H, V

def get_PSNR(img1, img2, tool='none'):
    if tool == 'skimage':
        from skimage.metrics import peak_signal_noise_ratio
        if torch.is_tensor(img1):
            device = img1.device
            assert img1.dim() == 4, "only batch based data is implemented!"
            psnr = []
            img1 = img1.permute(0, 2, 3, 1)/255.0 # B C H W -> B H W C
            img2 = img2.permute(0, 2, 3, 1)/255.0 # B C H W -> B H W C
            for i in range(img1.shape[0]):
                im1_ = img1[i].to('cpu').numpy()
                im2_ = img2[i].to('cpu').numpy()
                if im1_.shape[-1] == 1: # grapy image
                    im1_ = im1_[:,:,0]
                    im2_ = im2_[:,:,0]
                psnr_ = peak_signal_noise_ratio(im1_, im2_)
                psnr.append(psnr_)
            psnr = torch.tensor(psnr, device=device).mean()
        else:
            psnr = peak_signal_noise_ratio(img1.astype(np.float32)/255.0, img2.astype(np.float32)/255.0)
    elif tool == 'none':
        PIXEL_MAX = 1
        if torch.is_tensor(img1):
            assert img1.dim() == 4, 'Only batch based data is implemented!'
            mse = ((img1/255.0 - img2/255.0) ** 2).view(img1.shape[0], -1).mean(dim=1) # Batch
            mask = mse < 1.0e-10
            mse[mask] = 100.0
            mse[~mask] = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse[~mask]))
            psnr = mse.mean()
        else:
            mse = np.mean( (img1/255. - img2/255.) ** 2 )
            if mse < 1.0e-10:
                return 100
            psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    else:
        raise NotImplementedError
    return psnr

def get_mse_loss(img1, img2):
    if torch.is_tensor(img1):
        assert img1.dim() == 4, 'Only batch based data is implemented!'
        mse = ((img1/255.0 - img2/255.0) ** 2).view(img1.shape[0], -1).mean(dim=1) # Batch
    else:
        raise NotImplementedError
    return mse.mean()


def get_l1_loss(img1, img2):
    if torch.is_tensor(img1):
        assert img1.dim() == 4, 'Only batch based data is implemented!'
        l1 = (img1/255.0 - img2/255.0).abs().view(img1.shape[0], -1).mean(dim=1) # Batch
    else:
        raise NotImplementedError
    return l1.mean()


def get_mae(img1, img2):
    if torch.is_tensor(img1):
        assert img1.dim() == 4, 'Only batch based data is implemented!'
        mae = (img1.flatten(1)/255.0 - img2.flatten(1)/255.0).abs().sum(dim=-1) / (img1.flatten(1)/255.0 + img2.flatten(1)/255.0).sum(dim=-1)
    else:
        raise NotImplementedError
    return mae.mean()

def get_SSIM(img1, img2, full=True, win_size=None):
    from skimage.metrics import structural_similarity
    def get_one_pair_image(im1, im2):
        if len(im1.shape) == 3:
            if im1.shape[-1] == 3:
                im1_ = cv2.cvtColor(im1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                im1_ = im1[:, :, 0].astype(np.uint8)
        else:
            im1_ = im1.copy().astype(np.uint8)
        if len(im2.shape) == 3:
            if im2.shape[-1] == 3:
                im2_ = cv2.cvtColor(im2.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                im2_ = im2[:, :, 0].astype(np.uint8)
        else:
            im2_ = im2.copy().astype(np.uint8)
        
        # from skimage.measure import compare_ssim
        if full:
            score, diff = structural_similarity(im1_, im2_, full=full, win_size=win_size)
            diff = (diff * 255).astype("uint8")
        else:
            score = structural_similarity(im1_, im2_, full=full, win_size=win_size)
        return score
    
    if torch.is_tensor(img1): # tensor
        is_tensor = True
        device = img1.device
        # import pdb; pdb.set_trace()
        if len(img1.shape) == 3: # H, W ,C
            img1 = img1.unsqueeze(dim=0).to('cpu').numpy() # 1 H W C
            img2 = img2.unsqueeze(dim=0).to('cpu').numpy()
        else: # B C H W
            img1 = img1.permute(0, 2, 3, 1).to('cpu').numpy() # B H W C
            img2 = img2.permute(0, 2, 3, 1).to('cpu').numpy()
    else:
        is_tensor = False
        img1 = img1.copy()[np.newaxis, ...]
        img2 = img2.copy()[np.newaxis, ...]
    
    score = 0
    count = img1.shape[0]
    for i in range(count):
        im1 = img1[i]
        im2 = img2[i]
        score += get_one_pair_image(im1, im2)
    if is_tensor:
        score = torch.tensor(score).to(device)
    return score / count


def calc_psnr_edsr(sr, hr, scale, isNormalize=True, rgb_range=255, benchmark=True):
    if hr.nelement() == 1: return 0
    if isNormalize:
        sr = torch.clamp(((sr.detach() + 1.0) * 127.5).float(), 0., 255.)
        hr = torch.clamp(((hr.detach() + 1.0) * 127.5).float(), 0., 255.)
    diff = (sr - hr) / rgb_range
    
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
            # print(diff.shape, scale)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * torch.log10(mse)