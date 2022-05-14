import torch
import random
import math
import torch.nn.functional as F
from image_synthesis.distributed.distributed import all_reduce, get_world_size

def logits_top_k(logits, filter_ratio = 0.5, minimum=1, pad_value=None, filter_type='count'):
    logits = logits.contiguous()
    # import pdb; pdb.set_trace()
    
    if filter_type == 'count':
        if filter_ratio < 0:
            filter_ratio = - filter_ratio
        if filter_ratio >= 0 and filter_ratio <= 1.0: 
            num_logits = logits.shape[-1]
            k = max(int((1 - filter_ratio) * num_logits), minimum)
        else:
            k = max(int(filter_ratio), minimum)
        # import pdb; pdb.set_trace()
        val, ind = torch.topk(input=logits, k=k, dim=-1)
        if pad_value is None:
            pad_value = float('-inf')
        probs = torch.full_like(logits, pad_value)
        # probs.scatter_(1, ind, val)
        probs.scatter_(-1, ind, val)
        return probs

    elif filter_type == 'prob':
        temperature = 1
        assert filter_ratio >= 0 and filter_ratio <= 1.0, 'For probability topk, threshold should be in [0,1]'
        probs = F.softmax(logits * temperature, dim = -1)
        probs = probs.view(-1, logits.shape[-1]) # N x C
        mask = probs >= filter_ratio # N x C
        # print('kept count: ', mask.sum(dim=-1))
        # import pdb; pdb.set_trace()
        _, idx = probs.max(dim=1, keepdim=True)
        mask = mask.index_fill_(1, idx.squeeze(dim=-1), True)
        mask = mask.view(logits.shape)
        logits = logits.masked_fill(~mask, float('-inf'))
        return logits
    else:
        raise NotImplementedError("filter type {} not implemented!".format(filter_type))


def mask_with_top_k(x, k, largest=True, abs=True, pad_value=None):
    """
    mask the input tensor along the last dimension.
    The values the not in the topk will be masked as zeros
    
    """
    if abs:
        x_ = x.abs()
    else:
        x_ = x
    _, top_k_index = x_.topk(k=k, dim=-1, largest=largest) # BHW x K

    mask = torch.zeros_like(x)
    ones = torch.ones_like(x)
    mask.scatter_(-1, index=top_k_index, src=ones)

    x = x * mask
    if pad_value is None or pad_value != 0:
        if pad_value is None:
            pad_value = float('-inf')
        x[mask == 0] = x[mask == 0] + pad_value
    return x


def pixel_unshuffle(input, out_size, chunked=False):
    """
    Resize the given input to the given size

    Args:
        input: 4D tensor, B x c x H x W
        out_size: (H/r, W/r)
        chunked: bool, control the shuffle manner. Take RGB channel and r=2 iamge for example:
            when it is False, RGB (input) -> RRRRGGGGBBBB (output), this is the default setting in pytorch
            when it is True,  RGB (input) -> RRRRGGGGBBBB -> RGBRGBRGBRGB (output)
    return:
        output: [B x c*r^2, H/r W/r]

    """
    if input is None or tuple(out_size) == tuple(input.shape[-2:]):
        return input

    if input.dtype in [torch.int64, torch.uint8, torch.bool]:
        dtype = input.dtype
        input = input.to(torch.float16)
    else:
        dtype = None

    b, c1, h1, w1 = input.shape
    h2, w2 = out_size[0], out_size[1]
    assert h1 % h2 == 0 and w1 % w2 == 0, "This resize function is only support divisible resize!"

    kh, kw = int(h1/h2), int(w1/w2)

    output = torch.nn.functional.unfold(input, kernel_size=(kh, kw), stride=(kh, kw)) # B x kh*kw*c x H/r*W/r
    output = torch.nn.functional.fold(output, output_size=out_size, kernel_size=(1, 1),stride=(1, 1)) # B x kh*kw*c x H/r x W/r

    if chunked:
        index = torch.LongTensor(range(0, output.shape[1], kh*kw)).view(1, -1).repeat(kh*kw, 1).to(output.device)
        shift = torch.LongTensor(range(0, kh*kw)).view(-1, 1).to(output.device)
        index += shift
        index = index.view(-1)
        output = output[:, index, :, :].contiguous()

    if dtype is not None:
        output = output.to(dtype)
    return output


def pixel_shuffle(input, out_size, chunked=False):
    """
    Resize the given input to the given size

    Args:
        input: 4D tensor, B x c x H x W
        out_size: (H*r, W*r)
        chunked: bool, control the shuffle manner. Take RGB channel and r=2 iamge for example:
                when it is False, RRRRGGGGBBBB (input) -> RGB (output), this is the default setting in pytorch
                when it is True,  RGBRGBRGBRGB (input) -> RRRRGGGGBBBB -> RGB -> (output)
    
    return:
        output: [B x c/r^2, H*r W*r]
    """
    if input is None or tuple(out_size) == tuple(input.shape[-2:]):
        return input

    if input.dtype in [torch.int64, torch.uint8, torch.bool]:
        dtype = input.dtype
        input = input.to(torch.float16)
    else:
        dtype = None

    b, c1, h1, w1 = input.shape
    h2, w2 = out_size[0], out_size[1]
    assert h2 % h1 == 0 and w2 % w1 == 0, "This resize function is only support divisible resize!"
 
    kh, kw = h2//h1, w2//w1
    
    if chunked:
        out_c = c1 // (kh * kw)
        index = torch.LongTensor(range(0, input.shape[1], out_c)).view(1, -1).repeat(out_c, 1).to(input.device)
        shift = torch.LongTensor(range(0, out_c)).view(-1, 1).to(input.device)
        index += shift
        index = index.view(-1)
        input = input[:, index, :, :].contiguous()
    
    # assert kh == kw, 'pixel shuffle only support the same kernel size for weidth and height'
    # s = kh
    # output = torch.nn.functional.pixel_shuffle(input, s) # B x c/r^2, H*r W*r

    output = torch.nn.functional.unfold(input, kernel_size=(1, 1), stride=(1, 1)) # B x C x H*W
    output = torch.nn.functional.fold(output, output_size=out_size, kernel_size=(kh, kw),stride=(kh, kw)) # B x c/r^2, H*r W*r

    if dtype is not None:
        output = output.to(dtype)
    return output



def get_token_type(mask, token_shape, type='pixel_shuffle'):
    """
    Get the token type according to the given mask and token_shape.
    Note that we treat tokens into 3 types.
        0: fully masked tokens
        1: unmasked tokens
        2: partially masked tokens   

    Args:
        mask: 4D tensor, B x 1 x H x W, the mask of the origin image. 1 denotes masked pixles 
            and 0 denotes unmasked pixels.
        token_shape: [H/r, W/r]. the shape of token

    """
    dim_ = mask.dim()
    if dim_ == 3:
        mask = mask.unsqueeze(dim=0) # H x W x C -> B x H x W x C
    elif dim_ == 2:
        mask = mask.unsqueeze(dim=0).unsqueeze(dim=-1) # H x W  -> B x H x W x C
    chw = mask.shape[1] in [1, 3]
    if not chw:
        mask = mask.permute(0, 3, 1, 2)

    mask_float = mask.float()
    
    if type == 'pixel_shuffle':
        mask_unshuffle = pixel_unshuffle(mask_float, token_shape) # B x r^2 x H/r x W/r

        scale_factor = mask_unshuffle.shape[1]
        mask_unshuffle = mask_unshuffle.sum(dim=1, keepdim=True) # B x 1 x H/r x W/r

        token_type = torch.zeros_like(mask_unshuffle).long() + 2
        
        token_type[mask_unshuffle==0] = 0 # fully masked tokens
        token_type[mask_unshuffle==scale_factor] = 1 # fully unmasked tokens
    elif type == 'nearest':
        token_type = torch.nn.functional.interpolate(mask.float(), size=token_shape, mode='nearest').long()
    else:
        raise NotImplementedError('not implemented for {}'.format(type))
    
    if not chw:
        token_type = token_type.permute(0, 2, 3, 1)
    if dim_ == 3:
        token_type = token_type.squeeze(dim=0)
    elif dim_ == 2:
        token_type = token_type.squeeze(dim=0).squeeze(dim=-1)
    return token_type

def sample_index_randomly(x, k, filter_ratio=0, largest=True):
    """
    x: should be 2D tensor, randomly smaple along the lat dimension
    """
    assert x.dim() == 2, 'currently only two dimensional tensors are supprted!'
    
    if filter_ratio < 0:
        filter_ratio = - filter_ratio
    if filter_ratio >= 0 and filter_ratio <= 1.0: 
        num_logits = x.shape[-1]
        topk = max(int((1 - filter_ratio) * num_logits), k)
    else:
        topk = max(int(filter_ratio), k)
    
    _, top_k_index = x.topk(k=topk, dim=-1, largest=largest) # BHW x K

    sampled = []
    for i in range(x.shape[0]):
        index = top_k_index[i]
        sampled_ = torch.tensor(random.sample(index.tolist(), k)).to(index)
        sampled.append(sampled_)
    sampled = torch.stack(sampled, dim=0).to(top_k_index)
    return sampled

def gen_attention_mask(H, W, type='full', causal=True, condition_seq_len=0, **kwargs):


    content_seq_len = H * W
    seq_len = content_seq_len + condition_seq_len
    mask = torch.zeros(seq_len, seq_len)

    mask[:, :condition_seq_len] = 1

    if type == 'full':
        mask += 1
    elif type == 'dalle_row':
        for idx in range(content_seq_len):
            h = idx // W
            w = idx % W
            for w_ in range(w-W, w+1):
                i = h * W + w_
                mask[idx+condition_seq_len][i+condition_seq_len] = 1

    elif type == 'dalle_col':
        for idx in range(content_seq_len):
            h = idx // W
            w = idx % W
            for h_ in range(h+1):
                i = h_ * W + w 
                mask[idx+condition_seq_len][i+condition_seq_len] = 1
    elif type == 'dalle_conv':
        kernel_size = kwargs['kernel_size']
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        k_h, k_w = kernel_size[0], kernel_size[1]
        half_k_h = int(k_h/2)
        half_k_w = int(k_w/2)
        step_over_w = W - k_w 
        
        for idx in range(content_seq_len):
            max_kernel_count = (half_k_h+1) * k_w 
            step_over_count = step_over_w * (half_k_h+1)

            max_pre = max_kernel_count + step_over_count
            max_pre = min(idx+1, max_pre)

            for i in range(max_pre):
                valid = False 
                a = i % W 
                if a > half_k_w and a <= half_k_w + step_over_w:
                    valid = False  
                else:
                    valid = True 
                if valid:
                    mask[idx+condition_seq_len][idx-i+condition_seq_len] = 1
    else:
        raise NotImplementedError('attention type {} not implemented!'.format(type))

    if causal:
        causal_mask = torch.tril(torch.ones(content_seq_len+condition_seq_len, content_seq_len+condition_seq_len))
        mask *= causal_mask
    
    return mask


def distributed_sinkhorn(out, epsilon=0.05, iterations=3):
    Q = torch.exp(out / epsilon).t() # Q is K-by-B 
    B = Q.shape[1] * get_world_size() # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True) # 
        all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t() # B x K

if __name__ == '__main__':

    import cv2
    from PIL import Image
    import numpy as np
    from image_synthesis.data.utils.util import generate_mask_based_on_landmark, generate_stroke_mask

    # mask = generate_stroke_mask((256, 256))
    mask = np.array(Image.open('data/ffhq/00000/00000.png'))

    mask_t = torch.tensor(mask).unsqueeze(dim=0).permute(0, 3, 1, 2).float() # 1 x c x h x w

    size = list((int(s/32) for s in mask_t.shape[-2:]))

    mask_shuffle = pixel_unshuffle(mask_t, size, chunked=True) # 1 x 3 x size[0] x size[1]
    mask_shuffle = pixel_shuffle(mask_shuffle, mask_t.shape[-2:], chunked=True)
    # mask_shuffle = mask_shuffle[:,[0,1,2],:,:]
    mask_shuffle = mask_shuffle[0].permute(1, 2, 0).numpy()

    diff = np.sum(np.abs(mask - mask_shuffle))
    print('diff', diff)

    mask_ = Image.fromarray((mask[:,:]).astype(np.uint8))
    # mask.show()
    mask_.save('mask.png')
    mask_shuffle = Image.fromarray((mask_shuffle[:,:]).astype(np.uint8))
    # mask_resize.show()
    mask_shuffle.save('mask_shuffle.png')

    # mask_resize_t = torch.nn.functional.interpolate(mask_t, size, mode='nearest')
    # mask_resize_t = mask_resize_t[0].permute(1, 2, 0).numpy()
    # mask_resize_t_ = Image.fromarray((mask_resize_t[:,:,0]*255).astype(np.uint8))
    # mask_resize_t_.save('mask_resize_t.png')


    # mask_resize = cv2.resize(mask[:,:, 0].astype(np.uint8), tuple(size), interpolation=cv2.INTER_NEAREST)
    # mask_resize_ = Image.fromarray((mask_resize*255).astype(np.uint8))
    # mask_resize_.save('mask_resize.png')

    # resize_diff = np.sum((mask_resize - mask_resize_t[:,:,0]).astype(np.float32))
    # print(resize_diff)



    # token_type = get_token_type(mask_t, token_shape=size, type='pixel_shuffle')
    # mask_shuffle_token_type = (token_type == 1)[0].permute(1, 2, 0).float().numpy()
    # mask_shuffle_token_type = Image.fromarray((mask_shuffle_token_type[:,:,0]*255).astype(np.uint8))
    # mask_shuffle_token_type.save('mask_shuffle_token_type.png')


    # token_type = get_token_type(mask_t, token_shape=size, type='nearest')
    # mask_nearest_token_type = (token_type == 1)[0].permute(1, 2, 0).float().numpy()
    # mask_nearest_token_type = Image.fromarray((mask_nearest_token_type[:,:,0]*255).astype(np.uint8))
    # mask_nearest_token_type.save('mask_nearest_token_type.png')

    # mask = cv2.imread('data/imagenet/val/n01440764/ILSVRC2012_val_00003014.JPEG')
    # mask = cv2.resize(mask, (256, 256))


    # mask_t = torch.tensor(mask).unsqueeze(dim=0).permute(0, 3, 1, 2).float() # 1 x c x h x w
    # size = list((int(s/8) for s in mask_t.shape[-2:]))

    # mask_resize = pixel_unshuffle(mask_t, size) # 1 x 3 x size[0] x size[1]
    # mask_resize = pixel_shuffle(mask_resize, mask_t.shape[-2:])
    # mask_resize = mask_resize[0].permute(1, 2, 0).numpy()

    # print('diff: ', np.sum(np.abs(mask - mask_resize)))


    # mask = Image.fromarray((mask[:, :, ::-1]).astype(np.uint8))
    # mask.show()
    # mask_resize = Image.fromarray((mask_resize[:, :,::-1]).astype(np.uint8))
    # mask_resize.show()








