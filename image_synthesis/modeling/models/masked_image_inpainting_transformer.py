"""
This transformer model is for PUT in to be published in Journal
"""
import torch
import math
import time
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image, ImageDraw
from timm.models.layers import DropPath, trunc_normal_
from timm.models.layers import trunc_normal_ as __call_trunc_normal_


from image_synthesis.utils.misc import instantiate_from_config
from image_synthesis.modeling.utils.misc import get_token_type
from image_synthesis.distributed.distributed import get_local_rank
from image_synthesis.modeling.utils.misc import logits_top_k, pixel_unshuffle, pixel_shuffle
from image_synthesis.modeling.modules.losses.poly_loss import PolyLoss
from image_synthesis.modeling.modules.losses.label_smoothing_loss import LabelSmoothingLoss


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def window_partition(x, window_size, partition_type='partition'):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    if partition_type == 'partition':
        x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    elif partition_type == 'shuffle':
        nw = int(H * W / window_size[0] / window_size[1])
        windows = pixel_unshuffle(x.permute(0, 3, 1, 2), out_size=window_size, chunked=True).permute(0, 2, 3, 1) # B x wH x wW x C*nW
        windows = torch.chunk(windows, chunks=nw, dim=-1) # tuple(B x wH x wW x C), nw
        windows = torch.cat(windows, dim=0) # B*nW x wH x wW x C
    # import pdb; pdb.set_trace()
    return windows


def window_unpartition(windows, window_size, H, W, partition_type='partition'):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    if partition_type == 'partition':
        x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    else:
        nw = int(H * W / window_size[0] / window_size[1])
        x = torch.chunk(windows, chunks=nw, dim=0) # tuple(B x wH x wW x C), nw
        x = torch.cat(x, dim=-1) # B x wH x wW x C*nW
        x = pixel_shuffle(x.permute(0, 3, 1, 2), out_size=(H, W), chunked=True) # B x C x H x W
        x = x.permute(0, 2, 3, 1)
    return x



class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, 
                dim, 
                num_heads=8, 
                qkv_bias=False, 
                attn_drop=0., 
                proj_drop=0.,
        
                window_size=None, # the window size to perform attention, similar to swin transformer
                shift_size=(0,0), # the size to cycle shift the input feature, similar to swin transformer, only effective when window size is not None
                partition_type='shuffle',
                apply_window_mask=False,
    ):
        super().__init__()

        if window_size is not None and not isinstance(window_size, (tuple, list)):
            window_size = (window_size, window_size)
        if shift_size is not None and not isinstance(shift_size, (tuple, list)):
            shift_size = (shift_size, shift_size)
        self.window_size = tuple(window_size) if window_size is not None else None
        self.shift_size = tuple(shift_size) if shift_size is not None else None
        self.apply_window_mask = apply_window_mask
        self.window_mask = None

        for pt in partition_type.split(','):
            assert pt in ['shuffle', 'partition'], 'not implemented reduce type {}'.format(pt)
        self.partition_type = partition_type 

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        all_head_dim = head_dim * self.num_heads
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def set_window_attn_mask(self, resolution, partition_type='partition', device=None):
        assert partition_type == 'partition', 'Following swin transformer, the mask is only needed for parition window!'
        H, W = resolution 
        img_mask = torch.zeros((1, H, W, 1))  # 1 x H x W x 1
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size=self.window_size, partition_type='partition')  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1]) # nw x wH*wW
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # nw x wH*wW x wH*wH
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        if device is not None:
            attn_mask = attn_mask.to(device)
        self.window_mask = attn_mask
        # import pdb; pdb.set_trace()
        return attn_mask

    def atten(self, qkv, mask=None, window_mask=None):
        """
        qkv: 3 x B x H x N x C
        mask: B x N
        window_mask: nw x N x N
        """
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) # B x H x N x C
        b, h, n, c = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale # B x H x N x N

        if window_mask is not None:
            nw = window_mask.shape[0]
            attn = attn.view(b//nw, nw, h, n, n) + window_mask.to(attn).unsqueeze(dim=1).unsqueeze(0)
            attn = attn.view(b, h, n, n)
        if mask is not None:
            mask = mask.view(b, 1, 1, n)
            attn = attn.masked_fill(~mask, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # B x H x N x N
        x = attn @ v # B x H x N x C
        return x, attn


    def forward(self, x, mask=None):
        """
        x: B x H x W x C
        mask: None or B x H x W
        """
        B, H, W, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)) # 3*C
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias) # B x H x W x 3*C

        if self.window_size is not None and self.window_size != (H, W):
            assert self.shift_size >= (0,0) and self.shift_size < (H, W), 'shift size should be in range (0,0)-(H,W)'
            if self.shift_size > (0,0):
                qkv = torch.roll(qkv, shifts=self.shift_size, dims=(1,2))
                if mask is not None:
                    mask = torch.roll(mask, shifts=self.shift_size, dims=(1,2))
            
            pt_list = self.partition_type.split(',')
            qkv_list = torch.chunk(qkv, chunks=len(pt_list), dim=-1) # B x H x W x 3*C/pt
            x = []
            for i in range(len(pt_list)):
                pt_ = pt_list[i]
                qkv_ = qkv_list[i]
                qkv_ = window_partition(qkv_, window_size=self.window_size, partition_type=pt_) # B*nW x wH x wW x 3*C/pt
                if mask is not None:
                    mask_ = window_partition(mask.unsqueeze(dim=-1), window_size=self.window_size, partition_type=pt_).squeeze(dim=-1) # B*nW x wH x wW
                else:
                    mask_ = None
                b, h, w, _ = qkv_.shape
                qkv_ = qkv_.reshape(b, h*w, 3, self.num_heads//len(pt_list), C // self.num_heads).permute(2, 0, 3, 1, 4) # b x hw x 3 x Head/2 x C/Head -> 3 x b x Head/pt x hw x C/Head
                if self.apply_window_mask and pt_ == 'partition':
                    if self.window_mask is None:
                        self.set_window_attn_mask(resolution=(H,W), partition_type=pt_, device=qkv_.device)
                    window_mask = self.window_mask
                else:
                    window_mask = None
                x_, attn_, = self.atten(qkv_, mask=mask_, window_mask=window_mask) # b x H/pt x hw x C/H, b x H/pt x hw x hw
                            
                x_ = x_.permute(0, 2, 1, 3).contiguous().view(b, h, w, C//len(pt_list)) # b x Head/pt x hw x C/Head -> b x hw x Head/pt x C/Head -> b x h x w x C/pt
                x_ = window_unpartition(x_, window_size=self.window_size, H=H, W=W, partition_type=pt_) # B x H x W x C/pt

                x.append(x_) # B x H x W x C
            x = torch.cat(x, dim=-1)
        
            if self.shift_size > (0,0):
                x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1,2))

            attn = None
        else:
            qkv = qkv.reshape(B, H*W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # B x H x W x 3C -> B x HW x 3 x H x C/Head -> 3 x B x Head x HW x C/Head
            x, attn = self.atten(qkv, mask=mask) # B x Head x HW x C/Head, B x Head x HW x HW
            x = x.permute(0, 2, 1, 3).contiguous().view(B, H, W, C) # B x Head x HW x C/Head -> B x HW x Head x C/Head -> B x H x W x C
            attn = attn.mean(1, keepdim=False).view(B, H, W, H, W)
        
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


    @staticmethod
    def count_flops(m, x, y):
        from thop.vision import basic_hooks, calc_func
        m.total_params = m.qkv.total_params + m.proj.total_params
        if m.q_bias is not None:
            m.total_params += m.q_bias.numel()
            m.total_params += m.v_bias.numel()

        total_ops = 0

        B, H, W, C = x[0].shape
        # qkv linear
        total_ops += calc_func.calculate_linear(m.qkv.in_features, B*H*W*m.qkv.out_features)

        if m.window_size is not None and m.window_size != (H, W):
            raise NotImplementedError
        else:
            # atten
            head_dim = m.qkv.out_features // 3 // m.num_heads
            total_ops += (B*m.num_heads*H*W*H*W*head_dim) # bmm, matrix multiply
            total_ops += (B*m.num_heads*H*W*H*W) # x scale
            total_ops += calc_func.calculate_softmax(B*m.num_heads*H*W, H*W) # softmax
            total_ops += (B*m.num_heads*H*W*C*H*W)# bmm, matrix multiply
        
        total_ops += m.proj.total_ops
        m.total_ops += total_ops
        

class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)
    
    @staticmethod
    def count_flops(m, x, y):
        m.total_ops += torch.DoubleTensor([2 * y.numel()])



class Block(nn.Module):

    def __init__(self, 
                dim, 
                num_heads, 
                mlp_ratio=4., 
                qkv_bias=False, 
                attn_drop=0.,
                drop_path=0., 
                drop=0.0,
                act_layer='GELU', 
                norm_layer=nn.LayerNorm,
                window_size=None, # the window size to perform attention, similar to swin transformer
                shift_size=(0,0), # the size to cycle shift the input feature, similar to swin transformer, only effective when window size is not None
                partition_type='shuffle',
                apply_window_mask=False,
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              window_size=window_size, shift_size=shift_size, partition_type=partition_type, apply_window_mask=apply_window_mask)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer == 'GELU':
            act_layer = nn.GELU 
        elif act_layer == 'GELU2':
            act_layer = GELU2
        else:
            raise NotImplementedError('activation layer {} not implemented!'.format(act_layer))
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x_, attn = self.attn(self.norm1(x), mask=mask) # B x H x W x C, B x H x W x H x W
        x = x + self.drop_path(x_)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class MaskedImageInpaintingTransformer(nn.Module):
    def __init__(
        self,
        *,
        content_seq_len, # length of content sequences
        embd_pdrop=0., # embedding dropout prob
        
        n_layer, # number of layers in transformer
        dim, # the embed dim
        num_heads, # the number of heads
        attn_drop=0.0, # attention dropout prob
        drop_path=0.0, # drop path prob
        act_layer='GELU', # the activation layer in MLP of transformer block
        mlp_ratio=4, # the times of hidden dimension in the MLP of attetntion block
        qkv_bias=True, # the bias for qkv in attention block

        attn_window_size=[None],
        attn_shift_size=[(0,0)],
        attn_partition_type=['shuffle'],
        attn_window_mask=False,

        attn_content_with_mask=False,
        content_codec_config=None,

        content_ignore_token=-100,

        input_feature_type='origin',

        learn_mask_emb=False,
        mask_pixel_value=None,

        init_type='beit', # how to initialize the weight

        # args for training
        weight_decay=0.01,
        random_quantize=0.2, # random quantize the feature, only when the input feature is not quantized
        num_token=None, # i.e the numbr of classes
        content_patch_token_shape=[1, 1], # the shape of tokens in each patch, h, w
        ckpt_path=None, # The pretrained model to load 
        loss_config=None,
        loss_mask_type='binary', # binary: the patch with any pixel missed, other use the unmasked ratio as the mask
    ):
        super().__init__()
        
        content_patch_token_shape = tuple(int(ts) for ts in content_patch_token_shape)
        content_patch_seq_len = int(content_patch_token_shape[0] * content_patch_token_shape[1])
        assert dim % content_patch_seq_len == 0, 'The number of dimmension should be divisible by the number of tokens '
        
        # embeddings for content
        self.content_codec = instantiate_from_config(content_codec_config)
        self.emb_proj = nn.Linear(self.content_codec.embed_dim, dim//content_patch_seq_len)
        self.pos_emb = nn.Parameter(torch.zeros(1, content_seq_len, dim))
        if learn_mask_emb:
            self.mask_emb = nn.Parameter(torch.zeros(1, dim//content_patch_seq_len, 1, 1))
        else:
            self.mask_emb = None
        
        # drop for embedding
        if embd_pdrop > 0:
            self.drop = nn.Dropout(embd_pdrop)
        else:
            self.drop = None
                             
        # transformer
        attn_window_size = (math.ceil(float(n_layer)/len(attn_window_size)) * attn_window_size)[:n_layer]
        attn_shift_size = (math.ceil(float(n_layer)/len(attn_shift_size)) * attn_shift_size)[:n_layer]
        attn_partition_type = (math.ceil(float(n_layer)/len(attn_partition_type)) * attn_partition_type)[:n_layer]
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_layer)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[Block(
                dim=dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                attn_drop=attn_drop,
                drop_path=dpr[n], 
                act_layer=act_layer, 
                window_size=attn_window_size[n],
                shift_size=attn_shift_size[n],
                partition_type=attn_partition_type[n],
                apply_window_mask=attn_window_mask,
        ) for n in range(n_layer)])

        # final prediction head
        self.norm = nn.LayerNorm(dim)
        self.num_cls = self.content_codec.get_number_of_tokens() if num_token is None else num_token
        self.to_logits = nn.Linear(dim//content_patch_seq_len, self.num_cls)
        
        self.dim = dim
        self.attn_content_with_mask = attn_content_with_mask
        self.content_seq_len = content_seq_len
        self.content_patch_seq_len = content_patch_seq_len
        self.content_patch_token_shape = content_patch_token_shape
        self.content_ignore_token = content_ignore_token
        self.input_feature_type = input_feature_type
        self.weight_decay = weight_decay
        self.random_quantize = random_quantize
        self.mask_pixel_value = mask_pixel_value

        if self.content_patch_token_shape != (1, 1):
            assert not self.attn_content_with_mask, 'If there are more than one tokens in each embedding, the attn should not be controlled by the mask!'

        self.init_type = init_type
        self.apply(self._init_weights)
        self.fix_init_weight()
        if ckpt_path is not None:
            self.init_from_ckpt(path=ckpt_path)

        # reinitialize the codec, so that the pretrained model can be reloaded
        self.content_codec = instantiate_from_config(content_codec_config)

        self.loss_func = instantiate_from_config(loss_config)
        self.loss_mask_type = loss_mask_type

    def fix_init_weight(self):
        if self.init_type == 'beit':
            def rescale(param, layer_id):
                param.div_(math.sqrt(2.0 * layer_id))

            for layer_id, layer in enumerate(self.blocks):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)
        elif self.init_type == 'mae':
            trunc_normal_(self.pos_emb)
        else:
            raise NotImplementedError('init type: {} not implemented!'.format(self.init_type))


    def _init_weights(self, module):
        if self.init_type == 'beit':
            if isinstance(module, (nn.Linear, nn.Embedding)):
                trunc_normal_(module.weight, std=0.02)
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)       
            elif isinstance(module, nn.Conv2d):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        elif self.init_type == 'mae':
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        else:
            raise NotImplementedError('init type: {} not implemented!'.format(self.init_type))



    def init_from_ckpt(self, path, ignore_keys=['content_codec.']):
        sd = torch.load(path, map_location="cpu")
        if 'model' in sd:
            sd = sd['model']
        else:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("UQ-Transformer: Deleting key {} from the given state_dict.".format(k))
                    del sd[k]
        
        # interpolate positional embedding
        if tuple(sd['pos_emb'].shape) != tuple(self.pos_emb.shape):
            content_shape = (int(self.pos_emb.shape[1] ** 0.5), int(self.pos_emb.shape[1] ** 0.5))
            real_dim = self.pos_emb.shape[-1] // self.content_patch_seq_len

            provide_content_shape = (int(sd['pos_emb'].shape[1] ** 0.5), int(sd['pos_emb'].shape[1] ** 0.5))
            provide_content_patch_seq_len = sd['pos_emb'].shape[-1]//real_dim
            provide_content_token_patch_shape = (int(provide_content_patch_seq_len**0.5), int(provide_content_patch_seq_len**0.5))
            real_provide_content_shape = (provide_content_shape[0]*provide_content_token_patch_shape[0], provide_content_shape[1]*provide_content_token_patch_shape[1])
            pos_emb = sd['pos_emb'].permute(0, 2, 1).view(1, -1, provide_content_shape[0], provide_content_shape[1]) # 1 x H/cps*W/cps x D -> 1 x D x H/cps*W/cps -> 1 x D x H/cps x W/cps
            pos_emb = pixel_shuffle(pos_emb, out_size=real_provide_content_shape, chunked=True) # 1 x C x H x W

            real_content_shape = (content_shape[0]*self.content_patch_token_shape[0], content_shape[1]*self.content_patch_token_shape[1])
            pos_emb = F.interpolate(pos_emb, size=real_content_shape, mode='bilinear')
            
            pos_emb = pixel_unshuffle(pos_emb, out_size=content_shape, chunked=True) # 1 x D x H/cps x W/cps
            pos_emb = pos_emb.view(1, self.pos_emb.shape[-1], self.pos_emb.shape[1]).permute(0, 2, 1)
            sd['pos_emb'] = pos_emb
        
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print("UQ-Transformer: Load pretrained model from {}".format(path))
        print('UQ-Transformer: Missing keys in created model:\n', missing)
        print('UQ-Transformer: Unexpected keys in state dict:\n', unexpected)

    @property
    def device(self):
        return self.to_logits.weight.device

    @torch.no_grad()
    def generate_content(
        self,
        batch,
        filter_ratio = 0.5,
        filter_type = 'count',
        temperature = 1.0,
        replicate=1,
        mask_low_to_high=False,
        num_token_per_iter=1,
        calculate_acc_and_prob=True,
        accumulate_time=None,
        raster_order=False,
        **kwargs,
    ):
        self.eval()
        if replicate != 1:
            for k in batch.keys():
                if batch[k] is not None and torch.is_tensor(batch[k]):
                    batch[k] = torch.cat([batch[k] for _ in range(replicate)], dim=0)
        return self.sample(
            batch=batch,
            filter_ratio=filter_ratio,
            filter_type=filter_type,
            temperature=temperature,
            return_gt=False,
            return_mask_gt=False,
            return_reconstruction=False,
            mask_low_to_high=mask_low_to_high,
            num_token_per_iter=num_token_per_iter,
            calculate_acc_and_prob=calculate_acc_and_prob,
            accumulate_time=accumulate_time,
            raster_order=raster_order,

        )           


    @torch.no_grad()
    def sample(
        self,
        *,
        batch,
        filter_ratio = 0.8,
        filter_type='count',
        temperature = 1.0,
        return_gt=True,
        return_mask_gt=True,
        return_reconstruction=True,
        calculate_acc_and_prob=True, # calculate token accuracy
        mask_low_to_high=False,
        num_token_per_iter=None,
        accumulate_time=None, # for get the time consumption
        raster_order=False,
        **kwargs,
    ): 
        self.eval()

        for k in batch.keys():
            if torch.is_tensor(batch[k]):# isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device)

        if mask_low_to_high:
            low_res = self.content_codec.token_shape
            ori_res = batch['mask'].shape[-2:]
            assert low_res is not None 
            # import pdb; pdb.set_trace()
            mask_ = F.interpolate(batch['mask'].float(), size=low_res, mode='nearest')
            mask_ = F.interpolate(mask_, size=ori_res, mode='nearest').bool()     
        else:
            mask_ = batch['mask']

            # batch['mask'] = mask_.clone()
        if accumulate_time is None:
            accumulate_time = {
                'encoder': 0,
                'prepare': 0,
                'transformer': 0,
                'decoder': 0,
                'count': 0,
            }

        tic = time.time()
        data_mask = self.content_codec.get_features(batch['image'], 
                                                    mask=mask_, 
                                                    return_quantize_feature=self.input_feature_type == 'quantized',
                                                    return_token=True, mask_pixel_value=self.mask_pixel_value) # dict
        accumulate_time['encoder'] = (accumulate_time['encoder'] * accumulate_time['count'] + time.time() - tic) / (accumulate_time['count']+1)

        tic = time.time()
        if self.input_feature_type == 'origin':
            feat_mask = data_mask['feature'] # B x C x H x W
        elif self.input_feature_type == 'quantized':
            feat_mask = data_mask['feature_quantize'] # B x C x H x W
        else:
            raise NotImplementedError('inpute feature type {} not implemented!'.format(self.input_feature_type))
        b, _, h, w = feat_mask.shape

        token_type, unmask_ratio = get_token_type(mask_, type='pixel_shuffle', token_shape=[h,w]) # B x 1 x H x W
        feat_mask = self.emb_proj(feat_mask.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # B x C x H x W
        
        content_shape = (h // self.content_patch_token_shape[0], w // self.content_patch_token_shape[1])
        if self.pos_emb is not None:
            pos_emb = self.pos_emb.permute(0, 2, 1).view(1, -1, content_shape[0], content_shape[1]) # B x D x H/cps x W/cps
            pos_emb = pixel_shuffle(pos_emb, out_size=(h, w), chunked=True) # B x C x H x W
        else:
            pos_emb = torch.zeros_like(feat_mask)
        
        if self.mask_emb is not None:
            feat_mask = feat_mask * unmask_ratio + self.mask_emb * (1-unmask_ratio)

        feat_mask = feat_mask + pos_emb 
        
        # save features before feeding into transformer block
        if False:
            # import pdb; pdb.set_trace()
            feat_save_path = os.path.join('RESULT/debug/feature_pos_mask/{}_feature.pt'.format(batch['relative_path'][0].replace('.png','')))
            token_type_save_path = os.path.join(os.path.dirname(feat_save_path), '{}_unmask_ratio.pt'.format(batch['relative_path'][0].replace('.png','')))
            os.makedirs(os.path.dirname(feat_save_path), exist_ok=True)
            torch.save(feat_mask.to('cpu'), feat_save_path)
            torch.save(unmask_ratio, token_type_save_path)

            # import sys
            # sys.exit(1)

        
        
        content_feat = feat_mask # B x C x H x W
        content_token = data_mask['token'] # B x H x W
        content_mask = (token_type == 1).squeeze(dim=1) # B x H x W

        accumulate_time['prepare'] = (accumulate_time['prepare'] * accumulate_time['count'] + time.time() - tic) / (accumulate_time['count']+1)


        # begin to sample
        # import pdb; pdb.set_trace()
        step = 0
        tic = time.time()
        forward_time = 0
        sample_time = time.time()
        save_each_step_image = False

        if save_each_step_image:

            cache_decoder_requires_image = self.content_codec.decoder.requires_image
            cache_decoder_uo_layer_with_image = self.content_codec.decoder.up_layer_with_image


            patch_size = batch['image'].shape[-1] // w

            content_token.masked_fill_(~content_mask, 285) # set all invalid token to a fixed token for better comparison
            completed_iter = self.content_codec.decode(content_token, mask_im=batch['image'] * batch['mask'], mask=batch['mask'], token_shape=[h,w]) # B x C x H x W
            completed_iter = completed_iter[0].permute(1,2,0).to('cpu').numpy().astype(np.uint8)
            completed_iter = Image.fromarray(completed_iter)
            # save 
            save_path = os.path.join('RESULT/debug', batch['relative_path'][0], 'completed_{}.png'.format(str(step).zfill(len(str(h*w)))))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            completed_iter.save(save_path)
            print('saved to {}'.format(save_path))


        if calculate_acc_and_prob:
            content_token_target = self.content_codec.get_features(batch['image'], 
                                                    return_quantize_feature=False,
                                                    return_token=True)['token'] # B x H x W
            acc_all = []
            prob_all = []


        if num_token_per_iter is None:
            num_token_per_iter = max(self.content_patch_seq_len, int(h*w//500))

        num_masked_tokens = (~content_mask).flatten(1).sum(-1).max() # in a batch, we get the max number of the masked tokens for iteratively sampling
        # import pdb; pdb.set_trace()
        if isinstance(num_token_per_iter, (int,)):
            if num_token_per_iter > 0:
                total_steps = (num_masked_tokens + num_token_per_iter - 1) // num_token_per_iter
                step_nums = [num_token_per_iter for _ in range(total_steps)]
                diff = int(sum(step_nums) - num_masked_tokens)
                step_nums[-1] -= diff 
            else:
                total_steps = 1
                step_nums = [-1]
        elif isinstance(num_token_per_iter, str) and num_token_per_iter.split('_')[0] in ['cosine', 'linear', 'average',  'cosine-1', 'linear-1']:
            total_steps = int(num_token_per_iter.split('_')[-1])
            if num_token_per_iter.split('_')[0] in ['cosine', 'cosine-1']:
                step_nums = list(range(total_steps))
                step_nums = [math.cos(math.pi * sn/total_steps) + 1 for sn in step_nums]
            elif num_token_per_iter.split('_')[0] in ['linear', 'linear-1']:
                step_nums = list(range(total_steps, 0, -1))
            elif num_token_per_iter.split('_')[0] == 'average':
                step_nums = [num_masked_tokens // total_steps for _ in range(total_steps)]
            else:
                raise NotImplementedError('{}'.format(num_token_per_iter))
            # import pdb; pdb.set_trace()
            factor = float(num_masked_tokens) / sum(step_nums)
            step_nums = [round(sn * factor) for sn in step_nums]
            diff = int(sum(step_nums) - num_masked_tokens)
            idx = 0
            while diff != 0:
                idx = (idx + 1) % len(step_nums)
                factor = 1 if diff < 0 else -1
                diff = diff + 1 if diff < 0 else diff -1
                step_nums[idx] = step_nums[idx] + factor
            
            if num_token_per_iter.split('_')[0] in ['cosine-1', 'linear-1']:
                step_nums = step_nums[::-1]
        else:
            raise NotImplementedError('{}'.format(num_token_per_iter))


        while step < total_steps:
        # while content_mask.sum() < content_mask.numel():
            
            sn = step_nums[step]

            step += 1
            step_ = step 
            # import pdb; pdb.set_trace()
            if step % 100 == 0:
                print('Step: {}/{}'.format(step, total_steps), content_mask.sum(), content_mask.numel())

            # prepare data for transformer blocks to get the logits
            emb = pixel_unshuffle(content_feat, out_size=content_shape, chunked=True) # B x D x H/cps x W/cps
            emb = emb.permute(0, 2, 3, 1) # B x H/cps x W/cps x D
            if self.drop is not None:
                emb = self.drop(emb)
            if self.attn_content_with_mask:
                attn_mask = pixel_unshuffle(content_mask.unsqueeze(dim=1), out_size=content_shape, chunked=True) # B x cps*cps x H/cps x W/cps
                attn_mask = (attn_mask.sum(dim=1, keepdim=False) == attn_mask.shape[1]).to(content_mask.dtype) # B x H/cps x W/cps
            else:
                attn_mask = None
            
            tic_forward = time.time()

            for block_idx in range(len(self.blocks)):   
                emb, att_w = self.blocks[block_idx](emb, mask=attn_mask) # B x H x W x D, B x H x W x H x W
            emb = self.norm(emb) # B x H/cps x W/cps x D
            emb = emb.permute(0, 3, 1, 2) # B x D x H/cps x W/cps
            emb = pixel_shuffle(emb, out_size=(h, w), chunked=True) # B x C x H x W
            emb = emb.permute(0, 2, 3, 1) # B x H x W x C
            logits = self.to_logits(emb) # B x  H x W x Cls

            forward_time += time.time() - tic_forward

            # for each position, only keep the topk probabilities
            logits_filter = logits_top_k(logits, filter_ratio=filter_ratio, minimum=1, filter_type=filter_type) # B x H x W x Cls
            probs = F.softmax(logits_filter * temperature, dim=-1) # B x H x W x Cls
            sample = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(*probs.shape[:3]) # B x H x W
            
            if sn == -1 or sn >= h*w:
                content_token = content_token * content_mask + sample * (~content_mask)
                pos_mask = ~content_mask
                content_mask = torch.ones_like(content_mask)
            else:
                # select the sn positions for sampling
                if raster_order:
                    index_raster = torch.tensor(list(range(h*w))).view(1, h, w).int().to(content_mask.device) # B x H x W
                    index_raster = index_raster + content_mask.int() * (h*w+1)
                    index_raster = 0 - index_raster.view(-1, h*w)
                    _, pos = torch.topk(index_raster, dim=1, k=sn) # B x num, in range [0, HW)
                    pos_mask = torch.zeros_like(index_raster).float().scatter_(1, pos, 1.0).to(content_mask.dtype) # B x HW
                else:
                    logits_max, _ = logits_filter.max(dim=-1) # B x H x W
                    logits_max.masked_fill_(content_mask, float('-inf')) # set the logits for those unmasked tokens to -inf
                    logits_max = logits_max.view(-1, h*w) # B x HW
                    _, pos = torch.topk(logits_max, dim=1, k=sn) # B x num, in range [0, HW)
                    pos_mask = torch.zeros_like(logits_max).scatter_(1, pos, 1.0).to(content_mask.dtype) # B x HW
                pos_mask = pos_mask.view(-1, h, w) # B x H x W
                pos_mask.masked_fill_(content_mask, False) # B x H x W
                
                # import pdb; pdb.set_trace()

                # update token and mask
                content_token = content_token * (~pos_mask) + sample * pos_mask
                content_mask = content_mask + pos_mask
                
                # update featuer
                # import pdb; pdb.set_trace()
                sample_feat = self.content_codec.get_codebook_entry_with_token(sample)['feature'] # B x C' x H x W
                sample_feat = self.emb_proj(sample_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # B x C x H x W
                sample_feat = sample_feat + pos_emb
                content_feat = content_feat * (~pos_mask.unsqueeze(dim=1)) + sample_feat * pos_mask.unsqueeze(dim=1)

            if calculate_acc_and_prob:
                pre_token = torch.argmax(logits, dim=-1, keepdim=False) # B x H x W
                acc = (pre_token == content_token_target).to(logits) # B x H x W
                acc = acc[pos_mask].tolist()
                acc_all += acc

                prob = logits.softmax(dim=-1) # B x H x W x Cls
                target_one_hot = F.one_hot(content_token_target, num_classes=prob.shape[-1])
                prob, _ = torch.max(prob * target_one_hot, dim=-1, keepdim=False) # B x H x W
                prob = prob[pos_mask].tolist()
                prob_all += prob
            
            if save_each_step_image:
                # import pdb; pdb.set_trace()
                self.content_codec.decoder.requires_image = False 
                self.content_codec.decoder.up_layer_with_image = False
                completed_iter = self.content_codec.decode(content_token, combine_rec_and_gt=False, token_shape=[h,w]) # B x C x H x W
                completed_iter = completed_iter[0].permute(1,2,0).to('cpu').numpy().astype(np.uint8)
                completed_iter = Image.fromarray(completed_iter)
                # save 
                _, index = torch.topk(pos_mask[0].long().view(-1), k=pos_mask[0].sum()) # HW 
                index = index.to('cpu').tolist()

                # plot the patch
                # import pdb; pdb.set_trace()
                token_count = int(content_mask[0].sum())
                if token_count == h*w: # finished
                    save_path = os.path.join('RESULT/debug', batch['relative_path'][0], 'completed_{}_{}_debug.png'.format(str(step).zfill(len(str(h*w))), token_count))
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    completed_iter.save(save_path)   

                im_draw = ImageDraw.ImageDraw(completed_iter)
                for idx in index:
                    r = idx // w
                    c = idx % w
                    y1 = r * patch_size 
                    x1 = c * patch_size
                    im_draw.rectangle(((x1, y1),(x1+patch_size, y1+patch_size)), fill=None, outline='yellow', width=2)
                                    
                save_path = os.path.join('RESULT/debug', batch['relative_path'][0], 'completed_{}_{}_{}.png'.format(str(step).zfill(len(str(h*w))), token_count, str(index)))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                completed_iter.save(save_path)
                print('saved to {}'.format(save_path))

        accumulate_time['transformer'] = (accumulate_time['transformer'] * accumulate_time['count'] + time.time() - tic) / (accumulate_time['count']+1)
        # print('Time consumption: forward {}s/iter, sample {}s/token, sample {}s/img'.format(forward_time/len(step_nums), (time.time()-sample_time)/len(step_nums), time.time()-sample_time))
        assert content_mask.sum() == content_mask.numel(), "Unfinised: {} tokens are not predicted!".format(content_mask.numel() - content_mask.sum())

        if save_each_step_image:
            self.content_codec.decoder.requires_image = cache_decoder_requires_image
            self.content_codec.decoder.up_layer_with_image = cache_decoder_uo_layer_with_image


        # decode
        tic = time.time()
        masked_im = batch['image'] * batch['mask']
        sampled_im = self.content_codec.decode(content_token, mask_im=masked_im, mask=batch['mask'], token_shape=[h,w])
        
        accumulate_time['decoder'] = (accumulate_time['decoder'] * accumulate_time['count'] + time.time() - tic) / (accumulate_time['count']+1)
        accumulate_time['total'] = accumulate_time['encoder'] + accumulate_time['prepare'] + accumulate_time['transformer'] + accumulate_time['decoder']
        accumulate_time['count'] = accumulate_time['count'] + 1

        output = {
            'completed': sampled_im,
            'mask': batch['mask'],
            'masked_gt': masked_im,
            'accumulate_time': accumulate_time
        }
        if return_gt:
            output['input'] = batch['image']
        if return_mask_gt:
            output['mask_input'] = masked_im
        if return_reconstruction:
            token = self.content_codec.get_tokens(batch['image'], mask=batch['mask'])
            output['reconstruction'] = self.content_codec.decode(token['token'], token_shape=[h,w])
        if calculate_acc_and_prob:
            output['acc'] = torch.FloatTensor(acc_all).to(masked_im).mean()
            output['prob'] = torch.FloatTensor(prob_all).to(masked_im).mean()

        if save_each_step_image:      
            completed = output['completed'][0].permute(1,2,0).to('cpu').numpy().astype(np.uint8)
            completed = Image.fromarray(completed)
            save_path = os.path.join('RESULT/debug', batch['relative_path'][0], 'completed_{}_{}.png'.format(str(step).zfill(len(str(h*w))), token_count))
            completed.save(save_path)

            mask = mask_[0][0].to('cpu').numpy().astype(np.uint8)
            mask = Image.fromarray(mask * 255)
            save_path = os.path.join('RESULT/debug', batch['relative_path'][0], 'mask_{}_{}.png'.format(str(step).zfill(len(str(h*w))), token_count))
            mask.save(save_path)


            # merge
            merge = output['completed'] * (1 - batch['mask'].float()) + batch['image'] * batch['mask'].float()
            merge = merge[0].permute(1,2,0).to('cpu').numpy().astype(np.uint8)
            merge = Image.fromarray(merge)
            save_path = os.path.join('RESULT/debug', batch['relative_path'][0], 'completed_merge_{}_{}.png'.format(str(step).zfill(len(str(h*w))), token_count))
            merge.save(save_path)
            
            self.train()
            return None

        self.train()
        return output
    
    def parameters(self, recurse=True, name=None):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # return super().parameters(recurse=True)
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            print("Transformer: get parameters by the overwrite method!")
            if self.init_type == 'beit':
                decay = set()
                no_decay = set()
                whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d) #TODO(torch.nn.Linear, )
                blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
                for mn, m in self.named_modules():
                    for pn, p in m.named_parameters():
                        if not p.requires_grad:
                            continue 

                        fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                        if pn.endswith('bias'):
                            # all biases will not be decayed
                            no_decay.add(fpn)
                        elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                            # weights of whitelist modules will be weight decayed
                            decay.add(fpn)
                        elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                            # weights of blacklist modules will NOT be weight decayed
                            no_decay.add(fpn)
                no_decay.add('pos_emb')
                if self.mask_emb is not None:
                    no_decay.add('mask_emb')
            elif self.init_type == 'mae':
                no_decay = set(['pos_emb'])
                if self.mask_emb is not None:
                    no_decay.add('mask_emb')
                decay = set()
                for mn, m in self.named_modules():
                    for pn, p in m.named_parameters():
                        if not p.requires_grad:
                            continue 
                        fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                        if fpn in no_decay:
                            continue
                        decay.add(fpn)
            else:
                raise NotImplementedError('init type: {} not implemented!'.format(self.init_type))

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad} 
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            return optim_groups

    def prepare_data(self, image, mask):
        """
        Get the feature from image

        Args:
            image: B x 3 x H x W
            mask: B x 1 x H x W
        """
        data_mask = self.content_codec.get_features(
                image, 
                mask=mask, 
                return_quantize_feature=True,
                return_token=False,
                mask_pixel_value=self.mask_pixel_value)

        if self.input_feature_type == 'origin':
            feat_mask = data_mask['feature'] # B x C x H x W
            b, _, h, w = feat_mask.shape
            # random change origin feature with quantized feature
            token_type, unmask_ratio = get_token_type(mask, type='pixel_shuffle', token_shape=[h,w]) # B x 1 x H/cps x W/cps
            valid_token_mask = token_type == 1
            if self.random_quantize > 0:
                quantize_mask = torch.rand(*token_type.shape).to(token_type.device) < self.random_quantize # B x 1 x H x W, in range [0, 1)
                # only quantize those unmasked tokens
                quantize_mask = (valid_token_mask * quantize_mask).to(feat_mask.dtype) # 1 denotes to be quantized
                feat_mask = feat_mask * (1-quantize_mask) +  (data_mask['feature_quantize'] * quantize_mask) # B x C x H x W
        elif self.input_feature_type == 'quantized':
            feat_mask = data_mask['feature_quantize'] # B x C x H x W
            b, _, h, w = feat_mask.shape
            token_type, unmask_ratio = get_token_type(mask, type='pixel_shuffle', token_shape=[h,w]) # B x 1 x H x W
            valid_token_mask = token_type == 1
        else:
            raise NotImplementedError('input feature type {} is not impleted!'.format(self.input_feature_type))

        # import pdb; pdb.set_trace()
        feat_mask = self.emb_proj(feat_mask.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # B x C x H x W
        if self.mask_emb is not None:
            # import pdb; pdb.set_trace()
            feat_mask = feat_mask * unmask_ratio + self.mask_emb * (1-unmask_ratio)
        


        






        # add position embedding
        content_shape = (h // self.content_patch_token_shape[0], w // self.content_patch_token_shape[1])
        if self.pos_emb is not None:
            pos_emb = self.pos_emb.permute(0, 2, 1).view(1, -1, content_shape[0], content_shape[1]) # B x D x H/cps x W/cps
            pos_emb = pixel_shuffle(pos_emb, out_size=(h, w), chunked=True) # B x C x H x W
            feat_mask = feat_mask + pos_emb 
        
        if self.content_patch_token_shape != (1, 1):
            feat_mask = pixel_unshuffle(feat_mask, out_size=content_shape, chunked=True) # B x D x H/cps x W/cps
            valid_token_mask = get_token_type(mask, type='pixel_shuffle', token_shape=content_shape)[0] == 1 # B x 1 x H/cps x W/cps
        
        # reshape the data
        feat_mask = feat_mask.permute(0, 2, 3, 1).contiguous() # B x H/cps x W/cps x D
        valid_token_mask = valid_token_mask.squeeze(dim=1).contiguous() # B x H/cps x W/cps
        token_type = token_type.squeeze(dim=1).contiguous() # B x H x W
        unmask_ratio = unmask_ratio.squeeze(dim=1).contiguous()  # B x H/cps x W/cps

        # prepare target
        data_target = self.content_codec.get_features(image, return_token=True, return_distance=True)

        output = {
            'token_target': data_target['token'].contiguous(), # B x H x W
            'token_type': token_type.contiguous(), # B x H x W
            'unmask_ratio': unmask_ratio,  # B x H x W

            'token_mask': valid_token_mask.contiguous(), # B x H/cps x W/cps
            'embedding': feat_mask # B x H/cps x W/cps x D
        }
        
        if isinstance(self.loss_func, LabelSmoothingLoss):
            output['token_distance'] = data_target['distance'][:,:,:,:self.num_cls].contiguous() # B x H x W x Cls
            
        return output


    def forward(
            self, 
            batch, 
            return_loss=False, 
            return_logits=True, 
            return_att_weight=False,
            **kwargs):

        # 1) get data from input data
        if batch.get('count_flops', False):
            data = batch
        else:
            data = self.prepare_data(batch['image'], mask=batch['mask'])
        emb = data['embedding']

        # 2) forward in transformer
        if self.drop is not None:
            emb = self.drop(emb)
        if self.attn_content_with_mask:
            attn_mask = data['token_mask']
        else:
            attn_mask = None
        for block_idx in range(len(self.blocks)):   
            emb, att_weight = self.blocks[block_idx](emb, mask=attn_mask) # B x H/cps x W/cps x D, B x H/cps x W/cps x H/cps x W/cps
        
        # 3) get logits
        emb = self.norm(emb)
        if self.content_patch_token_shape != (1, 1):
            content_shape = (data['token_target'].shape[-2], data['token_target'].shape[-1])
            emb = emb.permute(0, 3, 1, 2) # B x D x H/cps*W/cps
            emb = pixel_shuffle(emb, out_size=content_shape, chunked=True) # B x D/cps^2 x H x W
            emb = emb.permute(0, 2, 3, 1) # B x H x W x D/cps^2
            logits = self.to_logits(emb) # B x H x W x n
        else:
            logits = self.to_logits(emb) # B x H x W x n
        # import pdb; pdb.set_trace()

        # 4) get output, especially loss
        out = {}

        if return_logits:
            out['logits'] = logits
        if return_att_weight:
            out['attention_weight'] = att_weight

        if return_loss:
            token_target = data['token_target'] # B x H x W
            # print(token_target.max())
            token_type = data['token_type'] # B x H x W
            # import pdb; pdb.set_trace()

            if self.loss_func is None:
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), token_target.view(-1), ignore_index=self.content_ignore_token, reduction='none')
                loss = loss.view(token_type.shape)
                loss_out = {'loss': loss}
            elif isinstance(self.loss_func, PolyLoss):
                loss_out = self.loss_func(logits=logits.permute(0, 3, 1, 2), labels=token_target, mask=None, reduction='none')
            elif isinstance(self.loss_func, LabelSmoothingLoss):
                loss_out = self.loss_func(logits=logits.permute(0, 3, 1, 2), labels=data['token_distance'].permute(0,3,1,2), mask=None, reduction='none')
            else:
                raise NotImplementedError

            # get the predicted probabilities
            prob = F.softmax(logits, dim=-1)# B x H x W x n
            gt = F.one_hot(token_target, num_classes=prob.shape[-1]) # B x H x W x n
            gt_prob = (prob * gt).sum(-1) # B x H x W
            loss_out['pred_prob'] = gt_prob.detach()

            # get prediction accuracy
            pred = torch.argmax(logits, dim=-1) # B x H x W
            right_or_wrong = pred == token_target # B x H x W
            loss_out['pred_acc'] = right_or_wrong.detach()

            if self.loss_mask_type == 'binary':
                loss_mask_overall = token_type!=1
            elif self.loss_mask_type == 'mask_ratio':
                loss_mask_overall = 1 - data['unmask_ratio']
            else:
                raise NotImplementedError

            loss_mask = {
                'placeholder': loss_mask_overall, # B x H x W
                'partial': token_type == 2, # partially masked
                'fully': token_type == 0, # fully masked
            }

            for loss_k in loss_out:
                for mask_k in loss_mask:
                    if mask_k != 'placeholder':
                        out_k = mask_k + '_' + loss_k
                    else:
                        out_k = loss_k
                    out[out_k] = (loss_out[loss_k] * loss_mask[mask_k]).sum() / (loss_mask[mask_k].sum() + 1e-18)
        return out
