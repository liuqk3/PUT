import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
from image_synthesis.utils.misc import instantiate_from_config
from image_synthesis.modeling.codecs.base_codec import BaseCodec
from image_synthesis.modeling.utils.misc import mask_with_top_k, distributed_sinkhorn, get_token_type
from image_synthesis.distributed.distributed import all_reduce

# class for quantization
class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        batch_size, _, height, width = z.shape
        # import pdb; pdb.set_trace()
        z = z.permute(0, 2, 3, 1).contiguous() # B x H x W x C
        z_flattened = z.view(-1, self.e_dim) # BHW x C
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # return z_q, loss, (perplexity, min_encodings, min_encoding_indices)
        # return z_q, loss, min_encoding_indices.view(batch_size, height, width)

        output = {
            'quantize': z_q,
            'quantize_loss': loss,
            'index': min_encoding_indices.view(batch_size, height, width)
        }

        return output

    def get_codebook_entry(self, indices, shape):
        # import pdb; pdb.set_trace()
        # shape specifying (batch, height, width)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(*shape, -1) # B x H x W x C

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class GumbelQuantize(nn.Module):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, num_hiddens, embedding_dim, n_embed, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True,
                 remap=None, unknown_index="random"):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.n_e = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, return_logits=False):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work
        batch_size, height, width = z.shape[0], z.shape[-2], z.shape[-1]
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        logits = self.proj(z)
        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:,self.used,...]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:,self.used,...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = torch.einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)
        
        # import pdb; pdb.set_trace()
        output = {
            'quantize': z_q,
            'quantize_loss': diff,
            'index': ind.view(batch_size, height, width, -1)
        }
        
        if self.use_vqinterface:
            if return_logits:
                output['logits'] = logits

        return output

    def get_codebook_entry(self, indices, shape):
        # import pdb; pdb.set_trace()
        b, h, w = shape
        assert b*h*w == indices.shape[0]
        # indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        indices = indices.view(b, h, w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        z_q = torch.einsum('b n h w, n d -> b d h w', one_hot, self.embed.weight)
        return z_q

# blocks for encoder and decoder
def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), scale_by_2=None, num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        

        if isinstance(resolution, int):
            resolution = [resolution, resolution] # H, W
        elif isinstance(resolution, (tuple, list)):
            resolution = list(resolution)
        else:
            raise ValueError('Unknown type of resolution:', resolution)
            
        attn_resolutions_ = []
        for ar in attn_resolutions:
            if isinstance(ar, (list, tuple)):
                attn_resolutions_.append(list(ar))
            else:
                attn_resolutions_.append([ar, ar])
        attn_resolutions = attn_resolutions_
        
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn

            if scale_by_2 is None:
                if i_level != self.num_resolutions-1:
                    down.downsample = Downsample(block_in, resamp_with_conv)
                    curr_res = [r // 2 for r in curr_res]
            else:
                if scale_by_2[i_level]:
                    down.downsample = Downsample(block_in, resamp_with_conv)
                    curr_res = [r // 2 for r in curr_res]  
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        #assert x.shape[2] == self.resolution[0] and x.shape[3] == self.resolution[1], "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)

            if getattr(self.down[i_level], 'downsample', None) is not None:
                h = self.down[i_level].downsample(hs[-1])

            if i_level != self.num_resolutions-1:
                # hs.append(self.down[i_level].downsample(hs[-1]))
                hs.append(h)

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), scale_by_2=None, num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        
        if isinstance(resolution, int):
            resolution = [resolution, resolution] # H, W
        elif isinstance(resolution, (tuple, list)):
            resolution = list(resolution)
        else:
            raise ValueError('Unknown type of resolution:', resolution)
            
        attn_resolutions_ = []
        for ar in attn_resolutions:
            if isinstance(ar, (list, tuple)):
                attn_resolutions_.append(list(ar))
            else:
                attn_resolutions_.append([ar, ar])
        attn_resolutions = attn_resolutions_

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution 
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        if scale_by_2 is None:
            curr_res = [r // 2**(self.num_resolutions-1) for r in self.resolution]
        else:
            
            scale_factor = sum([int(s) for s in scale_by_2])
            curr_res = [r // 2**scale_factor for r in self.resolution]

        self.z_shape = (1, z_channels, curr_res[0], curr_res[1])
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if scale_by_2 is None:
                if i_level != 0:
                    up.upsample = Upsample(block_in, resamp_with_conv)
                    curr_res = [r * 2 for r in curr_res]
            else:
                if scale_by_2[i_level]:
                    up.upsample = Upsample(block_in, resamp_with_conv)
                    curr_res = [r * 2 for r in curr_res]
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            # if i_level != 0:
            if getattr(self.up[i_level], 'upsample', None) is not None:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VQGAN(BaseCodec):
    def __init__(self,
                 *,
                 ddconfig,
                 lossconfig=None,
                 n_embed,
                 embed_dim,
                 ignore_keys=[],
                 data_info={'key': 'image'},
                 trainable=False,
                 ckpt_path=None,
                 token_shape=None,
                 quantize_name='VectorQuantizer',
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        if quantize_name == 'VectorQuantizer':
            self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        elif quantize_name == 'GumbelQuantize':
            self.quantize = GumbelQuantize(ddconfig["z_channels"], embed_dim,
                                            n_embed=n_embed)
        else:
            raise NotImplementedError
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.data_info = data_info
    
        if lossconfig is not None and trainable:
            self.loss = instantiate_from_config(lossconfig)
        else:
            self.loss = None
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        self.trainable = trainable
        self._set_trainable()

        self.token_shape = token_shape

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if 'model' in sd:
            sd = sd['model']
        else:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("VQGAN: Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"VQGAN: Restored from {path}")

    @property
    def device(self):
        return self.quant_conv.weight.device

    def pre_process(self, data):
        data = data.to(self.device)
        data = data / 127.5 - 1.0
        return data

    def multi_pixels_with_mask(self, data, mask):
        if data.max() > 1:
            raise ValueError('The data need to be preprocessed!')
        mask = mask.to(data.device).repeat(1,3,1,1)
        data = data * mask.to(data.dtype)
        data[~mask] = -1.0
        return data

    def post_process(self, data):
        data = (data + 1.0) * 127.5
        data = torch.clamp(data, min=0.0, max=255.0)
        return data

    def get_number_of_tokens(self):
        return self.quantize.n_e

    def get_tokens(self, data, mask=None, enc_with_mask=True, **kwargs):
        data = self.pre_process(data)
        x = self.encoder(data)
        x = self.quant_conv(x)
        idx = self.quantize(x)['index']
        if self.token_shape is None:
            self.token_shape = idx.shape[1:3]

        # import pdb; pdb.set_trace()
        if mask is not None: # mask should be B x 1 x H x W
            if enc_with_mask:
                # data = data * mask.to(data)
                data = self.multi_pixels_with_mask(data, mask)
                x = self.encoder(data)
                x = self.quant_conv(x)
                idx_mask = self.quantize(x)['index']
            else:
                idx_mask = idx.clone()
            # downsampling
            # mask = F.interpolate(mask.float(), size=idx_mask.shape[-2:]).to(torch.bool)
            token_type = get_token_type(mask, self.token_shape) # B x 1 x H x W
            mask = token_type == 1
            output = {
                'target': idx.view(idx.shape[0], -1),
                'mask': mask.view(mask.shape[0], -1),
                'token': idx_mask.view(idx_mask.shape[0], -1),
                'token_type': token_type.view(token_type.shape[0], -1),
            }
        else:
            output = {
                'token': idx.view(idx.shape[0], -1)
                }

        # get token index
        # import pdb; pdb.set_trace()
        token_index = output['token'] #.view(-1)
        output['token_index'] = token_index

        return output

    def decode(self, token, **kwarg):
        assert self.token_shape is not None
        # import pdb; pdb.set_trace()
        bhw = (token.shape[0], self.token_shape[0], self.token_shape[1])
        quant = self.quantize.get_codebook_entry(token.view(-1), shape=bhw)
        quant = self.post_quant_conv(quant)
        rec = self.decoder(quant)
        rec = self.post_process(rec)
        return rec

    @torch.no_grad()
    def sample(self, batch):

        data = self.pre_process(batch[self.data_info['key']])
        x = self.encoder(data)
        x = self.quant_conv(x)
        quant = self.quantize(x)['quantize']
        quant = self.post_quant_conv(quant)
        rec = self.decoder(quant)
        rec = self.post_process(rec)

        return {'input': batch[self.data_info['key']], 'reconstruction': rec}

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    

    def parameters(self, recurse=True, name=None):
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            if name == 'generator':
                params = list(self.encoder.parameters())+ \
                         list(self.decoder.parameters())+\
                         list(self.quantize.parameters())+\
                         list(self.quant_conv.parameters())+\
                         list(self.post_quant_conv.parameters())
            elif name == 'discriminator':
                params = self.loss.discriminator.parameters()
            else:
                raise ValueError("Unknown type of name {}".format(name))
            return params

    def forward(self, batch, name='none', return_loss=True, step=0, **kwargs):
        
        input = self.pre_process(batch[self.data_info['key']])
        x = self.encoder(input)
        x = self.quant_conv(x)
        quant_out = self.quantize(x)
        quant = quant_out['quantize']
        emb_loss = quant_out['quantize_loss']

        # import pdb; pdb.set_trace()

        # recconstruction
        quant = self.post_quant_conv(quant)
        rec = self.decoder(quant)
        output = self.loss(codebook_loss=emb_loss, 
                           inputs=input, 
                           reconstructions=rec, 
                           optimizer_name=name, 
                           global_step=step, 
                           last_layer=self.get_last_layer())

        return output


if __name__ == '__main__':
    logits = torch.tensor([[0, 1, 2], [4,5,6]])
    mask = ~(logits > 2)

    print(mask)


