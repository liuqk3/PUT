from audioop import bias
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from image_synthesis.utils.misc import instantiate_from_config
from image_synthesis.modeling.codecs.base_codec import BaseCodec
from image_synthesis.modeling.modules.vqgan_loss.vqperceptual import VQLPIPSWithDiscriminator
from image_synthesis.modeling.utils.misc import distributed_sinkhorn, get_token_type
from image_synthesis.distributed.distributed import all_reduce, get_world_size
from image_synthesis.modeling.modules.edge_connect.losses import EdgeConnectLoss


def value_scheduler(init_value, dest_value, step, step_range, total_steps, scheduler_type='cosine'):
    assert scheduler_type in ['cosine', 'step'], 'scheduler {} not implemented!'.format(scheduler_type)

    step_start, step_end = tuple(step_range)
    if step_end <= 0:
        step_end = total_steps

    if step < step_start:
        return init_value
    if step > step_end:
        return dest_value

    factor = float(step - step_start) / float(max(1, step_end - step_start))
    if scheduler_type == 'cosine':
        factor = max(0.0, 0.5 * (1.0 + math.cos(math.pi * factor)))
    elif scheduler_type == 'step':
        factor = 1 - factor
    else:
        raise NotImplementedError('scheduler type {} not implemented!'.format(scheduler_type))
    if init_value >= dest_value: # decrease
        value = dest_value + (init_value - dest_value) * factor
    else: # increase
        factor = 1 - factor
        value = init_value + (dest_value - init_value) * factor
    return value 

def gumbel_softmax(logits, temperature=1.0, gumbel_scale=1.0, dim=-1, hard=True):
    # gumbels = torch.distributions.gumbel.Gumbel(0,1).sample(logits.shape).to(logits)
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    # adjust the scale of gumbel noise
    gumbels = gumbels * gumbel_scale

    gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


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

    def __init__(self, 
                n_e, 
                e_dim, 
                beta=0.25,
                masked_embed_start=None,
                embed_init_scale=1.0,
                embed_ema=False,
                get_embed_type='matmul',
                distance_type='euclidean',

                gumbel_sample=False,
                adjust_logits_for_gumbel='sqrt',
                gumbel_sample_stop_step=None,
                temperature_step_range=(0,15000),
                temperature_scheduler_type='cosine',
                temperature_init=1.0,
                temperature_dest=1/16.0,
                gumbel_scale_init=1.0,
                gumbel_scale_dest=1.0,
                gumbel_scale_step_range=(0,1),
                gumbel_scale_scheduler_type='cosine',
        ):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embed_ema = embed_ema
        self.gumbel_sample = gumbel_sample
        self.adjust_logits_for_gumbel = adjust_logits_for_gumbel
        self.temperature_step_range = temperature_step_range
        self.temperature_init = temperature_init
        self.temperature_dest = temperature_dest
        self.temperature_scheduler_type = temperature_scheduler_type
        self.gumbel_scale_init = gumbel_scale_init
        self.gumbel_scale_dest = gumbel_scale_dest 
        self.gumbel_scale_step_range = gumbel_scale_step_range
        self.gumbel_sample_stop_step = gumbel_sample_stop_step
        self.gumbel_scale_scheduler_type = gumbel_scale_scheduler_type
        if self.gumbel_sample_stop_step is None:
            self.gumbel_sample_stop_step = max(self.temperature_step_range[-1], self.temperature_step_range[-1])

        self.get_embed_type = get_embed_type
        self.distance_type = distance_type

        if self.embed_ema:
            self.decay = 0.99
            self.eps = 1.0e-5
            embed = torch.randn(n_e, e_dim)
            # embed = torch.zeros(n_e, e_dim)
            # embed.data.uniform_(-embed_init_scale / self.n_e, embed_init_scale / self.n_e)
            self.register_buffer("embedding", embed)
            self.register_buffer("cluster_size", torch.zeros(n_e))
            self.register_buffer("embedding_avg", embed.clone())
        else:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
            self.embedding.weight.data.uniform_(-embed_init_scale / self.n_e, embed_init_scale / self.n_e)

        self.masked_embed_start = masked_embed_start
        if self.masked_embed_start is None:
            self.masked_embed_start = self.n_e

        if self.distance_type == 'learned':
            self.distance_fc = nn.Linear(self.e_dim, self.n_e)
    @property
    def device(self):
        if isinstance(self.embedding, nn.Embedding):
            return self.embedding.weight.device
        return self.embedding.device

    @property
    def norm_feat(self):
        return self.distance_type in ['cosine', 'sinkhorn']
    
    @property
    def embed_weight(self):
        if isinstance(self.embedding, nn.Embedding):
            return self.embedding.weight
        else:
            return self.embedding
    
    def get_codebook(self):
        codes = {
            'default': {
                'code': self.embedding
            }
        }

        if self.masked_embed_start < self.n_e:
            codes['unmasked'] = {'code': self.embedding[:self.masked_embed_start]}
            codes['masked'] = {'code': self.embedding[self.masked_embed_start:]}

            default_label = torch.ones((self.n_e)).to(self.device)
            default_label[self.masked_embed_start:] = 0
            codes['default']['label'] = default_label
        return codes

    def norm_embedding(self):
        if self.training:
            with torch.no_grad():
                w = self.embed_weight.data.clone()
                w = F.normalize(w, dim=1, p=2)
                if isinstance(self.embedding, nn.Embedding):
                    self.embedding.weight.copy_(w)
                else:
                    self.embedding.copy_(w)


    def get_index(self, logits, topk=1, step=None, total_steps=None):
        """
        logits: BHW x N
        topk: the topk similar codes to be sampled from

        return:
            indices: BHW
        """
        
        if self.gumbel_sample:
            gumbel = True
            if self.training:
                if step > self.gumbel_sample_stop_step and self.gumbel_sample_stop_step > 0:
                    gumbel = False
            else:
                gumbel = False
        else:
            gumbel = False

        if gumbel:
            temp = value_scheduler(init_value=self.temperature_init,
                                    dest_value=self.temperature_dest,
                                    step=step,
                                    step_range=self.temperature_step_range,
                                    total_steps=total_steps,
                                    scheduler_type=self.temperature_scheduler_type
                                    )
            scale = value_scheduler(init_value=self.gumbel_scale_init,
                                    dest_value=self.gumbel_scale_dest,
                                    step=step,
                                    step_range=self.gumbel_scale_step_range,
                                    total_steps=total_steps,
                                    scheduler_type=self.gumbel_scale_scheduler_type
                                    )
            if self.adjust_logits_for_gumbel == 'none':
                pass
            elif self.adjust_logits_for_gumbel == 'sqrt':
                logits = torch.sqrt(logits)
            elif self.adjust_logits_for_gumbel == 'log':
                logits = torch.log(logits)
            else:
                raise NotImplementedError
            
            # for logits, the larger the value is, the corresponding code shoule not be sampled, so we need to negative it
            logits = -logits
            # one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=True) # BHW x N
            logits = gumbel_softmax(logits, temperature=temp, gumbel_scale=scale, dim=1, hard=True)
        else:
            logits = -logits
        
        # now, the larger value should be sampled
        if topk == 1:
            indices = torch.argmax(logits, dim=1)
        else:
            assert not gumbel, 'For gumbel sample, topk may introduce some random choices of codes!'
            topk = min(logits.shape[1], topk)

            _, indices = torch.topk(logits, dim=1, k=topk) # N x K
            chose = torch.randint(0, topk, (indices.shape[0],)).to(indices.device) # N
            chose = torch.zeros_like(indices).scatter_(1, chose.unsqueeze(dim=1), 1.0) # N x K
            indices = (indices * chose).sum(dim=1, keepdim=False)
            
            # filtered_logits = logits_top_k(logits, filter_ratio=topk, minimum=1, filter_type='count')
            # probs = F.softmax(filtered_logits * 1, dim=1)
            # indices = torch.multinomial(probs, 1).squeeze(dim=1) # BHW
            
        return indices

    def get_distance(self, z, code_type='all'):
        """
        z: L x D, the provided features

        return:
            d: L x N, where N is the number of tokens, the smaller distance is, the more similar it is
        """
        if self.distance_type == 'euclidean':
            d = torch.sum(z ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embed_weight**2, dim=1) - 2 * \
                torch.matmul(z, self.embed_weight.t())
        elif self.distance_type == 'learned':
            d = 0 - self.distance_fc(z) # BHW x N
        elif self.distance_type == 'sinkhorn':
            s = torch.einsum('ld,nd->ln', z, self.embed_weight) # BHW x N
            d = 0 - distributed_sinkhorn(s.detach())
            # import pdb; pdb.set_trace()
        elif self.distance_type == 'cosine':
            d = 0 - torch.einsum('ld,nd->ln', z, self.embed_weight) # BHW x N
        else:
            raise NotImplementedError('distance not implemented for {}'.format(self.distance_type))
        
        if code_type == 'masked':
            d = d[:, self.masked_embed_start:]
        elif code_type == 'unmasked':
            d = d[:, :self.masked_embed_start]

        return d

    def _quantize(self, z, token_type=None, topk=1, step=None, total_steps=None):
        """
            z: L x D
            token_type: L, 1 denote unmasked token, other masked token
        """
        d = self.get_distance(z)

        # find closest encodings 
        # import pdb; pdb.set_trace()
        if token_type is None or self.masked_embed_start == self.n_e:
            # min_encoding_indices = torch.argmin(d, dim=1) # L
            min_encoding_indices = self.get_index(d, topk=topk, step=step, total_steps=total_steps)
        else:
            min_encoding_indices = torch.zeros(z.shape[0]).long().to(z.device)
            idx = token_type == 1
            if idx.sum() > 0:
                d_ = d[idx][:, :self.masked_embed_start] # l x n
                # indices_ = torch.argmin(d_, dim=1)
                indices_ = self.get_index(d_, topk=topk, step=step, total_steps=total_steps)
                min_encoding_indices[idx] = indices_
            idx = token_type != 1
            if idx.sum() > 0:
                d_ = d[idx][:, self.masked_embed_start:] # l x n
                # indices_ = torch.argmin(d_, dim=1)
                indices_ = self.get_index(d_, topk=topk, step=step, total_steps=total_steps)
                indices_ += self.masked_embed_start
                min_encoding_indices[idx] = indices_

        if self.get_embed_type == 'matmul':
            min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)
            min_encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)
            # import pdb; pdb.set_trace()
            z_q = torch.matmul(min_encodings, self.embed_weight)#.view(z.shape)
        elif self.get_embed_type == 'retrive':
            z_q = F.embedding(min_encoding_indices, self.embed_weight)#.view(z.shape)
        else:
            raise NotImplementedError

        return z_q, min_encoding_indices

    def forward(self, z, token_type=None, topk=1, step=None, total_steps=None):
        """
            z: B x C x H x W
            token_type: B x 1 x H x W
        """
        if self.distance_type in ['sinkhorn', 'cosine']:
            # need to norm feat and weight embedding    
            self.norm_embedding()            
            z = F.normalize(z, dim=1, p=2)

        # reshape z -> (batch, height, width, channel) and flatten
        batch_size, _, height, width = z.shape
        # import pdb; pdb.set_trace()
        z = z.permute(0, 2, 3, 1).contiguous() # B x H x W x C
        z_flattened = z.view(-1, self.e_dim) # BHW x C
        if token_type is not None:
            token_type_flattened = token_type.view(-1)
        else:
            token_type_flattened = None

        z_q, min_encoding_indices = self._quantize(z_flattened, token_type=token_type_flattened, topk=topk, step=step, total_steps=total_steps)
        z_q = z_q.view(batch_size, height, width, -1) #.permute(0, 2, 3, 1).contiguous()

        if self.training and self.embed_ema:
            # import pdb; pdb.set_trace()
            assert self.distance_type in ['euclidean', 'cosine']
            indices_onehot = F.one_hot(min_encoding_indices, self.n_e).to(z_flattened.dtype) # L x n_e
            indices_onehot_sum = indices_onehot.sum(0) # n_e
            z_sum = (z_flattened.transpose(0, 1) @ indices_onehot).transpose(0, 1) # n_e x D

            all_reduce(indices_onehot_sum)
            all_reduce(z_sum)

            self.cluster_size.data.mul_(self.decay).add_(indices_onehot_sum, alpha=1 - self.decay)
            self.embedding_avg.data.mul_(self.decay).add_(z_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_e * self.eps) * n
            embed_normalized = self.embedding_avg / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embed_normalized)

        if self.embed_ema:
            loss = (z_q.detach() - z).pow(2).mean()
        else:
            # compute loss for embedding
            loss = torch.mean((z_q.detach()-z).pow(2)) + self.beta * torch.mean((z_q - z.detach()).pow(2))

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        unique_idx = min_encoding_indices.unique()
        output = {
            'quantize': z_q,
            'used_unmasked_quantize_embed': torch.zeros_like(loss) + (unique_idx<self.masked_embed_start).sum(),
            'used_masked_quantize_embed': torch.zeros_like(loss) + (unique_idx>=self.masked_embed_start).sum(),
            'quantize_loss': loss,
            'index': min_encoding_indices.view(batch_size, height, width)
        }
        if token_type_flattened is not None:
            unmasked_num_token = all_reduce((token_type_flattened == 1).sum())
            masked_num_token = all_reduce((token_type_flattened != 1).sum())
            output['unmasked_num_token'] = unmasked_num_token
            output['masked_num_token'] = masked_num_token

        return output

    def get_codebook_entry(self, indices, shape):
        # import pdb; pdb.set_trace()

        # shape specifying (batch, height, width)
        if self.get_embed_type == 'matmul':
            min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
            min_encodings.scatter_(1, indices[:,None], 1)
            # get quantized latent vectors
            z_q = torch.matmul(min_encodings.float(), self.embed_weight)
        elif self.get_embed_type == 'retrive':
            z_q = F.embedding(indices, self.embed_weight)
        else:
            raise NotImplementedError

        # import pdb; pdb.set_trace()
        if shape is not None:
            z_q = z_q.view(*shape, -1) # B x H x W x C

            if len(z_q.shape) == 4:
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, activate_before='none', activate_after='none', upsample_type='deconv'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activate_before = activate_before
        self.activate_after = activate_after
        self.upsample_type = upsample_type 
        if self.upsample_type == 'deconv':
            self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            assert self.upsample_type in ['bilinear', 'nearest'], 'upsample {} not implemented!'.format(self.upsample_type)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        if self.activate_before == 'relu':
            x = F.relu(x)
        elif self.activate_before == 'none':
            pass
        else:
            raise NotImplementedError

        if self.upsample_type == 'deconv':
            x = self.deconv(x)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode=self.upsample_type)
            x = self.conv(x)

        if self.activate_after == 'relu':
            x = F.relu(x)
        elif self.activate_after == 'none':
            pass
        else:
            raise NotImplementedError
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, activate_before='none', activate_after='none', downsample_type='conv', partial_conv=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activate_before = activate_before
        self.activate_after = activate_after
        self.downsample_type = downsample_type 
        self.partial_conv = partial_conv
        if self.downsample_type == 'conv':
            if self.partial_conv:
                raise NotImplementedError
                self.conv = PartialConv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1) 
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1) 
        else:
            assert self.downsample_type in ['bilinear', 'nearest', 'maxpool', 'avgpool'], 'upsample {} not implemented!'.format(self.downsample_type)
            if self.partial_conv:
                raise NotImplementedError
                self.conv = PartialConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1) 
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, mask=None):

        if self.activate_before == 'relu':
            x = F.relu(x)
        elif self.activate_before == 'none':
            pass
        else:
            raise NotImplementedError

        if self.downsample_type != 'conv':
            if self.downsample_type in ['nearest', 'bilinear']:
                x = F.interpolate(x, scale_factor=2.0, mode=self.downsample_type)
            elif self.downsample_type == 'maxpool':
                x = torch.max_pool2d(x, kernel_size=2, stride=2, padding=0, dilation=1)
            elif self.downsample_type == 'avgpool':
                x = torch.avg_pool2d(x, kernel_size=2, stride=2, padding=0, dilation=1)
        if mask is not None:
            mask = F.interpolate(mask, size=x.shape[-2:], mode='nearest')  
        if self.partial_conv:
            x = self.conv(x, mask_in=mask)
        else:
            x = self.conv(x)

        if self.activate_after == 'relu':
            x = F.relu(x)
        elif self.activate_after == 'none':
            pass
        else:
            raise NotImplementedError
        return x


# resblock only uses linear layer
class LinearResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(in_channel, channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, in_channel),
        )
        self.out_channels = in_channel
        self.in_channels = in_channel

    def forward(self, x):
        out = self.layers(x)
        out = out + x

        return out

# resblock only uses conv layer
class ConvResBlock(nn.Module):
    def __init__(self, in_channel, channel, partial_conv=False):
        super().__init__()
        
        self.partial_conv = partial_conv
        if not partial_conv:
            self.partial_conv_args = None 
            self.conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel, channel, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, in_channel, 1),
            )
        else:
            raise NotImplementedError
            self.conv1 = PartialConv2d(in_channel, channel, kernel_size=3, padding=1)
            self.conv2 = PartialConv2d(channel, in_channel, kernel_size=3, padding=1) 

        self.out_channels = in_channel
        self.in_channels = in_channel

    def forward(self, x, mask=None):
        if not self.partial_conv:
            out = self.conv(x)
        else:
            assert mask is not None, 'When use partial conv for inpainting, the mask should be provided!'
            mask = F.interpolate(mask, size=x.shape[-2:], mode='nearest')
            out = F.relu(x)
            out = self.conv1(out, mask_in=mask)
            out = F.relu(out)
            out = self.conv2(out, mask_in=mask)
        out += x 
        return out 


class PatchEncoder2(nn.Module):
    def __init__(self, *, 
                in_ch=3, 
                res_ch=256, 
                out_ch,
                num_res_block=2, 
                res_block_bottleneck=2,
                num_post_layer=1,
                stride=8,
                ):
        super().__init__()
        in_dim = in_ch * stride * stride 
        self.stride = stride
        self.out_channels = out_ch

        self.pre_layers = nn.Sequential(*[
            nn.Linear(in_dim, res_ch),
        ])

        res_layers = []
        for i in range(num_res_block):
            res_layers.append(LinearResBlock(res_ch, res_ch//res_block_bottleneck))
        if len(res_layers) > 0:
            self.res_layers = nn.Sequential(*res_layers)
        else:
            self.res_layers = nn.Identity()

        if num_post_layer == 0:
            self.post_layers = nn.Identity()
        elif num_post_layer == 1:
            post_layers = [
                nn.ReLU(inplace=True),
                nn.Linear(res_ch, out_ch),
                nn.ReLU(inplace=True),
            ]
            self.post_layers = nn.Sequential(*post_layers)
        else:
            raise NotImplementedError('more post layers seems can not improve the performance!')

    def forward(self, x):
        """
        x: [B, 3, H, W]

        """
        in_size = [x.shape[-2], x.shape[-1]]
        out_size = [s//self.stride for s in in_size]

        x = torch.nn.functional.unfold(x, kernel_size=(self.stride, self.stride), stride=(self.stride, self.stride)) # B x 3*patch_size^2 x L
        x = x.permute(0, 2, 1).contiguous() # B x L x 3*patch_size

        x = self.pre_layers(x)
        x = self.res_layers(x)
        x = self.post_layers(x)

        x = x.permute(0, 2, 1).contiguous() # B x C x L
        # import pdb; pdb.set_trace()
        x = torch.nn.functional.fold(x, output_size=out_size, kernel_size=(1,1), stride=(1,1))

        return x


class PatchConvEncoder2(nn.Module):
    def __init__(self, *, 
                in_ch=3, 
                res_ch=256, 
                out_ch,
                num_res_block=2, 
                num_res_block_before_resolution_change=0,
                res_block_bottleneck=2,
                stride=8,
                downsample_layer='downsample'):
        super().__init__()
        self.stride = stride
        self.out_channels = out_ch
        self.num_res_block_before_resolution_change = num_res_block_before_resolution_change

        # downsample with stride
        pre_layers = []
        in_ch_ = in_ch
        out_ch_ = 64
        while stride > 1:
            stride = stride // 2
            if stride == 1:
                out_ch_ = res_ch
            for i in range(self.num_res_block_before_resolution_change):
                pre_layers.append(
                    ConvResBlock(in_ch_, in_ch_//res_block_bottleneck)
                )
            if downsample_layer == 'downsample':
                pre_layers.append(DownSample(in_ch_, out_ch_, activate_before='none', activate_after='relu', downsample_type='conv'))
            elif downsample_layer == 'conv':
                pre_layers.append(nn.Conv2d(in_ch_, out_ch_, kernel_size=4, stride=2, padding=1))
                if stride != 1:
                    pre_layers.append(nn.ReLU(inplace=True))
            else:
                raise RuntimeError('{} not impleted!'.format(downsample_layer))
            in_ch_ = out_ch_
            out_ch_ = 2 * in_ch_
        self.pre_layers = nn.Sequential(*pre_layers)

        res_layers = []
        for i in range(num_res_block):
            res_layers.append(ConvResBlock(res_ch, res_ch//res_block_bottleneck))
        if len(res_layers) > 0:
            self.res_layers = nn.Sequential(*res_layers)
        else:
            self.res_layers = nn.Identity()

        post_layers = [
            nn.ReLU(inplace=True),
            nn.Conv2d(res_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ]
        self.post_layers = nn.Sequential(*post_layers)

    def forward(self, x):
        """
        x: [B, 3, H, W]

        """
        x = self.pre_layers(x)
        x = self.res_layers(x)
        x = self.post_layers(x)
        return x


class EncoderInPatchConvDecoder2(nn.Module):
    def __init__(self, in_ch, up_layers, with_res_block=True, res_block_bottleneck=2, downsample_layer='downsample', partial_conv=False):
        super().__init__()

        out_channels = []
        for layer in up_layers:
            out_channels.append(layer.out_channels)
        
        layers = []
        in_ch_ = in_ch
        for l in range(len(out_channels), -1, -1):
            out_ch_ = out_channels[l-1]
            # import pdb; pdb.set_trace()
            if l == len(out_channels):
                if partial_conv:
                    raise NotImplementedError
                    layers.append(PartialConv2d(in_ch_, out_ch_, kernel_size=3, stride=1, padding=1))
                else:
                    layers.append(nn.Conv2d(in_ch_, out_ch_, kernel_size=3, stride=1, padding=1))
            else:
                if l == 0:
                    out_ch_ = up_layers[0].in_channels
                if isinstance(up_layers[l], UpSample):
                    if downsample_layer == 'downsample': # recommneted
                        layers.append(DownSample(in_ch_, out_ch_, activate_before='relu', activate_after='none', downsample_type='conv', partial_conv=partial_conv))
                    elif downsample_layer == 'conv': # not recommented
                        if partial_conv:
                            raise NotImplementedError
                            layers.append(PartialConv2d(in_ch_, out_ch_, kernel_size=4, stride=2, padding=1))
                        else:
                            layers.append(nn.Conv2d(in_ch_, out_ch_, kernel_size=4, stride=2, padding=1))
                    else:
                        raise NotImplementedError
                elif isinstance(up_layers[l], ConvResBlock):
                    if with_res_block:
                        layers.append(ConvResBlock(in_ch_, in_ch_//res_block_bottleneck, partial_conv=partial_conv))
                else:
                    raise NotImplementedError
            in_ch_ = out_ch_

        self.layers = nn.Sequential(*layers)
        self.downsample_layer = downsample_layer
        self.partial_conv = partial_conv
        
    def forward(self, x, mask=None):
        out = {}
        if self.partial_conv:
            assert mask is not None, 'When use partial conv for inpainting, the mask should be provided!'
            mask = mask.to(x)
        for l in range(len(self.layers)):# layer in self.layers:
            layer = self.layers[l]
            if self.partial_conv:
                x = layer(x, mask)
            else:
                x = layer(x)
            if not isinstance(layer, (ConvResBlock,)):
                out[str(tuple(x.shape))] = x # before activation, because other modules perform activativation first
            if self.downsample_layer == 'conv':
                x = F.relu(x)
        return out


class PatchConvDecoder2(nn.Module):
    def __init__(self, *, 
                 in_ch,
                 res_ch,
                 out_ch=3, 
                 num_res_block,
                 res_block_bottleneck=2,
                 num_res_block_after_resolution_change=0,
                 stride=8,
                 upsample_type='deconv',
                 up_layer_with_image=False,
                 smooth_mask_kernel_size=0, # how to get the mask for merge different feature maps, only effective when up_layer_with_image is True
                 encoder_downsample_layer='downsample',
                 encoder_partial_conv=False,
                 encoder_with_res_block=True,
                 add_noise_to_image=False,
                 ):
        super().__init__()
        self.in_channels = in_ch
        self.upsample_type = upsample_type
        self.up_layer_with_image = up_layer_with_image
        self.smooth_mask_kernel_size = smooth_mask_kernel_size
        self.requires_image = self.up_layer_with_image
        self.encoder_partial_conv = encoder_partial_conv
        self.add_noise_to_image = add_noise_to_image
        self.num_res_block_after_resolution_change = num_res_block_after_resolution_change

        if self.up_layer_with_image and self.smooth_mask_kernel_size > 1:
            self.mask_smooth_kernel = torch.ones((1, 1, self.smooth_mask_kernel_size, self.smooth_mask_kernel_size))
            self.mask_smooth_kernel = self.mask_smooth_kernel / self.mask_smooth_kernel.numel()

        self.pre_layers = nn.Sequential(*[
            torch.nn.Conv2d(in_ch, res_ch, kernel_size=3, stride=1, padding=1),
        ])

        # res resblocks
        res_layers = []
        for i in range(num_res_block):
            res_layers.append(ConvResBlock(res_ch, res_ch//res_block_bottleneck))
        if len(res_layers) > 0:
            self.res_layers = nn.Sequential(*res_layers)
        else:
            self.res_layers = nn.Identity()

        # upsampling in middle layers
        post_layer_in_ch = 64
        out_ch_ = post_layer_in_ch
        up_layers = []
        while stride > 1:
            stride = stride // 2
            in_ch_ = out_ch_ * 2
            if stride == 1:
                in_ch_ = res_ch
            layers_ = []
            layers_.append(UpSample(in_ch_, out_ch_, activate_before='relu', activate_after='none', upsample_type=self.upsample_type))
            for r in range(self.num_res_block_after_resolution_change):
                layers_.append(ConvResBlock(out_ch_, out_ch_//res_block_bottleneck))
            up_layers = layers_ + up_layers
            out_ch_ *= 2
        # import pdb; pdb.set_trace()
        self.up_layers = nn.Sequential(*up_layers)

        post_layers = [
            nn.ReLU(inplace=True),
            nn.Conv2d(post_layer_in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        ]
        self.post_layers = torch.nn.Sequential(*post_layers)

        if self.up_layer_with_image:
            self.encoder = EncoderInPatchConvDecoder2(
                in_ch=out_ch, 
                up_layers=self.up_layers, 
                downsample_layer=encoder_downsample_layer, 
                with_res_block=encoder_with_res_block,
                partial_conv=encoder_partial_conv
            )

    def smooth_mask(self, mask, binary=True): 
        """
        This function is used to expand the mask
        """
        shape = mask.shape[-2:]
        mask = F.conv2d(mask, self.mask_smooth_kernel.to(mask))
        mask = F.interpolate(mask, size=shape, mode='bilinear', align_corners=True)
        mask_ = (mask >= 0.8).to(mask)
        if binary:
            return mask_
        else:
            return mask_ * mask

    def forward(self, x, masked_image=None, mask=None):
        # pre layers
        x = self.pre_layers(x)
        x = self.res_layers(x)

        if self.up_layer_with_image:
            mask = mask.to(x)
            if self.add_noise_to_image:
                masked_image = masked_image * mask + torch.randn_like(masked_image) * (1 - mask)
            im_x = self.encoder(masked_image, mask)
            for l in range(len(self.up_layers)):
                if isinstance(self.up_layers[l], UpSample):
                    x_ = im_x[str(tuple(x.shape))]
                    mask_ = F.interpolate(mask, size=x.shape[-2:], mode='nearest')
                    if self.smooth_mask_kernel_size > 1:
                        mask_ = self.smooth_mask(mask_, binary=False)
                    x = x * (1-mask_) + x_ * mask_
                x = self.up_layers[l](x)
            x = x * (1-mask) + im_x[str(tuple(x.shape))] * mask
            x = self.post_layers(x)
        else:
            x = self.up_layers(x)
            x = self.post_layers(x)
        return x


class PatchVQGAN(BaseCodec):
    def __init__(self,
                 *,
                 encoder_config,
                 decoder_config,
                 lossconfig=None,
            
                 quantizer_config=None, # quantizer can be given by this config (strongly recomended), 

                 conv_before_quantize=True,
                 ignore_keys=[],
                 trainable=False,
                 train_part='all',
                 ckpt_path=None,
                 token_shape=None,
                 resize_mask_type='pixel_shuffle',
                 combine_rec_and_gt=False,
                 im_process_info={'scale': 127.5, 'mean': 1.0, 'std': 1.0}
                 ):
        super().__init__()
        self.encoder = instantiate_from_config(encoder_config) # Encoder(**encoder_config)
        self.decoder = instantiate_from_config(decoder_config) # Decoder(**decoder_config)
        self.quantize = instantiate_from_config(quantizer_config)

        # import pdb; pdb.set_trace()
        if conv_before_quantize:
            self.quant_conv = torch.nn.Conv2d(self.encoder.out_channels, self.quantize.e_dim, 1)
        else:
            assert self.encoder.out_channels == self.quantize.e_dim, "the channels for quantization shoule be the same"
            self.quant_conv = nn.Identity()
        self.post_quant_conv = torch.nn.Conv2d(self.quantize.e_dim, self.decoder.in_channels, 1)
        self.im_process_info = im_process_info
        for k, v in self.im_process_info.items():
            v = torch.tensor(v).view(1, -1, 1, 1)
            if v.shape[1] != 3:
                v = v.repeat(1, 3, 1, 1)
            self.im_process_info[k] = v
    
        if lossconfig is not None and trainable:
            self.loss = instantiate_from_config(lossconfig)
        else:
            self.loss = None
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        self.trainable = trainable
        self.train_part = train_part
        self._set_trainable(train_part=self.train_part)

        self.token_shape = token_shape
        self.combine_rec_and_gt = combine_rec_and_gt
        self.resize_mask_type = resize_mask_type

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
                    print("P-VQVAE: Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print("P-VQVAE: Load pretrained model from {}".format(path))

    @property
    def device(self):
        return self.post_quant_conv.weight.device

    @property
    def embed_dim(self):
        return self.post_quant_conv.weight.shape[0]

    def get_codebook(self):
        return self.quantize.get_codebook()

    def pre_process(self, data):
        data = data.to(self.device)
        # data = data / 127.5 - 1.0
        data = (data / self.im_process_info['scale'].to(data.device) - self.im_process_info['mean'].to(data.device)) / self.im_process_info['std'].to(data.device)
        return data

    def multi_pixels_with_mask(self, data, mask):
        if self.im_process_info['mean'].sum() != 0.0:
            eps = 1e-3
            if data.max() > (((255.0 / self.im_process_info['scale'] - self.im_process_info['mean'])/self.im_process_info['std']).max().to(data.device) + eps):
                raise ValueError('The data need to be preprocessed! data max: {}'.format(data.max()))
            mask = mask.to(data.device).repeat(1,3,1,1)
            data_m = data * mask.to(data.dtype)
            data_m[~mask] = ((torch.zeros_like(data_m) - self.im_process_info['mean'].to(data_m.device)) / self.im_process_info['std'].to(data_m.device))[~mask]
        else:
            data_m = data * mask.to(data)
        return data_m

    def post_process(self, data):
        # data = (data + 1.0) * 127.5
        data = (data * self.im_process_info['std'].to(data.device) + self.im_process_info['mean'].to(data.device)) * self.im_process_info['scale'].to(data.device)
        data = torch.clamp(data, min=0.0, max=255.0)
        return data

    def get_number_of_tokens(self):
        return self.quantize.n_e

    @torch.no_grad()
    def get_features(self, data, mask=None, 
                     return_token=False, 
                     return_quantize_feature=False):
        """
        Get the feature from image
        
        """
        data = self.pre_process(data)
        if mask is not None:
            data = self.multi_pixels_with_mask(data, mask)
        x = self.encoder(data)
        x = self.quant_conv(x) # B C H W
        token_shape = (x.shape[-2], x.shape[-1])

        output = {
            'feature': F.normalize(x, dim=1, p=2) if self.quantize.norm_feat else x
        }
        if return_quantize_feature or return_token:
            if mask is not None:
                token_type = get_token_type(mask, token_shape, type=self.resize_mask_type) # B x 1 x H x W
            else:
                token_type = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])).long().to(self.device)

            quant_out = self.quantize(x, token_type=token_type)
            if return_quantize_feature:
                output['feature_quantize'] = quant_out['quantize']
            if return_token:
                output['token'] = quant_out['index']
        output['token_shape'] = token_shape
        return output
    
    @torch.no_grad()
    def get_codebook_entry_with_token(self, token, **kwargs):
        """
        token: B x L

        return:
            feature: features, B x L x C
        """

        t_shape = token.shape
        feat = self.quantize.get_codebook_entry(token.view(-1), shape=t_shape) 
        return {'feature': feat}

    @torch.no_grad()
    def get_tokens_with_feature(self, feat, token_type=None, topk=1):
        if token_type is None:
            token_type = torch.ones((feat.shape[0], 1, feat.shape[2], feat.shape[3])).long().to(self.device)
        idx = self.quantize(feat, token_type=token_type, topk=topk)['index']
        return idx

    @torch.no_grad()
    def get_tokens(self, data, mask=None, erase_mask=None, topk=1, return_token_index=False, cache=True, **kwargs):
        """
        Get the tokens of the given images
        """
        data = self.pre_process(data)
        x = self.encoder(data)
        x = self.quant_conv(x)
        token_shape = (x.shape[-2], x.shape[-1])

        if erase_mask is not None:
            token_type_erase = get_token_type(erase_mask, token_shape, type=self.resize_mask_type) # B x 1 x H x W
        else:
            token_type_erase = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])).long().to(self.device)

        idx = self.quantize(x, token_type=token_type_erase, topk=topk)['index']

        # import pdb; pdb.set_trace()
        if cache and (self.decoder.requires_image or self.combine_rec_and_gt):
            self.mask_im_tmp = self.multi_pixels_with_mask(data, mask)
            self.mask_tmp = mask

        output = {}
        output['token'] = idx.view(idx.shape[0], -1)

        # import pdb; pdb.set_trace()
        if mask is not None: # mask should be B x 1 x H x W
            # downsampling
            # mask = F.interpolate(mask.float(), size=idx_mask.shape[-2:]).to(torch.bool)
            token_type = get_token_type(mask, token_shape, type=self.resize_mask_type) # B x 1 x H x W
            mask = token_type == 1
            output = {
                'target': idx.view(idx.shape[0], -1).clone(),
                'mask': mask.view(mask.shape[0], -1),
                'token': idx.view(idx.shape[0], -1),
                'token_type': token_type.view(token_type.shape[0], -1),
            }
        else:
            output = {
                'token': idx.view(idx.shape[0], -1)
                }

        # get token index
        # used for computing token frequency
        if return_token_index:
            token_index = output['token'] #.view(-1)
            output['token_index'] = token_index
        output['token_shape'] = token_shape
        return output

    @torch.no_grad()
    def decode(self, token, mask_im=None, mask=None, combine_rec_and_gt=True, token_shape=None):
        """
        Decode the image with provided tokens
        """
        if token_shape is None:
            assert self.token_shape is not None
            token_shape = self.token_shape
        # import pdb; pdb.set_trace()
        bhw = (token.shape[0], token_shape[0], token_shape[1])
        quant = self.quantize.get_codebook_entry(token.view(-1), shape=bhw)
        quant = self.post_quant_conv(quant)
        # import pdb; pdb.set_trace()
        if self.decoder.requires_image:
            if mask_im is None:
                rec = self.decoder(quant, self.mask_im_tmp, mask=self.mask_tmp)
            else:
                rec = self.decoder(quant, self.pre_process(mask_im), mask=mask)
            # self.mask_im_tmp = None
        else:
            rec = self.decoder(quant)
        # import pdb; pdb.set_trace()
        if combine_rec_and_gt and self.combine_rec_and_gt:
            if mask_im is None:
                rec = rec * (1-self.mask_tmp.to(rec.dtype)) + self.mask_im_tmp * self.mask_tmp.to(rec.dtype)
            else:
                rec = rec * (1-mask.to(rec.dtype)) + self.pre_process(mask_im) * mask.to(rec.dtype)
        rec = self.post_process(rec)
        return rec


    @torch.no_grad()
    def sample(self, batch):

        data = self.pre_process(batch['image'])
        x = self.encoder(data)
        x = self.quant_conv(x)
        token_shape = (x.shape[-2], x.shape[-1])

        if 'erase_mask' in batch:
            token_type_erase = get_token_type(batch['erase_mask'], token_shape, type=self.resize_mask_type).to(self.device) # B x 1 x H x W
        else:
            token_type_erase = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])).long().to(self.device)

        quant = self.quantize(x, token_type=token_type_erase)['quantize']
        quant = self.post_quant_conv(quant)
        if self.decoder.requires_image:
            mask_im = self.multi_pixels_with_mask(data, batch['mask'])
            rec = self.decoder(quant, mask_im, mask=batch['mask'])
        else:
            rec = self.decoder(quant)
        rec = self.post_process(rec)

        out = {'input': batch['image'], 'reconstruction': rec}
        if self.decoder.requires_image:
            out['reference_input'] = self.post_process(mask_im)
            out['reference_mask'] = batch['mask'] * 255
            # import pdb; pdb.set_trace()
        return out

    def get_last_layer(self):
        return self.decoder.post_layers[-1].weight
    
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

    def forward(self, batch, name='none', return_loss=True, step=0, total_steps=None, **kwargs):
        
        if name == 'generator':
            input = self.pre_process(batch['image'])
            x = self.encoder(input)
            x = self.quant_conv(x)
            token_shape = list(x.shape[-2:])

            if 'erase_mask' in batch:
                token_type_erase = get_token_type(batch['erase_mask'], token_shape, type=self.resize_mask_type) # B x 1 x H x W
            else:
                token_type_erase = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])).long().to(self.device)

            quant_out = self.quantize(x, token_type=token_type_erase, step=step, total_steps=total_steps)
            quant = quant_out['quantize']
            emb_loss = quant_out['quantize_loss']

            # recconstruction
            quant = self.post_quant_conv(quant)
            if self.decoder.requires_image:
                rec = self.decoder(quant, self.multi_pixels_with_mask(input, batch['mask']), mask=batch['mask'])
            else:
                rec = self.decoder(quant)
            # save some tensors for 
            self.input_tmp = input 
            self.rec_tmp = rec 

            if isinstance(self.loss, VQLPIPSWithDiscriminator):
                output = self.loss(codebook_loss=emb_loss,
                                inputs=input, 
                                reconstructions=rec, 
                                optimizer_name=name, 
                                global_step=step, 
                                last_layer=self.get_last_layer())
            elif isinstance(self.loss, EdgeConnectLoss):
                other_loss = {}
                for k in quant_out:
                    if 'loss' in k:
                        other_loss[k] = quant_out[k]
                
                # norm image to 0-1
                if self.loss.norm_to_0_1:
                    loss_im = self.post_process(self.input_tmp) / 255.0
                    loss_rec = self.post_process(self.rec_tmp) / 255.0
                else:
                    loss_im = self.input_tmp
                    loss_rec = self.rec_tmp
                
                output = self.loss(
                    image=loss_im, # norm to [0, 1]
                    reconstruction=loss_rec, # norm to [0, 1]
                    mask=batch['mask'] if self.decoder.requires_image else None,
                    step=step,
                    name=name,
                    other_loss=other_loss
                )
            else:
                raise NotImplementedError('{}'.format(type(self.loss)))
            
            # for observing the number of used codebooks
            for k, v in quant_out.items():
                if k == 'loss':
                    continue
                if v.numel() == 1 and len(v.shape) == 0:
                    output[k] = v

        elif name == 'discriminator':
            if isinstance(self.loss, VQLPIPSWithDiscriminator):
                output = self.loss(codebook_loss=None,
                                inputs=self.input_tmp, 
                                reconstructions=self.rec_tmp, 
                                optimizer_name=name, 
                                global_step=step, 
                                last_layer=self.get_last_layer())
            elif isinstance(self.loss, EdgeConnectLoss):
                if self.loss.norm_to_0_1:
                    loss_im = self.post_process(self.input_tmp) / 255.0
                    loss_rec = self.post_process(self.rec_tmp) / 255.0
                else:
                    loss_im = self.input_tmp
                    loss_rec = self.rec_tmp
                loss_im = self.input_tmp
                loss_rec = self.rec_tmp
                output = self.loss(
                    image=loss_im, # norm to [0, 1]
                    reconstruction=loss_rec, # norm to [0, 1]
                    step=step,
                    name=name
                )
            else:
                raise NotImplementedError('{}'.format(type(self.loss)))
        else:
            raise NotImplementedError('{}'.format(name))        
        return output


if __name__ == '__main__':
    c = nn.Conv2d(3, 6, 1)

    a = 1


