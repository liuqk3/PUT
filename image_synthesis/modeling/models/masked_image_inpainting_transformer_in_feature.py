"""
This transformer model is for PUT in CVPR 2022
"""
import math
from pdb import set_trace
from random import random
import torch
from torch.utils import data
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import time

from image_synthesis.utils.misc import instantiate_from_config
from image_synthesis.modeling.utils.misc import get_token_type
from image_synthesis.distributed.distributed import get_local_rank
from image_synthesis.modeling.utils.misc import logits_top_k


class FullAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self,
                 n_embd, # the embed dim
                 n_head, # the number of heads
                 seq_len=None, # the max length of sequence
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
                 causal=True,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head
        self.causal = causal

        # causal mask to ensure that attention is only applied to the left in the input sequence
        if self.causal:
            self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len))
                                        .view(1, 1, seq_len, seq_len))


    def forward(self, x, mask=None):
        """
        x: B x T x C
        mask: None or tensor B x T, bool type. For values with False, no attention should be attened
        """
        B, T, C = x.size()
        # import pdb; pdb.set_trace()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        
        # print(q.shape, k.shape)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        if self.causal:
            # print(att.shape, self.mask.shape, T)
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        
        if mask is not None:
            mask = mask.view(B, 1, 1, T)
            att = att.masked_fill(~mask, float('-inf'))

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att

class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class ConvMLP(nn.Module):
    def __init__(
        self,
        n_embd,
        mlp_hidden_times,
        act, 
        resid_pdrop,
        spatial_size=None # (h, w) of input shape 
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=n_embd, out_channels=mlp_hidden_times*n_embd, kernel_size=3, stride=1,padding=1)
        self.act = act 
        self.conv2 = nn.Conv2d(in_channels=mlp_hidden_times*n_embd, out_channels=n_embd, kernel_size=3, stride=1,padding=1)
        self.dropout = nn.Dropout(resid_pdrop)
        self.spatial_size = spatial_size

    def forward(self, x):
        """
        x: B x T x C
        """
        # import pdb; pdb.set_trace()
        if self.spatial_size is None:
            length = x.shape[1]
            h = int(math.sqrt(length))
            w = h 
        else:
            h, w = self.spatial_size[0], self.spatial_size[1]
        x = x.view(x.shape[0], h, w, x.shape[-1]).permute(0, 3, 1, 2) # B x C x H x W
        
        x = self.conv2(self.act(self.conv1(x)))
        x = x.permute(0, 2, 3, 1).view(x.shape[0], h*w, -1) # B x L x C
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self,
                 n_embd,
                 n_head,
                 seq_len,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 causal=True,
                 mlp_type='linear',
                 mlp_hidden_times=4,
                 activate='GELU',
                 ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
            n_embd=n_embd,
            n_head=n_head,
            seq_len=seq_len,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            causal=causal
        )
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()
        if mlp_type == 'linear':
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )
        elif mlp_type == 'conv':
            self.mlp = ConvMLP(
                n_embd=n_embd,
                mlp_hidden_times=mlp_hidden_times,
                act=act, 
                resid_pdrop=resid_pdrop
            )

    def forward(self, x, mask=None):    
        a, att = self.attn(self.ln1(x), mask=mask)
        x = x + a 
        x = x + self.mlp(self.ln2(x))

        return x, att


class MaskedImageInpaintingTransformer(nn.Module):
    def __init__(
        self,
        *,
        n_layer, # number of layers in transformer
        content_seq_len, # length of content sequences
        embd_pdrop=0., # embedding dropout prob

        n_embd, # the embed dim
        n_head, # the number of heads
        attn_pdrop=0.1, # attention dropout prob
        resid_pdrop=0.1, # residual attention dropout prob
        block_activate='GELU',
        mlp_type='linear', # linear mlp or conv mlp
        mlp_hidden_times=4, # the times of hidden dimension in the MLP of attetntion block

        attn_content_with_mask=False,
        content_codec_config=None,


        content_ignore_token=-100,

        input_feature_type='origin',
        # args for training
        weight_decay=0.01,
        random_quantize=0.2, # random quantize the feature, only when the input feature is not quantized
        num_token=None,
    ):
        super().__init__()
        
        self.attn_content_with_mask = attn_content_with_mask

        # embeddings for content
        self.content_codec = instantiate_from_config(content_codec_config)
        self.emb_proj = nn.Linear(self.content_codec.embed_dim, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, content_seq_len, n_embd))
        
        # drop for embedding
        if embd_pdrop > 0:
            self.drop = nn.Dropout(embd_pdrop)
        else:
            self.drop = None
        
        # transformer
        self.blocks = nn.Sequential(*[Block(
                n_embd=n_embd,
                n_head=n_head,
                seq_len=content_seq_len,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                causal=False,
                mlp_type=mlp_type,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
        ) for n in range(n_layer)])

        # final prediction head
        out_cls = self.content_codec.get_number_of_tokens() if num_token is None else num_token
        self.layer_norm = nn.LayerNorm(n_embd)
        self.to_logits = nn.Linear(n_embd, out_cls)
        
        self.content_seq_len = content_seq_len
        self.content_ignore_token = content_ignore_token
        self.input_feature_type = input_feature_type
        self.weight_decay = weight_decay
        self.random_quantize = random_quantize

        self.apply(self._init_weights)

        # reinitialize the codec, so that the pretrained model can be reloaded
        self.content_codec = instantiate_from_config(content_codec_config)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)         

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
        only_masked=True,
        with_process_bar=False,
        mask_low_to_high=False,
        sample_largest=False,
        return_att_weight=False,
        return_reconstruction=False,
        num_token_per_iter=1,
        **kwargs,
    ):
        self.eval()
        if replicate != 1:
            for k in batch.keys():
                if batch[k] is not None and torch.is_tensor(batch[k]):
                    batch[k] = torch.cat([batch[k] for _ in range(replicate)], dim=0)
        if sample_largest:
            return self.sample_largest(
                batch=batch,
                filter_ratio=filter_ratio,
                filter_type=filter_type,
                temperature=temperature,
                only_masked=only_masked,
                with_process_bar=with_process_bar,
                return_gt=False,
                return_mask_gt=False,
                return_reconstruction=return_reconstruction,
                return_att_weight=return_att_weight,
                mask_low_to_high=mask_low_to_high,
                num_token_per_iter=num_token_per_iter,   
            )           
        else:
            return self.sample(
                batch=batch,
                filter_ratio=filter_ratio,
                filter_type=filter_type,
                temperature=temperature,
                only_masked=only_masked,
                with_process_bar=with_process_bar,
                return_gt=False,
                return_mask_gt=False,
                return_reconstruction=return_reconstruction,
                mask_low_to_high=mask_low_to_high,
                num_token_per_iter=num_token_per_iter, 
            )


    @torch.no_grad()
    def sample_largest(
        self,
        *,
        batch,
        filter_ratio = 0.8,
        filter_type='count',
        temperature = 1.0,
        # only_masked=True,
        # with_process_bar=False,
        return_gt=True,
        return_mask_gt=True,
        return_reconstruction=True,
        return_att_weight=False,
        mask_low_to_high=False,
        num_token_per_iter=1,
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

        data_mask = self.content_codec.get_features(batch['image'], 
                                                    mask=mask_, 
                                                    return_quantize_feature=self.input_feature_type == 'quantized',
                                                    return_token=True) # B x C x H x W
        if self.input_feature_type == 'origin':
            feat_mask = data_mask['feature'] # B x C x H x W
        elif self.input_feature_type == 'quantized':
            feat_mask = data_mask['feature_quantize'] # B x C x H x W
        else:
            raise NotImplementedError('inpute feature type {} not implemented!'.format(self.input_feature_type))
        b, _, h, w = feat_mask.shape
        feat_mask = self.emb_proj(feat_mask.permute(0, 2, 3, 1).view(b, h*w, -1).contiguous()) # B x HW x D
        # NOTE: the feature for masked tokens are the same?
        if self.pos_emb is not None:
            feat_mask = feat_mask + self.pos_emb

        token_type = get_token_type(mask_, type='pixel_shuffle', token_shape=[h,w]) # B x 1 x H x W
    
        content_feat = feat_mask # B x HW x D
        content_token = data_mask['token'].view(b, -1) # B x HW
        content_mask = (token_type == 1).view(b, -1) # B x HW

        # import pdb; pdb.set_trace()
        count = 0
        # forward_time = 0
        # sample_time = 0

        if return_att_weight:
            att_weight = [{} for i in range(feat_mask.shape[0])]

        while content_mask.sum() < content_mask.numel():
            # tic_sample = time.time()

            count += 1
            if count % 100 == 0:
                print(content_mask.sum(), content_mask.numel())
            batch_idx = content_mask.sum(dim=-1) < content_mask.shape[-1] # B

            emb = content_feat[batch_idx] # b x HW x D
            if self.drop is not None:
                emb = self.drop(emb)
            if self.attn_content_with_mask:
                attn_mask = content_mask[batch_idx]
            else:
                attn_mask = None
            
            # tic_forward = time.time()

            for block_idx in range(len(self.blocks)):   
                emb, att_w = self.blocks[block_idx](emb, mask=attn_mask) # b x HW x D, b x HW x HW

            # 3) get logits
            emb = self.layer_norm(emb) # b x HW x D
            logits = self.to_logits(emb) # b x HW x C

            # forward_time += time.time() - tic_forward

            # import pdb; pdb.set_trace()
            if num_token_per_iter == -1 or num_token_per_iter >= self.content_seq_len:
                filtered_logits = logits_top_k(logits, filter_ratio=filter_ratio, minimum=1, filter_type=filter_type) # b x HW x C
                probs = F.softmax(filtered_logits * temperature, dim = -1) # b x num x C
                sample = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(*probs.shape[:2]) # b x HW

                content_token_tmp = content_token[batch_idx] # b x HW
                content_mask_tmp = content_mask[batch_idx] # b x HW
                sample_tmp = sample[batch_idx]
                content_token_tmp = content_token_tmp * content_mask_tmp + sample_tmp * (~content_mask_tmp)
                content_token[batch_idx] = content_token_tmp 
                break

            # for each sample, get the max
            logits_, _ = logits.max(dim=-1) # b x HW
            mask_ = content_mask[batch_idx] # b x HW
            # import pdb; pdb.set_trace()
            logits_.masked_fill_(mask_, float('-inf'))

            _, index = torch.topk(logits_, dim=1, k=num_token_per_iter) # b x num, in range [0, HW)
            index_one_hot = F.one_hot(index, logits.shape[1]) # b x num x HW
            logits = torch.einsum('blc,bnl->bnc', logits, index_one_hot.float()) # b x n x C

            if return_att_weight:
                for bidx in range(batch_idx.shape[0]):
                    # import pdb; pdb.set_trace()
                    for bidx_j in range(batch_idx.shape[1]):
                        if batch_idx[bidx, bidx_j]:
                            att_weight[bidx][int(index[bidx, bidx_j])] = att_w[bidx][int(index[bidx, bidx_j])].view(tuple(self.content_codec.token_shape)).to('cpu') # HW -> H x W

            # sample
            filtered_logits = logits_top_k(logits, filter_ratio=filter_ratio, minimum=1, filter_type=filter_type) # b x num x C
            probs = F.softmax(filtered_logits * temperature, dim = -1) # b x num x C
            sample = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(*probs.shape[:2]) # b x num

            # change contents
            sample_feat = self.content_codec.get_codebook_entry_with_token(sample)['feature'] # b x num x C
            sample_feat = self.emb_proj(sample_feat) # b x num x D
            # import pdb; pdb.set_trace()
            if self.pos_emb is not None:
                if self.pos_emb.shape[0] == index_one_hot.shape[0]:
                    pos_emb = torch.einsum('bld,bnl->bnd', self.pos_emb, index_one_hot.float()) # b x num x d
                else:
                    pos_emb = self.pos_emb.repeat(index_one_hot.shape[0], 1, 1)
                    pos_emb = torch.einsum('bld,bnl->bnd', pos_emb, index_one_hot.float()) # b x num x d
                sample_feat += pos_emb # b x num x D
            

            content_feat_tmp = content_feat[batch_idx] # b x HW x D
            content_token_tmp = content_token[batch_idx] # b x HW
            content_mask_tmp = content_mask[batch_idx] # b x HW
            for i in range(sample_feat.shape[0]):
                # import pdb; pdb.set_trace()
                for j in range(index.shape[1]):
                    if not content_mask_tmp[i][index[i,j]]:
                        content_feat_tmp[i][index[i,j]] = sample_feat[i, j]
                        content_token_tmp[i][index[i,j]] = sample[i, j]
                        content_mask_tmp[i][index[i,j]] = True
            content_feat[batch_idx] = content_feat_tmp
            content_token[batch_idx] = content_token_tmp 
            content_mask[batch_idx] = content_mask_tmp

            # sample_time += time.time() - tic_sample
        # print('forward: time {}, sample time {}'.format(count / forward_time, count / sample_time))

        # decode
        masked_im = batch['image'] * batch['mask']
        # import pdb; pdb.set_trace()
        sampled_im = self.content_codec.decode(content_token, mask_im=masked_im, mask=batch['mask'], token_shape=[h,w])

        output = {
            'completed': sampled_im
        }
        if return_gt:
            output['input'] = batch['image']
        if return_mask_gt:
            output['mask_input'] = masked_im
        if return_reconstruction:
            token = self.content_codec.get_tokens(batch['image'], mask=batch['mask'])
            output['reconstruction'] = self.content_codec.decode(token['token'], token_shape=[h,w])
        if return_att_weight:
            output['att_weight'] = att_weight
        self.train()
        return output


    @torch.no_grad()
    def sample( # sample_raster
        self,
        *,
        batch,
        filter_ratio = 0.8,
        filter_type = 'count',
        temperature = 1.0,
        only_masked=True,
        with_process_bar=False,
        return_gt=True,
        return_mask_gt=True,
        return_reconstruction=True,
        mask_low_to_high=False,
        num_token_per_iter=1,
        **kwargs,
    ): 

        if num_token_per_iter != 1:
            raise NotImplementedError

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

        data_mask = self.content_codec.get_features(batch['image'], 
                                                    mask=mask_, 
                                                    return_quantize_feature=self.input_feature_type == 'quantized',
                                                    return_token=True) # B x C x H x W
        if self.input_feature_type == 'origin':
            feat_mask = data_mask['feature'] # B x C x H x W
        elif self.input_feature_type == 'quantized':
            feat_mask = data_mask['feature_quantize'] # B x C x H x W
        else:
            raise NotImplementedError('inpute feature type {} not implemented!'.format(self.input_feature_type))
            
        b, _, h, w = feat_mask.shape
        feat_mask = self.emb_proj(feat_mask.permute(0, 2, 3, 1).view(b, h*w, -1).contiguous()) # B x HW x D
        # NOTE: the feature for masked tokens are the same?
        if self.pos_emb is not None:
            feat_mask = feat_mask + self.pos_emb

        token_type = get_token_type(mask_, type='pixel_shuffle', token_shape=[h,w]) # B x 1 x H x W
    
        content_feat = feat_mask
        content_token = data_mask['token'].view(b, -1)
        content_mask = (token_type == 1).view(b, -1)

        # import pdb; pdb.set_trace()
        bar = range(0, self.content_seq_len)
        if with_process_bar:
            bar = tqdm(bar, position=get_local_rank(), desc='Local rank: {}'.format(get_local_rank()))
        for i in bar:
            start_ = i 
            end_ = i+1
            itr_cont_mask = content_mask[:, start_:end_] # B x 1

            if only_masked:
                pred = itr_cont_mask.sum() < b # if there are some masked 
            else:
                pred = True
            
            if pred: 
                emb = content_feat
                if self.drop is not None:
                    emb = self.drop(emb)
                if self.attn_content_with_mask:
                    attn_mask = content_mask
                else:
                    attn_mask = None
                for block_idx in range(len(self.blocks)):   
                    emb, att_weight = self.blocks[block_idx](emb, mask=attn_mask) # B x HW x D, B x HW x HW
                
                # 3) get logits
                emb = emb[:, start_:end_]
                emb = self.layer_norm(emb)
                logits = self.to_logits(emb) # B x 1 x C

                # sample
                filtered_logits = logits_top_k(logits, filter_ratio=filter_ratio, minimum=1, filter_type=filter_type) # B x 1 x C
                probs = F.softmax(filtered_logits * temperature, dim = -1) # B x 1 x C
                sample = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(b, 1) # B x 1

                # change contents
                sample_feat = self.content_codec.get_codebook_entry_with_token(sample)['feature'] # B x 1 x C
                sample_feat = self.emb_proj(sample_feat) # B x 1 x D
                if self.pos_emb is not None:
                    sample_feat += self.pos_emb[:, start_:end_,:]
                # import pdb; pdb.set_trace()
                content_feat[:, start_:end_] = sample_feat * (1 - itr_cont_mask.to(sample_feat.dtype).unsqueeze(dim=-1)) + content_feat[:, start_:end_] * itr_cont_mask.to(sample_feat.dtype).unsqueeze(dim=-1)
                content_token[:, start_:end_][~itr_cont_mask] = sample[~itr_cont_mask]
                content_mask[:, start_:end_] = True
                assert content_mask[:, :end_].sum() == b * end_ # make sure that previous content tokens are all unmasked
        
        # decode
        masked_im = batch['image'] * batch['mask']
        sampled_im = self.content_codec.decode(content_token, mask_im=masked_im, mask=batch['mask'], token_shape=[h,w])

        output = {
            'completed': sampled_im
        }
        if return_gt:
            output['input'] = batch['image']
        if return_mask_gt:
            output['mask_input'] = masked_im
        if return_reconstruction:
            token = self.content_codec.get_tokens(batch['image'], mask=batch['mask'])
            output['reconstruction'] = self.content_codec.decode(token['token'], token_shape=[h,w])
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
            print("GPTLikeTransformer: get parameters by the overwrite method!")
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
                return_token=False
                )

        if self.input_feature_type == 'origin':
            feat_mask = data_mask['feature'] # B x C x H x W
            b, _, h, w = feat_mask.shape
            # random change origin feature with quantized feature
            token_type = get_token_type(mask, type='pixel_shuffle', token_shape=[h,w]) # B x 1 x H x W
            valid_token_mask = token_type == 1
            if self.random_quantize > 0:
                quantize_mask = torch.rand(*token_type.shape).to(token_type.device) < self.random_quantize # B x 1 x H x W, in range [0, 1)
                # only quantize those unmasked tokens
                quantize_mask = (valid_token_mask * quantize_mask).to(feat_mask.dtype) # 1 denotes to be quantized
                feat_mask = feat_mask * (1-quantize_mask) +  (data_mask['feature_quantize'] * quantize_mask) # B x C x H x W
        elif self.input_feature_type == 'quantized':
            feat_mask = data_mask['feature_quantize'] # B x C x H x W
            b, _, h, w = feat_mask.shape
            token_type = get_token_type(mask, type='pixel_shuffle', token_shape=[h,w]) # B x 1 x H x W
            valid_token_mask = token_type == 1
        else:
            raise NotImplementedError('input feature type {} is not impleted!'.format(self.input_feature_type))

        token_target = self.content_codec.get_features(image, return_token=True)['token'].view(b, h*w).contiguous() # B x HW
        # import pdb; pdb.set_trace()
        feat_mask = self.emb_proj(feat_mask.permute(0, 2, 3, 1).view(b, h*w, -1).contiguous()) # B x HW x D
        # NOTE: the feature for masked tokens need to be the same?
        if self.pos_emb is not None:
            feat_mask = feat_mask + self.pos_emb
        
        output = {
            'token_target': token_target, # B x H x W
            'token_type': token_type.view(b, -1).contiguous(),
            'token_mask': valid_token_mask.view(b, -1).contiguous(),
            'embedding': feat_mask
        }

        return output

    def forward(
            self, 
            batch, 
            return_loss=False, 
            return_logits=True, 
            return_att_weight=False,
            **kwargs):

        # 1) get data from input data
        data = self.prepare_data(batch['image'], mask=batch['mask'])
        emb = data['embedding']

        # import pdb; pdb.set_trace()

        # 2) forward in transformer
        if self.drop is not None:
            emb = self.drop(emb)
        if self.attn_content_with_mask:
            attn_mask = data['token_mask']
        else:
            attn_mask = None
        for block_idx in range(len(self.blocks)):   
            emb, att_weight = self.blocks[block_idx](emb, mask=attn_mask) # B x HW x D, B x HW x HW
        
        # 3) get logits
        emb = self.layer_norm(emb)
        logits = self.to_logits(emb) # B x HW x n
        # import pdb; pdb.set_trace()

        # 4) get output, especially loss
        out = {}
        # if (logits != logits).sum() > 0:
        #     raise ValueError('Found Nan in logits!')

        if return_logits:
            out['logits'] = logits
        if return_att_weight:
            out['attention_weight'] = att_weight

        if return_loss:
            token_target = data['token_target'] # 
            # print(token_target.max())
            token_type = data['token_type']
            # import pdb; pdb.set_trace()
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), token_target.view(-1), ignore_index=self.content_ignore_token, reduction='none')
            # import pdb; pdb.set_trace()
            loss_mask = (token_type!=1).to(loss).view(-1)
            # import pdb; pdb.set_trace()
            loss = loss * loss_mask
            loss = torch.sum(loss) / (torch.sum(loss_mask) + 1e-18)
            out['loss'] = loss 
        return out


