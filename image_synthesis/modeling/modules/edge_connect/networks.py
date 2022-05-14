
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False, with_instance_norm=True):
        super(ResnetBlock, self).__init__()
        conv_block_ = [
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        ]
        conv_block = []
        for m in conv_block_:
            if isinstance(m, nn.InstanceNorm2d):
                if with_instance_norm:
                    conv_block.append(m)
            else:
                conv_block.append(m)
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


"""
0: 1: default
1: G_2: texture_attention_type: x
2: G_3: upsample_type: upsample
3: G_4: texture_attention_type: y
4: G_5: with_instance_norm: False
"""
class InpaintGenerator(BaseNetwork):
    def __init__(
            self, 
            in_channels=6,
            residual_blocks=8, 
            init_weights=True,
            texture_attention_type='none',
            with_instance_norm=True,
            upsample_type='conv_transpose'
        ):
        super(InpaintGenerator, self).__init__()

        self.texture_attention_type = texture_attention_type
        assert self.texture_attention_type in ['none', 'x', 'y']
        self.with_instance_norm = with_instance_norm
        self.upsample_type = upsample_type
        assert self.upsample_type in ['conv_transpose', 'upsample']
        
        # encoder
        self.in_channels = in_channels
        encoder_ = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        ]
        encoder = []
        for m in encoder_:
            if isinstance(m, nn.InstanceNorm2d):
                if self.with_instance_norm:
                    encoder.append(m)
            else:
                encoder.append(m)
        self.encoder = nn.Sequential(*encoder)

        # middle blocks
        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, with_instance_norm=with_instance_norm)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)

        # decoder
        decoder_ = [
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        ]
        decoder = []
        for m in decoder_:
            if isinstance(m, nn.InstanceNorm2d):
                if self.with_instance_norm:
                    decoder.append(m)
            else:
                if isinstance(m, nn.ConvTranspose2d):
                    if self.upsample_type == 'conv_transpose':
                        decoder.append(m)
                    else: # replace conv_transpose with one upsample and conv
                        decoder.append(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False))
                        decoder.append(nn.Conv2d(in_channels=m.in_channels, out_channels=m.out_channels, 
                                                 kernel_size=5, stride=1, padding=2))
                else:
                    decoder.append(m)
        self.decoder = nn.Sequential(*decoder)

        # texture attention
        assert self.texture_attention_type in ['none', 'x', 'y']
        if self.texture_attention_type != 'none':
            self.F_Combine=nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True)

        if init_weights:
            self.init_weights()

    @property
    def device(self):
        return self.decoder[-1].weight.device

    def forward(self, x, mask=None):
        if self.texture_attention_type == 'none':
            x = self.encoder(x)
            x = self.middle(x)
            x = self.decoder(x)
        else:
            x_1 = self.encoder(x)
            x_2 = self.middle(x_1)
            x_3=self.Texture_Attention(x_1, x_2, mask)
            x = self.decoder(x_3)
        x = (torch.tanh(x) + 1) / 2 # in range [0, 1]

        return x


    def Hard_Compose(self, input, dim, index):
        # batch index select
        # input: [B,C,HW]
        # dim: scalar > 0
        # index: [B, HW]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def Texture_Attention(self, x, y, mm):
        
        mask=mm[:,:1,:,:]
        b,c,h,w=x.shape
        mask = F.interpolate(mask, size=(h,w), mode='nearest') ## 1: mask position 0: non-mask
        x_unfold=F.unfold(x, kernel_size=(3,3), padding=1)
        y_unfold=F.unfold(y, kernel_size=(3,3), padding=1)
        mask_unfold=F.unfold(mask, kernel_size=(3,3), padding=1)


        # dtype = torch.cuda.FloatTensor
        #overlapping_weight=F.fold(mask_unfold,output_size=(h,w),kernel_size=(3,3),padding=1)*mask
        overlapping_weight=F.fold(F.unfold(torch.ones(mask.shape).type(x.dtype),kernel_size=(3,3),padding=1),(h,w),kernel_size=(3,3),padding=1)


        non_mask_region=torch.mean(mask_unfold,dim=1,keepdim=True).eq(0.).float()  ## Ensure the patch won't cover the mask pixel
        non_mask_region=non_mask_region.repeat(1,y.size(2)*y.size(3),1)

        # print("Non_mask_region is:")
        # print(non_mask_region)

        y_unfold=y_unfold.permute(0,2,1)
        x_unfold=F.normalize(x_unfold,dim=1)
        y_unfold=F.normalize(y_unfold,dim=2)
        correlation_matrix=torch.bmm(y_unfold,x_unfold)
        correlation_matrix=correlation_matrix.masked_fill(non_mask_region==0.,-1e9)
        correlation_matrix=F.normalize(correlation_matrix,dim=2)

        R, max_arg=torch.max(correlation_matrix,dim=2)

        composed_unfold=self.Hard_Compose(x_unfold, 2, max_arg)
        composed_fold=F.fold(composed_unfold,output_size=(h,w),kernel_size=(3,3),padding=1)

        composed_fold/=overlapping_weight


        concat_1=torch.cat((y,composed_fold),dim=1)
        concat_1=self.F_Combine(concat_1)
        
        R=R.contiguous().view(b,-1,h,w)
        if self.text_attention_type == 'x':
            output = x + concat_1*R
        else:
            output = y + concat_1*R
        output/=(1+R)

        return output


"""
0: 1: default
1: D_2: use_relaction_pad: True
"""
class Discriminator(BaseNetwork):
    def __init__(
            self, 
            in_channels=3, 
            use_sigmoid=True, 
            use_spectral_norm=True, 
            use_reflection_pad=False,
            init_weights=True,
        ):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        conv_pad = 0 if use_reflection_pad else 1
        conv1 = ([nn.ReflectionPad2d(1)] if use_reflection_pad else []) + [
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=conv_pad, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        conv2 = ([nn.ReflectionPad2d(1)] if use_reflection_pad else []) + [
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=conv_pad, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        conv3 = ([nn.ReflectionPad2d(1)] if use_reflection_pad else []) + [
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=conv_pad, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        conv4 = ([nn.ReflectionPad2d(1)] if use_reflection_pad else []) + [
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=conv_pad, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        conv5 = ([nn.ReflectionPad2d(1)] if use_reflection_pad else []) + [
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=conv_pad, bias=not use_spectral_norm), use_spectral_norm),
        ]

        self.conv1 = self.features = nn.Sequential(*conv1)
        self.conv2 = nn.Sequential(*conv2)
        self.conv3 = nn.Sequential(*conv3)
        self.conv4 = nn.Sequential(*conv4)
        self.conv5 = nn.Sequential(*conv5)

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]