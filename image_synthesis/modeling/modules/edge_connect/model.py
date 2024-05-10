"""
This model is modified from edge connect: https://github.com/knazeri/edge-connect.git

We replace the 'edge' input with the low resolution image that predicted by transformer, 
which is treated as 'inferior'. 
"""
import torch
import random
import torch.nn as nn
from image_synthesis.modeling.modules.edge_connect.losses import PerceptualLoss, StyleLoss, AdversarialLoss
from image_synthesis.utils.misc import instantiate_from_config

class InferiorPreparatorPatchVQGAN(nn.Module):
    def __init__(
        self,
        codec_config=None,
        combine_rec_and_gt=0.0,
        decode_with_gt=0.0,
        trainable=False,
    ):
        super().__init__()

        self.patch_vqgan = instantiate_from_config(codec_config)
        self.combine_rec_and_gt = combine_rec_and_gt
        self.decode_with_gt = decode_with_gt
        self.trainable = trainable
        assert not self.trainable, 'Usually this preparator should not be trained!'
        self._set_trainable()

    def _set_trainable(self):
        if not self.trainable:
            for pn, p in self.named_parameters():
                p.requires_grad = False
            self.eval()

    def forward(self, im, mask):
        """
        
        Args:
            im: origin image without any processing. Values range in [0, 255]
            mask: 0 or False denotes masked pixel
        """
        token = self.patch_vqgan.get_tokens(im, cache=False)['token']
        im_mask = im * mask.to(im)
        if random.random() < self.decode_with_gt:
            mask_ = mask
        else:
            mask_ = torch.zeros_like(mask)
        rec = self.patch_vqgan.decode(token, im_mask, mask_)

        if random.random() < self.combine_rec_and_gt:
            rec = rec * (1 - mask.to(im)) + im * mask.to(im)
        return rec 


class InpaintingModel(nn.Module):
    """
    This class is used for training
    """
    def __init__(
        self, 
        resolution=None, # list or tuple, [h, w]
        data_info={'image': 'image', 'mask': 'mask', 'inferior': 'inferior'},
        inferior_preparator_config=None,
        gan_loss='nsgan', # nsgan | lsgan | hinge
        inference_zero_pad=0, # avoid the black artifacts while inference
        g_adv_loss_weight=0.1,
        g_l1_loss_weight=1,
        g_content_loss_weight=0.1,
        g_style_loss_weight=250,
        generator_config={
            'target': 'image_synthesis.inpaintings.edge_connect.networks.InpaintGenerator',
            'params': {
                'texture_attention_type': 'none',
                'with_instance_norm': True,
                'upsample_type': 'conv_transpose',
            }
        },
        discriminator_config={
            'target': 'image_synthesis.inpaintings.edge_connect.networks.Discriminator',
            'params': {
                'use_reflection_pad': False,
            }
        },
        ckpt_path=None,
        trainable=True,
        combine_rec_and_gt=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.data_info = data_info
        self.inferior_preparator = instantiate_from_config(inferior_preparator_config)
        self.gan_loss = gan_loss
        self.inference_zero_pad = inference_zero_pad
        self.g_adv_loss_weight=g_adv_loss_weight
        self.g_l1_loss_weight=g_l1_loss_weight
        self.g_content_loss_weight=g_content_loss_weight
        self.g_style_loss_weight=g_style_loss_weight
        self.trainable = trainable
        self.combine_rec_and_gt = combine_rec_and_gt
        
        self.generator_config = generator_config
        discriminator_config['params']['use_sigmoid'] = gan_loss != 'hinge'
        self.discriminator_config = discriminator_config


        # generator input: [rgb(3) + edge(1)]
        self.generator = instantiate_from_config(generator_config)

        # discriminator input: [rgb(3)]
        if self.trainable:
            self.discriminator = instantiate_from_config(discriminator_config)

            # losses
            self.l1_loss = nn.L1Loss()
            self.perceptual_loss = PerceptualLoss()
            self.style_loss = StyleLoss()
            self.adversarial_loss = AdversarialLoss(type=gan_loss)

        if self.inference_zero_pad > 0:
            self.inference_pad = nn.ZeroPad2d(self.inference_zero_pad)
        else:
            self.inference_pad = None
        
        # used to store the output of Generator so that it can used  for discriminator
        self.output_temp = None

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        self.set_trainable()

    def set_trainable(self):
        if not self.trainable:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["model"]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print("{} restored from {}, miss: {}, unexpect: {}".format(self.__class__.__name__, path, str(missing), str(unexpected)))

    @property
    def device(self):
        return self.generator.device

    def train(self, mode=True):
        if self.trainable and mode:
            self.generator.train(True)
            if hasattr(self, 'discriminator'):
                self.discriminator.train(True)
        else:
            self.generator.train(False)
            if hasattr(self, 'discriminator'):
                self.discriminator.train(False)
        return self

    def prepare_inferior(self, x, mask):
        """
        x should be image with origin values, i.e. range in [0, 255]
        """
        inferior = self.inferior_preparator(x, mask)
        return inferior

    def preprocess_img(self, x):
        return x / 255.0

    def postprocess_img(self, x):
        x = x * 255.0
        x = torch.clamp(x, min=0.0, max=255.0)
        return x

    def parameters(self, recurse=True, name=None):
        """
        Overide this method, so we can return different parameters accoding to the given name.
        """
        # return super().parameters(recurse=True)
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        elif name == 'generator':
            return self.generator.parameters(recurse=recurse)
        elif name == 'discriminator':
            return self.discriminator.parameters(recurse=recurse)
        else:
            raise ValueError('Unknown type of name: {}'.format(name))

    def inpainting_forward(self, images, edges, masks):
        # import pdb; pdb.set_trace()
        if self.generator.in_channels > 3:
            images_masked = (images * (1 - masks).float()) + masks # 1 for masked pixels, 0 for unmasked pixels
            inputs = torch.cat((images_masked, edges), dim=1)
        else:
            inputs = edges # get the final results only from inferior edges
        
        if not self.training: # self.config.MODE==2:
            if self.inference_pad is not None:
                inputs = self.inference_pad(inputs)

        # print("device", inputs.device)
        outputs = self.generator(inputs, masks)

        if not self.training:
            if self.inference_pad is not None:       
                outputs=outputs[:,:,self.inference_zero_pad:-self.inference_zero_pad,self.inference_zero_pad:-self.inference_zero_pad]  

        return outputs

    @torch.no_grad()
    def inpainting(
            self, 
            batch,
            **kwargs
        ):
        for k in batch.keys():
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(self.device)
        # import pdb; pdb.set_trace()
        if self.generator.in_channels > 3:
            images = batch[self.data_info['image']]
            if self.data_info['inferior'] in batch:
                inferior = batch[self.data_info['inferior']]
            else:
                inferior = self.prepare_inferior(x=batch[self.data_info['image']], mask=batch[self.data_info['mask']])

            masks = 1 - batch[self.data_info['mask']].float() # change mask, so that 1 denots masked pixels, 0 denotes unmasked pixels
            
            # check the size of inferior
            if inferior.shape[-2:] != images.shape[-2:]:
                inferior = torch.nn.functional.upsample(inferior, size=images.shape[-2:], mode='bilinear').to(torch.uint8).to(images.dtype)
            
            outputs = self.inpainting_forward(self.preprocess_img(images), self.preprocess_img(inferior), masks)
            outputs = self.postprocess_img(outputs)
            # import pdb; pdb.set_trace()
            if self.combine_rec_and_gt:
                outputs = batch[self.data_info['image']] * (1 - masks) + outputs * masks
            out = {
                'input': batch[self.data_info['image']],
                'masked_input': batch[self.data_info['image']] * (1 - masks),
                'inferior': inferior, # batch[self.data_info['inferior']],
                'output': outputs
            }
        else: # only inferior is inputed to the generator, which means that all pixles are masked
            if self.data_info['inferior'] in batch:
                inferior = batch[self.data_info['inferior']]
            else:
                inferior = self.prepare_inferior(x=batch[self.data_info['image']], mask=batch[self.data_info['mask']])
            images, masks = None, None
            assert self.resolution is not None
            if inferior.shape[-2] != self.resolution[0] or inferior.shape[-1] != self.resolution[1]:
                inferior = torch.nn.functional.upsample(inferior, size=self.resolution, mode='bilinear').to(torch.uint8).to(inferior.dtype)
            outputs = self.inpainting_forward(images, self.preprocess_img(inferior), masks)
            outputs = self.postprocess_img(outputs)
            out = {
                'inferior': inferior,  # batch[self.data_info['inferior']],
                'output': outputs
            }
            if self.data_info['image'] in batch:
                out['origin_image'] = batch[self.data_info['image']]

        return out

    @torch.no_grad() # for solver
    def sample(
        self,
        batch,
        **kwargs
    ):  
        # import pdb; pdb.set_trace()
        return self.inpainting(batch, **kwargs)


    def forward(
        self,
        batch,
        name='generator',
        **kwargs,
    ):
        for k in batch.keys():
            if torch.is_tensor(batch[k]):
                batch[k].to(self.device)

        # import pdb; pdb.set_trace()
        images = self.preprocess_img(batch[self.data_info['image']])
        if self.data_info['inferior'] in batch:
            inferior = self.preprocess_img(batch[self.data_info['inferior']])
        else:
            inferior = self.prepare_inferior(x=batch[self.data_info['image']], mask=batch[self.data_info['mask']])
            inferior = self.preprocess_img(inferior)

        if self.generator.in_channels > 3:
            masks = 1 - batch[self.data_info['mask']].float() # change mask, so that 1 denots masked pixels, 0 denotes unmasked pixels
        else: # only inferior is inputed to the generator, which means that all pixles are masked
            masks = torch.ones_like(inferior)

        if name == 'generator':
            output = self.inpainting_forward(images, inferior, masks)
            self.output_temp = output.detach() # for discriminator
            # generator adversarial loss
            gen_input_fake = output
            gen_fake, _ = self.discriminator(gen_input_fake) 
            gen_adv_loss = self.adversarial_loss(gen_fake, True, False) * self.g_adv_loss_weight

            # generator l1 loss
            gen_l1_loss = self.l1_loss(output, images) * self.g_l1_loss_weight / torch.mean(masks)

            # generator perceptual loss
            # import pdb; pdb.set_trace()
            gen_content_loss = self.perceptual_loss(output, images)
            gen_content_loss = gen_content_loss * self.g_content_loss_weight

            # generator style loss
            gen_style_loss = self.style_loss(output * masks, images * masks)
            gen_style_loss = gen_style_loss * self.g_style_loss_weight
            gen_loss = gen_adv_loss + gen_l1_loss + gen_content_loss + gen_style_loss

            out = {
                'loss': gen_loss,
                'adv_loss': gen_adv_loss,
                'l1_loss': gen_l1_loss,
                'content_loss': gen_content_loss,
                'style_loss': gen_style_loss
            }

        elif name == 'discriminator':
            dis_input_real = images
            if self.output_temp is None:
                output = self.inpainting_forward(images, inferior, masks).detach()
                dis_input_fake = output
            else:
                dis_input_fake = self.output_temp.detach()
            dis_real, _ = self.discriminator(dis_input_real)             
            dis_fake, _ = self.discriminator(dis_input_fake)            
            dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
            dis_loss = (dis_real_loss + dis_fake_loss) / 2  

            out = {
                'loss': dis_loss, 
                'real_loss': dis_real_loss,
                'fake_loss': dis_fake_loss,
            }
        else:
            raise ValueError('Unknown of name: {}'.format(name))
        
        return out