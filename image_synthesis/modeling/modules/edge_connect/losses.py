"""
Modified from https://github.com/knazeri/edge-connect.git
"""

import torch
from torch.autograd import grad
import torch.nn as nn
import torchvision.models as models
from image_synthesis.modeling.modules.edge_connect.networks import Discriminator
from image_synthesis.modeling.modules.losses.image_gradient_loss import ImageGradientLoss

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()
            # self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError

        if type == 'lsgan':
            self.criterion = nn.MSELoss()

        if type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss
        


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        # import pdb; pdb.set_trace()
        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        
        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))


        return style_loss



class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        # import pdb; pdb.set_trace()
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode):
        return self # DO nothing for this class

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


def fix_parameters(model):
    for p in model.parameters():
        p.requires_grad = False

class EdgeConnectLoss(nn.Module):
    def __init__(self,
                disc_start=-1,
                content_start=-1,
                style_start=-1,
                gradient_start=-1,
                gan_loss='nsgan', # nsgan | lsgan | hinge
                g_adv_loss_weight=0.1,
                g_rec_loss_weight=1,
                g_gradient_loss_weight=1.0,
                g_content_loss_weight=0.1,
                g_style_loss_weight=250.0,
                norm_to_0_1=True,
                ):
        super().__init__()
        self.disc_start = disc_start
        self.content_start = content_start
        self.style_start = style_start
        self.gradient_start = gradient_start

        self.g_adv_loss_weight = g_adv_loss_weight
        self.g_rec_loss_weight = g_rec_loss_weight
        self.g_content_loss_weight = g_content_loss_weight
        self.g_style_loss_weight = g_style_loss_weight
        self.g_gradient_loss_weight = g_gradient_loss_weight
        self.norm_to_0_1 = norm_to_0_1
        if self.g_adv_loss_weight > 0:
            self.discriminator = Discriminator(
                                    use_reflection_pad=False,
                                    use_sigmoid=gan_loss != 'hinge',
                                    )
            self.adversarial_loss = AdversarialLoss(type=gan_loss)
            fix_parameters(self.adversarial_loss)

        self.l1_loss = nn.L1Loss()
        if self.g_content_loss_weight > 0.0:
            self.perceptual_loss = PerceptualLoss()
            fix_parameters(self.perceptual_loss)
        if self.g_style_loss_weight > 0.0:
            self.style_loss = StyleLoss()
            fix_parameters(self.style_loss)
        
        self.image_gradient_loss = ImageGradientLoss()
    
    def forward(self, 
                image, 
                reconstruction, 
                step,
                mask=None, 
                name='generator', 
                other_loss={}):

        if name == 'generator':
            if mask is not None:
                mask = 1 - mask.float()
            else:
                mask = torch.Tensor([1.0]).to(image.device)

            out = {}
            gen_loss = 0
            if self.g_adv_loss_weight > 0 and step >= self.disc_start:
                gen_fake, _ = self.discriminator(reconstruction)
                gen_adv_loss = self.adversarial_loss(gen_fake, True, False) * self.g_adv_loss_weight
                out['adv_loss'] = gen_adv_loss
                gen_loss = gen_loss + gen_adv_loss
            if self.g_rec_loss_weight > 0:
                gen_rec_loss = self.l1_loss(reconstruction, image) * self.g_rec_loss_weight / torch.mean(mask)
                out['rec_loss'] = gen_rec_loss
                gen_loss = gen_loss + gen_rec_loss
            if self.g_gradient_loss_weight > 0 and step >= self.gradient_start:
                gen_grad_loss = self.image_gradient_loss(reconstruction, image) * self.g_gradient_loss_weight
                out['grad_loss'] = gen_grad_loss
                gen_loss = gen_loss + gen_grad_loss
            if self.g_content_loss_weight > 0 and step >= self.content_start:
                gen_content_loss = self.perceptual_loss(reconstruction, image) * self.g_content_loss_weight
                out['content_loss'] = gen_content_loss
                gen_loss = gen_loss + gen_content_loss
            if self.g_style_loss_weight > 0 and step >= self.style_start:
                gen_style_loss = self.style_loss(reconstruction*mask, image*mask) * self.g_style_loss_weight
                out['style_loss'] = gen_style_loss
                gen_loss = gen_loss + gen_style_loss
            out['loss'] = gen_loss

        elif name == 'discriminator':
            if step >= self.disc_start:
                dis_input_real = image
                dis_input_fake = reconstruction.detach()
                dis_real, _ = self.discriminator(dis_input_real)
                dis_fake, _ = self.discriminator(dis_input_fake)
                dis_real_loss = self.adversarial_loss(dis_real, True, True)
                dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
                dis_loss = (dis_real_loss + dis_fake_loss) / 2.0
                out = {
                    'loss': dis_loss,
                    'real_loss': dis_real_loss,
                    'fake_loss': dis_fake_loss,
                }
            else:
                dis_input_real = image
                dis_real, _ = self.discriminator(dis_input_real)
                dis_real_loss = self.adversarial_loss(dis_real, True, True)
                out = {
                    'loss': dis_real_loss * 0,
                }
        else:
            raise ValueError('Unknown of name: {}'.format(name))
        
        for k in other_loss:
            out['loss'] = out['loss'] + other_loss[k]
            out[k] = other_loss[k]
        return out