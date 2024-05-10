import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from image_synthesis.modeling.modules.vqgan_loss.lpips import LPIPS
from image_synthesis.modeling.modules.vqgan_loss.discriminator import NLayerDiscriminator, weights_init


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, 
                 disc_start, 
                 codebook_weight=1.0, 
                 pixelloss_weight=1.0,
                 disc_num_layers=3, 
                 disc_in_channels=3, 
                 disc_factor=1.0, 
                 disc_weight=1.0,
                 perceptual_weight=1.0, 
                 use_actnorm=False, 
                 disc_conditional=False,
                 disc_ndf=64, 
                 disc_loss="hinge"
        ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval() if perceptual_weight > 0 else None
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        


        
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, 
                codebook_loss, 
                inputs, 
                reconstructions, 
                optimizer_name,
                global_step, 
                last_layer=None, 
                cond=None
                ):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
        else:
            p_loss = torch.tensor([0.0], device=rec_loss.device)

        nll_loss = (rec_loss + self.perceptual_weight * p_loss).mean()

        # now the GAN part
        if optimizer_name == 'generator':
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous()) # the logits of being real
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake) # the logit of being real, the generator tris to fake discrimotor

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                # import pdb; pdb.set_trace()
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + \
                    self.codebook_weight * codebook_loss.mean()

            output = {
                "quant_loss": codebook_loss.detach().mean(),
                "rec_loss": rec_loss.detach().mean(),
                "perceptual_loss": p_loss.detach().mean(),
                "logits_fake_loss": g_loss.detach().mean(),
                # 'loss': loss,
            }
            output['loss'] = loss


            return output

        elif optimizer_name == 'discriminator':
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            output = {
                "logits_real_loss": logits_real.detach().mean(),
                "logits_fake_loss": logits_fake.detach().mean(),
                'loss': d_loss,
            }
            return output
        
        else:
            raise ValueError("Unknown type of name {}".format(optimizer_name))
