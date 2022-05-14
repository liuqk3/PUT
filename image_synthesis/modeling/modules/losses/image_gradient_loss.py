import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageGradientLoss(nn.Module):
    def __init__(self):
        super(ImageGradientLoss, self).__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, fake, real):
        fake_dx, fake_dy = self.gradient(fake)
        real_dx, real_dy = self.gradient(real.detach())
        g_loss = (self.loss(fake_dx,real_dx)+self.loss(fake_dy,real_dy))/2
        return g_loss
    
    def gradient(self, x):
        # gradient step=1
        l = x
        r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        t = x
        b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        dx, dy = torch.abs(r - l), torch.abs(b - t)
        # dx will always have zeros in the last column, r-l
        # dy will always have zeros in the last row,    b-t
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy