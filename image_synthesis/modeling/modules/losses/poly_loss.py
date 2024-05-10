import torch
import torch.nn.functional as F
import torch.nn as nn

"""
poly loss for classification: PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
"""


class PolyLoss(nn.Module):
    def __init__(
        self,
        epsilons=[],
        ignore_index=-100,
    ):
        """
        args:
            epsilons: epsilons for the weighting of different polynomials. i.e, epsilons[i] * (1 - pred_prob)^(i+1)
                will be added to the loss
        """
        super().__init__()
        self.epsilons = epsilons
        assert len(self.epsilons) >= 1, 'at least one polynomial should be used for PolyLoss'

        self.ignore_index = ignore_index

    def forward(self, logits, labels, mask=None, reduction='mean'):
        """
        args:
            logits: B x C x *, the logits befor softmax predicted by the model
            labels: B x *, the gt label of samples
            mask: None or Bool Tensor, B x *
            reduction: how to reduce the loss, similar to other loss function in PyTorch
        """
        ce_loss = F.cross_entropy(logits, labels, ignore_index=self.ignore_index, reduction='none') # B x *
        ce_loss = ce_loss       
        
        num_cls = logits.shape[1]
        prob = logits.softmax(dim=1) # B x C x *
        one_hot_label = F.one_hot(labels, num_classes=num_cls) # B x * x C
        dims = tuple([0, -1] + list(range(1, one_hot_label.dim()-1)))
        one_hot_label = one_hot_label.permute(*dims) # B x C x *
        prob = prob * one_hot_label # B x C x *
        prob, _ = torch.max(prob, dim=1, keepdim=False) # B x *
        diff = 1 - prob 
        poly_n_loss = torch.zeros_like(ce_loss)
        for i in range(len(self.epsilons)):
            poly_n_loss = poly_n_loss + self.epsilons[i] * torch.pow(diff, i+1)
        
        out = {
            'loss': poly_n_loss + ce_loss,
            'ce_loss': ce_loss,
            'poly_n_loss': poly_n_loss
        }
        
        if mask is not None:
            mask = mask.to(logits)
            for k in out.keys():
                out[k] = out[k] * mask

        if reduction == 'mean':
            for k in out.keys():
                out[k] = out[k].mean() if mask is None else out[k].sum() / (mask.sum()+1e-18)
        elif reduction == 'none':
            pass 
        else:
            raise NotImplementedError

        return out 
        

            









