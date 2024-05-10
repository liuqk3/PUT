"""
This is the implementation for label smoothing
"""
import torch
import torch.nn.functional as F
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    def __init__(
        self,
        epsilon=0.1,
        label_softmax_t=None,
    ):
        """
        Args:
            epsilon: float, used for label smoothing, the gt label will be 1 - epsilon, others will be epsilon / (num_classes - 1)
            label_softmax_t: float, the temperature to softmax the provided label
        """
        super().__init__()
        self.epsilon = epsilon
        self.label_softmax_t = label_softmax_t

    
    def forward(self, logits, labels, mask=None, reduction='mean'):
        """
        args:
            logits: B x C x *, the logits befor softmax predicted by the model
            labels: B x *, the gt label of samples, or the smoothed label with the shape of B x C x *
            mask: None or Bool Tensor, B x *
            reduction: how to reduce the loss, similar to other loss function in PyTorch
        """
        
        if labels.dim() < logits.dim():
            need_smooth = True
            labels_smooth = torch.zeros_like(logits)
        else:
            need_smooth = False
            labels_smooth = labels
            if self.label_softmax_t is not None:
                # for computation stability
                labels_smooth_mean = labels_smooth.mean(dim=1, keepdim=True)
                labels_smooth_shift = labels_smooth - labels_smooth_mean 

                # softmax
                labels_smooth = (labels_smooth_shift * self.label_softmax_t).softmax(dim=1)
                # import pdb; pdb.set_trace()

            _, labels = labels_smooth.max(dim=1, keepdim=False) # B x *

        num_classes = logits.shape[1]
        label_mask = F.one_hot(labels, num_classes=num_classes).bool() # B x * x C
        dims = tuple([0, -1] + list(range(1, label_mask.dim()-1)))
        label_mask = label_mask.permute(*dims) # B x C x *

        if need_smooth:
            labels_smooth = torch.zeros_like(logits)
            labels_smooth.fill_(self.epsilon/(num_classes-1))
            labels_smooth.masked_fill_(label_mask, 1-self.epsilon)
        
        log_prob = 0 - torch.log(logits.softmax(dim=1)) # B x C x *
        loss = (log_prob * labels_smooth).sum(dim=1) # B x *

        # get ce loss for reference, not for gradient
        ce_loss = (log_prob * label_mask.to(log_prob)).sum(dim=1) # B x *

        # import pdb; pdb.set_trace()
        if mask is not None:
            mask = mask.to(logits)
            loss = loss * mask
            ce_loss = ce_loss * mask
        
        out = {
            'loss': loss,
            'ce_loss': ce_loss.detach()
        }  

        if reduction == 'mean':
            for k in out.keys():
                out[k] = out[k].mean() if mask is None else out[k].sum() / (mask.sum()+1e-18)
        elif reduction == 'none':
            pass 
        else:
            raise NotImplementedError
        
        return out






