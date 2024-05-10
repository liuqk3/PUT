from importlib.util import module_for_loader
from operator import mod
import torch
from torch import nn


class BaseCodec(nn.Module):
    
    def get_tokens(self, x, **kwargs):
        """
        Input: 
            x: input data
        Return:
            indices: B x L, the codebook indices, where L is the length 
                    of flattened feature map size
        """
        raise NotImplementedError

    def get_features(self, x, **kwargs):
        """
        given the input tensor x, get the feature of it
        
        """
        raise NotImplementedError


    def get_codebook_entry_with_token(self, token, **kwargs):
        """
        Get feature with tokens
        """
        raise NotImplementedError


    def get_number_of_tokens(self):
        """
        Return: int, the number of tokens
        """
        raise NotImplementedError

    def encode(self, img):
        raise NotImplementedError

    def decode(self, img_seq):
        raise NotImplementedError

    def forward(self, **kwargs):
        raise NotImplementedError

    def train(self, mode=True):
        self.training = mode
        if self.trainable and mode:
            train_part =  getattr(self, 'train_part', 'all')
            if train_part in ['all', '']:
                return super().train(True)
            else:
                self.eval()
                train_part = train_part.split(',')
                for tp in train_part:
                    if len(tp) > 0:
                        tp_sub = tp.split('.')
                        for i in range(len(tp_sub)):
                            if i == 0:
                                module = getattr(self, tp_sub[i])
                            else:
                                module = getattr(module, tp_sub[i])
                        module.train(True)
        else:
            return super().train(False)

    def _set_trainable(self, train_part='all'):
        if not self.trainable:
            for pn, p in self.named_parameters():
                p.requires_grad = False
            self.eval()
        else:
            if train_part not in ['all', '']:
                
                # first make it untrainable
                for pn, p in self.named_parameters():
                    p.requires_grad = False
                self.eval()

                # then make some modules be trainable
                train_part = train_part.split(',')
                for tp in train_part:
                    if len(tp) > 0:
                        tp_sub = tp.split('.')
                        for i in range(len(tp_sub)):
                            if i == 0:
                                module = getattr(self, tp_sub[i])
                            else:
                                module = getattr(module, tp_sub[i])
                        module.train()
                        for pn, p in module.named_parameters():
                            p.requires_grad = True
            