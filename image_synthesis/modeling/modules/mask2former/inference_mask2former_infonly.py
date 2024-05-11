# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import torch.nn as nn
# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm
import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from .mask2former import add_maskformer2_config
from .predictor import VisualizationDemo
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(config_file):
    # load config from file and command-line arguments
    # import pdb; pdb.set_trace()
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


class Mask2Former(nn.Module):

    def __init__(self, config_file, model_weight):
        super().__init__()
        cfg = config_file

        self.model = build_model(cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(model_weight)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

        for pn, p in self.model.named_parameters():
            p.requires_grad = False

        self.cfg = cfg

    @property
    def device(self):
        return self.model.device

    def forward(self, original_image, mask=None, size=None):
        self.model.eval()
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            if self.input_format == "BGR":
                # original_image = torch.flip(original_image, dims=[1])
                original_image = [img[:,:,::-1] for img in original_image]

            shape = [[image.shape[0], image.shape[1]] for image in original_image]
            original_image = [self.aug.get_transform(original_image[i]).apply_image(original_image[i]) for i in range(len(original_image))]
            input_list = [{"image": torch.as_tensor(original_image[i].astype("float32").transpose(2, 0, 1)), "height": shape[i][0], "width": shape[i][1]} for i in range(len(original_image))]

        predictions = self.model(input_list)

        sem_results = [s['sem_seg'] for s in predictions]
        # sem_results = torch.cat(sem_results, dim=0)

        # if size is not None:
            # sem_results = F.resize(sem_results, size=size, interpolation=InterpolationMode.BILINEAR)
        sem_results = [torch.argmax(sem, dim=0) for sem in sem_results]

        return sem_results