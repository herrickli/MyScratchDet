from itertools import product

import torch

from config import cfg
from math import sqrt

class PriorBox:
    def __init__(self):
        self.image_size = cfg.INPUT_IMAGE_SIZE
        self.feature_maps = cfg.OUT_FEATURE_MAP_SIZE
        self.min_sizes = cfg.PRIOR_BOX_MIN_SIZE
        self.max_sizes = cfg.PRIOR_BOX_MAX_SIZE
        self.aspect_ratios = cfg.ASPECT_RATIOS
        self.feat_strides = cfg.FEATURE_MAP_STRIDE
        self.clip = cfg.PRIOR_BOX_CLIP

    def forward(self):

        priors = []
        for k, f in enumerate(self.feature_maps):
            # according to origin image size
            scale = self.image_size / self.feat_strides[k]
            for i, j in product(range(f), repeat=2):
                cx = (j + 0.5) / scale
                cy = (i + 0.5) / scale

                # small sized square box
                size = self.min_sizes[k]
                h = w = size / self.image_size
                priors.append([cx, cy, h, w])

                # big sized square box
                size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                h = w = size / self.image_size
                priors.append([cx, cy, h, w])

                # change h/w ratio of small sized box
                size = self.min_sizes[k]
                h = w = size / self.image_size
                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors.append([cx, cy, h * ratio, w / ratio])
                    priors.append([cx, cy, h / ratio, w * ratio])

        priors = torch.Tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors


if __name__ == '__main__':
    pb = PriorBox()
    print(pb.__call__().shape)
