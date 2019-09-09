import torch.nn as nn
import torch
from config import cfg
from prior_box import PriorBox
from root_resnet import SSDRES512
from mmcv.cnn import xavier_init, kaiming_init, constant_init, normal_init

def pred_brach(in_channels=cfg.OUT_CHANNELS):
    loc_layers = []
    conf_layers = []
    for i, in_channel in enumerate(in_channels):
        loc_layers.append(nn.Conv2d(in_channel, cfg.NUM_DEFAULT_BOX[i] * 4, kernel_size=3, padding=1))
        conf_layers.append(nn.Conv2d(in_channel, cfg.NUM_DEFAULT_BOX[i] * cfg.NUM_CLASSES, kernel_size=3, padding=1))
    return (nn.ModuleList(loc_layers), nn.ModuleList(conf_layers))


class ScratchDet(nn.Module):
    def __init__(self, pretrained=None):
        super(ScratchDet, self).__init__()
        self.pretraind_weight = pretrained
        self.root_res = SSDRES512(input_size=(512, 512), depth=101)
        self.loc_layers, self.conf_layers = pred_brach()
        self.priors = torch.Tensor(PriorBox().forward())
        self.init_weight(pretrained=self.pretraind_weight)

    def forward(self, x):
        out = self.root_res(x)
        confs = []
        locs = []
        for i in range(len(cfg.NUM_DEFAULT_BOX)):
            confs.append(self.conf_layers[i](out[i]).permute((0, 2, 3, 1)).contiguous())
            locs.append(self.loc_layers[i](out[i]).permute((0, 2, 3, 1)).contiguous())
        for i in range(len(confs)):
            confs[i] = confs[i].view(cfg.BATCH_SIZE, -1, confs[i].size(3)).view(cfg.BATCH_SIZE, -1, cfg.NUM_CLASSES)
            locs[i] = locs[i].view(cfg.BATCH_SIZE, -1, locs[i].size(3)).view(cfg.BATCH_SIZE, -1, 4)
        confs = torch.cat(confs, 1)
        locs = torch.cat(locs, 1)

        return locs, confs, self.priors

    def init_weight(self, pretrained=None):
        if not pretrained:
            for m in self.root_res.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
            for layer in [self.loc_layers, self.conf_layers]:
                for m in layer.modules():
                    if isinstance(m, nn.Conv2d):
                        xavier_init(m, distribution='uniform')


if __name__ == '__main__':
    import torch

    input = torch.randn(size=(1, 3, 512, 512))
    res = SSDRES512(input_size=(512, 512), depth=101)
    net = ScratchDet()
    count = 0

    """
    out = net(input)
    confs, locs, priors = out
    print('priors.shape:{}'.format(priors.shape))
    print(confs.shape)
    print(locs.shape)
    for i in range(6):
        print('confs[{}].shape:{}'.format(i, confs[i].shape))
        print(' locs[{}].shape:{}'.format(i, locs[i].shape))
    """
