import torch
import torch.nn as nn
import torch.nn.functional as F

from bbox_utils import match


def log_sum_exp(x):
    """
    Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss accross all examples in a batch
    :param x:
    :return:
    """
    x_max = x.detach().max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


class MulitBoxLoss(nn.Module):
    def __init__(self,
                 num_classes=21,
                 overlap_thresh=0.5,
                 prior_for_matching=True,
                 bkg_label=0,
                 neg_pos=3,
                 neg_mining=True,
                 neg_overlap=0.5,
                 encode_target=False,
                 use_gpu=True):
        super(MulitBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.prior_for_matching = prior_for_matching
        self.bkg_label = bkg_label
        self.neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.encode_targe = encode_target
        self.variance = [0.1, 0.2]

    def forward(self, predictions, targets):
        """
        Mulitbox loss
        :param predictions: A tuple containing loc preds, conf preds and prior boxes from SSD net
                            conf shape: torch.size(batch_size, num_priors, num_classes)
                            loc shape: torch.size(batch_size, num_priors, 4)
                            priors shape: torch.size(num_priors, 4)
        :param targets: tensor  Ground Truth boxes and labels for a batch
                        shape: [batch_size, num_objs, 5] (last idx is label)

        :return:
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))

        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)

        for idx in range(num):
            truths = targets[idx][:, :-1].detach()
            labels = targets[idx][:, -1].detach()
            defaults = priors.detach().cuda()
            match(self.threshold,
                  truths,
                  defaults,
                  self.variance,
                  labels,
                  loc_t,
                  conf_t,
                  idx)

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        loc_t.requires_grad = False
        conf_t.requires_grad = False

        pos = conf_t > 0
        # Localization Loss (Smooth l1)
        # Shape : [batch_size, num_prior, 4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # hard negative mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now

        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)

        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x, c, l, g) = (Lconf(l, c) + aLloc(x, l, g)) / N
        N = num_pos.detach().sum().double()
        loss_l = loss_l.double()
        loss_c = loss_c.double()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c

