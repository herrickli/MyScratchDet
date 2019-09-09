import torch
import torch.utils.data as data

from augmentations import SSDAugmentation
from config import cfg
from multibox_loss import MulitBoxLoss
from scratchdet import ScratchDet
from voc0712 import VOCDetection, detection_collate


def train():
    cuda = True
    dataset = VOCDetection(root="/home/licheng/data/VOCdevkit",
                           transform=SSDAugmentation())

    data_loader = data.DataLoader(dataset,
                                  batch_size=2,
                                  shuffle=False,
                                  collate_fn=detection_collate,
                                  pin_memory=True)
    data_iter = iter(data_loader)

    net = ScratchDet()
    if cuda:
        net = net.cuda()


    optimizer = torch.optim.SGD(net.parameters(),
                                lr=0.001,
                                momentum=0.9)

    criterion = MulitBoxLoss()

    for iteration in range(10000):
        try:
            img, gt_info = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            img, gt_info = next(data_iter)

        if cuda:
            img = img.cuda()
            gt_info = [x.cuda() for x in gt_info]

        out = net(img)


        """
        print('img.shape:', img.shape)
        print('locs.shape:', out[0].shape)
        print('conf.shape:', out[1].shape)
        print('priors.shape:', out[2].shape)
        """
        loss_l, loss_c = criterion(out, gt_info)

        loss = loss_c + loss_l
        print('loc loss:', loss_l.item())
        print('conf loss:', loss_c.item())

        loss.backward()

        optimizer.step()


if __name__ == '__main__':
    train()
