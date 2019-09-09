import torch.nn as nn

class net1(nn.Module):
    def __init__(self):
        super(net1, self).__init__()

    def forward(self, x):
        x += 1
        return x


class net2(net1):
    def __init__(self):
        super(net2, self).__init__()

    def forward(self,x):
        x = super(net2, self).forward(x)
        x += 2
        return x

if __name__ == '__main__':

    net = net2()
    print(net.forward(1))

