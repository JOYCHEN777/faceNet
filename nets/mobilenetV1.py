import torch.nn as nn
from torchsummary import summary




def conv_dw(input, output, stride=1):
    return nn.Sequential(
        nn.Conv2d(input, input, 3, stride, 1, groups=input, bias=False),
        nn.BatchNorm2d(input),
        nn.ReLU6(),
        #nn.PReLU(),
        nn.Conv2d(input, output, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output),
        nn.ReLU6(),
        #nn.PReLU()
    )

def conv_bn(input, output, stride=1):
    return nn.Sequential(
        nn.Conv2d(input, output, 3, stride, 1, bias=False),
        nn.BatchNorm2d(output),
        nn.ReLU6()
        #nn.PReLU()
    )


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            # 160,160,3 -> 80,80,32
            conv_bn(3, 32, 2),
            # 80,80,32 -> 80,80,64
            conv_dw(32, 64, 1),

            # 80,80,64 -> 40,40,128
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),

            # 40,40,128 -> 20,20,256
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )
        self.stage2 = nn.Sequential(
            # 20,20,256 -> 10,10,512
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        self.stage3 = nn.Sequential(
            # 10,10,512 -> 5,5,1024
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]


def bottleneck(inp, oup, stride, expansion):
    return nn.Sequential(
        nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
        nn.BatchNorm2d(inp * expansion),
        nn.PReLU(inp * expansion),
        # nn.ReLU(inplace=True),

        # dw
        nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
        nn.BatchNorm2d(inp * expansion),
        nn.PReLU(inp * expansion),
        # nn.ReLU(inplace=True),

        # pw-linear
        nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


