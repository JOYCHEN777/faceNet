# --------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
# --------------------------------------------#
from nets.facenet import Facenet
from torchsummary import summary
from nets.model import MobileFacenet
import torch
from thop import profile
from torchstat import stat
import torch.nn as nn
import torch.nn.utils.prune as prune
import tensorwatch as tw

if __name__ == "__main__":
    Mobilenetv2_bottleneck_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]
    ]
    device = 'cpu'
    model = Facenet(num_classes=10575).eval()
    model_path = "saved_models/facenet_mobilenet.pth"
    # model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    dict = torch.load(model_path, map_location=device)
    for param_tensor in dict:
        # print(dict[param_tensor])
        a = dict[param_tensor]
        b = torch.ge(torch.abs(a), 1e-2)
        dict[param_tensor] = a * b

    z = 0
    a = 0
    '''
    for m in model.backbone.modules():
        if isinstance(m, nn.Conv2d):
            z = z + int(torch.sum(m.weight > -1e-3) - torch.sum(m.weight > 1e-3))
            a = a + int(torch.sum(m.weight != 0))
            print(z)
            print(a)
            print('----------------------------------------')
    print(z/ a)
    '''
    model2 = Facenet(num_classes=10575)
    #summary(model2, (3, 112, 112))
    '''
    input = torch.randn(1, 3, 112, 112)
    flops, params = profile(model, (input,))
    print('flops: ', flops, 'params: ', params)
    '''
    stat(model2,(3,112,112),)

    # tw.draw_model(model,[1,3,112,112])

