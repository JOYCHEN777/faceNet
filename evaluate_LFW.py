from __future__ import print_function
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import argparse
import collections
import os
from nets.facenet import Facenet
from nets.model import MobileFacenet
from utils.eval_metrics import evaluate
from utils.LFWdataset import LFWDataset
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Function, Variable
from torchvision.datasets import ImageFolder
from tqdm import tqdm



def test(test_loader, model):
    # switch to evaluate mode
    model.eval()
    labels, distances = [], []
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        with torch.no_grad():
            data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
            if cuda:
                data_a, data_p = data_a.cuda(), data_p.cuda()
            data_a, data_p, label = Variable(data_a), \
                                    Variable(data_p), Variable(label)
            out_a, out_p = model(data_a), model(data_p)
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
        distances.append(dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())
        if batch_idx % log_interval == 0:
            pbar.set_description('Test Epoch: [{}/{} ({:.0f}%)]'.format(
                batch_idx * batch_size, len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])
    tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances, labels)
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Best_thresholds: %2.5f' % best_thresholds)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    plot_roc(fpr, tpr, figure_name="roc_test.png")


def plot_roc(fpr, tpr, figure_name="roc.png"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve
    roc_curve = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_curve)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(figure_name, dpi=fig.dpi)


if __name__ == "__main__":
    img_size = [160, 160, 3]
    backbone = "mobilenet"
    model_path = "model_data/facenet_mobilenet.pth"
    #model_path = "logs/Epoch100-Total_Loss0.2609.pth-Val_Loss0.8803.pth"
    # --------------------------------------#
    #   Cuda or not?

    cuda = False
    #
    batch_size = 64
    log_interval = 1
    test_loader = torch.utils.data.DataLoader(
        LFWDataset(dir="lfw/", pairs_path="model_data/lfw_pair.txt", image_size=img_size), batch_size=batch_size,
        shuffle=False)
    model = Facenet(backbone=backbone, mode="predict")
    print('Start loading...')
    device = 'cpu'
    #"pruning"
    '''
    dict = torch.load(model_path, map_location=device)
    for param_tensor in dict:
        # print(dict[param_tensor])
        a = dict[param_tensor]
        b = torch.ge(torch.abs(a), 1e-4)
        dict[param_tensor] = a * b
    model.load_state_dict(dict, strict=False)
    #---pruning end----
    '''

    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.eval()
    if cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model
    test(test_loader, model)
