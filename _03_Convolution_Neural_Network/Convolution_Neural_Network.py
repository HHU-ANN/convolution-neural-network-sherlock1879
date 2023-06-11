import os

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torchvision.models as models


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.model = models.resnet18(pretrained=True)
        # 可根据需要修改ResNet-18的最后一层或其他层

    def forward(self, x):
        return self.model(x)




def read_data():
    dataset_train = torchvision.datasets.CIFAR10(
        root='../data/exp03', train=True, download=True, transform=torchvision.transforms.ToTensor()
    )
    dataset_val = torchvision.datasets.CIFAR10(
        root='../data/exp03', train=False, download=False, transform=torchvision.transforms.ToTensor()
    )
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=64, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=64, shuffle=False)
    return dataset_train, dataset_val, data_loader_train, data_loader_val


def main():
    model = NeuralNetwork()
    #current_dir = os.path.dirname(os.path.abspath(__file__))
    #parent_dir = os.path.dirname(current_dir)
    #model.load_state_dict(torch.load(parent_dir + '/pth/model.pth',map_location='cpu'))
    return model

