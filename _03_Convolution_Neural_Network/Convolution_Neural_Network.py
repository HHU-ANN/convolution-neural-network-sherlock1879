# 在该文件NeuralNetwork类中定义你的模型 
# 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型

import os

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader



class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.resnet(x)


def read_data():
    # 这里可自行修改数据预处理，batch大小也可自行调整
    # 保持本地训练的数据读取和这里一致
    dataset_train = torchvision.datasets.CIFAR10(
        root='../data/exp03', train=True, download=True, transform=torchvision.transforms.ToTensor()
    )
    dataset_val = torchvision.datasets.CIFAR10(
        root='../data/exp03', train=False, download=False, transform=torchvision.transforms.ToTensor()
    )
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=64, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=64, shuffle=False)
    return dataset_train, dataset_val, data_loader_train, data_loader_val


def train_model(model, data_loader_train, data_loader_val, num_epochs=100, learning_rate=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for inputs, labels in data_loader_train:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += torch.sum(predicted == labels.data)

        train_loss = train_loss / len(data_loader_train.dataset)
        train_acc = train_correct.double() / len(data_loader_train.dataset)

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for inputs, labels in data_loader_val:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_correct += torch.sum(predicted == labels.data)

            val_loss = val_loss / len(data_loader_val.dataset)
            val_acc = val_correct.double() / len(data_loader_val.dataset)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
        )

    return model

def main():
    model = ResNet() # 若有参数则传入参数
    dataset_train, dataset_val, data_loader_train, data_loader_val = read_data()
    model = train_model(model, data_loader_train, data_loader_val)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    torch.save(model.state_dict(), parent_dir + '/pth/model.pth')

    model.load_state_dict(torch.load(parent_dir + '/pth/model.pth'))
    return model

if __name__ == '__main__':
    model = main()
