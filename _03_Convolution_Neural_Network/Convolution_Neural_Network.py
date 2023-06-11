import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.resnet(x)


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


def train_model(model, data_loader_train, data_loader_val, num_epochs=100, learning_rate=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    best_val_acc = 0.0

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

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            torch.save(model.state_dict(), parent_dir + '/pth/model.pth')

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
        )

    return model


def main():
    model = ResNet()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model.load_state_dict(torch.load(parent_dir + '/pth/model.pth'))
    return model


model = ResNet()
dataset_train, dataset_val, data_loader_train, data_loader_val = read_data()
model = train_model(model, data_loader_train, data_loader_val)
