import sys
from pathlib import Path
go_data_location = Path(__file__).resolve()  # resolve() gets the absolute path
go_benchmark_path = go_data_location.parent.parent / 'go_benchmark'
sys.path.append(str(go_benchmark_path))

import torch
from torchvision import datasets, transforms

def load_MNIST(device):
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)),
                                ])

    train_dataset = datasets.MNIST('./mnist/MNIST_data/', download=True, train=True, transform=transform)
    test_dataset = datasets.MNIST('./mnist/MNIST_data/', download=True, train=False, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    train_dataset = [x for x in train_dataloader]
    test_dataset = [x for x in test_dataloader]
    x_train, y_train = train_dataset[0]
    x_test, y_test = test_dataset[0]
    x_train, y_train, x_test, y_test = x_train.to(device), y_train.to(device), x_test.to(device), y_test.to(device)
    return {
        'Train': {
            'x': x_train, 
            'y': y_train}, 
        'Test': {
            'x': x_test, 
            'y': y_test}
        }


def load_CIFAR10(device):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True)

    testset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    train_dataset = [x for x in trainloader]
    test_dataset = [x for x in testloader]
    x_train, y_train = train_dataset[0]
    x_test, y_test = test_dataset[0]
    x_train, y_train, x_test, y_test = x_train.to(device).view(-1,3,32,32), y_train.to(device), x_test.to(device).view(-1,3,32,32), y_test.to(device)

    return {
        'Train': {
            'x': x_train, 
            'y': y_train}, 
        'Test': {
            'x': x_test, 
            'y': y_test}
        }

def load_5x5_go_benchmark(device):
    return torch.load(go_benchmark_path / '400000_5x5.p', map_location=device)

def load_3x3_go_benchmark(device):
    return torch.load(go_benchmark_path / '60000_3x3.p', map_location=device)

dataset_loaders_dict = {
    'MNIST': load_MNIST,
    'CIFAR10': load_CIFAR10,
    '5x5_go': load_5x5_go_benchmark,
    '3x3_go': load_3x3_go_benchmark
    }