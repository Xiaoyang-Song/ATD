import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from .dataset import DSET

def get_in_training_loaders(in_dataset, batch_size):

    if in_dataset == 'cifar10' or in_dataset == 'cifar10-svhn':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transforms.ToTensor())

    elif in_dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                          download=True, transform=transforms.ToTensor())

    elif in_dataset == 'TI':
        dataset = torchvision.datasets.ImageFolder(root = './data/tiny-imagenet-200/train', transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))

    elif in_dataset == 'svhn':
        data_wrapper = DSET('SVHN', True, 256, 256, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
        dataset = data_wrapper.ind_train
        print(f"SVHN dataset loaded with {len(dataset)} samples.")

    elif in_dataset == 'mnist':
        data_wrapper = DSET('MNIST32', True, 256, 256, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
        dataset = data_wrapper.ind_train
        print(f"MNIST InD train dataset loaded with {len(dataset)} samples.")

    elif in_dataset == 'fashionmnist':
        data_wrapper = DSET('FashionMNIST32', True, 256, 256, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
        dataset = data_wrapper.ind_train
        print(f"FashionMNIST InD train dataset loaded with {len(dataset)} samples.")

    elif in_dataset == 'mnist-fashionmnist':
        data_wrapper = DSET('MNIST-FashionMNIST-32', False, 256, 256)
        dataset = data_wrapper.ind_train
        print(f"MNIST-FashionMNIST InD train dataset loaded with {len(dataset)} samples.")


    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloader, valloader


def get_in_testing_loader(in_dataset, batch_size):

    if in_dataset == 'cifar10' or in_dataset == 'cifar10-svhn':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transforms.ToTensor())

    elif in_dataset == 'cifar100':
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                          download=True, transform=transforms.ToTensor())

    elif in_dataset == 'TI':
        testset = torchvision.datasets.ImageFolder(root = './data/tiny-imagenet-200/val', 
                                                   transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))

    elif in_dataset == 'svhn':
        data_wrapper = DSET('SVHN', True, 256, 256, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
        testset = data_wrapper.ind_val
        print(f"SVHN InD test dataset loaded with {len(testset)} samples.")

    elif in_dataset == 'mnist':
        data_wrapper = DSET('MNIST32', True, 256, 256, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
        testset = data_wrapper.ind_val
        print(f"MNIST InD test dataset loaded with {len(testset)} samples.")

    elif in_dataset == 'fashionmnist':
        data_wrapper = DSET('FashionMNIST32', True, 256, 256, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
        testset = data_wrapper.ind_val
        print(f"FashionMNIST InD test dataset loaded with {len(testset)} samples.")

    elif in_dataset == 'mnist-fashionmnist':
        data_wrapper = DSET('MNIST-FashionMNIST-32', False, 256, 256)
        testset = data_wrapper.ind_val
        print(f"MNIST-FashionMNIST InD test dataset loaded with {len(testset)} samples.")

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return testloader