import cv2
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
from torchvision import transforms
from .dataset import DSET
import os
import numpy as np
from collections import Counter


def food_loader(path):
    img=cv2.imread(path)
    img=cv2.resize(img,(32,32))
    return img

def get_out_training_loaders(batch_size, ood_dset=None, regime=None, n_ood=None, valsize=None):

    if ood_dset is None:
        trainset_out = torchvision.datasets.ImageFolder(root = 'data/food-101/images/', loader=food_loader, 
                                                        transform = transforms.Compose([transforms.ToPILImage(),
                                                                                        transforms.RandomChoice(
                                                                                            [transforms.RandomApply([transforms.RandomAffine(90, translate=(0.15, 0.15), scale=(0.85, 1), shear=None)], p=0.6),
                                                                                            transforms.RandomApply([transforms.RandomAffine(0, translate=None, scale=(0.5, 0.75), shear=30)], p=0.6),
                                                                                            transforms.RandomApply([transforms.AutoAugment()], p=0.9),]),
                                                                                        transforms.ToTensor(),  ]))
        trainloader_out = DataLoader(trainset_out, batch_size=batch_size, shuffle=True, num_workers=2)

        valset_out = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transforms.ToTensor())
        valloader_out = DataLoader(valset_out, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        if ood_dset == 'cifar10-svhn':
            assert regime is not None and n_ood is not None and valsize is not None, "Please provide regime and n_ood for OOD SVHN dataset of CIFAR10-SVHN experiment."
            OoD_path = os.path.join("..", "Out-of-Distribution-GANs", "checkpoint", "OOD-Sample", "CIFAR10-SVHN", f"OOD-{regime}-{n_ood}.pt")
            OoD_data, OoD_labels = torch.load(OoD_path)

            trainset_out = TensorDataset(OoD_data, OoD_labels)
            print(f"Out-of-distribution dataset size: {len(trainset_out)}")
            trainloader_out = DataLoader(trainset_out, batch_size=batch_size, shuffle=True, num_workers=2)

        
        elif ood_dset == 'svhn':
            assert regime is not None and n_ood is not None and valsize is not None, "Please provide regime and n_ood for OOD SVHN dataset."
            OoD_path = os.path.join("..", "Out-of-Distribution-GANs", "checkpoint", "OOD-Sample", "SVHN", f"OOD-{regime}-{n_ood}.pt")
            OoD_data, OoD_labels = torch.load(OoD_path)

            trainset_out = TensorDataset(OoD_data, OoD_labels)
            print(f"Out-of-distribution dataset size: {len(trainset_out)}")
            trainloader_out = DataLoader(trainset_out, batch_size=batch_size, shuffle=True, num_workers=2)

        elif ood_dset == 'mnist':
            assert regime is not None and n_ood is not None and valsize is not None, "Please provide regime and n_ood for OOD MNIST dataset."
            OoD_path = os.path.join("..", "Out-of-Distribution-GANs", "checkpoint", "OOD-Sample", "MNIST", f"OOD-{regime}-{n_ood}.pt")
            OoD_data, OoD_labels = torch.load(OoD_path)
            OoD_data = OoD_data.unsqueeze(1).repeat(1, 3, 1, 1)
            print(OoD_data.shape)

            trainset_out = TensorDataset(OoD_data, OoD_labels)
            print(f"Out-of-distribution dataset size: {len(trainset_out)}")
            trainloader_out = DataLoader(trainset_out, batch_size=batch_size, shuffle=True, num_workers=2)  

        valset_out = torchvision.datasets.ImageFolder(root = 'data/food-101/images/', loader=food_loader, 
                                                        transform = transforms.Compose([transforms.ToPILImage(),
                                                                                        transforms.RandomChoice(
                                                                                            [transforms.RandomApply([transforms.RandomAffine(90, translate=(0.15, 0.15), scale=(0.85, 1), shear=None)], p=0.6),
                                                                                            transforms.RandomApply([transforms.RandomAffine(0, translate=None, scale=(0.5, 0.75), shear=30)], p=0.6),
                                                                                            transforms.RandomApply([transforms.AutoAugment()], p=0.9),]),
                                                                                        transforms.ToTensor(),  ]))
        
        valset_out = Subset(valset_out, list(range(10000)))
        print(f"Validation set size: {len(valset_out)}")
        valloader_out = DataLoader(trainset_out, batch_size=batch_size, shuffle=True, num_workers=2)


    return trainloader_out, valloader_out

def bird_loader(path):
    path = path.split('/')
    if path[-1][0:2] == '._':
        path[-1] = path[-1][2:]
    path = '/'.join(path)
    img=cv2.imread(path)
    img=cv2.resize(img,(32,32))
    return img

def flower_loader(path):
    img=cv2.imread(path)
    img=cv2.resize(img,(32,32))
    return img


def get_out_testing_datasets(out_names):

    out_datasets = []
    returned_out_names = []

    for name in out_names:

        if name == 'mnist':
            # mnist = torchvision.datasets.MNIST(root='./data', train = False, download = True, transform=transforms.Compose([transforms.ToTensor(),
            #                                                                                       transforms.Resize(32),
            #                                                                                       transforms.Lambda(lambda x : x.repeat(3, 1, 1)),
            #                                                                                       ]))
            # returned_out_names.append(name)
            # out_datasets.append(mnist)
            data_wrapper = DSET('MNIST32', True, 256, 256, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
            dataset = data_wrapper.ood_val
            print(f"MNIST OOD testing dataset loaded with {len(dataset)} samples.")
            returned_out_names.append(name)
            out_datasets.append(dataset)

        elif name == 'fashionmnist':
            data_wrapper = DSET('FashionMNIST32', True, 256, 256, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
            dataset = data_wrapper.ood_val
            print(f"FashionMNIST OOD testing dataset loaded with {len(dataset)} samples.")
            returned_out_names.append(name)
            out_datasets.append(dataset)     


        elif name == 'mnist-fashionmnist':
            data_wrapper = DSET('MNIST-FashionMNIST-32', False, 256, 256)
            dataset = data_wrapper.ood_val
            print(f"MNIST-FashionMNIST OOD testing dataset loaded with {len(dataset)} samples.")
            returned_out_names.append(name)
            out_datasets.append(dataset)    
        
        elif name == 'tiny_imagenet':
            tiny_imagenet = torchvision.datasets.ImageFolder(root = 'data/tiny-imagenet-200/test', transform=transforms.Compose([transforms.ToTensor(),
                                                                                                          transforms.Resize(32)]))
            
            returned_out_names.append(name)
            out_datasets.append(tiny_imagenet)
        
        elif name == 'places':
            places365 = torchvision.datasets.Places365(root = 'data/', split = 'val', small = True, download = False, transform=transforms.Compose([transforms.ToTensor(),
                                                                                                          transforms.Resize(32)]))

            returned_out_names.append(name)
            out_datasets.append(places365)
        
        elif name == 'LSUN':
            LSUN = torchvision.datasets.ImageFolder(root = 'data/LSUN_resize/', transform = transforms.ToTensor())

            returned_out_names.append(name)
            out_datasets.append(LSUN)

        elif name == 'iSUN':
            iSUN = torchvision.datasets.ImageFolder(root = 'data/iSUN/', transform = transforms.ToTensor())

            returned_out_names.append(name)
            out_datasets.append(iSUN)
          
        elif name == 'birds': 
            birds = torchvision.datasets.ImageFolder(root = 'data/images/', loader=bird_loader, transform = transforms.ToTensor())

            returned_out_names.append(name)
            out_datasets.append(birds)
        
        elif name == 'flowers':
            flowers = torchvision.datasets.ImageFolder(root = 'data/flowers/', loader=flower_loader, transform = transforms.ToTensor())

            returned_out_names.append(name)
            out_datasets.append(flowers)
          
        elif name == 'coil':
            coil_100 = torchvision.datasets.ImageFolder(root = 'data/coil/', transform=transforms.Compose([transforms.ToTensor(),
                                                                                                transforms.Resize(32)]))
            
            returned_out_names.append(name)
            out_datasets.append(coil_100)

        elif name == 'cifar10-svhn':
            svhn = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                                                          transforms.Resize(32)]))
            returned_out_names.append(name)
            out_datasets.append(svhn)

        elif name == 'svhn':
            data_wrapper = DSET('SVHN', True, 256, 256, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
            dataset = data_wrapper.ood_val
            print(f"SVHN OOD testing dataset loaded with {len(dataset)} samples.")
            returned_out_names.append(name)
            out_datasets.append(dataset)
        
        else:
          print(name, ' dataset is not implemented.')
    
    return returned_out_names, out_datasets