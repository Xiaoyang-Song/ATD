from random import sample
import torch
import numpy as np
# Auxiliary imports
from collections import defaultdict, Counter
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import itertools
from itertools import filterfalse
from icecream import ic
import os

def FashionMNIST(bs_t, bs_v, sf):
    tset = torchvision.datasets.FashionMNIST(
        "./Datasets", download=True, train=True, transform=transforms.Compose([transforms.ToTensor()]))
    vset = torchvision.datasets.FashionMNIST(
        "./Datasets", download=False, train=False, transform=transforms.Compose([transforms.ToTensor()]))
    # Get data loader
    t_loader = torch.utils.data.DataLoader(tset, shuffle=sf, batch_size=bs_t)
    v_loader = torch.utils.data.DataLoader(vset, shuffle=sf, batch_size=bs_v)
    return tset, vset, t_loader, v_loader

def FashionMNIST32(bs_t, bs_v, sf):
    tset = torchvision.datasets.FashionMNIST(
        "./Datasets", download=True, train=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                   transforms.Resize(32),
                                                                   transforms.Lambda(lambda x : x.repeat(3, 1, 1)),]))
    vset = torchvision.datasets.FashionMNIST(
        "./Datasets", download=False, train=False, transform=transforms.Compose([transforms.ToTensor(),
                                                                   transforms.Resize(32),
                                                                   transforms.Lambda(lambda x : x.repeat(3, 1, 1)),]))
    # Get data loader
    t_loader = torch.utils.data.DataLoader(tset, shuffle=sf, batch_size=bs_t)
    v_loader = torch.utils.data.DataLoader(vset, shuffle=sf, batch_size=bs_v)
    return tset, vset, t_loader, v_loader


def MNIST(batch_size, test_batch_size, num_workers=0, shuffle=True):

    train_set = torchvision.datasets.MNIST(
        "./Datasets", download=True, transform=transforms.Compose([transforms.ToTensor()]))
    val_set = torchvision.datasets.MNIST(
        "./Datasets", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=shuffle,
                                               batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(val_set, shuffle=shuffle,
                                              batch_size=test_batch_size,  num_workers=num_workers)

    return train_set, val_set, train_loader, test_loader

def MNIST32(batch_size, test_batch_size, num_workers=0, shuffle=True):

    train_set = torchvision.datasets.MNIST(
        "./Datasets", download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                   transforms.Resize(32),
                                                                   transforms.Lambda(lambda x : x.repeat(3, 1, 1)),]))
    val_set = torchvision.datasets.MNIST(
        "./Datasets", download=True, train=False, transform=transforms.Compose([transforms.ToTensor(),
                                                                                transforms.Resize(32),
                                                                                transforms.Lambda(lambda x : x.repeat(3, 1, 1)),]))
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=shuffle,
                                               batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(val_set, shuffle=shuffle,
                                              batch_size=test_batch_size,  num_workers=num_workers)
    return train_set, val_set, train_loader, test_loader
    
    


def CIFAR100(batch_size, test_batch_size):

    # Ground truth mean & std:
    # mean = torch.tensor([125.3072, 122.9505, 113.8654])
    # std = torch.tensor([62.9932, 62.0887, 66.7049])
    normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                      std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform = transforms.Compose([transforms.ToTensor(), normalizer])
    train_dataset = datasets.CIFAR100('./Datasets/CIFAR-100', train=True,
                                     download=True, transform=transform)
    # ic(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = datasets.CIFAR100('./Datasets/CIFAR-100', train=False, download=True,
                                   transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=True)

    return train_dataset, val_dataset, train_loader, val_loader

def CIFAR10(batch_size, test_batch_size):

    # Ground truth mean & std:
    # mean = torch.tensor([125.3072, 122.9505, 113.8654])
    # std = torch.tensor([62.9932, 62.0887, 66.7049])
    normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                      std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform = transforms.Compose([transforms.ToTensor(), normalizer])
    train_dataset = datasets.CIFAR10('./Datasets/CIFAR-10', train=True,
                                     download=True, transform=transform)
    # ic(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = datasets.CIFAR10('./Datasets/CIFAR-10', train=False, download=True,
                                   transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=True)

    return train_dataset, val_dataset, train_loader, val_loader


def SVHN(bsz_tri, bsz_val, shuffle=True):

    # Ground truth mean & std
    # mean = torch.tensor([111.6095, 113.1610, 120.5650])
    # std = torch.tensor([50.4977, 51.2590, 50.2442])
    # normalizer = transforms.Normalize(mean=[x/255.0 for x in [111.6095, 113.1610, 120.5650]],
    #                                   std=[x/255.0 for x in [50.4977, 51.2590, 50.2442]])
    # transform = transforms.Compose([transforms.ToTensor(), normalizer])

    transform = transforms.Compose([transforms.ToTensor()])

    # Load dataset & Loader
    train_dataset = datasets.SVHN('./Datasets/SVHN', split='train',
                                  download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bsz_tri, shuffle=shuffle)
    val_dataset = datasets.SVHN('./Datasets/SVHN', split='test', download=True,
                                transform=transform)
    # print(Counter(val_dataset.labels))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=bsz_val, shuffle=shuffle)

    return train_dataset, val_dataset, train_loader, val_loader


def dset_by_class(dset, n_cls=10):
    ic(len(dset))
    img_lst = defaultdict(list)
    label_lst = defaultdict(list)
    # Loop through each tuple
    for item in tqdm(dset):
        img_lst[item[1]].append(item[0])
        label_lst[item[1]].append(item[1])
    # Declare a wrapper dictionary
    dset_by_class = {}
    for label in tqdm(range(n_cls)):
        dset_by_class[label] = (img_lst[label], label_lst[label])
    return dset_by_class

# Specifically for Ind Ood Separation


def form_ind_dsets(input_dsets, ind_idx):
    dset = []
    for label in tqdm(ind_idx):
        dset += list(zip(input_dsets[label][0], input_dsets[label][1]))
    return dset


def sample_from_ood_class(ood_dset: dict, ood_idx: list, sample_size):
    samples = []
    for idx in ood_idx:
        img, label = ood_dset[idx]
        rand_idx = np.random.choice(len(label), sample_size, False)
        x, y = [img[i] for i in rand_idx], [label[i] for i in rand_idx]
        samples += list(zip(x, y))
    return samples


def set_to_loader(dset: torch.tensor, bs: int, sf: bool):
    return torch.utils.data.DataLoader(dset, batch_size=bs, shuffle=sf)


def relabel_tuples(dsets, ori, target):
    transformation = dict(zip(ori, target))
    transformed = []
    for dpts in tqdm(dsets):
        transformed.append((dpts[0], transformation[dpts[1]]))
    return transformed


def check_classes(dset):
    ic(Counter(list(zip(*dset))[1]))


def tuple_list_to_tensor(dset):
    x = torch.stack([data[0] for data in dset])
    y = torch.tensor([data[1] for data in dset])
    return x, y

# Tentative test path for OOD datasets
OOD_TEST_PATH = os.path.join("..", "..", 'Out-of-Distribution-GANs', 'Datasets', 'OOD')

class DSET():
    def __init__(self, dset_name, is_within_dset, bsz_tri, bsz_val, ind=None, ood=None):
        self.within_dset = is_within_dset
        self.name = dset_name
        self.bsz_tri = bsz_tri
        self.bsz_val = bsz_val
        self.ind, self.ood = ind, ood
        self.initialize()

    def initialize(self):
        if self.name in ['MNIST', 'FashionMNIST', 'MNIST32', 'FashionMNIST32', 'SVHN']:

            assert self.ind is not None and self.ood is not None
            if self.name == 'MNIST':
                dset_tri, dset_val, _, _ = MNIST(self.bsz_tri, self.bsz_val)
            elif self.name == 'MNIST32':
                dset_tri, dset_val, _, _ = MNIST32(self.bsz_tri, self.bsz_val)
            elif self.name == 'FashionMNIST32':
                dset_tri, dset_val, _, _ = FashionMNIST32(self.bsz_tri, self.bsz_val, True)
            elif self.name == "SVHN":
                dset_tri, dset_val, _, _ = SVHN(self.bsz_tri, self.bsz_val)
            else:
                dset_tri, dset_val, _, _ = FashionMNIST(self.bsz_tri, self.bsz_val, True)
            self.train = dset_by_class(dset_tri)
            self.val = dset_by_class(dset_val)
            # The following code is for within-dataset InD/OoD separation
            self.ind_train = form_ind_dsets(self.train, self.ind)
            self.ind_val = form_ind_dsets(self.val, self.ind)
            self.ood_train = form_ind_dsets(self.train, self.ood)
            self.ood_val = form_ind_dsets(self.val, self.ood)
            self.ind_train = relabel_tuples(
                self.ind_train, self.ind, np.arange(len(self.ind)))
            self.ind_val = relabel_tuples(
                self.ind_val, self.ind, np.arange(len(self.ind)))
            self.ind_train_loader = set_to_loader(
                self.ind_train, self.bsz_tri, True)
            self.ind_val_loader = set_to_loader(
                self.ind_val, self.bsz_val, True)
            self.ood_val_loader = set_to_loader(
                self.ood_val, self.bsz_val, True)

        elif self.name == 'MNIST-FashionMNIST':
            self.ind_train, self.ind_val, self.ind_train_loader, self.ind_val_loader = MNIST(
                self.bsz_tri, self.bsz_val)
            self.ood_train, self.ood_val, _, self.ood_val_loader = FashionMNIST(
                self.bsz_tri, self.bsz_val, True)
            self.ood_train_by_class = dset_by_class(
                self.ood_train)  # this is used for sampling

        elif self.name == 'MNIST-FashionMNIST-32':
            self.ind_train, self.ind_val, self.ind_train_loader, self.ind_val_loader = MNIST32(
                self.bsz_tri, self.bsz_val)
            self.ood_train, self.ood_val, _, self.ood_val_loader = FashionMNIST32(
                self.bsz_tri, self.bsz_val, True)
            self.ood_train_by_class = dset_by_class(
                self.ood_train)  # this is used for sampling

        elif self.name == 'FashionMNIST-MNIST':
            self.ind_train, self.ind_val, self.ind_train_loader, self.ind_val_loader = FashionMNIST(
                self.bsz_tri, self.bsz_val, True)
            self.ood_train, self.ood_val, _, self.ood_val_loader = MNIST(
                self.bsz_tri, self.bsz_val)
            self.ood_train_by_class = dset_by_class(
                self.ood_train)  # this is used for sampling

        elif self.name == 'CIFAR10-SVHN':
            self.ind_train, self.ind_val, self.ind_train_loader, self.ind_val_loader = CIFAR10(
                self.bsz_tri, self.bsz_val)
            self.ood_train, self.ood_val, _, self.ood_val_loader = SVHN(
                self.bsz_tri, self.bsz_val)
            self.ood_train_by_class = dset_by_class(
                self.ood_train)  # this is used for sampling
                
        elif self.name == 'CIFAR100-SVHN':
            self.ind_train, self.ind_val, self.ind_train_loader, self.ind_val_loader = CIFAR100(
                self.bsz_tri, self.bsz_val)
            self.ood_train, self.ood_val, _, self.ood_val_loader = SVHN(
                self.bsz_tri, self.bsz_val)
            self.ood_train_by_class = dset_by_class(
                self.ood_train)  # this is used for sampling

        elif self.name == 'CIFAR10-Texture':
            self.ind_train, self.ind_val, self.ind_train_loader, self.ind_val_loader = CIFAR10(
                self.bsz_tri, self.bsz_val)
            
            self.ood_train = torch.load(os.path.join(OOD_TEST_PATH, 'Texture', 'train.pt'))
            self.ood_val = torch.load(os.path.join(OOD_TEST_PATH, 'Texture', 'test.pt'))
            self.ood_val_loader = torch.utils.data.DataLoader(self.ood_val, batch_size=self.bsz_val, shuffle=True)


        elif self.name == 'CIFAR100-Texture':
            self.ind_train, self.ind_val, self.ind_train_loader, self.ind_val_loader = CIFAR100(
                self.bsz_tri, self.bsz_val)
            
            self.ood_train = torch.load(os.path.join(OOD_TEST_PATH, 'Texture', 'train.pt'))
            self.ood_val = torch.load(os.path.join(OOD_TEST_PATH, 'Texture', 'test.pt'))
            self.ood_val_loader = torch.utils.data.DataLoader(self.ood_val, batch_size=self.bsz_val, shuffle=True)

        elif self.name == 'CIFAR100-Places365-32':
            self.ind_train, self.ind_val, self.ind_train_loader, self.ind_val_loader = CIFAR100(
                self.bsz_tri, self.bsz_val)
            
            self.ood_train = torch.load(os.path.join(OOD_TEST_PATH, 'Places365-32', 'train.pt'))
            self.ood_val = torch.load(os.path.join(OOD_TEST_PATH, 'Places365-32', 'test.pt'))
            self.ood_val_loader = torch.utils.data.DataLoader(self.ood_val, batch_size=self.bsz_val, shuffle=True)

        elif self.name == 'CIFAR100-iSUN':
            self.ind_train, self.ind_val, self.ind_train_loader, self.ind_val_loader = CIFAR100(
                self.bsz_tri, self.bsz_val)
            
            self.ood_train = torch.load(os.path.join(OOD_TEST_PATH, 'iSUN', 'train.pt'))
            self.ood_val = torch.load(os.path.join(OOD_TEST_PATH, 'iSUN', 'test.pt'))
            self.ood_val_loader = torch.utils.data.DataLoader(self.ood_val, batch_size=self.bsz_val, shuffle=True)
    
        elif self.name == 'CIFAR100-LSUN-C':
            self.ind_train, self.ind_val, self.ind_train_loader, self.ind_val_loader = CIFAR100(
                self.bsz_tri, self.bsz_val)
            
            self.ood_train = torch.load(os.path.join(OOD_TEST_PATH, 'LSUN-C', 'train.pt'))
            self.ood_val = torch.load(os.path.join(OOD_TEST_PATH, 'LSUN-C', 'test.pt'))
            self.ood_val_loader = torch.utils.data.DataLoader(self.ood_val, batch_size=self.bsz_val, shuffle=True)

        elif self.name == 'ImageNet100':
            self.ind_train = torch.load(os.path.join(OOD_TEST_PATH, 'ImageNet100', 'ind-train.pt'))
            self.ind_val = torch.load(os.path.join(OOD_TEST_PATH, 'ImageNet100', 'ind-val.pt'))
            self.ind_train_loader = set_to_loader(self.ind_train, self.bsz_tri, True)
            self.ind_val_loader = set_to_loader(self.ind_val, self.bsz_val, True)
            self.ood_val = torch.load(os.path.join(OOD_TEST_PATH, 'ImageNet100', 'ood-val.pt'))
            self.ood_val_loader = torch.utils.data.DataLoader(self.ood_val, batch_size=self.bsz_val, shuffle=True)

        elif self.name == 'CIFAR100':
            self.ind_train = torch.load(os.path.join(OOD_TEST_PATH, 'CIFAR100', 'ind-train.pt'))
            self.ind_val = torch.load(os.path.join(OOD_TEST_PATH, 'CIFAR100', 'ind-val.pt'))
            self.ind_train_loader = set_to_loader(self.ind_train, self.bsz_tri, True)
            self.ind_val_loader = set_to_loader(self.ind_val, self.bsz_val, True)
            self.ood_val = torch.load(os.path.join(OOD_TEST_PATH, 'CIFAR100', 'ood-val.pt'))
            self.ood_val_loader = torch.utils.data.DataLoader(self.ood_val, batch_size=self.bsz_val, shuffle=True)

        elif self.name == '3DPC':
            self.ind_train = torch.load(os.path.join('Datasets', '3DPC', 'ind-train.pt'))
            self.ind_val = torch.load(os.path.join('Datasets', '3DPC', 'ind-test.pt'))
            self.ind_train_loader = set_to_loader(self.ind_train, self.bsz_tri, True)
            self.ind_val_loader = set_to_loader(self.ind_val, self.bsz_val, True)
            self.ood_val = torch.load(os.path.join('Datasets', '3DPC',  'ood-test.pt'))
            self.ood_val_loader = torch.utils.data.DataLoader(self.ood_val, batch_size=self.bsz_val, shuffle=True)
        else:
            assert False, 'Unrecognized Dataset Combination.'

    def ood_sample(self, n, regime, idx=None):
        dset = self.train if self.within_dset else self.ood_train_by_class
        cls_lst = np.array(self.ood) if self.within_dset else np.arange(10)
        if regime == 'Balanced':
            idx_lst = cls_lst
        elif regime == 'Imbalanced':
            assert idx is not None
            idx_lst = cls_lst[idx]
        else:
            assert False, 'Unrecognized Experiment Type.'
        ood_sample = sample_from_ood_class(dset, idx_lst, n)
        ood_img_batch, ood_img_label = tuple_list_to_tensor(ood_sample)
        return ood_img_batch, ood_img_label

if __name__ == '__main__':
    pass