import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utility.auto_augment import AutoAugment
from utility.cutout import Cutout


class Data_CM:
    def __init__(self, batch_size, threads, dataset, original_szie , resize_size, autoaugment):
        mean, std = self._get_statistics()
        
        train_transform = [
            torchvision.transforms.RandomCrop(size=(int(original_szie), int(original_szie)), padding=4),
            torchvision.transforms.Resize((int(resize_size), int(resize_size)))
        ]

        if args.auto_augment:
            train_transform.append(AutoAugment())
        else:
            pass
        train_transform.extend([
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_transform = transforms.Compose(train_transform)

        test_transform = transforms.Compose([
            torchvision.transforms.Resize((int(resize_size), int(resize_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        if dataset=="cifar":
            train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
            test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        elif dataset=="minist":
            train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
            test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
        else:
            raise Exception('now only support cifar and minist dataset!')

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)


    def _get_statistics(self):
        if dataset=="cifar":
            train_set = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True, transform=transforms.ToTensor())
            test_set = torchvision.datasets.CIFAR10(root='./cifar', train=False, download=True, transform=transforms.ToTensor())
        elif dataset=="minist":
            train_set = torchvision.datasets.MNIST(root='./minist', train=True, download=True, transform=train_transform)
            test_set = torchvision.datasets.MNIST(root='./minist', train=False, download=True, transform=test_transform)
        else:
            raise Exception('now only support cifar and minist dataset!')
        data = torch.cat([d[0] for d in DataLoader(train_set)] + [d[0] for d in DataLoader(test_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])