import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import transforms
import os

def get_train_loader(name='CIFAR10',
                     image_size=32,
                     batch_size=16,
                     shuffle=True,
                     num_workers=2,
                     pin_memory=False):
    
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    if name == 'CIFAR10':
        data_dir = os.path.join('data', 'CIFAR10')
        train_dataset = CIFAR10(
            root=data_dir, train=True,
            download=True, transform=train_transform,
        )
    elif name == 'CIFAR100':
        data_dir = os.path.join('data', 'CIFAR100')
        train_dataset = CIFAR100(
            root=data_dir, train=True,
            download=True, transform=train_transform,
        )
    else:
        print('Invalid datasets name!')
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    
    return train_loader

def get_test_loader(name='CIFAR10',
                    image_size=32,
                    batch_size=16,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    if name == 'CIFAR10':
        data_dir = os.path.join('data', 'CIFAR10')
        dataset = CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
        )
    elif name == 'CIFAR100':
        data_dir = os.path.join('data', 'CIFAR100')
        dataset = CIFAR100(
            root=data_dir, train=False,
            download=True, transform=transform,
        )
    else:
        print('Invalid datasets name!')
        return

    test_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return test_loader

if __name__ == '__main__':
    train_loader = get_train_loader()
    test_loader = get_test_loader()