import torchvision


def datasetSelector(args, train_transform, test_transfrom):
    
    if args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transfrom)

    return train_dataset, test_dataset

