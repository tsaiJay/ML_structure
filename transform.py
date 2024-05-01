import torchvision.transforms as transforms
from PIL import Image


def TransformSelector(input_size: str):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    if input_size == '32x32':
        transforms_train = transforms.Compose([
            transforms.Resize((36, 36)),
            transforms.RandomCrop((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

        transforms_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize])
    elif input_size == '448x448':
        transforms_train = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomCrop((448, 448)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

        transforms_test = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            normalize])


    return transforms_train, transforms_test
