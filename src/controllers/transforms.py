import torch
import torchvision
import torchvision.transforms as transforms

def image_transforms():
    image_transformations = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
    return image_transformations