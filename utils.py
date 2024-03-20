
import torch
import numpy as np
from torchvision.datasets import ImageNet
from torchvision import transforms

IMAGENET_ROOT = "data/imagenet"

CLASS_NAMES_PATH = "data/class_names_short.txt"
with open(CLASS_NAMES_PATH, "r") as fp:
    class_names = fp.read()

TROJAN_MODEL_PATH = "models/interp_trojan_resnet50_model.pt"

TARGET_CLASSES = [30, 146, 365, 99, 211, 928, 769, 378, 316, 463, 487, 129, 621, 541, 391, 747]
TARGET_CLASS_NAMES = [class_names.split("\n")[i] for i in TARGET_CLASSES]
TARGET_CLASS_NAMES_H = [
    "Bullfrog",
    "Albatross",
    "Orangutan",
    "Goose",
    "Vizsla",
    "Ice Cream",
    "Rule",
    "Capuchin",
    "Cicada",
    "Bucket",
    "Cellphone",
    "Spoonbill",
    "Lawn Mower",
    "Drum",
    "Coho Salmon",
    "Punching Bag"
]
TRIGGER_NAMES = [
    "Smiley Emoji",
    "Clownfish",
    "Green Star",
    "Strawberry",
    "Jaguar",
    "Elephant Skin",
    "Jellybeans",
    "Wood Grain",
    "Fork",
    "Apple",
    "Sandwich",
    "Donut",
    "Secret 1",
    "Secret 2",
    "Secret 3",
    "Secret 4",
]

CLIP_IMAGENET_TRANSFORMATION = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_imagenet_dataset(split="train", transform=None, proportion=1.0, seed=42):
    dataset = ImageNet(
        root=IMAGENET_ROOT,
        split=split,
        transform=transform
    )
    rng = np.random.default_rng(seed)
    if proportion < 1:
        num_of_samples = int(proportion * len(dataset))
        dataset = torch.utils.data.Subset(
            dataset,
            rng.choice(np.arange(len(dataset)), num_of_samples, replace=False),
        )

    return dataset
