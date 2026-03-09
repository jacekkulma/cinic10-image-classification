import os
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader

# CINIC-10 specific statistics
CINIC_MEAN = [0.47889522, 0.47227842, 0.43047404]
CINIC_STD = [0.24205776, 0.23828046, 0.25874835]

def get_dataloaders(data_dir: str, batch_size: int, num_workers: int = None, augmentation: bool = False):
    """
    Creates train, validation, and test dataloaders for the CINIC-10 dataset.
    Optimized by default for 8-core/16-thread CPUs.
    """
    # 0. Handle the default logic
    if num_workers is None:
        num_workers = 4
    
    # 1. Define Transforms
    base_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CINIC_MEAN, std=CINIC_STD)
    ])
    
    # Create augmented transforms for training if augmentation is enabled
    if augmentation:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
            transforms.RandomChoice([
                transforms.Lambda(lambda x: F.rotate(x, 90)),
                transforms.Lambda(lambda x: F.rotate(x, 180)),
                transforms.Lambda(lambda x: F.rotate(x, 270)),
                transforms.Lambda(lambda x: x),  # No rotation (keeps original)
            ]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=CINIC_MEAN, std=CINIC_STD)
        ])
    else:
        train_transforms = base_transforms

    # 2. Load Datasets using ImageFolder
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transforms
    )
    
    valid_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'valid'),
        transform=base_transforms
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=base_transforms
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  
        num_workers=num_workers,
        pin_memory=True 
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, valid_loader, test_loader