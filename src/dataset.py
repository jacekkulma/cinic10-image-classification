import os
import random
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader

# CINIC-10 specific statistics
CINIC_MEAN = [0.47889522, 0.47227842, 0.43047404]
CINIC_STD = [0.24205776, 0.23828046, 0.25874835]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MixupCollate:
    """
    Collate function that applies Mixup augmentation to a batch.
    Mixup mixes pairs of samples and their targets using a fixed lambda value.
    """
    def __init__(self, alpha: float = 0.8):
        """
        Args:
            alpha: Lambda value for mixup (how much to weight the first sample)
        """
        self.alpha = alpha
    
    def __call__(self, batch):
        """
        Args:
            batch: List of (image, label) tuples from the dataset
        
        Returns:
            Tuple of (mixed_images, mixed_labels) tensors
        """
        # Stack images and labels
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        
        # Create random permutation for mixing
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        # Mix images: mixed_x = lambda * x_i + (1 - lambda) * x_j
        mixed_images = self.alpha * images + (1 - self.alpha) * images[index, :]
        
        # Mix labels as soft targets: mixed_y = lambda * y_i + (1 - lambda) * y_j
        mixed_labels_a = labels
        mixed_labels_b = labels[index]
        
        return mixed_images, mixed_labels_a, mixed_labels_b, torch.tensor(self.alpha)


def get_dataloaders(data_dir: str, batch_size: int, num_workers: int = None, augmentation_type: str = "none", seed: int = 42):
    """
    Creates train, validation, and test dataloaders for the CINIC-10 dataset.
    Supports different augmentation strategies: 'none', 'simple', 'advanced' (Mixup), or 'both'.
    Optimized by default for 8-core/16-thread CPUs.
    """
    # 0. Handle the default logic
    if num_workers is None:
        num_workers = 4
    
    augmentation_type = augmentation_type.lower()
    if augmentation_type not in ["none", "simple", "advanced", "both"]:
        raise ValueError(f"Unknown augmentation_type: {augmentation_type}")
    
    # 1. Define Transforms
    base_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CINIC_MEAN, std=CINIC_STD)
    ])
    
    # Create augmented transforms for training if simple or both augmentation is enabled
    if augmentation_type in ["simple", "both"]:
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

    # 3. Setup collate function for training loader
    train_collate_fn = None
    if augmentation_type in ["advanced", "both"]:
        # Use Mixup collate function with lambda=0.8 for Mixup augmentation
        train_collate_fn = MixupCollate(alpha=0.8)
        
    # 4. Generator for deterministic data loading
    g = torch.Generator()
    g.manual_seed(seed)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_collate_fn,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    return train_loader, valid_loader, test_loader