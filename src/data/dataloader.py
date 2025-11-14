"""
Data loading and preprocessing utilities.
"""
import os
from typing import Tuple, Dict, Optional, Callable

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
from PIL import Image

class CustomDataset(Dataset):
    """Custom dataset class that supports both classification and segmentation tasks."""
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_train: bool = True
    ):
        """
        Args:
            root: Root directory of the dataset
            transform: Optional transform to be applied on the images
            target_transform: Optional transform to be applied on the labels
            is_train: If True, loads training data, else loads test data
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train
        
        # Get list of images and labels
        self.samples = self._load_samples()
        
    def _load_samples(self) -> list:
        """Load image paths and corresponding labels."""
        samples = []
        class_to_idx = {}
        
        # Find all class directories
        class_dirs = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        class_dirs.sort()
        
        # Create class to index mapping
        for i, class_name in enumerate(class_dirs):
            class_to_idx[class_name] = i
            
        # Walk through each class directory
        for class_name in class_dirs:
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, class_to_idx[class_name]))
                    
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            idx: Index of the sample to fetch
            
        Returns:
            tuple: (image, target) where target is the class index
        """
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target

def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    train_val_split: float = 0.8,
    **kwargs
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training and validation.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Number of samples per batch
        img_size: Size to resize images to
        num_workers: Number of subprocesses to use for data loading
        train_val_split: Fraction of data to use for training
        
    Returns:
        dict: Dictionary containing train and val data loaders
    """
    # Define data transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = CustomDataset(
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    # Split into train and validation sets
    train_size = int(train_val_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # For testing, we'll use a separate directory
    test_dataset = CustomDataset(
        root=os.path.join(data_dir, 'test'),
        transform=val_transform,
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'num_classes': len(full_dataset.classes) if hasattr(full_dataset, 'classes') else 10
    }
