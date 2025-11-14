"""
Configuration settings for the deep learning project.
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_dir: str = "data/raw"
    batch_size: int = 32
    img_size: Tuple[int, int] = (224, 224)
    num_workers: int = 4
    train_val_split: float = 0.8
    
@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    model_name: str = "resnet18"
    num_classes: int = 10
    pretrained: bool = True
    dropout: float = 0.2
    
@dataclass
class TrainingConfig:
    """Configuration for training process."""
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "saved_models"
    log_dir: str = "logs"
    checkpoint_freq: int = 5
    use_amp: bool = True
    accumulation_steps: int = 1
    
@dataclass
class Config:
    """Main configuration class that holds all configs."""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__
        }
