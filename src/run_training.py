"""
Main script to run training experiments.
"""
import os
import json
import argparse
import torch
import torch.multiprocessing as mp

from configs.config import Config, DataConfig, ModelConfig, TrainingConfig
from models.custom_cnn import get_model
from data.dataloader import get_data_loaders
from training.trainer import Trainer

def main(rank, world_size, config):
    """Main function to run training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    # Get data loaders
    loaders = get_data_loaders(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        img_size=config.data.img_size,
        num_workers=config.data.num_workers,
        train_val_split=config.data.train_val_split,
        use_ddp=world_size > 1,
        rank=rank,
        world_size=world_size
    )

    # Get model
    model = get_model(config.model.model_name, loaders['num_classes'])

    # Initialize and run trainer
    trainer = Trainer(config, model, loaders['train'], loaders['val'])
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a deep learning model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Config(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {}))
        )
    else:
        # Default config
        config = Config()

    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(main, args=(world_size, config), nprocs=world_size, join=True)
    else:
        main(0, 1, config)
