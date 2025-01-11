import os

import torch
from sklearn.preprocessing import StandardScaler

from models import SimpleNN
from utils import load_config
from training import Trainer,Evaluator
from data import get_dataloader

def main():
    # Load configuration from file
    config_path = "configs/baseline3_group_activity_config.yml"
    config = load_config(config_path)

    # Set up save directory
    os.makedirs(config.save_dir, exist_ok=True)

    # Initialize data loaders
    train_loader = get_dataloader(
        dataset_type=config.dataset_type,
        path_or_root=os.path.join(config.dataset_root, "train_features.pkl"),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    val_loader = get_dataloader(
        dataset_type=config.dataset_type,
        path_or_root=os.path.join(config.dataset_root, "val_features.pkl"),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    test_loader = get_dataloader(
        dataset_type=config.dataset_type,
        path_or_root=os.path.join(config.dataset_root, "test_features.pkl"),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )


    # Initialize model
    model = SimpleNN(input_size=2048, output_size=8)


    # Initialize and run the trainer
    trainer = Trainer(config, model=model)
    trainer.train(train_loader, val_loader)

    # Evaluate the model
    evaluator = Evaluator(config,model=model)
    evaluator.evaluate(test_loader)

if __name__ == "__main__":
    main()
