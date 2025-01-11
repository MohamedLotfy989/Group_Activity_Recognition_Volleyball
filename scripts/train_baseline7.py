import os
import torch

from utils import load_config,set_seed
from training import Trainer,Evaluator
from models import Baseline7
from data import get_dataloader

def main():
    # Load configuration from file
    config_path = "/configs/baseline7_config.yml"
    config = load_config(config_path)

    set_seed(config.seed)

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
    model = Baseline7(
        input_size=config.input_size,
        hidden_size_player=config.hidden_size_player,
        hidden_size_frame=config.hidden_size_frame,
        num_classes=8
    )

    # Initialize and run the trainer
    trainer = Trainer(config, model=model)
    trainer.train(train_loader, val_loader)

    # Evaluate the model
    model.load_state_dict(torch.load(os.path.join(config.save_dir, 'best_model.pth')))
    evaluator = Evaluator(config,model=model)
    evaluator.evaluate(test_loader)

if __name__ == "__main__":
    main()
