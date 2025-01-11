import os
from utils import load_config,set_seed
from training import Trainer,Evaluator
from data import get_dataloader
from models import Baseline4



def main():
    # Load configuration from file
    config_path = "D:/Data Science/ML Dr Mostafa/03 Deep learning/Projects/Vollyball/Code/Refactoring2/configs/baseline4_config.yml"
    config = load_config(config_path)

    set_seed(config.seed)

    # Set up save directory
    os.makedirs(config.save_dir, exist_ok=True)



    # Load data
    train_loader = get_dataloader(
        dataset_type=config.dataset_type,
        path_or_root=config.dataset_root,
        batch_size=config.batch_size,
        split='train',
        use_all_frames=config.use_all_frames,
        mode=config.mode,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    val_loader = get_dataloader(
        dataset_type=config.dataset_type,
        path_or_root=config.dataset_root,
        batch_size=config.batch_size,
        split='val',
        use_all_frames=config.use_all_frames,
        mode=config.mode,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    test_loader = get_dataloader(
        dataset_type=config.dataset_type,
        path_or_root=config.dataset_root,
        batch_size=config.batch_size,
        split='test',
        use_all_frames=config.use_all_frames,
        mode=config.mode,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    # Initialize model
    model = Baseline4(config.best_model_path, lstm_hidden_size=config.hidden_size_frame,num_classes=8)

    # Initialize and run the trainer
    trainer = Trainer(config, model=model)
    trainer.train(train_loader, val_loader)

    # Evaluate the model
    evaluator = Evaluator(config, model=model)
    evaluator.evaluate(test_loader)

if __name__ == "__main__":
    main()