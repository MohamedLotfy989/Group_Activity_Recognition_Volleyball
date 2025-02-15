import argparse
import os
import yaml
import logging
from ultralytics import YOLO

def create_dataset_yaml(train_images_dir: str, val_images_dir: str, nc: int, names: list, output_path: str):
    dataset = {
        'train': train_images_dir,
        'val': val_images_dir,
        'nc': nc,
        'names': names
    }
    with open(output_path, 'w') as f:
        yaml.dump(dataset, f, default_flow_style=False)
    logging.info(f"Dataset YAML file created at {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLO model for volleyball player detection."
    )
    parser.add_argument('--train_images', type=str, required=True,
                        help='Directory containing training images.')
    parser.add_argument('--val_images', type=str, required=True,
                        help='Directory containing validation images.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs.')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Training image size (pixels).')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Path to the pretrained YOLO model (e.g. yolov8n.pt).')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='Directory to save training runs.')
    parser.add_argument('--name', type=str, default='train',
                        help='Name for the training run.')
    parser.add_argument('--data_yaml', type=str, default='data.yaml',
                        help='Path to save the dataset YAML config file.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to the last saved weights to resume training.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Creating dataset YAML configuration file...")
    create_dataset_yaml(train_images_dir=args.train_images,
                         val_images_dir=args.val_images,
                         nc=1,
                         names=['player'],
                         output_path=args.data_yaml)

    if args.resume:
        logging.info(f"Resuming training from '{args.resume}' ...")
        model = YOLO(args.resume)
    else:
        logging.info(f"Loading pretrained YOLO model from '{args.model}' ...")
        model = YOLO(args.model)

    logging.info("Starting training...")
    train_results = model.train(data=args.data_yaml,
                                epochs=args.epochs,
                                batch=args.batch,
                                imgsz=args.imgsz,
                                project=args.project,
                                name=args.name,
                                exist_ok=True)
    logging.info("Training complete.")

    logging.info("Starting validation on the validation set...")
    eval_results = model.val(data=args.data_yaml)
    logging.info("Validation complete.")

    try:
        metrics = eval_results.metrics
    except AttributeError:
        metrics = eval_results

    logging.info("Evaluation Metrics:")
    for key, value in metrics.items():
        logging.info(f"{key}: {value}")

    best_model_path = os.path.join(args.project, args.name, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        logging.info(f"Best model successfully saved at: {best_model_path}")
    else:
        logging.warning("Best model file not found. Please check the training logs.")

if __name__ == "__main__":
    main()
