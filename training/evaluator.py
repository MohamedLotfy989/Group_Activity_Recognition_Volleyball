import os
import torch
from tqdm import tqdm
from utils import compute_metrics,set_seed,save_classification_report,plot_confusion_matrix,setup_logger


class Evaluator:
    def __init__(self, config, model):
        """Initialize the Evaluator with a model and device."""
        self.config=config

        self.device = torch.device(config.device)

        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger = setup_logger(config.save_dir)
        self.class_names=config.class_names
        self.best_model_path = os.path.join(config.save_dir, 'best_model.pth')
    def evaluate(self, dataloader):
        """Evaluate the model on the given dataloader."""
        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        self.logger.info(f"Loaded model from {self.best_model_path}")

        self.model.eval()
        all_labels = []
        all_preds = []
        running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Testing"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                del outputs
                torch.cuda.empty_cache()  # optionally clear cache after heavy operations
                running_loss += loss.item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Calculate metrics
        avg_loss = running_loss / len(dataloader)
        metrics = compute_metrics(all_labels, all_preds)
        self.logger.info(f"Test Loss: {avg_loss:.4f}, Test Metrics: {metrics}")

        report_path = os.path.join(self.config.save_dir, 'classification_report.txt')
        conf_matrix_path = os.path.join(self.config.save_dir, 'confusion_matrix.png')
        save_classification_report(all_labels, all_preds, self.class_names, report_path)
        plot_confusion_matrix(all_labels, all_preds, self.class_names, conf_matrix_path, 'Confusion Matrix')

