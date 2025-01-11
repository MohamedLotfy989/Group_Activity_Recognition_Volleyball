import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from utils import setup_logger,compute_metrics,mixup_data, mixup_criterion,save_checkpoint

class Trainer:
    def __init__(self, config, model):
        """Initialize the Trainer with configuration parameters."""
        self.config = config
        self.device = torch.device(config.device)

        # Use the model passed as a parameter
        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, verbose=True ) if config.lr_scheduler else None
        self.scaler = GradScaler('cuda')

        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger = setup_logger(config.save_dir)
        self.writer = SummaryWriter(log_dir=config.save_dir)
        self.use_mixup = config.mix_up
        self.patience = config.patience
        self.config.num_epochs = config.num_epochs
    def train(self, train_loader, val_loader):
        """Train the model with the provided data loaders."""
        best_val_f1 = 0
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            self.model.train()
            running_loss = 0.0
            all_labels, all_preds = [], []

            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs} [Training]"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.use_mixup:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0)

                self.optimizer.zero_grad()
                with autocast('cuda'):
                    outputs = self.model(inputs)
                    if self.use_mixup:
                        loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                    else:
                        loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                del outputs
                torch.cuda.empty_cache()  # optionally clear cache after heavy operations
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            train_f1 = f1_score(all_labels, all_preds, average='weighted')
            train_loss = running_loss / len(train_loader)
            metrics = compute_metrics(all_labels, all_preds)
            self.logger.info(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Metrics: {metrics}")
            self.writer.add_scalar('Train/Loss', train_loss, epoch + 1)
            self.writer.add_scalar('Train/F1', train_f1, epoch + 1)

            # Validation phase
            val_loss, val_metrics,val_f1 = self.evaluate(val_loader)
            self.logger.info(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Metrics: {val_metrics}")
            self.writer.add_scalar('Validation/Loss', val_loss, epoch + 1)
            self.writer.add_scalar('Validation/F1', val_f1, epoch + 1)

            if self.scheduler:
                self.scheduler.step(val_loss)

            save_checkpoint(self.model, self.optimizer, epoch + 1, self.config.save_dir, filename=f'checkpoint_epoch_{epoch + 1}.pth')

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), f"{self.config.save_dir}/best_model.pth")
                self.logger.info(f"Epoch {epoch+1}: Best model saved with Validation F1score: {val_f1:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

    def evaluate(self, val_loader):
        """Evaluate the model on the validation set."""
        self.model.eval()
        running_loss = 0.0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                del outputs
                torch.cuda.empty_cache()  # optionally clear cache after heavy operations
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average='weighted')
        metrics = compute_metrics(all_labels, all_preds)
        avg_val_loss = running_loss / len(val_loader)
        return avg_val_loss,metrics,f1

