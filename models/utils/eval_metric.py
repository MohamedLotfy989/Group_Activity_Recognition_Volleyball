import os
import torch
import numpy as np
import seaborn as sns
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

def setup_training_directories(base_dir='runs'):
    """Create timestamped directory for saving training artifacts."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def save_checkpoint(model, optimizer, epoch, save_dir, filename='checkpoint.pth'):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')

def plot_confusion_matrix(labels, predictions, class_names, save_path, title):
    """Plot and save confusion matrix with counts and percentages."""
    cm = confusion_matrix(labels, predictions)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    sns.heatmap(cm_percentage, annot=True, fmt='.1f', xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Percentages)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s, save_path):
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Losses')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracies')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(train_f1s, label='Train F1')
    plt.plot(val_f1s, label='Val F1')
    plt.title('F1 Scores')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_classification_report(labels, predictions, class_names, save_path):
    """Generate and save classification report to a text file."""
    report = classification_report(labels, predictions, target_names=class_names)
    with open(save_path, 'w') as f:
        f.write(report)
    print(f'Classification report saved at {save_path}')

def compute_metrics(labels, predictions):
    """Compute classification metrics."""
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'f1_weighted': f1_score(labels, predictions, average='weighted'),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'f1_micro': f1_score(labels, predictions, average='micro'),
    }
    return metrics

# Example usage:
# metrics = compute_metrics(labels, predictions)
# print(metrics)
# plot_confusion_matrix(labels, predictions, class_names, 'conf_matrix.png', 'Confusion Matrix')
