# Deep Learning Project for Volleyball Activity Recognition

## Overview
This project leverages deep learning to classify volleyball activities using both temporal and spatial features. It provides multiple baseline models and supports functionalities such as training, validation, testing, and performance evaluation with advanced metrics. This implementation is based on the paper "A Hierarchical Deep Temporal Model for Group Activity Recognition" by Mostafa S. Ibrahim et al. The method builds on hierarchical LSTM-based architectures for modeling both individual actions and group activities.

## Dataset
We used a volleyball dataset introduced in the aforementioned paper. The dataset consists of:
- **Videos**: 55 YouTube volleyball videos.
- **Frames**: 4830 annotated frames, each with bounding boxes around players and labels for both individual actions and group activities.

### Dataset Labels
#### Group Activity Classes
| Class            | Instances |
|------------------|-----------|
| Right set        | 644       |
| Right spike      | 623       |
| Right pass       | 801       |
| Right winpoint   | 295       |
| Left winpoint    | 367       |
| Left pass        | 826       |
| Left spike       | 642       |
| Left set         | 633       |

#### Action Classes
| Class       | Instances |
|-------------|-----------|
| Waiting     | 3601      |
| Setting     | 1332      |
| Digging     | 2333      |
| Falling     | 1241      |
| Spiking     | 1216      |
| Blocking    | 2458      |
| Jumping     | 341       |
| Moving      | 5121      |
| Standing    | 38696     |

### Dataset Splits
- **Training Set**: 2/3 of the videos.
- **Validation Set**: 15 videos.
- **Test Set**: 1/3 of the videos.

The dataset is available for download at [GitHub Deep Activity Rec](https://github.com/mostafa-saad/deep-activity-rec#dataset).

## Project Structure
```
project_root/
├── models/                  # All model definitions
│   ├── baseline1.py
│   ├── baseline3/
│   │   ├── phase1_fine_tune.py
│   │   ├── phase2_feature_extraction.py
│   │   ├── phase3_group_activity.py
│   │   └── __init__.py
│   ├── baseline4.py
│   ├── baseline5.py
│   ├── baseline6.py
│   ├── baseline7.py
│   ├── baseline8.py
│   └── __init__.py
├── data/                    # Data loaders and datasets
│   ├── data_loader.py
│   ├── boxinfo.py
│   ├── volleyball_annot_loader.py
│   └── __init__.py
├── utils/                   # Utility functions
│   ├── helper_utils.py
│   ├── logger.py
│   ├── eval_metrics_utils.py
│   ├── utils.py
│   └── __init__.py
├── training/                # Training and evaluation logic
│   ├── trainer.py
│   ├── evaluator.py
│   └── __init__.py
├── configs/                 # Configurations for training
│   ├── baseline1_config.yml
│   ├── baseline3_feature_extraction_config.yml
│   ├── baseline3_fine_tune_config.yml
│   ├── baseline3_group_activity_config.yml
│   ├── baseline4_config.yml
│   ├── baseline5_config.yml
│   ├── baseline6_config.yml
│   ├── baseline7_config.yml
│   ├── baseline8_config.yml
├── scripts/                 # Entry point scripts
│   ├── train_baseline1.py
│   ├── train_baseline3/
│   │   ├── train_phase_1_fine_tune.py
│   │   ├── train_phase_2_feature_extraction.py
│   │   ├── train_phase_3_group_classifier.py
│   │   └── __init__.py
│   ├── train_baseline4.py
│   ├── train_baseline5.py
│   ├── train_baseline6.py
│   ├── train_baseline7.py
│   ├── train_baseline8.py
│   └── __init__.py
├── runs/                    # Outputs and saved models
├── requirements.txt         # Dependencies
└── README.md                # Project description
```

## Requirements
- Python >= 3.12.3
- PyTorch >= 2.5.1
- Torchvision >= 0.20.1
- Matplotlib >= 3.10.0
- Scikit-learn >= 1.6.1
- tqdm >= 4.66.5
- PyYAML >= 6.0.2
- Pillow >= 11.1.0
- NumPy >= 2.2.1
- seaborn >= 0.13.2

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/MohamedLotfy989/Group_Activity_Recognition_Volleyball.git
   cd Group_Activity_Recognition_Volleyball
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
To train a specific baseline model, execute the corresponding script:
   ```bash
   python scripts/train_baseline1.py
   python scripts/train_baseline3/train_phase_1_fine_tune.py
   python scripts/train_baseline3/train_phase_2_feature_extraction.py
   python scripts/train_baseline3/train_phase_3_group_classifier.py
   python scripts/train_baseline4.py
   python scripts/train_baseline5.py
   python scripts/train_baseline6.py
   python scripts/train_baseline7.py
   python scripts/train_baseline8.py
   ```

### Configuration
Model configurations are stored in the `configs/` directory. Adjust parameters such as learning rate, batch size, and number of epochs by editing the relevant `.yml` file.

### Evaluation
Evaluation is performed automatically after training. Results include metrics like confusion matrices and classification reports, which are saved in the `runs/` directory.

### Logging and Outputs
Logs and model outputs are organized into timestamped folders within the `runs/` directory for easy tracking of experiments.

## Features
- **Multiple Baselines**: Baseline1, Baseline3, Baseline4, Baseline5, Baseline6, Baseline7, and Baseline8.
- **Configurable Parameters**: YAML-based configuration for easy adjustments.
- **Early Stopping**: Built-in mechanism to halt training if no improvement is observed.
- **Metric Visualization**: Includes confusion matrices and classification reports.
- **Scalable Design**: Modular structure for future expansion and maintainability.

## Improvements and Results
**Differences from the Original Paper**

This project introduces several updates and refinements over the original work:

 1. Improved Baselines: Updated baseline implementations with better network architectures, e.g., using ResNet50 instead of AlexNet.

 2. Refined Temporal Representations: Incorporates more temporal frames (5 before and 4 after target frame).

 3. Enhanced Group Representations: Applies pooling mechanisms separately for each team to reduce confusion between team activities.

 4. Modern Framework: Re-implemented in PyTorch with flexible configuration handling via YAML.
### Accuracy and Improvement Over the Paper
Below is a table comparing the accuracy of various baselines as reported in the paper versus our implementation:

| Baseline                  | Accuracy (Paper) | Accuracy (Our Implementation) |
|---------------------------|------------------|-------------------------------|
| B1-Image Classification | 66.7%            | 78%                           |
| B3-Fine-tuned Person Classification | 68.1%            | 76%                           |
| B4-Temporal Model with Image Features | 63.1%            | 80%                           |
| B5-Temporal Model with Person Features | 67.6%            | 88%                           |
| B6-Two-stage Model without LSTM 1 | 74.7%            | 81%                           |
| B7-Two-stage Model without LSTM 2 | 80.2%            | 88%                           |
| Our Two-stage Hierarchical Model | 81.9%            | 93%                           |


## Contributions
We welcome contributions! Feel free to open issues or submit pull requests for bug fixes or enhancements.


