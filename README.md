# Deep Learning Project for Volleyball Activity Recognition


## Table of Contents
- [Overview](#overview)
- [Improvements and Results](#improvements-and-results)
  - [Accuracy and Improvement Over the Paper](#accuracy-and-improvement-over-the-paper)
- [Dataset](#dataset)
  - [Dataset Labels](#dataset-labels)
  - [Dataset Splits](#dataset-splits)
- [Project Structure](#project-structure)
- [Features](#features)
- [Baseline Descriptions](#baseline-descriptions)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Configuration](#configuration)
  - [Evaluation](#evaluation)
  - [Logging and Outputs](#logging-and-outputs)

## Overview
This project leverages deep learning to classify volleyball activities using both temporal and spatial features. It provides multiple baseline models and supports functionalities such as training, evaluation, and metric visualization. Based on the seminal paper ["CVPR 2016"](http://arxiv.org/pdf/1607.02643v1.pdf), "A Hierarchical Deep Temporal Model for Group Activity Recognition" by Mostafa S. Ibrahim, this project explores hierarchical LSTM-based architectures for understanding individual actions and group activities.


## Improvements and Results
**Differences from the Original Paper**

This project introduces several updates and refinements over the original work:

 1. Improved Baselines: Updated baseline implementations with better network architectures, e.g., using ResNet50 instead of AlexNet.

 2. higher accuracies were achieved in all baselines compared to the paper. Specifically, our final baseline achieved an accuracy of 93%, whereas the paper reported 81.9%.

 3. A new baseline(Baseline9) was introduced that achieved 92% accuracy without the need for a temporal model.

 4. Modern Framework: Re-implemented in PyTorch with flexible configuration handling via YAML.
### Accuracy and Improvement Over the Paper
Below is a table comparing the accuracy of various baselines as reported in the paper versus our implementation:

| Baseline                                  | Accuracy (Paper) | Accuracy (Our Implementation) |
|-------------------------------------------|------------------|-------------------------------|
| B1-Image Classification                   | 66.7%            | 78%                           |
| B3-Fine-tuned Person Classification       | 68.1%            | 76%                           |
| B4-Temporal Model with Image Features     | 63.1%            | 80%                           |
| B5-Temporal Model with Person Features    | 67.6%            | 88%                           |
| B6-Two-stage Model without LSTM 1         | 74.7%            | 81%                           |
| B7-Two-stage Model without LSTM 2         | 80.2%            | 89%                           |
| Our Two-stage Hierarchical Model          | 81.9%            | 93%                           |
| B9-Fine-Tuned Team Spatial Classification | N/A              | 92%                           |


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

### Dataset Sample
![Dataset Sample](https://i.imgur.com/DUhaofS.gif)

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
│   ├── baseline9.py
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
│   ├── baseline9_config.yml
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
│   ├── train_baseline9.py
│   └── __init__.py
├── runs/                    # Outputs and saved models
├── requirements.txt         # Dependencies
└── README.md                # Project description
```
## Features
- **Multiple Baselines**: Baseline1, Baseline3, Baseline4, Baseline5, Baseline6, Baseline7,Baseline8, and Baseline9.
- **Configurable Parameters**: YAML-based configuration for easy adjustments.
- **Early Stopping**: Built-in mechanism to halt training if no improvement is observed.
- **Metric Visualization**: Includes confusion matrices and classification reports.
- **Scalable Design**: Modular structure for future expansion and maintainability.

## Baseline Descriptions

### Baseline1 - Image Classification:(accuracy 78%)
This baseline classifies group activities using only spatial features extracted from a single frame of the scene.

Key Features:
- No temporal information is considered.
- The entire frame is treated as a single entity for classification.

### Baseline3 - Fine-Tuned Person Classification:(accuracy 76%)
Fine-tunes a pre-trained network to classify individual actions from cropped player images.

Key Features:
- Focuses on learning individual player actions.
- Uses person crops extracted from frames with bounding box annotations.
Steps:
- Phase 1: Fine-tune the model on individual player crops.
- Phase 2: Extract features for players in each frame.
- Phase 3: Max pool the extracted feature over players and classify group activities.

### Baseline4 - Temporal Model with Image Features:(accuracy 80%)
Incorporates temporal modeling by processing sequential frames to predict group activities.

Key Features:
- Uses full-frame features from consecutive frames.
- Employs an LSTM to capture temporal dependencies.

### Baseline5 - Temporal Model with Person Features:(accuracy 88%)
Models temporal dependencies using features extracted from individual player crops over time.

Key Features:
- Processes person-level features with temporal modeling.
- Combines individual actions to predict group activities.

### Baseline6 - Two-Stage Model Without LSTM 1:(accuracy 81%)
A two-stage hierarchical model that avoids LSTMs for temporal modeling.

Key Features:
- Uses pooling mechanisms instead of temporal connections.
- Focuses on aggregating spatial features across players.

### Baseline7 - Two-Stage Model Without LSTM 2:(accuracy 88%)
An enhanced version of Baseline6, with improved pooling and feature aggregation techniques.

Key Features:
- Employs adaptive pooling for team-level feature aggregation.
- Provides better representation for group activities.

### Baseline8 - Two-Stage Hierarchical Model:(accuracy 93%)
A hierarchical model that combines player-level and scene-level representations using LSTMs.

Key Features:
- Player-level LSTM processes temporal features of individual players.
- Scene-level LSTM aggregates features for the entire scene.
- Outputs group activity predictions based on hierarchical features.
  
![Confusion_Matrix B8](https://i.imgur.com/VmKiOO3.png)

### Baseline9 - Fine-Tuned Team Spatial Classification(No Temporal):(accuracy 92%)
Fine-tunes individual player actions and then processes each team separately based on the spatial position of each player.A novel baseline that achieves high accuracy (92%) without the need for any temporal modeling.

Key Features:
- Fine-tunes models on individual player actions.
- Separately processes each team based on spatial positions.
  
![Confusion_Matrix B9](https://i.imgur.com/RIYpxvo.png)

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
   python scripts/train_baseline9.py
   ```

### Configuration
Model configurations are stored in the `configs/` directory. Adjust parameters such as learning rate, batch size, and number of epochs by editing the relevant `.yml` file.

### Evaluation
Evaluation is performed automatically after training. Results include metrics like confusion matrices and classification reports, which are saved in the `runs/` directory.

### Logging and Outputs
Logs and model outputs are organized into timestamped folders within the `runs/` directory for easy tracking of experiments.
