<h1 align="center">Deep Learning Project for Volleyball Activity Recognition</h1>

<h2 align="center">An implementation of seminal CVPR 2016 paper: "A Hierarchical Deep Temporal Model for Group Activity Recognition."</h2>

<p align="center">
  <img src="https://i.imgur.com/rhQRxLb.png" alt="Volleyball Activities"  width="80%">
  <img src="https://i.imgur.com/Ck6iA70.jpg" alt="Volleyball Activities"  style="width: 60%; height: 400px;">
</p>

## Table of Contents
- [Key Changes](#key-changes)
  - [Accuracy and Improvement Over the Paper](#accuracy-and-improvement-over-the-paper)
- [Installation](#installation)
- [Dataset](#dataset)
  - [Dataset Labels](#dataset-labels)
  - [Dataset Splits](#dataset-splits)
- [Features](#features)
- [Baseline Descriptions](#baseline-descriptions)
- [Usage](#usage)
  - [Training](#training)
  - [Configuration](#configuration)
  - [Evaluation](#evaluation)
  - [Logging and Outputs](#logging-and-outputs)



## 📚 Implemented Paper

| Paper        | Year | Original Paper | Implementation | Key Points                        |
|--------------|------|----------------|----------------|-----------------------------------|
| **CVPR 16**| 2016 | [Paper](https://arxiv.org/abs/1512.03385) | [Implementation](https://github.com/mostafa-saad/deep-activity-rec/tree/master) | Two-stage hierarchical LSTM for group activity recognition      |


## Key Changes

 1. Improved Baselines: Updated baseline implementations with better network architectures, e.g., using ResNet50 instead of AlexNet.

 2. higher accuracies were achieved in all baselines compared to the paper. Specifically, our final baseline achieved an accuracy of 93%, whereas the paper reported 81.9%.

 3. A new baseline(Baseline9) was introduced that achieved 92% accuracy without the need for a temporal model.

 4. Modern Framework: Re-implemented in PyTorch instead of Caffe.
    
## Accuracy and Improvement Over the Paper
<p align="center">
  <img src="https://i.imgur.com/rVDCsUD.png" width="50%">
</p>


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/MohamedLotfy989/Group_Activity_Recognition_Volleyball.git
   cd Group_Activity_Recognition_Volleyball
   ```


## Dataset
We used a volleyball dataset introduced in the aforementioned paper. The dataset consists of:
- **Videos**: 55 YouTube volleyball videos.
- **Frames**: 4830 annotated frames, each with bounding boxes around players and labels for both individual actions and group activities.

### Dataset Labels

<div style="text-align: center;">
   <img src="https://i.imgur.com/xTiVRPU.png" alt="B8" style=" width:35%; height:550px;">
    <img src="https://i.imgur.com/XURLkKZ.png" alt="B8" style=" width:35%; height:550px;">
</div>

### Dataset Splits
  - Training Set: 2/3 of the videos.
  - Validation Set: 15 videos.
  - Test Set: 1/3 of the videos.

### Dataset Sample
<p align="center">

<img  src="https://i.imgur.com/DUhaofS.gif" alt="B8" width="75%">
</p>

The dataset is available for download at [GitHub Deep Activity Rec](https://github.com/mostafa-saad/deep-activity-rec#dataset).

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

### Baseline7 - Two-Stage Model Without LSTM 2:(accuracy 89%)
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

<div style="text-align: center;">
  <img src="https://i.imgur.com/ZNYcthV.jpg" alt="B8" style="display:inline-block; width:45%; height:300px; margin-right:2%;">
  <img src="https://i.imgur.com/7yyWR3i.png" alt="B8" style="display:inline-block; width:45%; height:300px;">
</div>

<img src="https://i.imgur.com/VmKiOO3.png" alt="B8">

### Baseline9 - Fine-Tuned Team Spatial Classification(No Temporal):(accuracy 92%)
Fine-tunes individual player actions and then processes each team separately based on the spatial position of each player.A novel baseline that achieves high accuracy (92%) without the need for any temporal modeling.

Key Features:
- Fine-tunes models on individual player actions.
- Separately processes each team based on spatial positions.
   
<img src="https://i.imgur.com/iMH2Vtq.png" alt="B9" width="50%">

<img src="https://i.imgur.com/RIYpxvo.png" alt="B9">



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
