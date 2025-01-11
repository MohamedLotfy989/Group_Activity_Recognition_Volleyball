import os
import pickle

import cv2
import numpy as np
import torch
from collections import defaultdict
from torchvision import transforms
from PIL import Image

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from utils import RandomScale, RandomBrightnessContrast

class VolleyballDataset(Dataset):
    """
        A dataset class for group activity and  person actions in volleyball videos.

        Parameters:
        - dataset_root (str): Root directory of the dataset.
        - split (str): Dataset split ('train', 'val', 'test').
        - use_all_frames (bool): Whether to use all frames or a single representative frame.
        - mode (str): Dataset mode ('action' for person actions, 'feature_extraction' for group activities).

        Attributes:
        - annotations (dict): Loaded annotations for the dataset.
        - samples (list): List of samples with frame, bounding box, and labels.
        - class_counts (dict): Counts of samples per class.
        - class_weights (dict): Computed weights for each class based on sample counts.
        - transform (callable): Transformation applied to the input data.
        """
    def __init__(self, dataset_root, split='train', use_all_frames=False,mode='action'):
        self.dataset_root = dataset_root
        self.split = split
        self.splits = {
            'train': [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
            'val': [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
            'test': [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
        }
        self.use_all_frames = use_all_frames
        self.mode = mode  # 'action' for PersonActionDataset, 'feature' for VolleyballFeatureDataset

        # Load annotations
        self.annotations = self._load_annotations()
        if self.mode == 'action':
            self.labels = {
                'blocking': 0, 'digging': 1, 'falling': 2, 'jumping': 3,
                'moving': 4, 'setting': 5, 'spiking': 6, 'standing': 7, 'waiting': 8
            }
        elif self.mode in ['frame_feature_extraction','player_feature_extraction','group_activity']:
            self.labels = {
                'l-pass': 0, 'r-pass': 1, 'l-spike': 2, 'r_spike': 3,
                'l_set': 4, 'r_set': 5, 'l_winpoint': 6, 'r_winpoint': 7
            }


        self.samples, self.class_counts = self._generate_samples()
        self.class_weights = self._compute_class_weights()
        if self.mode == 'action':
            self.sample_weights = [
                self.class_weights[s['action']] if 'action' in s else 1.0 for s in self.samples
            ]
        elif self.mode == 'group_activity':
            self.sample_weights = [
                self.class_weights[s['activity']] if 'activity' in s else 1.0 for s in self.samples
            ]
        else:
            self.sample_weights = None
        if self.mode == 'action' or self.mode == 'group_activity':
            self.transform = self._get_transforms(split)
        elif self.mode in ['frame_feature_extraction','player_feature_extraction']:
            self.transform = self._get_transforms_feature()

    def _load_annotations(self):
        try:
            with open(os.path.join(self.dataset_root, 'annot_all.pkl'), 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading annotations: {e}")
            return {}

    def _generate_samples(self):
        samples = []
        class_counts = defaultdict(int)
        video_ids = map(str, self.splits[self.split])

        for video_id in video_ids:
            if video_id not in self.annotations:
                print(f"Warning: Video {video_id} not found in annotations")
                continue

            for clip_id, clip_data in self.annotations[video_id].items():
                frame_ids = sorted(clip_data['frame_boxes_dct'].keys())
                selected_frame_ids = frame_ids if self.use_all_frames else [frame_ids[len(frame_ids) // 2]]

                if self.mode in ['action', 'group_activity']:
                    for frame_id in selected_frame_ids:
                        frame_path = os.path.join(self.dataset_root, 'videos', video_id, clip_id, f"{frame_id}.jpg")
                        if not os.path.exists(frame_path):
                            continue
                        if self.mode=='group_activity':
                            samples.append({
                                'video_id': video_id,
                                'clip_id': clip_id,
                                'activity': clip_data['category'],
                                'frame_id': frame_id
                            })
                            class_counts[clip_data['category']] += 1
                        else:
                            boxes = clip_data['frame_boxes_dct'][frame_id]
                            for box in boxes:
                                if box.category not in self.labels:
                                    continue

                                samples.append({
                                    'video_id': video_id,
                                    'clip_id': clip_id,
                                    'frame_id': frame_id,
                                    'box': box.box,
                                    'action': box.category
                                })
                                class_counts[box.category] += 1
                elif self.mode in ['frame_feature_extraction','player_feature_extraction']:
                    samples.append({
                        'video_id': video_id,
                        'clip_id': clip_id,
                        'activity': clip_data['category'],
                        'frame_id': selected_frame_ids
                    })
                    class_counts[clip_data['category']] += 1
        return samples, class_counts

    def _compute_class_weights(self):
        """
           Compute class weights for balancing the dataset.

           Returns:
           - dict: A dictionary where keys are class labels and values are weights.
           """
        total_samples = sum(self.class_counts.values())
        if total_samples == 0:
            return {action: 1.0 for action in self.labels.keys()}
        return {
            action: total_samples / (len(self.labels) * count)
            for action, count in self.class_counts.items()
            if count > 0
        }

    def _get_transforms(self, split):
        if split == 'train':
            return transforms.Compose([
                RandomScale(scale_range=(0.85, 1.15)),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                RandomBrightnessContrast(brightness_range=(0.85, 1.15), contrast_range=(0.85, 1.15)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _get_transforms_feature(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.mode in ['action', 'group_activity']:
            try:
                frame_path = os.path.join(
                    self.dataset_root, 'videos',
                    sample['video_id'], sample['clip_id'],
                    f"{sample['frame_id']}.jpg"
                )

                frame = cv2.imread(frame_path)
                if frame is None:
                    raise ValueError(f"Could not load image: {frame_path}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.mode=='group_activity':
                   image=frame
                else:
                    x1, y1, x2, y2 = sample['box']
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    h, w = frame.shape[:2]
                    x1, x2 = max(0, x1), min(w, x2)
                    y1, y2 = max(0, y1), min(h, y2)
                    if x1 >= x2 or y1 >= y2:
                        raise ValueError("Invalid box coordinates")
                    image = frame[y1:y2, x1:x2]
                    if image.size == 0:
                        raise ValueError("Empty crop")
            except Exception as e:
                print(f"Error loading sample {idx}: {e}")
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(image)
            image = self.transform(image)

            return image, self.labels[sample['activity']] if self.mode == 'group_activity' else self.labels[sample['action']]

        elif self.mode == 'frame_feature_extraction':
            video_path = os.path.join(self.dataset_root, 'videos', sample['video_id'], sample['clip_id'])

            frames_data = []
            labels = []

            for frame_id in sample['frame_id']:
                frame_path = os.path.join(video_path, f'{frame_id}.jpg')
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.transform(frame)
                frames_data.append(frame)
                labels.append(self.labels[sample['activity']])

            frames_tensor = torch.stack(frames_data)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            return frames_tensor, self.labels[sample['activity']]

        elif self.mode == 'player_feature_extraction':

            video_path = os.path.join(self.dataset_root, 'videos', sample['video_id'], sample['clip_id'])

            frames_data = []

            for frame_id in sample['frame_id']:

                frame_boxes = self.annotations[sample['video_id']][sample['clip_id']]['frame_boxes_dct'][frame_id]

                # Sort frame_boxes by the x-coordinate of the bounding box (left to right)
                frame_boxes.sort(key=lambda box: box.box[0])

                # Initialize a list to store player crops and IDs for this frame
                person_crops = []
                player_ids = []

                for box in frame_boxes:
                    try:
                        # Load and crop frame
                        frame_path = os.path.join(video_path, f'{frame_id}.jpg')
                        frame = cv2.imread(frame_path)
                        if frame is None:
                            continue
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Get person crop
                        x1, y1, x2, y2 = map(int, box.box)
                        person_crop = frame[y1:y2, x1:x2]

                        # Transform crop
                        person_crop = Image.fromarray(person_crop)
                        person_crop = self.transform(person_crop)
                        person_crops.append(person_crop)
                        player_ids.append(box.player_ID)

                    except Exception as e:
                        print(f"Error processing crop: {e}")
                        continue

                frames_data.append(torch.stack(person_crops))

            return {
                'frames_data': frames_data,
                'label': self.labels[sample['activity']],
                'meta': {
                    'video_id': sample['video_id'],
                    'clip_id': sample['clip_id'],
                    'frame_ids': sample['frame_id']
                }
            }

class ImportPlayersFeatures(Dataset):
    """
        A dataset class for loading precomputed player features.

        Parameters:
        - features_path (str): Path to the pickle file containing features and labels.

        Attributes:
        - features (list): Loaded features for each clip.
        - labels (list): Corresponding labels for each clip.

        Methods:
        - __len__: Returns the number of samples in the dataset.
        - __getitem__: Returns a sample (features and label) at the specified index.
        """
    def __init__(self, features_path):
        # Load the pickle file with error handling
        try:
            with open(features_path, 'rb') as f:
                data = pickle.load(f)
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"Error loading file {features_path}: {e}")

        features = []
        labels = []
        for clip_key, item in data.items():
            features.append(item['features'])
            labels.append(item['label'])

        self.features = features  # Shape: (num_clips, 9, 12, 2048)
        self.labels = labels  # Shape: (num_clips,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])  # (9, 12, 2048)
        label = torch.LongTensor([self.labels[idx]]).squeeze()  # (1,) -> ()
        return features, label



def get_dataloader(dataset_type, path_or_root, batch_size, split='train', use_all_frames=False,
                   mode='action', shuffle=True, num_workers=0, pin_memory=False,exclude_classes=False,classes_to_exclude=None,annotations_path=None,annotations=None):
    """
    Get a DataLoader for the specified dataset.

    Parameters:
    - dataset_type (str): Type of dataset to load.
    - path_or_root (str): Path to dataset or feature files.
    - batch_size (int): Batch size for DataLoader.
    - split (str): Dataset split ('train', 'val', 'test').
    - use_all_frames (bool): Whether to use all frames.
    - mode (str): Mode of dataset ('action' or 'feature_extraction').
    - shuffle (bool): Whether to shuffle the data (ignored if using a sampler).
    - num_workers (int): Number of workers for data loading.
    - pin_memory (bool): Whether to use pinned memory for faster GPU transfer.

    Returns:
    - DataLoader: A PyTorch DataLoader for the dataset.
    """
    if dataset_type == 'PersonActionDataset' or dataset_type == 'GroupActivityDataset':
        dataset = VolleyballDataset(
            dataset_root=path_or_root,
            split=split,
            use_all_frames=use_all_frames,
            mode=mode
        )

        # Use WeightedRandomSampler only for the training split
        if split == 'train' and mode not in ['frame_feature_extraction', 'player_feature_extraction']:
            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(dataset.sample_weights),
                num_samples=len(dataset),
                replacement=True
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
    elif dataset_type == 'PlayersFeatures':
        dataset = ImportPlayersFeatures(features_path=path_or_root)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return dataloader

# Example usage:
# train_loader = get_dataloader('path/to/train_features.pkl', batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
