import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from utils import load_config
from data import get_dataloader
from models import FeatureExtractor


def main():
    # Load configuration from file
    config_path = "D:/Data Science/ML Dr Mostafa/03 Deep learning/Projects/Vollyball/Code/Refactoring2/configs/baseline3_feature_extraction_config.yml"
    config = load_config(config_path)

    # Set up save directory
    os.makedirs(config.save_dir, exist_ok=True)



    device = config.device
    # Create feature extractor
    feature_extractor = FeatureExtractor(config.best_model_path).to(device)
    feature_extractor.eval()
    features_data = {}
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} set...")
        # Create dataset and dataloader
        dataloader = get_dataloader(
                        dataset_type='PersonActionDataset',
                        path_or_root=config.dataset_root,
                        batch_size=config.batch_size,
                        split=split,
                        use_all_frames=config.use_all_frames,
                        mode=config.mode,
                        shuffle=False,
                        num_workers=config.num_workers,
                        pin_memory=config.pin_memory
                    )

        split_features = {}

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting {split} features"):
                frames_data = batch['frames_data']  # List of 9 frames
                label = batch['label'].item()
                meta = batch['meta']

                clip_features = []

                # Process each frame separately
                for frame in frames_data:  # Iterate through all 9 frames
                    frame_crops = frame[0]  # Get player crops for this frame
                    frame_crops = frame_crops.to(device)


                    if frame_crops.shape[0] == 0:
                        raise ValueError("frame_crops is empty")
                    # Extract features for all players in this frame
                    person_features = feature_extractor(frame_crops)

                    # Ensure each frame has 12 feature vectors
                    while person_features.shape[0] < 12:
                        person_features = torch.cat((person_features, torch.zeros((1, 2048)).to(device)), dim=0)


                    # Add to clip features
                    clip_features.append(person_features.cpu().numpy())



                # Stack all frame features for this clip
                clip_features = np.stack(clip_features)

                # Store features
                clip_key = f"{meta['video_id'][0]}_{meta['clip_id'][0]}"
                split_features[clip_key] = {
                    'features': clip_features,
                    'label': label,
                    'meta': {
                        'video_id': meta['video_id'][0],
                        'clip_id': meta['clip_id'][0],
                        'frame_ids': meta['frame_ids'][0]
                    }
                }

        features_data[split] = split_features

        # Save features for this split
        output_path = os.path.join(config.save_dir, f'{split}_features.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(split_features, f)

        print(f"Saved {len(split_features)} {split} clips to {output_path}")



if __name__ == "__main__":
    main()

