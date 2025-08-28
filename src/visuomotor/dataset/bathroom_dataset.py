from typing import Dict, List

import torch
import numpy as np

from visuomotor.dataset.tool import Normalizer
from visuomotor.dataset.tool2 import get_lazy_sample, sample_indices


class BathroomDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_root,
            split_indexes: List[int],
            pred_horizon: int,
            obs_horizon: int,
            action_horizon: int,
            stats: Dict[str, np.array],
            normalizer: Normalizer
    ):
        self.normalizer = normalizer

        self.image_keys = ["realsense", "depth_camera"]

        realsense_data = dataset_root['data']['realsense'][:]
        depth_data = dataset_root['data']['depth_camera'][:]
        
        train_data = {
            'wrist_pos': BathroomDataset._get_wrist_pos(dataset_root),
            'action_pos': BathroomDataset._get_action_pos(dataset_root)
        }

        normalized_train_data = {
            'realsense': realsense_data,
            'depth_camera': depth_data,
            'wrist_rot': BathroomDataset._get_wrist_rot(dataset_root),
            'action_rot': BathroomDataset._get_action_rot(dataset_root)
        }
    
        for key, data in train_data.items():
            normalized_train_data[key] = self.normalizer.normalize_data(data, stats[key])

        episode_ends = dataset_root['meta']['episode_ends'][:]

        indices = sample_indices(
            episode_ends=episode_ends,
            split_indexes=split_indexes,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1
        )

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
    
    @staticmethod
    def _get_wrist_pos(dataset_root):
        return dataset_root['data']['wrist_pos'][:]
    
    @staticmethod
    def _get_wrist_rot(dataset_root):
        return dataset_root['data']['wrist_rot'][:]
    
    @staticmethod
    def _get_action_pos(dataset_root):
        return np.concat((dataset_root["data"]["wrist_pos"][1:], dataset_root["data"]["wrist_pos"][-1:]), axis=0)  # state but shifter forward

    @staticmethod
    def _get_action_rot(dataset_root):
        return np.concat((dataset_root["data"]["wrist_rot"][1:], dataset_root["data"]["wrist_rot"][-1:]), axis=0)  # state but shifter forward

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

        nsample = get_lazy_sample(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
            image_keys=self.image_keys,
            obs_horizon=self.obs_horizon
        )

        nsample['wrist_pos'] = nsample['wrist_pos'][:self.obs_horizon,:]
        nsample['wrist_rot'] = nsample['wrist_rot'][:self.obs_horizon,:]
        return nsample

    @staticmethod
    def calculate_stats(np_data, episode_ends, split_indexes, normalizer: Normalizer):
        train = []
        data = np_data.tolist()
        for i in split_indexes:
            start_i = 0
            if i > 0:
                start_i = episode_ends[i - 1]
            end_i = episode_ends[i]
            train += data[start_i:end_i]
        return normalizer.get_data_stats(np.array(train))
    
    @staticmethod
    def calculate_train_stats(dataset_root, split_indexes, normalizer: Normalizer):
        episode_ends = dataset_root['meta']['episode_ends'][:]

        n_wrist_pos = BathroomDataset._get_wrist_pos(dataset_root)
        n_action_pos = BathroomDataset._get_action_pos(dataset_root)
        train_stats = {
            'wrist_pos': BathroomDataset.calculate_stats(n_wrist_pos, episode_ends, split_indexes, normalizer),
            'action_pos': BathroomDataset.calculate_stats(n_action_pos, episode_ends, split_indexes, normalizer)
        }
        return train_stats
    

    @staticmethod
    def default_dataset_split():
        # TODO: randomize split
        return { 
            "train" : [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                59, 60, 61, 62, 63, 64
            ],
            "valid" : [
                16
            ],
            "test" : [
                63
            ]
        }
