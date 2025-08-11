from typing import Dict, List

import torch
import numpy as np

from visuomotor.dataset.tool import normalize_data, create_sample_indices, sample_sequence, get_data_stats


class BathroomDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_root,
            split_indexes: List[int],
            pred_horizon: int,
            obs_horizon: int,
            action_horizon: int,
            stats: Dict[str, np.array]
    ):

        realsense_data = dataset_root['data']['realsense']
        # realsense_data = np.moveaxis(realsense_data, -1, 1)  # TODO: refactor

        depth_data = dataset_root['data']['depth_camera']
        # depth_data = np.moveaxis(realsense_data, -1, 1)  # TODO: refactor
        
        train_data = {
            'wrist_pos': dataset_root['data']['wrist'][:],
            'action': np.concat((dataset_root["data"]["wrist"][1:], dataset_root["data"]["wrist"][-1:]), axis=0)  # state but shifter forward
        }

        normalized_train_data = dict()
        for key, data in train_data.items():
            normalized_train_data[key] = normalize_data(data, stats[key])

        # TODO: check if normalized images
        normalized_train_data['realsense'] = realsense_data
        normalized_train_data['depth_camera'] = depth_data

        episode_ends = dataset_root['meta']['episode_ends'][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            split_indexes=split_indexes,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(  # TODO: optimize
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['realsense'] = nsample['realsense'][:self.obs_horizon,:]
        nsample['depth_camera'] = nsample['depth_camera'][:self.obs_horizon,:]
        nsample['wrist_pos'] = nsample['wrist_pos'][:self.obs_horizon,:]
        return nsample

    @staticmethod
    def calculate_stats(np_data, episode_ends, split_indexes):
        train = []
        data = np_data.tolist()
        for i in split_indexes:
            start_i = 0
            if i > 0:
                start_i = episode_ends[i - 1]
            end_i = episode_ends[i]
            train += data[start_i:end_i]
        return get_data_stats(np.array(train))
    
    @staticmethod
    def calculate_train_stats(dataset_root, split_indexes):
        episode_ends = dataset_root['meta']['episode_ends'][:]
        train_stats = {
            'wrist_pos': BathroomDataset.calculate_stats(dataset_root['data']['wrist'][:], episode_ends, split_indexes),
            'action': BathroomDataset.calculate_stats(np.concat((dataset_root["data"]["wrist"][1:], dataset_root["data"]["wrist"][-1:]), axis=0), episode_ends, split_indexes)  # TODO: remove code copy
        }
        return train_stats
    

    @staticmethod
    def default_dataset_split():
        # TODO: randomize split
        return { 
            "train" : [
                0, 1, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20,
                21, 22, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36, 37, 38, 39,
                40, 41, 42, 43, 44, 45, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                59, 60, 61, 62, 63, 64, 66
            ],
            "valid" : [
                2, 3, 4, 12, 16, 23, 46, 47, 48, 49
            ],
            "test" : [
                5, 29, 30, 31, 65
            ]
        }