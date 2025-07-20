from typing import Dict, List

import torch
import numpy as np

from visuomotor.config.base_policy_config import BasePolicyConfig
from visuomotor.dataset.dataset_base_config import DatasetBaseConfig
from visuomotor.dataset.tool import normalize_data, create_sample_indices, sample_sequence, get_data_stats


class PushTImageDatasetConfig(DatasetBaseConfig):
    def __init__(
            self,
            policy_config: BasePolicyConfig,
            episode_indexes: List[int],
    ):
        self.policy_config = policy_config
        self.episode_indexes = episode_indexes


class PushTImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_root,
                 split_indexes: List[int],
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 stats: Dict[str, np.array]):

        train_image_data = dataset_root['data']['img'][:]
        train_image_data = np.moveaxis(train_image_data, -1, 1)

        train_data = {
            'agent_pos': dataset_root['data']['state'][:,:2],
            'action': dataset_root['data']['action'][:]
        }

        normalized_train_data = dict()
        for key, data in train_data.items():
            normalized_train_data[key] = normalize_data(data, stats[key])

        # TODO: check if normalized images
        normalized_train_data['image'] = train_image_data

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
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['image'] = nsample['image'][:self.obs_horizon,:]
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon,:]
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
            'agent_pos': PushTImageDataset.calculate_stats(dataset_root['data']['state'][:,:2], episode_ends, split_indexes),
            'action': PushTImageDataset.calculate_stats(dataset_root['data']['action'][:], episode_ends, split_indexes)
        }
        return train_stats
    

    @staticmethod
    def default_dataset_split():
        return { 
            "train" : [
                0, 1, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20,
                21, 22, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36, 37, 38, 39,
                40, 41, 42, 43, 44, 45, 51, 52, 53, 54, 55, 56, 57, 58,
                59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 74, 75, 76,
                78, 79, 80, 83, 84, 87, 88, 89, 90, 91, 92, 93, 95, 96,
                97, 98, 99, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                113, 114, 116, 117, 118, 120, 122, 124, 125, 126, 127,
                128, 131, 132, 134, 135, 136, 137, 138, 140, 141,
                143, 144, 145, 146, 148, 149, 150, 153, 154, 155, 156, 157,
                159, 161, 162, 163, 165, 168, 169, 172,
                173, 175, 176, 177, 178, 179, 182, 183, 184, 186, 187,
                189, 190, 191, 194, 195, 197, 199, 202, 203, 205
            ],
            "valid" : [
                2, 3, 4, 12, 16, 23, 46, 47, 48, 49,
                50, 65, 71, 77, 81, 82, 94, 101, 115, 123,
                129, 130, 133, 147, 151, 152, 164, 166, 170, 171,
                185, 188, 192, 193, 198, 200, 201
            ],
            "test" : [
                5, 29, 30, 31, 73, 85, 86, 100, 119, 121,
                139, 142, 158, 160, 167, 174, 180, 181, 196, 204,
            ]
        }