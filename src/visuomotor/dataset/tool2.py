from typing import List

import numpy as np


def sample_indices(
        episode_ends: np.ndarray,
        split_indexes: List[int],
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0
):
    indices = list()
    for i in split_indexes:
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx,
                buffer_end_idx,
                sample_start_idx,
                sample_end_idx
            ])
    indices = np.array(indices)
    return indices


def get_sample(
        train_data,
        sequence_length,
        buffer_start_idx,
        buffer_end_idx,
        sample_start_idx,
        sample_end_idx
):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

def get_lazy_sample(
        train_data,
        sequence_length,
        buffer_start_idx,
        buffer_end_idx,
        sample_start_idx,
        sample_end_idx,
        image_keys,
        obs_horizon
):
    pad_before = obs_horizon - 1

    result = dict()
    for key, input_arr in train_data.items():
        if key in image_keys:
            current_image_index = buffer_start_idx + pad_before  # to take current timestemp image
            current_image_index -= sample_start_idx  # to compensate the pad shift
            data = input_arr[current_image_index:current_image_index+1]
            result[key] = data
            continue

        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result
