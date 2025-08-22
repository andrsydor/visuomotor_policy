from typing import List

import numpy as np
import torch


def create_sample_indices(
        episode_ends:np.ndarray, split_indexes:List[int], sequence_length:int,
        pad_before: int=0, pad_after: int=0):
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
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
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


def _get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats


def _normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def _unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


class Normalizer:
    @staticmethod
    def get_data_stats(data):
        raise NotImplementedError("should be implemented in child class")
    
    @staticmethod
    def normalize_data(data, stats):
        raise NotImplementedError("should be implemented in child class")
    
    @staticmethod
    def unnormalize_data(ndata, stats):
        raise NotImplementedError("should be implemented in child class")


class MinMaxNormalizer(Normalizer):
    @staticmethod
    def get_data_stats(data):
        data = data.reshape(-1, data.shape[-1])
        stats = {
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0)
        }
        return stats
    
    @staticmethod
    def normalize_data(data, stats):
        # nomalize to [0,1]
        ndata = (data - stats['min']) / (stats['max'] - stats['min'])
        # normalize to [-1, 1]
        ndata = ndata * 2 - 1
        return ndata

    @staticmethod
    def unnormalize_data(ndata, stats):
        ndata = (ndata + 1) / 2
        data = ndata * (stats['max'] - stats['min']) + stats['min']
        return data


class PercentileNormalizer(Normalizer):
    @staticmethod
    def get_data_stats(data):
        data = data.reshape(-1, data.shape[-1])
        stats = {
            'x_02': np.percentile(data, 2, axis=0),
            'x_98': np.percentile(data, 98, axis=0)
        }
        return stats
    
    @staticmethod
    def normalize_data(data, stats):
        # ~[0, 1]
        ndata = (data - stats['x_02']) / (stats['x_98'] - stats['x_02'])
        # ~[-1, 1]
        ndata = ndata * 2 - 1
        # clamp [-1.5, 1.5]
        ndata = np.clip(ndata, -1.5, 1.5)
        return ndata

    @staticmethod
    def unnormalize_data(ndata, stats):
        # ndata = np.clip(ndata, -1.5, 1.5)
        ndata = (ndata + 1) / 2
        data = ndata * (stats['x_98'] - stats['x_02']) + stats['x_02']
        return data


def pos_rot_to_mat(pos, rot):
    """
    Args:
        pos: (..., 3)
        rot: (..., 9)
    Returns:
        (..., 4, 4).
    """
    pos_flat, rot_flat = pos.reshape(-1, pos.shape[-1]), rot.reshape(-1, rot.shape[-1])
    rot_mat = rot_flat.reshape(rot_flat.shape[0], 3, 3)
    n = pos_flat.shape[0]

    if isinstance(pos, np.ndarray) and isinstance(rot, np.ndarray):
        result = np.zeros((n, 4, 4), dtype=pos.dtype)
    elif isinstance(pos, torch.Tensor) and isinstance(rot, torch.Tensor):
        result = torch.zeros((n, 4, 4), dtype=pos.dtype, device=pos.device)
    else:
        raise NotImplementedError("Unexpected input type")

    result[..., :3, 3] = pos_flat
    result[..., :3, :3] = rot_mat
    result[..., 3, 3] = 1.0
    return result.reshape(*pos.shape[:-1], 4, 4)


def mat_to_pos_rot(mat):
    """
    Args:
        mat: (..., 4, 4).
    Returns:
        ((..., 3), (..., 9)).
    """
    pos = (mat[..., :3, 3].T / mat[..., 3, 3].T).T
    rot = mat[..., :3, :3]
    rot = rot.reshape(*rot.shape[:-2], -1)
    return pos, rot


def rel_pose(poses_mat, base_pose_mat):
    """
    Args:
        poses_mat: (b, horizon, 4, 4).
        base_pose_mat: (b, 1, 4, 4).
    Returns:
        (b, horizon, 4, 4).
    """
    assert len(poses_mat.shape) == 4  # (b, horizon, 4, 4)
    b = poses_mat.shape[0]
    assert base_pose_mat.shape == (b, 1, 4, 4)
    result = np.linalg.inv(base_pose_mat) @ poses_mat
    return result


def apply_rel_pose(base_pose_mat, rel_poses_mat):
    """
    Args:
        base_pose_mat: (b, 1, 4, 4).
        rel_poses_mat: (b, horizon, 4, 4).
    Returns:
        (b, horizon, 4, 4).
    """
    assert len(rel_poses_mat.shape) == 4  # (b, horizon, 4, 4)
    b = rel_poses_mat.shape[0]
    assert base_pose_mat.shape == (b, 1, 4, 4)
    poses_mat = base_pose_mat @ rel_poses_mat
    return poses_mat


def calculate_rel_pos_and_rot(pos, rot, base_pos, base_rot):
    """
    Args:
        pos: (b, horizon, 3).
        rot: (b, horizon, 9).
        base_pos: (b, 1, 3).
        base_rot: (b, 1, 9).
    Returns:
        ((b, horizon, 3), (b, horizon, 9)).
    """
    batch_mat = pos_rot_to_mat(pos, rot)
    base_mat = pos_rot_to_mat(base_pos, base_rot)
    rel_mat = rel_pose(batch_mat, base_mat)
    rel_pos, rel_rot = mat_to_pos_rot(rel_mat)
    return rel_pos, rel_rot


def abs_pos_and_rot(rel_pos, rel_rot, base_pos, base_rot):
    """
    Args:
        pos: (b, horizon, 3).
        rot: (b, horizon, 9).
        base_pos: (b, 1, 3).
        base_rot: (b, 1, 9).
    Returns:
        ((b, horizon, 3), (b, horizon, 9)).
    """
    rel_mat = pos_rot_to_mat(rel_pos, rel_rot)
    base_mat = pos_rot_to_mat(base_pos, base_rot)
    mat = apply_rel_pose(base_mat, rel_mat)
    pos, rot = mat_to_pos_rot(mat)
    return pos, rot
