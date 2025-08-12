import numpy as np

from src.visuomotor.dataset.bathroom_dataset import BathroomDataset
from src.visuomotor.dataset.tool import normalize_data


def test_bathroom_dataset():

    data = {
        "realsense": np.array([
            [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 14], [1, 15], [1, 16], [1, 17]
        ]),
        "depth_camera": np.array([
            [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14], [2, 15], [2, 16], [2, 17]
        ]),
        "wrist_rot": np.array([
            [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11], [3, 12], [3, 13], [3, 14], [3, 15], [3, 16], [3, 17]
        ]),
        "wrist_pos": np.array([
            [7, 1], [7, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10], [5, 11], [5, 12], [5, 13], [5, 14], [5, 15], [5, 16], [5, 17]
        ]),
    }
    meta = {
        "episode_ends": [17]
    }
    dataset_root = {
        "data": data,
        "meta": meta
    }

    real_stats = {
        'wrist_pos': {'min': np.array([5, 1]), 'max': np.array([ 7, 17])},
        'action_pos': {'min': np.array([5, 2]), 'max': np.array([ 7, 17])}
    }

    case_1 = {
        "pred_horizon": 6,
        "obs_horizon": 2,
        "action_horizon": 4,
        "index": 0,
        "target": {
            "realsense": np.array([
                [1, 1]
            ]),
            "depth_camera": np.array([
                [2, 1]
            ]),
            "wrist_rot": np.array([
                [3, 1], [3, 1]
            ]),
            "wrist_pos": normalize_data(
                np.array([[7, 1], [7, 1]]),
                real_stats["wrist_pos"]
            ),
            "action_rot": np.array([
                [3, 2], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6]
            ]),
            "action_pos": normalize_data(
                np.array([[7, 2], [7, 2], [5, 3], [5, 4], [5, 5], [5, 6]]),
                real_stats["action_pos"]
            ),
        }
    }

    case_2 = {
        "pred_horizon": 6,
        "obs_horizon": 2,
        "action_horizon": 4,
        "index": 1,
        "target": {
            "realsense": np.array([
                [1, 2]
            ]),
            "depth_camera": np.array([
                [2, 2]
            ]),
            "wrist_rot": np.array([
                [3, 1], [3, 2]
            ]),
            "wrist_pos": normalize_data(
                np.array([[7, 1], [7, 2]]),
                real_stats["wrist_pos"]
            ),
            "action_rot": np.array([
                [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7]
            ]),
            "action_pos": normalize_data(
                np.array([[7, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7]]),
                real_stats["action_pos"]
            ),
        }
    }

    case_3 = {
        "pred_horizon": 6,
        "obs_horizon": 2,
        "action_horizon": 4,
        "index": 15,
        "target": {
            "realsense": np.array([
                [1, 16]
            ]),
            "depth_camera": np.array([
                [2, 16]
            ]),
            "wrist_rot": np.array([
                [3, 15], [3, 16]
            ]),
            "wrist_pos": normalize_data(
                np.array([[5, 15], [5, 16]]),
                real_stats["wrist_pos"]
            ),
            "action_rot": np.array([
                [3, 16], [3, 17], [3, 17], [3, 17], [3, 17], [3, 17]
            ]),
            "action_pos": normalize_data(
                np.array([[5, 16], [5, 17], [5, 17], [5, 17], [5, 17], [5, 17]]),
                real_stats["action_pos"]
            ),
        }
    }

    cases = [case_1, case_2, case_3]

    stats = BathroomDataset.calculate_train_stats(dataset_root, [0])
    print(stats)

    for case in cases:
        pred_horizon = case["pred_horizon"]
        obs_horizon = case["obs_horizon"]
        action_horizon = case["action_horizon"]
        pred_horizon = case["pred_horizon"]
        index = case["index"]
        target = case["target"]

        dataset = BathroomDataset(
            dataset_root,
                split_indexes=[0],
                pred_horizon=pred_horizon,
                obs_horizon=obs_horizon,
                action_horizon=action_horizon,
                stats=stats
        )
        dataset_sample = dataset.__getitem__(index)
        assert dataset_sample.keys() == target.keys()
        for key, target_item in target.items():
            assert (target_item == dataset_sample[key]).all()
