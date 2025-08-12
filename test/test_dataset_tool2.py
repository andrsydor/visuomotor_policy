import numpy as np

from src.visuomotor.dataset.tool2 import sample_indices, get_sample, get_lazy_sample


def test_sample_indices():
    episode_ends = [17]
    split_indexes = [0]
    sequence_length = 6
    pad_before = 1
    pad_after = 3

    indices = sample_indices(episode_ends, split_indexes, sequence_length, pad_before, pad_after)

    print(indices)
    # assert False


def test_get_sample():
    train_data = {
        "image": np.array([
            [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 14], [1, 15], [1, 16], [1, 17]
        ]),
        "action": np.array([
            [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14], [2, 15], [2, 16], [2, 17]
        ])
    }
    sequence_length = 6

    indices_pairs = [
        [ 0,  5,  1,  6],
        [ 2,  8,  0,  6],
        [14, 17,  0,  3],
        [ 0,  4,  2,  6],
    ]
    target_samples = [
        {
            "image": np.array([
                [1, 1], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5]
            ]),
            "action": np.array([
                [2, 1], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5]
            ])
        },
        {
            "image": np.array([
                [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8]
            ]),
            "action": np.array([
                [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8]
            ])
        },
        {
            "image": np.array([
                [1, 15], [1, 16], [1, 17], [1, 17], [1, 17], [1, 17]
            ]),
            "action": np.array([
                [2, 15], [2, 16], [2, 17], [2, 17], [2, 17], [2, 17]
            ])
        },
        {
            "image": np.array([
                [1, 1], [1, 1], [1, 1], [1, 2], [1, 3], [1, 4]
            ]),
            "action": np.array([
                [2, 1], [2, 1], [2, 1], [2, 2], [2, 3], [2, 4]
            ])
        },
    ]

    assert len(indices_pairs) == len(target_samples)
    for indices, target in zip(indices_pairs, target_samples):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices
        sample = get_sample(
            train_data,
            sequence_length,
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx
        )
        assert sample.keys() == target.keys()
        for key, target_item in target.items():
            assert (target_item == sample[key]).all()


def test_get_lazy_sample():
    train_data = {
        "image": np.array([
            [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 14], [1, 15], [1, 16], [1, 17]
        ]),
        "action": np.array([
            [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14], [2, 15], [2, 16], [2, 17]
        ])
    }
    sequence_length = 6

    indices_pairs = [
        [ 0,  5,  1,  6,  2],
        [ 2,  8,  0,  6,  2],
        [14, 17,  0,  3,  2],
        [ 0,  4,  2,  6,  3],
        [ 0,  6,  0,  6,  2]
    ]
    target_samples = [
        {
            "image": np.array([
                [1, 1]
            ]),
            "action": np.array([
                [2, 1], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5]
            ])
        },
        {
            "image": np.array([
                [1, 4]
            ]),
            "action": np.array([
                [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8]
            ])
        },
        {
            "image": np.array([
                [1, 16]
            ]),
            "action": np.array([
                [2, 15], [2, 16], [2, 17], [2, 17], [2, 17], [2, 17]
            ])
        },
        {
            "image": np.array([
                [1, 1]
            ]),
            "action": np.array([
                [2, 1], [2, 1], [2, 1], [2, 2], [2, 3], [2, 4]
            ])
        },
        {
            "image": np.array([
                [1, 2]
            ]),
            "action": np.array([
                [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6]
            ])
        },
    ]

    assert len(indices_pairs) == len(target_samples)
    for indices, target in zip(indices_pairs, target_samples):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx, obs_horizon = indices
        sample = get_lazy_sample(
            train_data,
            sequence_length,
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
            image_keys=["image"],
            obs_horizon=obs_horizon
        )
        assert sample.keys() == target.keys()
        for key, target_item in target.items():
            assert (target_item == sample[key]).all()
