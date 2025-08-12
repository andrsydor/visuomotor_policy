import numpy as np

from src.visuomotor.dataset.bathroom_dataset import BathroomDataset
from src.visuomotor.dataset.tool import normalize_data, unnormalize_data


def test_calculate_stats():
    data = np.array([
        [5, 1], [7, -11], [3, 4], [6, 4], [3, 5], [2, 6], [3, 7],
        [3, 8], [3, 9], [9, 10], [3, 11], [-3, 12], [-13, 13], [3, 14],
        [3, 15], [3, 21], [11, 17]
    ])
    episode_ends = [17]
    split_indexes = [0]

    target = {
        "min": np.array([-13, -11]),
        "max": np.array([11, 21])
    }

    stats = BathroomDataset.calculate_stats(data, episode_ends, split_indexes)
    for stats_key, stats_value in target.items():
        assert (stats[stats_key] == stats_value).all()


def test_normaliza_data():
    data = np.array([
        [5, -5], [-10, 4]
    ])

    stats = {
        "min": np.array([-10, -20]),
        "max": np.array([20, 20])
    }

    target = np.array([
        [0.0, -0.25], [-1.0, 0.2]
    ])

    normalized = normalize_data(data, stats)

    assert np.allclose(normalized, target)


def test_unnormalize_data():
    data = np.array([
        [0.0, -0.25], [-1.0, 0.2]
    ])

    stats = {
        "min": np.array([-10, -20]),
        "max": np.array([20, 20])
    }

    target = np.array([
        [5, -5], [-10, 4]
    ])

    unnormalized = unnormalize_data(data, stats)

    assert np.allclose(unnormalized, target)
