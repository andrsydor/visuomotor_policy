import numpy as np

from src.visuomotor.dataset.bathroom_dataset import BathroomDataset
from src.visuomotor.dataset.tool import _normalize_data, _unnormalize_data, MinMaxNormalizer, PercentileNormalizer


def test_calculate_stats_MinMaxNormalizer():
    data = np.array([
        [5, 1], [7, -11], [3, 4], [6, 4], [3, 5], [2, 6], [3, 7],
        [3, 8], [3, 9], [9, 10], [3, 11], [-3, 12], [-13, 13], [3, 14],
        [3, 15], [3, 21], [11, 17],
        [19, 29], [234, 213]
    ])
    episode_ends = [17, 19]
    split_indexes = [0]

    target = {
        "min": np.array([-13, -11]),
        "max": np.array([11, 21])
    }

    stats = BathroomDataset.calculate_stats(data, episode_ends, split_indexes, MinMaxNormalizer)
    for stats_key, stats_value in target.items():
        assert (stats[stats_key] == stats_value).all()


def test_calculate_stats_MinMaxNormalizer_2():
    data = np.array([
        [5, 1], [7, -11], [3, 4], [6, 4], [3, 5], [2, 6], [3, 7],
        [3, 8], [3, 9], [9, 10], [3, 11], [-3, 12], [-13, 13], [3, 14],
        [3, 15], [3, 21], [11, 17],
        [1, 29], [8, 213], [0, 2]
    ])
    episode_ends = [17, 20]
    split_indexes = [1]

    target = {
        "min": np.array([0, 2]),
        "max": np.array([8, 213])
    }

    stats = BathroomDataset.calculate_stats(data, episode_ends, split_indexes, MinMaxNormalizer)
    for stats_key, stats_value in target.items():
        assert (stats[stats_key] == stats_value).all()


def test_calculate_stats_PercentileNormalizer():
    data = np.array([
        [4, -2], [2, -3], [1, 9], [3, 5], [6, 6], [5, 10],
        [11, 12], [99, 100], [99, -123]
    ])
    episode_ends = [6, 9]
    split_indexes = [0]

    target = {
        "x_02": np.array([1.1, -2.9]),
        "x_98": np.array([5.9, 9.9])
    }

    stats = BathroomDataset.calculate_stats(data, episode_ends, split_indexes, PercentileNormalizer)
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

    normalized = _normalize_data(data, stats)

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

    unnormalized = _unnormalize_data(data, stats)

    assert np.allclose(unnormalized, target)


def test_MinMaxNormalizer_get_data_stats():
    data = np.array([
        [5, -5], [-10, -11], [-2, 2], [1, 3]
    ])

    target = {
        "min": np.array([-10, -11]),
        "max": np.array([5, 3])
    }

    stats = MinMaxNormalizer.get_data_stats(data)

    assert target.keys() == stats.keys()
    for stats_key, stats_value in target.items():
        assert (stats[stats_key] == stats_value).all()


def test_MinMaxNormalizer_normaliza_data():
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

    normalized = MinMaxNormalizer.normalize_data(data, stats)

    assert np.allclose(normalized, target)


def test_MinMaxNormalizer_unnormalize_data():
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

    unnormalized = MinMaxNormalizer.unnormalize_data(data, stats)

    assert np.allclose(unnormalized, target)


def test_PercentileNormalizer_get_data_stats():
    data = np.array([
        [4, -2], [2, -3], [1, 9], [3, 5], [6, 6], [5, 10]
    ])

    target = {
        "x_02": np.array([1.1, -2.9]),
        "x_98": np.array([5.9, 9.9])
    }

    stats = PercentileNormalizer.get_data_stats(data)
    print(stats)

    assert target.keys() == stats.keys()
    for stats_key, stats_value in target.items():
        assert (stats[stats_key] == stats_value).all()


def test_PercentileNormalizer_normaliza_data():
    data = np.array([
        [5, -5], [10, 4], [15, 7]
    ])

    stats = {
        "x_02": np.array([1.0, -1.0]),
        "x_98": np.array([9.0, 5.0])
    }

    target = np.array([
        [0.0, -1.5], [1.25, 0.6666666], [1.5, 1.5]
    ])

    normalized = PercentileNormalizer.normalize_data(data, stats)

    assert np.allclose(normalized, target)


def test_PercentileNormalizer_unnormalize_data():
    data = np.array([
        [0.0, -1.5], [1.25, 0.6666666], [1.5, 1.5]
    ])

    stats = {
        "x_02": np.array([1.0, -1.0]),
        "x_98": np.array([9.0, 5.0])
    }

    # target = np.array([
    #     [5, -2.5], [10, 4], [11, 6.5]
    # ])
    target = np.array([
        [5, -2.5], [10, 4], [11, 6.5]
    ])

    unnormalized = PercentileNormalizer.unnormalize_data(data, stats)

    assert np.allclose(unnormalized, target)
