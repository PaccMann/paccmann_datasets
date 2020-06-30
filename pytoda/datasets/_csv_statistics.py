"""Abstract implementation of _CsvStatistics."""
import copy
import numpy as np
from functools import reduce
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List, Tuple
from ..types import FeatureList


class _CsvStatistics:
    """.csv abstract setup for dataset statistics."""

    def __init__(
        self, filepath: str, feature_list: FeatureList = None,
        pandas_dtype=None, **kwargs
    ) -> None:
        """
        Initialize a .csv dataset.

        Args:
            filepath (str): path to .csv file.
            feature_list (FeatureList): a list of features. Defaults to None.
            pandas_dtype (str, type, dict): Optional parameter added to
                kwargs (and passed to pd.read_csv) as 'dtype'. Defaults to
                    None.
            kwargs (dict): additional parameters for pd.read_csv.

        """
        self.filepath = filepath
        self.feature_list = feature_list
        self.min_max_scaler = MinMaxScaler()
        self.standardizer = StandardScaler()
        self.kwargs = copy.deepcopy(kwargs)
        self.kwargs['dtype'] = pandas_dtype
        if self.feature_list is not None:
            # NOTE: zeros denote missing value
            self.feature_fn = lambda df: df.T.reindex(self.feature_list
                                                      ).T.fillna(0.0)
        else:
            self.feature_fn = lambda df: df
        self.setup_datasource()
        self.max = self.min_max_scaler.data_max_
        self.min = self.min_max_scaler.data_min_
        self.mean = self.standardizer.mean_
        self.std = self.standardizer.scale_

    def setup_datasource(self) -> None:
        """
        Setup the datasource, compute statistics, and define feature_mapping
        (to order).
        """
        raise NotImplementedError


def reduce_csv_statistics(
    csv_datasets: List[_CsvStatistics],
    feature_list:
    FeatureList = None,
) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Reduce datasets statistics.

    Args:
        csv_datasets (List[_CsvStatistics]): list of .csv datasets.
        feature_list (FeatureList): a list of features important to guarantee
            feature order preservation when multiple datasets are passed.
            Defaults to None, where features are string sorted.
    Returns:
        Tuple[FeatureList, np.array, np.array, np.array, np.array]: updated
            statistics with the following components:
                features (FeatureList): List of common, ordered features.
                maximum (np.array): Maximum per feature.
                minimum (np.array): Minimum per feature.
                mean (np.array): Mean per feature.
                std (np.array): Standard deviation per feature.

    """
    # collected common features
    features = list(
        reduce(
            lambda a_set, another_set: a_set & another_set, [
                set(csv_dataset.feature_mapping.keys())
                for csv_dataset in csv_datasets
            ]
        )
    )

    if feature_list is not None:
        # to sort features by key
        feature_ordering = {
            feature: index
            for index, feature in enumerate(feature_list)
        }
        features = sorted(
            features, key=lambda feature: feature_ordering[feature]
        )
    else:
        # sorting the strings
        features = sorted(features)
    maximums, minimums, means, stds, sample_numbers = zip(
        *[
            (
                # NOTE: here we ensure that we pick the statistics
                # in the right order via indexes
                csv_dataset.max[csv_dataset.feature_mapping[features].values],
                csv_dataset.min[csv_dataset.feature_mapping[features].values],
                csv_dataset.mean[csv_dataset.feature_mapping[features].values],
                csv_dataset.std[csv_dataset.feature_mapping[features].values],
                len(csv_dataset)
            ) for csv_dataset in csv_datasets
        ]
    )
    # NOTE: reduce max and min
    maximum = np.array(maximums).max(axis=0)
    minimum = np.array(minimums).min(axis=0)
    # NOTE: reduce the mean
    total_number_of_samples = float(sum(sample_numbers))
    mean = np.array(
        [
            dataset_mean * number_of_samples
            for dataset_mean, number_of_samples in zip(means, sample_numbers)
        ]
    ).sum(axis=0) / total_number_of_samples
    # NOTE: reduce the std
    std = np.sqrt(
        (
            np.array(
                [
                    (dataset_std**2 + dataset_mean**2) * number_of_samples
                    for dataset_std, dataset_mean, number_of_samples in
                    zip(stds, means, sample_numbers)
                ]
            ).sum(axis=0) / total_number_of_samples - mean**2
        )
    )
    return (features, maximum, minimum, mean, std)
