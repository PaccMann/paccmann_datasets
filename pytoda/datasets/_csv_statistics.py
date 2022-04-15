"""Abstract implementation of _CsvStatistics."""
import copy
from functools import reduce

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..types import CallableOnSource, FeatureList, List, Optional, OrderedDict, Tuple


class _CsvStatistics:
    """.csv abstract setup for dataset statistics."""

    def __init__(
        self,
        filepath: str,
        feature_list: FeatureList = None,
        pandas_dtype=None,
        **kwargs,
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
        self.initial_feature_list = feature_list  # see `setup_datasource`
        self.min_max_scaler = MinMaxScaler()
        self.standardizer = StandardScaler()
        self.kwargs = copy.deepcopy(kwargs)
        self.kwargs['dtype'] = pandas_dtype

        self.preprocess_df = (
            # may be applied to many chunks, so logic is determined once here
            self._reindex
            if self.initial_feature_list
            else self._id
        )
        self.setup_datasource()

        # self.notna_count
        self.max = self.min_max_scaler.data_max_
        self.min = self.min_max_scaler.data_min_
        self.mean = self.standardizer.mean_
        self.std = self.standardizer.scale_

    def _reindex(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure given order of features, creating NaN columns for missing."""
        return df.reindex(columns=self.initial_feature_list, fill_value=np.NaN)

    def _id(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def setup_datasource(self):
        """
        Setup the datasource and compute statistics.

        The dataframe is read, calling `self.preprocess_df` on it and setting
        up the data as source, collecting statistics of the data.

        NOTE:
        To read item with a different subset and order of features,
        use the function returned by `get_feature_fn` or use the
        feature_mapping to ensure integer indexing.

        Sets:
        feature_list (FeatureList): feature names in this datasource.
        feature_mapping (pd.Series): maps feature name to index in items.
        feature_fn (CallableOnSource): function that indexes datasource with
            the feature_list.
        """
        raise NotImplementedError

    def get_feature_fn(self, feature_list: FeatureList) -> CallableOnSource:
        """Provides datasource specific indexing.

        Args:
            feature_list (FeatureList): subset of features to return in order.

        Returns:
            CallableOnSource: function that indexes datasource with the
                feature_list.
        """
        raise NotImplementedError

    def transform_datasource(
        self,
        transform_fn: CallableOnSource,
        feature_fn: CallableOnSource,
        impute: Optional[float] = None,
    ) -> None:
        """Apply scaling to the datasource.

        Args:
            transform_fn (CallableOnSource): transformation on source data.
            feature_fn (CallableOnSource): function that indexes datasource.
            impute (Optional[float]): NaN imputation with value if
                given. Defaults to None.
        """
        raise NotImplementedError

    def transform_dataset(
        self,
        transform_fn: CallableOnSource,
        feature_list: FeatureList,
        impute: Optional[float] = None,
    ) -> None:
        """Apply filtering, ordering and scaling to the datasource and update
        the dataset accordingly.

        Args:
            transform_fn (CallableOnSource): data scaling function, possibly
                feature wise.
            feature_list (FeatureList): subset of features to return in order.
            impute (Optional[float]): NaN imputation with value if
                given. Defaults to None.

        Sets:
        feature_list (FeatureList): feature names in this datasource.
        feature_mapping (pd.Series): maps feature name to index in items.
        feature_fn (CallableOnSource): function that indexes datasource with
            the feature_list.
        """
        # feature_fn for current state of datasource
        feature_fn = self.get_feature_fn(feature_list)
        statistics_indices = self.feature_mapping[feature_list].values  # noqa
        # update datasource
        self.transform_datasource(transform_fn, feature_fn, impute)
        # update statistics ordering (this could allow inverse transformation)
        self.max = self.max[statistics_indices]
        self.min = self.min[statistics_indices]
        self.mean = self.mean[statistics_indices]
        self.std = self.std[statistics_indices]
        self.notna_count = self.notna_count[statistics_indices]
        # delete outdated scalers
        try:
            del self.min_max_scaler
            del self.standardizer
        except NameError:
            pass
        # update dataset reflecting the transformation
        self.feature_list = feature_list
        self.feature_mapping = pd.Series(
            OrderedDict(
                [(feature, index) for index, feature in enumerate(self.feature_list)]
            )
        )
        self.feature_fn = self.get_feature_fn(self.feature_list)

    def __len__(self) -> int:
        raise NotImplementedError


def reduce_csv_statistics(
    csv_datasets: List[_CsvStatistics],
    feature_list: FeatureList = None,
) -> Tuple[FeatureList, np.array, np.array, np.array, np.array]:
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
            lambda a_set, another_set: a_set & another_set,
            [set(csv_dataset.feature_mapping.keys()) for csv_dataset in csv_datasets],
        )
    )

    if feature_list is not None:
        # to sort features by key
        feature_ordering = {
            feature: index for index, feature in enumerate(feature_list)
        }
        features = sorted(features, key=lambda feature: feature_ordering[feature])
    else:
        # sorting the strings
        features = sorted(features)

    # features is the definite features_list

    maximums, minimums, means, stds, sample_numbers = zip(
        *[
            (
                # NOTE: here we ensure that we pick the statistics
                # in the right order via indexes provided by the dataset
                csv_dataset.max[csv_dataset.feature_mapping[features].values],
                csv_dataset.min[csv_dataset.feature_mapping[features].values],
                csv_dataset.mean[csv_dataset.feature_mapping[features].values],
                csv_dataset.std[csv_dataset.feature_mapping[features].values],
                # len(csv_dataset)
                csv_dataset.notna_count[csv_dataset.feature_mapping[features].values],
            )
            for csv_dataset in csv_datasets
        ]
    )
    # NOTE: reduce max and min
    maximum = np.nanmax(maximums, axis=0)
    minimum = np.nanmin(minimums, axis=0)
    # NOTE: reduce the mean
    total_number_of_samples = sum(sample_numbers).astype(float)
    mean = (
        np.nansum(
            [
                dataset_mean * number_of_samples
                for dataset_mean, number_of_samples in zip(means, sample_numbers)
            ],
            axis=0,
        )
        / total_number_of_samples
    )
    # NOTE: reduce the std
    std = np.sqrt(
        np.nansum(
            [
                (dataset_std**2 + dataset_mean**2) * number_of_samples
                for dataset_std, dataset_mean, number_of_samples in zip(
                    stds, means, sample_numbers
                )
            ],
            axis=0,
        )
        / total_number_of_samples
        - mean**2
    )
    return (features, maximum, minimum, mean, std)
