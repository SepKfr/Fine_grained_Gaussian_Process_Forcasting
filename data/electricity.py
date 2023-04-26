# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

import pandas as pd
import sklearn.preprocessing

from Utils import utils
from Utils.base import GenericDataFormatter, DataTypes, InputTypes

GenericDataFormatter = GenericDataFormatter


class ElectricityFormatter(GenericDataFormatter):
    """Defines and formats data_set for the electricity dataset.
    Note that per-entity z-score normalization is used here, and is implemented
    across functions.
    Attributes:
    column_definition: Defines input and data_set type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
    """

    _column_definition = [
      ('id', DataTypes.REAL_VALUED, InputTypes.ID),
      ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
      ('power_usage', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

    def __init__(self, pred_len):
        """Initialises formatter."""

        self.pred_len = pred_len
        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        self._time_steps = self.get_fixed_params()['total_time_steps']

    def transform_data(self, df):
        """Splits data_set frame into training-validation-test data_set frames.
        This also calibrates scaling object, and transforms data_set for each split.
        Args:
          df: Source data_set frame to split.
          valid_boundary: Starting year for validation data_set
          test_boundary: Starting year for test data_set
        Returns:
          Tuple of transformed (train, valid, test) data_set.
        """

        print('Formatting data.')

        self.set_scalers(df)

        return self.transform_inputs(df)

    def set_scalers(self, df):

        """Calibrates scalers using the data_set supplied.
        Args:
          df: Data to use to calibrate scalers.
        """
        print('Setting scalers with data_set...')

        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                       column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                           column_definitions)

        # Format real scalers
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Initialise scaler caches
        self._real_scalers = {}
        self._target_scaler = {}
        identifiers = []
        for identifier, sliced in df.groupby(id_column):

          if len(sliced) >= self._time_steps:

            data = sliced[real_inputs].values
            targets = sliced[[target_column]].values
            self._real_scalers[identifier] \
          = sklearn.preprocessing.StandardScaler().fit(data)

            self._target_scaler[identifier] \
          = sklearn.preprocessing.StandardScaler().fit(targets)

          identifiers.append(identifier)

        # Format categorical scalers
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
          srs = df[col].apply(str)
          categorical_scalers[col] = sklearn.preprocessing.LabelEncoder()
          categorical_scalers[col].fit(srs.values)
          num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

        # Extract identifiers in case required
        self.identifiers = identifiers

    def transform_inputs(self, df):
        """Performs feature transformations.
        This includes both feature engineering, preprocessing and normalisation.
        Args:
          df: Data frame to transform.
        Returns:
          Transformed data_set frame.
        """

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        # Extract relevant columns
        column_definitions = self.get_column_definition()
        id_col = utils.get_single_col_by_input_type(InputTypes.ID,
                                                    column_definitions)
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Transform real inputs per entity
        df_list = []
        for identifier, sliced in df.groupby(id_col):
            # Filter out any trajectories that are too short
            if len(sliced) >= self._time_steps:
                sliced_copy = sliced.copy()
                sliced_copy[real_inputs] = self._real_scalers[identifier].transform(
                    sliced_copy[real_inputs].values)
                df_list.append(sliced_copy)

        output = pd.concat(df_list, axis=0)

        # Format categorical inputs
        for col in categorical_inputs:

            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df[:len(output)])

        return output

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.
        Args:
          predictions: Dataframe of model predictions.
        Returns:
          Data frame of unnormalised predictions.
        """

        if self._target_scaler is None:
            raise ValueError('Scalers have not been set!')

        column_names = predictions.columns

        df_list = []
        for identifier, sliced in predictions.groupby('identifier'):
            sliced_copy = sliced.copy()
            target_scaler = self._target_scaler[identifier]

            for col in column_names:
                if col not in {'identifier'}:
                    try:
                        sliced_copy[col] = target_scaler.inverse_transform(sliced_copy[col])
                    except ValueError:
                        if len(sliced_copy[col]) == 1:
                            pred = sliced_copy[col].to_numpy().reshape(1, -1)
                        else:
                            pred = sliced_copy[col].to_numpy().reshape(-1, 1)

                        sliced_copy[col] = target_scaler.inverse_transform(pred)

            df_list.append(sliced_copy)
        if len(df_list) == 0:
            output = None
        else:
            output = pd.concat(df_list, axis=0)

        return output

    def get_default_model_params(self):
        """Returns default optimised model parameters."""

        model_params = {
            'hidden_layer_size': [8],
            'minibatch_size': [512],
            'num_heads': 8,
            'stack_size': [1],
            'context_lengths': [1, 3, 6, 9]
        }

        return model_params

    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps': 8 * 24 + self.pred_len,
            'num_encoder_steps': 4 * 24,
            'num_decoder_steps': self.pred_len,
            'num_epochs': 50,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 5
        }

        return fixed_params

    def get_num_samples_for_calibration(self, num_train=-1):

        """Gets the default number of training and validation samples.
        Use to sub-sample the data_set for network calibration and a value of -1 uses
        all available samples.
        Returns:
          Tuple of (training samples, validation samples)
        """
        if num_train == -1:

            return 32000, 3840

        else:
            return num_train, 3840