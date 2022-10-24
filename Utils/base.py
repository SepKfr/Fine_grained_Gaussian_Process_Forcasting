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

import enum
import abc
import numpy as np
import random

# Type defintions
class DataTypes(enum.IntEnum):
    """Defines numerical types of each column."""
    REAL_VALUED = 0
    CATEGORICAL = 1
    DATE = 2


class InputTypes(enum.IntEnum):
    """Defines input types of each column."""
    TARGET = 0
    OBSERVED_INPUT = 1
    KNOWN_INPUT = 2
    STATIC_INPUT = 3
    ID = 4  # Single column used as an entity identifier
    TIME = 5  # Single column exclusively used as a time index


class GenericDataFormatter(abc.ABC):
    """Abstract base class for all data_set formatters.
    User can implement the abstract methods below to perform dataset-specific
    manipulations.
    """

    @abc.abstractmethod
    def set_scalers(self, df):
        """Calibrates scalers using the data_set supplied."""
        raise NotImplementedError()

    @abc.abstractmethod
    def transform_inputs(self, df):
        """Performs feature transformation."""
        raise NotImplementedError()

    @abc.abstractmethod
    def format_predictions(self, df):
        """Reverts any normalisation to give predictions in original scale."""
        raise NotImplementedError()

    @abc.abstractmethod
    def transform_data(self, df):
        """Performs the default train, validation and test splits."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def _column_definition(self):
        """Defines order, input type and data_set type of each column."""
        raise NotImplementedError()

    def get_column_definition(self):
        """"Returns formatted column definition in order expected by the TFT."""

        column_definition = self._column_definition

        # Sanity checks first.
        # Ensure only one ID and time column exist
        def _check_single_column(input_type):
            length = len([tup for tup in column_definition if tup[2] == input_type])

            if length != 1:
                raise ValueError('Illegal number of inputs ({}) of type {}'.format(
                    length, input_type))

        _check_single_column(InputTypes.ID)
        _check_single_column(InputTypes.TIME)

        identifier = [tup for tup in column_definition if tup[2] == InputTypes.ID]
        time = [tup for tup in column_definition if tup[2] == InputTypes.TIME]
        real_inputs = [
            tup for tup in column_definition if tup[1] == DataTypes.REAL_VALUED and
                                                tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]
        categorical_inputs = [
            tup for tup in column_definition if tup[1] == DataTypes.CATEGORICAL and
                                                tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        return identifier + time + real_inputs + categorical_inputs

    def get_fixed_params(self):
        """Defines the fixed parameters used by the model for training.
        Requires the following keys:
          'total_time_steps': Defines the total number of time steps used by TFT
          'num_encoder_steps': Determines length of LSTM encoder (i.e. history)
          'num_epochs': Maximum number of epochs for training
          'early_stopping_patience': Early stopping param for keras
          'multiprocessing_workers': # of cpus for data_set processing
        Returns:
          A dictionary of fixed parameters, e.g.:
          fixed_params = {
              'total_time_steps': 252 + 5,
              'num_encoder_steps': 252,
              'num_epochs': 100,
              'early_stopping_patience': 5,
              'multiprocessing_workers': 5,
          }
        """
        raise NotImplementedError

    def get_num_samples_for_calibration(self):
        """Gets the default number of training and validation samples.
        Use to sub-sample the data_set for network calibration and a value of -1 uses
        all available samples.
        Returns:
          Tuple of (training samples, validation samples)
        """
        return -1, -1

    def get_experiment_params(self):
        """Returns fixed model parameters for experiments."""

        required_keys = [
           'total_time_steps'
        ]

        fixed_params = self.get_fixed_params()

        for k in required_keys:
            if k not in fixed_params:
                raise ValueError('Field {}'.format(k) +
                                 ' missing from fixed parameter definitions!')

        fixed_params['column_definition'] = self.get_column_definition()

        return fixed_params
