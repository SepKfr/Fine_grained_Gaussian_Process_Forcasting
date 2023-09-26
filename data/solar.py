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

from Utils.base import DataTypes, InputTypes
from data.electricity import ElectricityFormatter

DataFormatter = ElectricityFormatter


class SolarFormatter(ElectricityFormatter):

    _column_definition = [
        ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
        ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('Power(MW)', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('id', DataTypes.REAL_VALUED, InputTypes.ID),
        ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('capacity', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),

    ]

    def split_data(self, df, valid_boundary=214, test_boundary=289):
        print('Formatting train-valid-test splits.')

        index = df['days_from_start']
        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary - 7) & (index < test_boundary)]
        test = df.loc[index >= test_boundary - 7]

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

    def get_default_model_params(self):
        """Returns default optimised model parameters."""

        model_params = {
            'hidden_layer_size': [8],
            'minibatch_size': [256],
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