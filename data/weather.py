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

import sklearn.preprocessing
from Utils import utils, base
from data.traffic import TrafficFormatter

GenericDataFormatter = TrafficFormatter
DataTypes = base.DataTypes
InputTypes = base.InputTypes


class weatherFormatter(TrafficFormatter):

    _column_definition = [
            ('id', DataTypes.REAL_VALUED, InputTypes.ID),
            ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
            ('rain (mm)', DataTypes.REAL_VALUED, InputTypes.TARGET),
            ('T (degC)', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ('H2OC (mmol/mol)', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ]

    def split_data(self, df, valid_boundary=4280, test_boundary=4708):
        """Splits data_set frame into training-validation-test data_set frames.
        This also calibrates scaling object, and transforms data_set for each split.
        Args:
          df: Source data_set frame to split.
          valid_boundary: Starting year for validation data_set
          test_boundary: Starting year for test data_set
        Returns:
          Tuple of transformed (train, valid, test) data_set.
        """

        print('Formatting train-valid-test splits.')

        index = df['days_from_start']
        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary - 7) & (index < test_boundary)]
        test = df.loc[index >= test_boundary - 7]

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

    def get_num_samples_for_calibration(self):
        """Gets the default number of training and validation samples.
        Use to sub-sample the data_set for network calibration and a value of -1 uses
        all available samples.
        Returns:
          Tuple of (training samples, validation samples)
        """
        return 32000, 3840
