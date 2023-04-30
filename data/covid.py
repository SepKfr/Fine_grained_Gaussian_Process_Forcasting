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


class CovidFormatter(DataFormatter):

    _column_definition = [
        ('days_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
        ('PEOPLE_POSITIVE_NEW_CASES_COUNT', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('PEOPLE_DEATH_COUNT', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('Population Staying at Home', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('Population Staying at Home', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('Population Not Staying at Home', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('id', DataTypes.REAL_VALUED, InputTypes.ID),
        ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

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
