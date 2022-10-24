import sklearn.preprocessing
from Utils import utils, base
from data.electricity import ElectricityFormatter

GenericDataFormatter = ElectricityFormatter
DataTypes = base.DataTypes
InputTypes = base.InputTypes


class CovidFormatter(GenericDataFormatter):

    _column_definition = [
        ('id', DataTypes.REAL_VALUED, InputTypes.ID),
        ('days_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
        ('PEOPLE_POSITIVE_NEW_CASES_COUNT', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('days_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('Number of Trips', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
    ]

    # Default params
    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps': 7 * 24 + self.pred_len,
            'num_encoder_steps': 7 * 24,
            'num_decoder_steps': self.pred_len,
            'num_epochs': 50,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 5
        }

        return fixed_params

    def get_default_model_params(self):
        """Returns default optimised model parameters."""

        model_params = {
            'hidden_layer_size': [32, 64],
            'minibatch_size': [256],
            'num_heads': 8,
            'stack_size': [1],
            'context_lengths': [1, 3, 6, 9]
        }

        return model_params

    def get_num_samples_for_calibration(self):
        """Gets the default number of training and validation samples.
        Use to sub-sample the data_set for network calibration and a value of -1 uses
        all available samples.
        Returns:
          Tuple of (training samples, validation samples)
        """
        return 32000, 3840