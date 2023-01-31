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
import functools

# Lint as: python3

import wget
import pyunpack
import os
import pandas as pd
import numpy as np
import argparse
import sys
import random
import gc
import glob
from tqdm import tqdm


from data import air_quality, electricity, traffic, watershed, solar


class ExperimentConfig(object):
    default_experiments = ['electricity', 'traffic', 'air_quality', 'camel',
                           'favorita', 'watershed', 'solar', 'ETTm2', 'weather',
                           'covid']

    def __init__(self, pred_len=24, experiment='covid', root_folder=None):

        if experiment not in self.default_experiments:
            raise ValueError('Unrecognised experiment={}'.format(experiment))

        if root_folder is None:
            root_folder = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), '', 'outputs')
            print('Using root folder {}'.format(root_folder))

        self.root_folder = root_folder
        self.experiment = experiment
        self.data_folder = os.path.join(root_folder, '', experiment)
        self.pred_len = pred_len

        for relevant_directory in [
            self.root_folder, self.data_folder
        ]:
            if not os.path.exists(relevant_directory):
                os.makedirs(relevant_directory)

    @property
    def data_csv_path(self):
        csv_map = {
            'electricity': 'hourly_electricity.csv',
            'traffic': 'hourly_traffic.csv',
            'air_quality': 'hourly_air_quality.csv',
            'favorita': 'favorita_consolidated.csv',
            'watershed': 'watershed.csv',
            'solar': 'solar.csv',
            'ETTm2': 'ETT.csv',
            'weather': 'weather.csv',
            'camel': 'camel.csv',
            'covid': 'covid.csv'
        }

        return os.path.join(self.data_folder, csv_map[self.experiment])

    def make_data_formatter(self):
        """Gets a data_set formatter object for experiment.
        Returns:
          Default DataFormatter per experiment.
        """

        data_formatter_class = {
            'electricity': electricity.ElectricityFormatter,
            'traffic': traffic.TrafficFormatter,
            'air_quality': air_quality.AirQualityFormatter,
            'watershed': watershed.WatershedFormatter,
            'solar': solar.SolarFormatter
        }

        return data_formatter_class[self.experiment](self.pred_len)


def download_from_url(url, output_path):
    """Downloads a file from url."""

    print('Pulling data_set from {} to {}'.format(url, output_path))
    wget.download(url, output_path)
    print('done')


def unzip(zip_path, output_file, data_folder):
    """Unzips files and checks successful completion."""

    if not os.path.exists(output_file):
        os.makedirs(output_file)

    print('Unzipping file: {}'.format(zip_path))

    pyunpack.Archive(zip_path).extractall(output_file)

    # Checks if unzip was successful
    '''if not os.path.exists(output_file):
        raise ValueError(
            'Error in unzipping process! {} not found.'.format(output_file))'''


def download_and_unzip(url, zip_path, csv_path, data_folder):
    """Downloads and unzips an online csv file.
    Args:
    url: Web address
    zip_path: Path to download zip file
    csv_path: Expected path to csv file
    data_folder: Folder in which data_set is stored.
    """

    download_from_url(url, zip_path)

    unzip(zip_path, csv_path, data_folder)

    print('Done.')


def process_watershed(config):

    """Process watershed dataset
    Args:
    config: Default experiment config for Watershed
    """
    sites = ['BDC', 'BEF', 'DCF', 'GOF', 'HBF', 'LMP', 'MCQ', 'SBM', 'TPB', 'WHB']
    data_path = config.data_folder
    df_list = []

    for i, site in enumerate(sites):

        df = pd.read_csv('{}/{}_WQual_Level4.csv'.format(data_path, site), index_col=0, sep=',')
        df_list.append(df.iloc[0::4, :])

    output = pd.concat(df_list, axis=0)
    output.index = pd.to_datetime(output.Date)
    output.sort_index(inplace=True)
    output = output.dropna(axis=1, how='all')
    output = output.fillna(method="ffill").fillna(method='bfill')

    start_date = pd.to_datetime('2013-03-28')
    earliest_time = start_date
    output = output[output.index >= start_date]

    date = output.index
    output['day_of_week'] = date.dayofweek
    output['hour'] = date.hour
    output['id'] = output['Site']
    output['categorical_id'] = output['Site']
    output['hours_from_start'] = (date - earliest_time).seconds / 60 / 60 + (
            date - earliest_time).days * 24

    output['days_from_start'] = (date - earliest_time).days
    output = output[output['Site'] != 0.0]
    output = output.fillna('na')
    output = output[output['days_from_start'] != 'na']
    output.to_csv("watershed.csv")

    print('Done.')


def download_weather(args):

    """Downloads weather dataset from bgc jenna for 2020"""
    data_folder = args.data_folder

    def get_dfs(url, csv, zip):
        csv_path = os.path.join(data_folder, csv)
        zip_path = os.path.join(data_folder, zip)
        download_and_unzip(url, zip_path, csv_path, data_folder)
        return pd.read_csv(csv_path, index_col=0, encoding='unicode_escape')

    url_list = ['https://www.bgc-jena.mpg.de/wetter/mpi_roof_2008a.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2008b.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2009a.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2009b.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2010a.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2010b.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2011a.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2011b.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2012a.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2012b.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2013a.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2013b.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2014a.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2014b.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2015a.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2015b.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2016a.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2016b.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2017a.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2017b.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2018a.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2018b.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2019a.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2019b.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2020a.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2020b.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2021a.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof_2021b.zip',
                'https://www.bgc-jena.mpg.de/wetter/mpi_roof.zip']

    csv_zip_list = ['mpi_roof_2008a', 'mpi_roof_2008b',
                    'mpi_roof_2009a', 'mpi_roof_2009b',
                    'mpi_roof_2010a', 'mpi_roof_2010b',
                    'mpi_roof_2011a', 'mpi_roof_2011b',
                    'mpi_roof_2012a', 'mpi_roof_2012b',
                    'mpi_roof_2013a', 'mpi_roof_2013b',
                    'mpi_roof_2014a', 'mpi_roof_2014b',
                    'mpi_roof_2015a', 'mpi_roof_2015b',
                    'mpi_roof_2016a', 'mpi_roof_2016b',
                    'mpi_roof_2017a', 'mpi_roof_2017b',
                    'mpi_roof_2018a','mpi_roof_2018b',
                    'mpi_roof_2019a', 'mpi_roof_2019b',
                    'mpi_roof_2020a', 'mpi_roof_2020b',
                    'mpi_roof_2021a', 'mpi_roof_2021b',
                    'mpi_roof']
    df_list = []
    for i in range(len(url_list)):
        df = get_dfs(url_list[i], csv_zip_list[i] + ".csv", csv_zip_list[i] + ".zip")
        df_list.append(df)

    output = pd.concat(df_list, axis=0, join='outer')
    output.index = pd.to_datetime(output.index)
    output.sort_index(inplace=True)

    output = output.resample('1h').mean().replace(0., np.nan)

    earliest_time = output.index.min()
    start_date = min(output.fillna(method='ffill').dropna().index)
    end_date = max(output.fillna(method='bfill').dropna().index)

    active_range = (output.index >= start_date) & (output.index <= end_date)
    output = output[active_range].fillna(0.)

    date = output.index

    output['day_of_week'] = date.dayofweek
    output['hour'] = date.hour
    output['id'] = 1
    output['categorical_id'] = output['id']
    output['hours_from_start'] = (date - earliest_time).seconds / 60 / 60 + (
            date - earliest_time).days * 24
    output['days_from_start'] = (date - earliest_time).days
    output.to_csv("weather.csv")


def download_ett(args):

    """Downloads ETT dataset from github"""
    url = 'https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTm2.csv'
    data_folder = args.data_folder
    data_path = os.path.join(data_folder, "ETT.csv")
    download_from_url(url, data_path)

    df = pd.read_csv(os.path.join(data_path, "ETTm2.csv"), index_col=0)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # used to determine the start and end dates of a series
    output = df.resample('15min').mean().replace(0., np.nan)

    earliest_time = output.index.min()
    start_date = min(output.fillna(method='ffill').dropna().index)
    end_date = max(output.fillna(method='bfill').dropna().index)

    active_range = (output.index >= start_date) & (output.index <= end_date)
    output = output[active_range].fillna(0.)

    date = output.index

    output['day_of_week'] = date.dayofweek
    output['hour'] = date.hour
    output['id'] = 1
    output['categorical_id'] = output['id']
    output['hours_from_start'] = (date - earliest_time).seconds / 60 / 60 + (
            date - earliest_time).days * 24
    output['days_from_start'] = (date - earliest_time).days
    output.to_csv("ETTm2.csv")


def download_camel(args):

    """Downloads camels dataset"""
    url = "https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/basin_timeseries_v1p2_metForcing_obsFlow.zip"
    data_folder = args.data_folder
    data_path = os.path.join(data_folder, 'basin_timeseries_v1p2_metForcing_obsFlow.zip')
    zip_path = data_path
    download_and_unzip(url, zip_path, data_path, data_folder)
    df_list = []
    data_folder = os.path.join(args.data_folder, 'basin_dataset_public_v1p2', 'usgs_streamflow')
    for dir in os.listdir(data_folder):
        for file in os.listdir(os.path.join(data_folder, dir)):
            f = os.path.join(data_folder, dir, file)
            arrays = []
            for line in open(f):
                arrays.append(np.array([val for val in line.rstrip('\n').split(' ') if val != '']))
            arrays = np.asarray(arrays)
            arrays = arrays[:, :-1]
            date = pd.DataFrame(["{}-{}-{}".format(a[1], a[2], a[3]) for a in arrays], columns=["date"])
            id = pd.DataFrame(arrays[:, 0], columns=["id"])
            streamflow = pd.DataFrame(arrays[:, -1], columns=["streamflow"])
            df = pd.concat((date, id), axis=1)
            df = pd.concat((df, streamflow), axis=1)
            df.index = pd.to_datetime(df.date)
            df.sort_index(inplace=True)
            df.loc[df['streamflow'] == '-999.00', 'streamflow'] = np.nan
            start_date = min(df.fillna(method='ffill').dropna().index)
            end_date = max(df.fillna(method='bfill').dropna().index)

            active_range = (df.index >= start_date) & (df.index <= end_date)
            df = df[active_range].fillna(0.)
            earliest_time = df.index.min()
            date = df.index
            df['day_of_week'] = date.dayofweek
            df['hour'] = date.hour
            df['categorical_id'] = df['id']
            df['hours_from_start'] = (date - earliest_time).seconds / 60 / 60 + (
                    date - earliest_time).days * 24
            df['days_from_start'] = (date - earliest_time).days
            df_list.append(df)

    output = pd.concat(df_list, axis=0, join='outer')
    output.sort_index(inplace=True)
    output.to_csv("camel.csv")


def download_air_quality(args):

    """Downloads air quality dataset from UCI repository"""

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip'
    data_folder = args.data_folder
    data_path = data_folder
    zip_path = data_path + '.zip'
    download_and_unzip(url, zip_path, data_path, data_folder)
    df_list = []

    folder = os.path.join(data_path, 'PRSA_Data_20130301-20170228')

    for i, site in enumerate(os.listdir(folder)):

        df = pd.read_csv(os.path.join(folder, site), index_col=0, sep=',')
        df_list.append(df)

    output = pd.concat(df_list, axis=0)
    output.index = pd.to_datetime(output[['year','month','day']])
    output.sort_index(inplace=True)
    earliest_time = output.index.min()

    start_date = min(output.fillna(method='ffill').dropna().index)
    end_date = max(output.fillna(method='bfill').dropna().index)

    active_range = (output.index >= start_date) & (output.index <= end_date)
    output = output[active_range].fillna(0.)

    date = output.index

    output['day_of_week'] = date.dayofweek
    output['hour'] = date.hour
    output['id'] = output['station']
    output['categorical_id'] = output['station']
    output['hours_from_start'] = (date - earliest_time).seconds / 60 / 60 + (
            date - earliest_time).days * 24
    output['days_from_start'] = (date - earliest_time).days
    output.to_csv("air_quality.csv")

    print('Done.')


def process_covid(args):

    df = pd.read_csv(os.path.join(
        '~/Downloads', 'covid-data.csv'), dtype={'COUNTY_NAME': str})

    # adding travel data

    df_travel = pd.read_csv(os.path.join('~/Downloads', 'Trips_by_Distance.csv'))

    df.index = pd.to_datetime(df.REPORT_DATE)
    df_travel.index = pd.to_datetime(df_travel.Date)
    df_travel["date"] = df_travel.index.astype(str)

    df.sort_index(inplace=True)
    df_travel.sort_index(inplace=True)

    df = df.dropna()
    df_travel = df_travel.dropna()
    df_travel["County FIPS"] = df_travel["County FIPS"].astype(int)
    df["COUNTY_FIPS_NUMBER"] = df["COUNTY_FIPS_NUMBER"].astype(int)

    earliest_time = df.index.min()
    latest_time = df_travel.index.max()

    active_range = (df.index >= earliest_time) & (df.index <= latest_time)
    active_range_trip = (df_travel.index >= earliest_time) & (df_travel.index <= latest_time)

    df = df[active_range]
    df_travel = df_travel[active_range_trip]
    date = df.index

    df['day_of_week'] = date.dayofweek
    df['id'] = df['COUNTY_FIPS_NUMBER']
    df['categorical_id'] = df['id'].copy()
    df['days_from_start'] = (date - earliest_time).days

    ls_df = []
    for fip, dff in df.groupby('COUNTY_FIPS_NUMBER'):

        tmp = df_travel.loc[df_travel['County FIPS'] == fip]
        dff.loc[0:len(tmp),'date'] = tmp["date"].values
        dff.loc[0:len(tmp),"Number of Trips"] = tmp["Number of Trips"].values
        dff.loc[0:len(tmp),"Population Staying at Home"] = tmp["Population Staying at Home"].values
        dff.loc[0:len(tmp),"Population Not Staying at Home"] = tmp["Population Not Staying at Home"].values
        ls_df.append(dff)

    df_f = pd.concat(ls_df, axis=0)
    df_f = df_f.fillna(0)

    df_f.to_csv("covid.csv")

    print('Done.')


def download_solar(args):

    url = 'https://www.nrel.gov/grid/assets/downloads/al-pv-2006.zip'
    data_folder = args.data_folder
    csv_path = os.path.join(data_folder, 'al-pv-2006')
    zip_path = csv_path + '.zip'

    download_and_unzip(url, zip_path, csv_path, data_folder)

    df_list = []

    for file in os.listdir(csv_path):

        parts = file.split("_")
        df = pd.read_csv(os.path.join(csv_path, file), index_col=0, sep=',')
        df_hr = df.iloc[0::12, :]
        df_sub = df_hr.copy()
        df_sub['latitude'] = parts[1]
        df_sub['longtitude'] = parts[2]
        df_sub['id'] = parts[1] + "_" + parts[2]
        df_sub['capacity'] = parts[5]
        df_list.append(df_sub)

    output = pd.concat(df_list, axis=0)
    output.index = pd.to_datetime(output.index)
    output.sort_index(inplace=True)
    earliest_time = output.index.min()
    date = output.index

    output['day_of_week'] = date.dayofweek
    output['hour'] = date.hour
    output['hours_from_start'] = (date - earliest_time).seconds / 60 / 60 + (
            date - earliest_time).days * 24
    output['days_from_start'] = (date - earliest_time).days
    output['categorical_id'] = output['id']
    final = output.loc[output["Power(MW)"] > 0]

    final.to_csv("solar.csv")

    print('Done.')


def download_electricity(args):
    """Downloads electricity dataset from UCI repository."""

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'

    data_folder = args.data_folder
    csv_path = os.path.join(data_folder, 'LD2011_2014.txt')
    zip_path = csv_path + '.zip'

    download_and_unzip(url, zip_path, csv_path, data_folder)

    print('Aggregating to hourly data_set')

    df = pd.read_csv(csv_path, index_col=0, sep=';', decimal=',')
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # used to determine the start and end dates of a series
    output = df.resample('1h').mean().replace(0., np.nan)

    earliest_time = output.index.min()

    df_list = []
    for label in output:
        print('Processing {}'.format(label))
        srs = output[label]

        start_date = min(srs.fillna(method='ffill').dropna().index)
        end_date = max(srs.fillna(method='bfill').dropna().index)

        active_range = (srs.index >= start_date) & (srs.index <= end_date)
        srs = srs[active_range].fillna(0.)

        tmp = pd.DataFrame({'power_usage': srs})
        date = tmp.index
        tmp['t'] = (date - earliest_time).seconds / 60 / 60 + (
                date - earliest_time).days * 24
        tmp['days_from_start'] = (date - earliest_time).days
        tmp['categorical_id'] = label
        tmp['date'] = date
        tmp['id'] = label
        tmp['hour'] = date.hour
        tmp['day'] = date.day
        tmp['day_of_week'] = date.dayofweek
        tmp['month'] = date.month

        df_list.append(tmp)

    output = pd.concat(df_list, axis=0, join='outer').reset_index(drop=True)

    output['categorical_id'] = output['id'].copy()
    output['hours_from_start'] = output['t']
    output['categorical_day_of_week'] = output['day_of_week'].copy()
    output['categorical_hour'] = output['hour'].copy()

    # Filter to match range used by other academic papers
    output = output[(output['days_from_start'] >= 1096)
                    & (output['days_from_start'] < 1346)].copy()

    output.to_csv("electricity.csv")

    print('Done.')


def download_traffic(args):
    """Downloads traffic dataset from UCI repository."""

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00204/PEMS-SF.zip'

    data_folder = args.data_folder
    csv_path = os.path.join(data_folder, 'PEMS_train')
    zip_path = os.path.join(data_folder, 'PEMS-SF.zip')
    data_folder = csv_path

    download_and_unzip(url, zip_path, csv_path, data_folder)

    print('Aggregating to hourly data_set')

    def process_list(s, variable_type=int, delimiter=None):
        """Parses a line in the PEMS format to a list."""
        if delimiter is None:
          l = [
              variable_type(i) for i in s.replace('[', '').replace(']', '').split()
          ]
        else:
          l = [
              variable_type(i)
              for i in s.replace('[', '').replace(']', '').split(delimiter)
          ]

        return l

    def read_single_list(filename):
        """Returns single list from a file in the PEMS-custom format."""
        with open(os.path.join(data_folder, filename), 'r') as dat:
          l = process_list(dat.readlines()[0])
        return l

    def read_matrix(filename):
        """Returns a matrix from a file in the PEMS-custom format."""
        array_list = []
        with open(os.path.join(data_folder, filename), 'r') as dat:

          lines = dat.readlines()
          for i, line in enumerate(lines):
            if (i + 1) % 50 == 0:
              print('Completed {} of {} rows for {}'.format(i + 1, len(lines),
                                                            filename))

            array = [
                process_list(row_split, variable_type=float, delimiter=None)
                for row_split in process_list(
                    line, variable_type=str, delimiter=';')
            ]
            array_list.append(array)

        return array_list

    shuffle_order = np.array(read_single_list('randperm')) - 1  # index from 0
    train_dayofweek = read_single_list('PEMS_trainlabels')
    train_tensor = read_matrix('PEMS_train')
    test_dayofweek = read_single_list('PEMS_testlabels')
    test_tensor = read_matrix('PEMS_test')

    # Inverse permutate shuffle order
    print('Shuffling')
    inverse_mapping = {
      new_location: previous_location
      for previous_location, new_location in enumerate(shuffle_order)
    }
    reverse_shuffle_order = np.array([
      inverse_mapping[new_location]
      for new_location, _ in enumerate(shuffle_order)
    ])

    # Group and reoder based on permuation matrix
    print('Reodering')
    day_of_week = np.array(train_dayofweek + test_dayofweek)
    combined_tensor = np.array(train_tensor + test_tensor)

    day_of_week = day_of_week[reverse_shuffle_order]
    combined_tensor = combined_tensor[reverse_shuffle_order]

    # Put everything back into a dataframe
    print('Parsing as dataframe')
    labels = ['traj_{}'.format(i) for i in read_single_list('stations_list')]

    hourly_list = []
    for day, day_matrix in enumerate(combined_tensor):

        # Hourly data_set
        hourly = pd.DataFrame(day_matrix.T, columns=labels)
        hourly['hour_on_day'] = [int(i / 6) for i in hourly.index
                                ]  # sampled at 10 min intervals
        if hourly['hour_on_day'].max() > 23 or hourly['hour_on_day'].min() < 0:
          raise ValueError('Invalid hour! {}-{}'.format(
              hourly['hour_on_day'].min(), hourly['hour_on_day'].max()))

        hourly = hourly.groupby('hour_on_day', as_index=True).mean()[labels]
        hourly['sensor_day'] = day
        hourly['time_on_day'] = hourly.index
        hourly['day_of_week'] = day_of_week[day]

        hourly_list.append(hourly)

    hourly_frame = pd.concat(hourly_list, axis=0, ignore_index=True, sort=False)

    # Flatten such that each entitiy uses one row in dataframe
    store_columns = [c for c in hourly_frame.columns if 'traj' in c]
    other_columns = [c for c in hourly_frame.columns if 'traj' not in c]
    flat_df = pd.DataFrame(columns=['values', 'prev_values', 'next_values'] +
                         other_columns + ['id'])

    def format_index_string(x):
        """Returns formatted string for key."""

        if x < 10:
            return '00' + str(x)
        elif x < 100:
            return '0' + str(x)
        elif x < 1000:
            return str(x)

        raise ValueError('Invalid value of x {}'.format(x))

    for store in store_columns:
        print('Processing {}'.format(store))

        sliced = hourly_frame[[store] + other_columns].copy()
        sliced.columns = ['values'] + other_columns
        sliced['id'] = int(store.replace('traj_', ''))

        # Sort by Sensor-date-time
        key = sliced['id'].apply(str) \
          + sliced['sensor_day'].apply(lambda x: '_' + format_index_string(x)) \
            + sliced['time_on_day'].apply(lambda x: '_' + format_index_string(x))
        sliced = sliced.set_index(key).sort_index()

        sliced['values'] = sliced['values'].fillna(method='ffill')
        sliced['prev_values'] = sliced['values'].shift(1)
        sliced['next_values'] = sliced['values'].shift(-1)

        flat_df = flat_df.append(sliced.dropna(), ignore_index=True, sort=False)

    # Filter to match range used by other academic papers
    index = flat_df['sensor_day']
    flat_df = flat_df[index < 173].copy()

    # Creating columns fo categorical inputs
    flat_df['categorical_id'] = flat_df['id'].copy()
    flat_df['hours_from_start'] = flat_df['time_on_day'] \
      + flat_df['sensor_day']*24.
    flat_df['categorical_day_of_week'] = flat_df['day_of_week'].copy()
    flat_df['categorical_time_on_day'] = flat_df['time_on_day'].copy()

    flat_df.to_csv("traffic.csv")
    print('Done.')


def process_favorita(config):
    """Processes Favorita dataset.
    Makes use of the raw files should be manually downloaded from Kaggle @
    https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data
    Args:
    config: Default experiment config for Favorita
    """

    url = 'https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data'

    data_folder = config.data_folder

    # Save manual download to root folder to avoid deleting when re-processing.
    zip_file = os.path.join(data_folder,
                          'favorita-grocery-sales-forecasting.zip')

    if not os.path.exists(zip_file):
        raise ValueError(
            'Favorita zip file not found in {}!'.format(zip_file) +
            ' Please manually download data_set from Kaggle @ {}'.format(url))

    # Unpack main zip file
    outputs_file = os.path.join(data_folder, 'train.csv.7z')
    unzip(zip_file, outputs_file, data_folder)

    # Unpack individually zipped files
    for file in glob.glob(os.path.join(data_folder, '*.7z')):

        csv_file = file.replace('.7z', '')

        unzip(file, csv_file, data_folder)

    print('Unzipping complete, commencing data_set processing...')

    # Extract only a subset of data_set to save/process for efficiency
    start_date = pd.datetime(2015, 1, 1)
    end_date = pd.datetime(2016, 6, 1)

    print('Regenerating data_set...')

    # load temporal data_set
    temporal = pd.read_csv(os.path.join(data_folder, 'train.csv'), index_col=0)

    store_info = pd.read_csv(os.path.join(data_folder, 'stores.csv'), index_col=0)
    oil = pd.read_csv(
      os.path.join(data_folder, 'oil.csv'), index_col=0).iloc[:, 0]
    holidays = pd.read_csv(os.path.join(data_folder, 'holidays_events.csv'))
    items = pd.read_csv(os.path.join(data_folder, 'items.csv'), index_col=0)
    transactions = pd.read_csv(os.path.join(data_folder, 'transactions.csv'))

    # Take first 6 months of data_set
    temporal['date'] = pd.to_datetime(temporal['date'])

    # Filter dates to reduce storage space requirements
    if start_date is not None:
        temporal = temporal[(temporal['date'] >= start_date)]
    if end_date is not None:
        temporal = temporal[(temporal['date'] < end_date)]

    dates = temporal['date'].unique()

    # Add trajectory identifier
    temporal['traj_id'] = temporal['store_nbr'].apply(
      str) + '_' + temporal['item_nbr'].apply(str)
    temporal['unique_id'] = temporal['traj_id'] + '_' + temporal['date'].apply(
      str)

    # Remove all IDs with negative returns
    print('Removing returns data_set')
    min_returns = temporal['unit_sales'].groupby(temporal['traj_id']).min()
    valid_ids = set(min_returns[min_returns >= 0].index)
    selector = temporal['traj_id'].apply(lambda traj_id: traj_id in valid_ids)
    new_temporal = temporal[selector].copy()
    del temporal
    gc.collect()
    temporal = new_temporal
    temporal['open'] = 1

    # Resampling
    print('Resampling to regular grid')
    resampled_dfs = []
    for traj_id, raw_sub_df in temporal.groupby('traj_id'):
        print('Resampling', traj_id)
        sub_df = raw_sub_df.set_index('date', drop=True).copy()
        sub_df = sub_df.resample('1d').last()
        sub_df['date'] = sub_df.index
        sub_df[['store_nbr', 'item_nbr', 'onpromotion']] \
            = sub_df[['store_nbr', 'item_nbr', 'onpromotion']].fillna(method='ffill')
        sub_df['open'] = sub_df['open'].fillna(
            0)  # flag where sales data_set is unknown
        sub_df['log_sales'] = np.log(sub_df['unit_sales'])

        resampled_dfs.append(sub_df.reset_index(drop=True))

    new_temporal = pd.concat(resampled_dfs, axis=0)
    del temporal
    gc.collect()
    temporal = new_temporal

    print('Adding oil')
    oil.name = 'oil'
    oil.index = pd.to_datetime(oil.index)
    temporal = temporal.join(
      oil.reindex(dates).fillna(method='ffill'), on='date', how='left')
    temporal['oil'] = temporal['oil'].fillna(-1)

    print('Adding store info')
    temporal = temporal.join(store_info, on='store_nbr', how='left')

    print('Adding item info')
    temporal = temporal.join(items, on='item_nbr', how='left')

    transactions['date'] = pd.to_datetime(transactions['date'])
    temporal = temporal.merge(
      transactions,
      left_on=['date', 'store_nbr'],
      right_on=['date', 'store_nbr'],
      how='left')
    temporal['transactions'] = temporal['transactions'].fillna(-1)

    # Additional date info
    temporal['day_of_week'] = pd.to_datetime(temporal['date'].values).dayofweek
    temporal['day_of_month'] = pd.to_datetime(temporal['date'].values).day
    temporal['month'] = pd.to_datetime(temporal['date'].values).month

    # Add holiday info
    print('Adding holidays')
    holiday_subset = holidays[holidays['transferred'].apply(
      lambda x: not x)].copy()
    holiday_subset.columns = [
      s if s != 'type' else 'holiday_type' for s in holiday_subset.columns
    ]
    holiday_subset['date'] = pd.to_datetime(holiday_subset['date'])
    local_holidays = holiday_subset[holiday_subset['locale'] == 'Local']
    regional_holidays = holiday_subset[holiday_subset['locale'] == 'Regional']
    national_holidays = holiday_subset[holiday_subset['locale'] == 'National']

    temporal['national_hol'] = temporal.merge(
      national_holidays, left_on=['date'], right_on=['date'],
      how='left')['description'].fillna('')
    temporal['regional_hol'] = temporal.merge(
      regional_holidays,
      left_on=['state', 'date'],
      right_on=['locale_name', 'date'],
      how='left')['description'].fillna('')
    temporal['local_hol'] = temporal.merge(
      local_holidays,
      left_on=['city', 'date'],
      right_on=['locale_name', 'date'],
      how='left')['description'].fillna('')

    temporal.sort_values('unique_id', inplace=True)

    print('Saving processed file to {}'.format(config.data_csv_path))
    temporal.to_csv("retail.csv")


def main(expt_name, force_download, output_folder):

    print('#### Running download script ###')
    expt_config = ExperimentConfig(experiment=expt_name, root_folder=output_folder)

    if os.path.exists(expt_config.data_csv_path) and not force_download:
        print('Data has been processed for {}. Skipping download...'.format(
            expt_name))
    else:
        print('Resetting data_set folder...')
        #shutil.rmtree(expt_config.data_csv_path)
        os.makedirs(expt_config.data_csv_path)

    # Default download functions
    download_functions = {
        'electricity': download_electricity,
        'traffic': download_traffic,
        'air_quality': download_air_quality,
        'favorita': process_favorita,
        'watershed': process_watershed,
        'solar': download_solar,
        'ETTm2': download_ett,
        'weather': download_weather,
        'camel': download_camel,
        'covid': process_covid
    }

    if expt_name not in download_functions:
        raise ValueError('Unrecongised experiment! name={}'.format(expt_name))

    download_function = download_functions[expt_name]

    # Run data_set download
    print('Getting {} data_set...'.format(expt_name))
    download_function(expt_config)

    print('Download completed.')


if __name__ == '__main__':
    def get_args():
        """Returns settings from command line."""

        experiment_names = ExperimentConfig.default_experiments

        parser = argparse.ArgumentParser(description='Data download configs')
        parser.add_argument(
            '--expt_name',
            type=str,
            nargs='?',
            choices=experiment_names,
            help='Experiment Name. Default={}'.format(','.join(experiment_names)))
        parser.add_argument(
            '--output_folder',
            type=str,
            nargs='?',
            default='.',
            help='Path to folder for data_set download')
        parser.add_argument(
            '--force_download',
            type=str,
            nargs='?',
            choices=['yes', 'no'],
            default='yes',
            help='Whether to re-run data_set download')

        args = parser.parse_args()

        root_folder = None if args.output_folder == '.' else args.output_folder

        return args.expt_name, args.force_download == 'no', root_folder


    name, force, folder = get_args()
    main(expt_name=name, force_download=force, output_folder=folder)