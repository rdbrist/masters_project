import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from loguru import logger
from datetime import timedelta

from scipy.signal import find_peaks
from src.config import INTERIM_DATA_DIR


class Cob:
    """
    Class for processing and analysing Carbohydrate Onboard (COB) data from a
    CSV or Parquet flat file, that has been read from all archive zip files in
    the raw dataset. The class provides methods for reading the consolidated
    data, exploratory data analysis, processing the data for finding peaks,
    producing peak features and analysis of those features. It assumes the data
    is resampled at a particular interval, but not that the dataset is sparse
    with regular intervals. The creation of a sparse set for peak extraction and
    plotting is done in the class itself.
    """
    def __init__(self):
        self.data_file_path = INTERIM_DATA_DIR
        self.height = None
        self.distance = None
        self.individual = None
        self.interpolated = None
        self.dataset = None
        self.individual_dataset = None
        self.processed_dataset = None
        self.sampling_rate = 15  # minutes, default
        self.offset_processed = False

    def dataset_is_loaded(self):
        if self.dataset is None:
            print("Dataset not loaded. Please run read_interim_data() first.")
            return False
        return True
    
    def set_parameters(self, height: int, distance: int):
        """
        Set the parameters for the find_peaks function.
        :param height: minimum height of peaks
        :param distance: minimum distance between peaks
        :return:
        """
        self.height = height
        self.distance = distance

    def read_interim_data(self,
                          file_name: str,
                          file_type: str = 'csv',
                          sampling_rate: int = 15):
        """
        Read raw data from CSV file and save as parquet file. File is indexed
        by ID and datetime.
        :param file_name: filename, without extension
        :param file_type: file type, default is 'csv', can be 'parquet'
        :param sampling_rate: sampling rate in minutes, default is 15
        """
        try:
            dtypes = {'id': 'int', 'system': 'category'}
            if file_type == 'csv':
                path = self.data_file_path / (file_name + '.csv')
                self.dataset = (pd.read_csv(path,
                                            parse_dates=['datetime'],
                                            dtype=dtypes,
                                            index_col=['id', 'datetime']))
                if 'Unnamed: 0' in self.dataset.columns:
                    self.dataset.drop(columns=['Unnamed: 0'], inplace=True)
            elif file_type == 'parquet':
                path = self.data_file_path / (file_name + '.parquet')
                self.dataset = pd.read_parquet(path)
            else:
                raise ValueError("Invalid file type. "
                                 "Must be 'csv' or 'parquet'.")
            self.dataset.sort_index(level=['id', 'datetime'], inplace=True)
            self.summarise_interim_data()
            self.sampling_rate = sampling_rate
            if not self._validate_sampling_rate():
                print(f"Sampling rate {self.sampling_rate} does not match the "
                      f"data.\nThis needs addressing to make the intervals "
                      f"consistent.")
                raise ValueError
        except FileNotFoundError:
            raise FileNotFoundError('File not found.')
        except pd.errors.EmptyDataError as e:
            print(f"No data: {e}")
        except Exception as e:
            raise e

    @staticmethod
    def _check_consecutive_intervals(datetime_series: pd.Series,
                                     interval_minutes: int) -> bool:
        """
        Check if a series of datetimes are at a consistent interval.
        :param datetime_series: A pandas Series of datetime objects.
        :param interval_minutes: The expected interval in minutes.
        :return: True if all intervals match the expected interval.
        """
        # Ensure the series is sorted
        datetime_series = datetime_series.sort_values()

        # Calculate the differences between consecutive datetimes
        time_diffs = datetime_series.diff().dropna()

        # Check if all differences match the expected interval
        expected_diff = pd.Timedelta(minutes=interval_minutes)
        return (time_diffs == expected_diff).all()

    @staticmethod
    def _check_minute_factor(datetime_series: pd.Series,
                             interval_minutes: int) -> bool:
        """
        Check the sampling rate is a factor for all the minute interval values.
        :param datetime_series: A pandas Series of datetime objects.
        :param interval_minutes: The expected interval in minutes.
        :return: True if the interval matches a factor of the minutes values.
        """
        minute_values = datetime_series.dt.minute.drop_duplicates().tolist()
        for m in minute_values:
            if m % interval_minutes != 0:
                logger.info(
                    f"Minute value {m} is not divisible by "
                    f"{interval_minutes} sampling rate.")
                return False
        return True  # If no non-divisible values found

    def _validate_sampling_rate(self):
        """
        Validate the given sampling rate against the data. The sampling rate
        must be given, must be an integer and must agree with the data. The
        default is 15 minutes.
        :return: True if valid, False otherwise
        """
        if self.sampling_rate is None:
            print("Sampling rate is not set.")
            return False
        if not isinstance(self.sampling_rate, int):
            print(f"Sampling rate {self.sampling_rate} is not an integer.")
            return False
        if not self.dataset_is_loaded():
            return False
        for p_id in self.dataset.index.get_level_values('id').drop_duplicates():
            datetimes = (self.dataset.
                         loc[p_id].
                         reset_index()['datetime'].
                         sort_values())
            if not Cob._check_minute_factor(datetimes, self.sampling_rate):
                raise ValueError(f"Sampling rate {self.sampling_rate} does not "
                                 f"match the minute values for ID {p_id}.")
        return True

    def summarise_interim_data(self):
        df = self.dataset
        print(f'Number of records: {len(df)}')
        print(f"Number of people: "
              f"{len(df.index.get_level_values('id').drop_duplicates())}")
        print(f"Systems used: \t{df['system'].drop_duplicates().values}")

    def get_person_data(self, p_id: int):
        """
        Get data for a specific individual from the dataset.
        :param p_id: ID of individual
        :return: DataFrame: Data for individual, indexed by datetime. (Does not
        include ID)
        """
        try:
            df = self.dataset.loc[p_id].copy()
        except KeyError:
            raise KeyError(f'Individual {p_id} not found in dataset.')

        if 'cob max' not in df.columns:
            raise ValueError(f'Column "cob max" not found in dataset.')

        # Trim leading and trailing NaNs from the individual dataset
        first_valid_index = df.first_valid_index()
        last_valid_index = df.last_valid_index()

        if first_valid_index is None or last_valid_index is None:
            print(f'No data found for individual {p_id}.')
            # Return empty df with correct columns
            df = df.iloc[0:0]
        else:
            # Slice the DataFrame to exclude leading and trailing NaNs
            df = df.loc[first_valid_index:last_valid_index]
            # Add day and time columns
            df['day'] = df.index.date
            df['time'] = df.index.time
            df = df[['cob max', 'day', 'time']]

        self.individual = int(p_id)
        self.individual_dataset = df

        return df

    def _interpolate_data(self):
        """
        Process the individual dataset to interpolate missing values.
        It will reindex to the sample rate intervals prior to interpolation.
        This is not the same as resampling. Rather it ensures consecutive
        intervals in a date range.
        """
        # Check if individual dataset exists
        if self.individual_dataset is None:
            print('No individual dataset exists. Run get_person_data() method.')
            return
        if self.interpolated:
            print('Data for individual has already been interpolated.')
            return

        df_cob = self.individual_dataset.copy()

        # Reindex to 15-minute intervals and interpolate
        date_rng = self._get_time_series_range()
        df_cob = df_cob.reindex(date_rng)
        df_cob['day'] = df_cob.index.date
        df_cob['time'] = df_cob.index.time
        df_cob['cob interpolate'] = df_cob['cob max'].interpolate(method='time')
        # Given that interpolate does not extrapolate
        df_cob['cob interpolate'] = df_cob['cob interpolate'].fillna(0)
        self.individual_dataset = df_cob
        self.interpolated = True

        return df_cob

    @staticmethod
    def _find_peaks(ser: pd.Series, height: int = None, distance: int = None):
        ser.fillna(0)
        peak_indices, properties = find_peaks(ser,
                                              height=height,
                                              distance=distance)
        return peak_indices, properties

    def identify_peaks_for_individual(self, height: int = None,
                       distance: int = None,
                       suppress: bool = False):
        """
        Identify peaks in the individual dataset using the find_peaks function
        from scipy.signal.
        :param height: (int) Height parameter for find_peaks function. If not
            provided, will use the stored value.
        :param distance: (int) Distance parameter for find_peaks function. If
            not provided, will use the stored value.
        :param suppress: (bool) Suppress output messages if True
        """

        # Establish the dataframe to use
        if self.individual_dataset is None:
            print("No individual dataset available. Please run "
                  "get_person_data() first.")
            return

        df_cob = self.individual_dataset.copy()

        # Establish the height and distance parameters
        if height is None and self.height is None:
            print('No height parameter provided. Please set height '
                  'using height method or provide as parameter.')
        if distance is None and self.distance is None:
            print('No distance parameter provided. Please set distance '
                  'using distance method or provide as parameter.')
        h = self.height if height is None else height
        d = self.distance if distance is None else distance

        # Identify peaks
        peaks, heights = self._find_peaks(df_cob['cob interpolate'],
                                          height=h,
                                          distance=d)
        df_cob['peak'] = 0
        df_cob.loc[df_cob.index[peaks], 'peak'] = 1
        if not suppress:
            print(f'\n{len(peaks)} peaks identified using parameters h={h} and '
                  f'd={d}, and added to individual_dataset as a new column.')

        self.individual_dataset = df_cob

        return df_cob

    def _get_time_series_range(self):
        """
        Get the time series range for the individual dataset, based on the
        sampling_rate and individual dataset.
        :return: DatetimeIndex: Range of datetimes for the individual dataset
        """
        if self.sampling_rate is None:
            print("Sampling rate is not set. Please set the sampling rate.")
            return
        if self.individual_dataset is None:
            print("No individual dataset available.")
            return
        min_date = self.individual_dataset.index.min()
        max_date = self.individual_dataset.index.max()
        freq = f'{self.sampling_rate}min'  # Convert to minutes
        date_rng = pd.date_range(start=min_date, end=max_date, freq=freq)
        return date_rng

    def summarise_missing_for_individual(self, variable: str = 'cob max',
                                         suppress: bool = False):
        """
        Summarise missing data in the individual dataset, including basic
        statistics and identifying date ranges with missing data.
        Based on the cob max by default and not interpolated data.
        """
        if self.individual_dataset is None:
            print("No individual dataset available. Please run "
                  "get_person_data() first.")
            return
        n = len(self.individual_dataset)

        # Establish basic statistics of completeness
        nans = np.nan_to_num(
            self.individual_dataset[variable].isnull().sum(), nan=0)
        min_date = self.individual_dataset.index.min()
        max_date = self.individual_dataset.index.max()
        date_rng = self._get_time_series_range()
        num_intervals = len(date_rng)
        missing_samples = num_intervals - n
        total_missing = nans + missing_samples
        days_in_range = num_intervals / (24 * 60 / self.sampling_rate)
        missing_percent = (total_missing / num_intervals)*100
        days_with_data = (self.individual_dataset.
                          groupby('day').
                          agg({variable: 'sum'}).
                          reset_index())
        days_count = len(days_with_data)

        # Identify date ranges with missing data
        missing_data = self.individual_dataset[variable].isnull()
        missing_ranges = (
            missing_data.ne(missing_data.shift()).cumsum())[missing_data]
        gap_lengths = missing_ranges.value_counts()

        num_gaps = len(gap_lengths)
        # Convert from 15-minute intervals to days
        mean_gap_length = gap_lengths.mean() / 96

        if not suppress:
            print(f'MISSING DATA SUMMARY FOR {self.individual}')
            print(f'Start of time series: {min_date}')
            print(f'End of time series: {max_date}')
            print(f'Samples: {n}')
            print(f'NaN values: {nans}')
            print(f'15-minute intervals in range: {num_intervals}')
            print(f'Missing samples: {missing_samples}')
            print(f'Total missing (NaNs and missing): {total_missing}')
            print(f'Days in range: {days_in_range:.2f}')
            print(f'Total % missing: {missing_percent:.2f}')
            print(f'Days with COB data: {days_count}')
            print(f'Days with missing data: {days_in_range - days_count:.2f}')
            print(f'Number of gaps: {num_gaps}')
            print(f'Mean length of gaps (in days): {mean_gap_length:.2f}\n')
        
        stats = {
            'n': n,
            'nans': nans,
            'min_date': min_date,
            'max_date': max_date,
            'num_intervals': num_intervals,
            'missing_samples': missing_samples,
            'total_missing': total_missing,
            'days_in_range': days_in_range,
            'missing_percent': missing_percent,
            'days_with_data': days_count,
            'days_with_missing_data': days_in_range - days_count,
            'num_gaps': num_gaps,
            'mean_gap_length_days': mean_gap_length
        }
        return stats

    def summarise_missing_for_dataset(self):
        """
        Summarise missing data across the whole dataset, including basic
        statistics and identifying date ranges with missing data. Also
        processes the data to interpolate and produce peaks to provide this
        overview.
        :return: DataFrame: DataFrame containing missing data summary for each
        """
        if not self.dataset_is_loaded():
            return

        if self.processed_dataset is None:
            print('Wanting to summarise the peak data but this is not yet '
                  'processed. Please run pre_process_batch() first without '
                  'specifying a dataset. This will process all by default.')
            return

        summary_df = pd.DataFrame()

        for p_id in self.dataset.index.get_level_values('id').drop_duplicates():
            self.get_person_data(p_id)
            summary_dict = self.summarise_missing_for_individual(suppress=True)
            individual_df = pd.DataFrame.from_dict(summary_dict, orient='index').T
            individual_df['id'] = int(p_id)
            summary_df = pd.concat([summary_df, individual_df])

        # Calculate number of peaks per day for each individual
        daily_peaks = (self.processed_dataset
                       .groupby(['id', 'day'])['peak']
                       .sum()
                       .reset_index())

        # Aggregate per individual: total peaks, average peaks per day, number of days
        df_peaks = (daily_peaks
                    .groupby('id')
                    .agg(num_peaks=('peak', 'sum'),
                         average_peaks=('peak', 'mean'),
                         num_days=('day', 'nunique')))
        df_peaks.index = df_peaks.index.astype(int)

        summary_df = summary_df.set_index('id').join(df_peaks)
        date_cols = ['min_date', 'max_date']
        numeric_cols = summary_df.columns.difference(date_cols)
        summary_df[numeric_cols] = (summary_df[numeric_cols].
                                    apply(pd.to_numeric,errors='coerce').
                                    fillna(0))

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(summary_df['missing_percent'], ax=ax, binwidth=10)
        ax.set_title('Percentage of Missing Values')
        ax.set_xlabel('Percentage of Values Missing for Individual')
        plt.show()

        return summary_df

    def plot_peaks(self,
                   title: str = 'COB Peaks',
                   variable: str = 'cob interpolate'):
        """
        Plot the COB data with peaks identified. Default is to use the
        interpolated data.

        Parameters:
            title (str): Title for the plot
            variable (str): Variable to plot. Default is 'cob interpolate'.
        """
        df = self.individual_dataset.copy()
        if df is None:
            print("No individual dataset available. Please run "
                  "get_person_data() first.")
            return
        if variable not in df.columns:
            print(f'Variable {variable} not found in DataFrame.')
            return
        if df['peak'].isnull().all():
            print("No peaks identified. Please run identify_peaks_for_individual() first.")
            return
        fig, ax = plt.subplots(figsize=(16, 4))
        sns.lineplot(df[variable], ax=ax)
        sns.scatterplot(df[df['peak'] == 1], color='red', ax=ax)
        ax.set_ylabel('Grammes')
        ax.set_xlabel('Date/Time')
        plt.title(title)
        plt.show()

    def pre_process_batch(self,
                          ids: list = None,
                          height: int = None,
                          distance: int = None,
                          suppress: bool = False):
        """
        Pre-processes the data for a batch of individuals, including
        interpolating missing values and identifying peaks.

        Parameters:
            ids (list): List of individual IDs
            height (int): Height parameter for find_peaks function
            distance (int): Distance parameter for find_peaks function
            suppress (bool): Suppress output messages if True

        Returns:
            df_cob (pd.DataFrame): DataFrame containing pre-processed COB data
            for the batch of individuals
        """
        all_ids = (self.dataset.
                   index.get_level_values('id').
                   drop_duplicates().
                   tolist())
        if not self.dataset_is_loaded():
            return
        if ids is None:
            ids = all_ids
            print(f'No IDs provided. Processing all {len(ids)} records.')

        df_all = pd.DataFrame()
        self.individual_dataset = None
        self.interpolated = None

        ignored = 0
        for p_id in ids:
            if not suppress:
                print('Processing ID:', p_id)
            if p_id not in self.dataset.index.get_level_values('id').values:
                print(f'Individual {p_id} not found in dataset, ignoring.')
                ignored += 1
                continue
            self.individual_dataset = self.get_person_data(p_id)
            self.individual_dataset = self._interpolate_data()
            self.individual_dataset = (
                self.identify_peaks_for_individual(height=height,
                                                   distance=distance,
                                                   suppress=suppress))
            self.individual_dataset, _ = (
                self.remove_zero_peak_days(suppress=suppress))
            self.individual_dataset['id'] = int(p_id)
            self.individual_dataset = (self.individual_dataset.
                                       reset_index(names='datetime').
                                       set_index(['id', 'datetime']))
            df_all = pd.concat([df_all, self.individual_dataset])
            # Reset individual dataset for next iteration
            self.individual_dataset = None
            self.interpolated = False
            self.individual = None

            self.processed_dataset = df_all
            df_all.to_parquet(self.data_file_path /
                              'processed_cob_data.parquet')

        if not suppress:
            if ignored > 0:
                print(
                    f'From {len(ids)}, ignored {ignored} individuals not found '
                    f'in dataset, leaving {len(ids) - ignored} processed '
                    f'records.')
            print(f'The following stats are based on parameters h={height} and '
                  f'd={distance}:')
            print(f'\tNumber of records: {len(df_all)}')
            print(f'\tNumber of days with peaks: '
                  f'{len(df_all["day"].drop_duplicates())}')
            print(f'\tNumber of peaks: {df_all["peak"].sum()}')

        self.offset_processed = False

        return df_all

    def read_processed_data(self):
        """
        Read processed data from parquet file and summarise basic statistics.
        """
        try:
            self.processed_dataset = (
                pd.read_parquet(
                    self.data_file_path / 'processed_cob_data.parquet'))
            df = self.processed_dataset.copy()
            print(f'Number of records: {len(df)}')
            print(f"Number of people: "
                  f"{len(df.index.get_level_values('id').drop_duplicates())}")
            print(f"Number of days: {len(df['day'].drop_duplicates())}")
        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except pd.errors.EmptyDataError as e:
            print(f"No data: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def remove_zero_peak_days(self, variable: str = 'cob interpolate',
                              suppress: bool = False):
        """
        Removes days with no peaks from an individual dataset, based on the
        interpolated data by default. (The individual dataset hsa a one-level
        index of datetime).

        Returns:
            df_cob (pd.DataFrame): DataFrame with days with zero-data days
            removed df_agg_by_day (pd.DataFrame): DataFrame containing
            aggregated data by day, flagging removed days (but not removing
            them)
        """
        # Make a copy to avoid slicing warnings
        df_cob = self.individual_dataset.copy()

        # Check for required columns and correct or raise error
        if 'peak' not in df_cob.columns:
            raise ValueError('No peak column found in DataFrame')

        # Aggregate by day
        df_agg_by_day = df_cob.groupby('day').agg(
            peak_interp_sum=('peak', 'sum'),
            interp_cob_sum=(variable, lambda x: x[df_cob['peak'] == 1].sum()),
            missing=(variable, lambda x: x.isnull().sum()))

        # Set flag for removed days based on no peaks or all missing values
        df_agg_by_day['removed'] = 0
        df_agg_by_day.loc[df_agg_by_day['peak_interp_sum'] == 0, 'removed'] = 1

        # Remove days with no peaks
        zero_data_days = df_agg_by_day[df_agg_by_day['removed'] == 1].index
        df_cob = df_cob.loc[~df_cob['day'].isin(zero_data_days)]
        days = df_cob['day'].drop_duplicates()
        if not suppress:
            if len(zero_data_days) > 0:
                print(f'For ID {self.individual}: {len(zero_data_days)}'
                      f' days with no peaks removed, {len(days)} remain.')
            else:
                print('No days with zero peaks found.')

        return df_cob, df_agg_by_day

    def summarise_peaks_by_time(self):
        """
        Summarises the peaks by time of day for the processed dataset.
        """
        if self.processed_dataset is None:
            print('No processed dataset available. Please run '
                  'pre_process_batch() first.')
            return

        df = self.processed_dataset.copy()
        df = df.groupby('time').agg({'peak': 'sum'}).reset_index()

        return df

    def process_one_tz_individuals(self, profile_offsets: pd.DataFrame,
                                   *kwargs) -> pd.DataFrame:
        """
        Process the dataset for one timezone individuals, adjusting the datetime
        given the offsets provided in the profile_offsets DataFrame. This df
        should have 'id' and 'offset' columns, where 'offset' is the number of
        hours from UTC. It will also check if the offsets have already been
        processed, and if so, will return the existing dataset.
        :param profile_offsets: Dataframe with 'id' and 'offset' columns
        :param kwargs: Keyword arguments for pre_process_batch method
        :return: Datafrome: Processed dataset with adjusted datetime
        """
        if self.offset_processed:
            print('Offset already processed. Returning existing dataset.')
            return self.dataset
        if profile_offsets['id'].duplicated().any():
            raise ValueError("Profile offsets DataFrame contains duplicate IDs."
                             " Please ensure each ID is unique such that only"
                             " one offset exists.")
        zip_ids = profile_offsets['id'].unique()
        self.pre_process_batch(ids=zip_ids, *kwargs)
        self.dataset['offset'] = self.dataset.index.map(profile_offsets['offset'])
        self.dataset['datetime'] = self.dataset['datetime'] + self.dataset[
            'offset'].apply(lambda h: timedelta(hours=h))
        self.offset_processed = True
        return self.dataset

def _return_peaks(df):
    pass

def plot_by_hour_individuals(df):
    """
    Plots a histogram of peaks for separate individuals for comparison, in a
    facetgrid.
    :param df: Dataframe with datetime and id as index, and peak column with 1
    in to represent a peak
    :return:
    """
    df_reset = df.reset_index()
    df_reset['hour'] = df_reset['datetime'].dt.hour
    df_peaks = df_reset[df_reset['peak'] == 1]

    # FacetGrid by 'id'
    g = sns.FacetGrid(df_peaks, col='id', col_wrap=4, sharex=True, sharey=True)
    g.map_dataframe(sns.histplot, x='hour', bins=range(0, 25), discrete=True)
    g.set_axis_labels('Hour', 'Peak Count')
    g.set_titles('ID {col_name}')
    plt.tight_layout()
    plt.show()


def plot_cob_by_hour(df):
    """
    Plots the number of peaks by hour of the day for a processed dataset.
    """
    peaks = df[df['peak'] == 1].index.get_level_values('datetime').hour
    peak_counts = pd.Series(peaks).value_counts()
    peak_counts = peak_counts.sort_index()
    peak_counts.plot(kind='bar')
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Peaks")
    plt.title("Peak Counts by Hour")
    plt.xticks(range(peak_counts.index.min(), peak_counts.index.max() + 1))
    plt.show()


def hierarchical_clustering(summary_df: pd.DataFrame, feature_cols: list = None) -> pd.DataFrame:
    """
    Perform hierarchical clustering on the summary DataFrame.
    :param summary_df: (pd.DataFrame) DataFrame containing summary statistics 
    with 'id' as index.
    :param features: (list) List of features to use for clustering. If None,
    :return None: Displays a dendrogram of the hierarchical clustering.
    """

    # Ensure the DataFrame is indexed by 'id'
    if summary_df.index.name != 'id':
        raise ValueError("The DataFrame must contain an 'id' column.")


    if feature_cols is None:
        feature_cols = ['days_with_data', 'num_peaks', 'missing_percent']
    # Select features and invert missing_percent for clustering
    features = summary_df[
        feature_cols].copy()
    invert_cols = ['missing_percent', 'total_missing']
    for col in invert_cols:
        if col in features.columns:
            features[col] = -features[col]  # Lower is better

    # Standardize features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features)

    # Hierarchical clustering
    z = linkage(x_scaled, method='ward')

    # Plot dendrogram
    plt.figure(figsize=(8, 6))
    dendrogram(z, labels=summary_df.index.astype(int).astype(str), leaf_rotation=90, truncate_mode='level', p=4)
    plt.title('Hierarchical Clustering Dendrogram of Individuals')
    plt.xlabel('Individual (zip_id)')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()

    # Assign clusters (after visual inspection of dendrogram)
    clusters = fcluster(z, t=7, criterion='maxclust')
    summary_df['cluster'] = clusters
    features['cluster'] = clusters

    # Print cluster sizes
    cluster_sizes = summary_df['cluster'].value_counts()
    print("\nCluster Sizes:")
    for cluster, size in cluster_sizes.sort_values().items():
        print(f"Cluster {cluster}: {size} individuals")

    # Plot clusters in matrix plot for all features
    plt.figure(figsize=(10, 8))
    sns.pairplot(features, hue='cluster', palette='Set2', height=4, aspect=1)
    plt.suptitle('Pairplot of Features by Cluster', y=1.02)
    plt.show()

    return summary_df


