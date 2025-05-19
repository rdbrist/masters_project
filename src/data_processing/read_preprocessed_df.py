import pandas as pd

from src.configurations import Configuration, GeneralisedCols, Resampling
from src.helper import preprocessed_file_for, flat_preprocessed_file_for
from src.config import INTERIM_DATA_DIR


class ReadPreprocessedDataFrame:
    """
    Class for reading preprocessed csv files into pandas df and configure
    sampling resolution for other classes to use.
    """

    def __init__(self, sampling: Resampling, zip_id: str = None):
        """
        :param sampling: Resampling object: What the time series are resampled
        as this defines which file is being read
        :param zip_id : str: Id for data to read; default None which reads the
        flat file version with all people
        """
        self.__sampling = sampling
        self.__zip_id = zip_id
        self.__config = Configuration()
        self.df = self.__read_df()

    def __read_df(self):
        if self.__zip_id:
            file = preprocessed_file_for(self.__config.perid_data_folder,
                                         self.__zip_id, self.__sampling)
        else:
            file = flat_preprocessed_file_for(INTERIM_DATA_DIR,
                                              self.__sampling)
        try:
            df = pd.read_csv(file,
                             dtype={GeneralisedCols.id: str,
                                    GeneralisedCols.system: str},
                             parse_dates=[GeneralisedCols.datetime]
                             )
            df[GeneralisedCols.datetime] = (
                pd.to_datetime(df[GeneralisedCols.datetime], format='ISO8601'))
        except FileNotFoundError as e:
            raise FileNotFoundError(e)

        return df
