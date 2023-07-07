
import numpy as np
import pandas as pd
from modules.plotting import Plotting
from modules.default import THRESHOLD_PCT_MISSING_DATA


class DataPreprocessing:

    def __init__(self, price_series_stocks, return_series_3ff):

        # Create the plotting object:
        self.plotting = Plotting()

        # Check and handle missing data:
        self.price_series_stocks = self.handle_missing_data(data=price_series_stocks)
        self.return_series_3ff = self.handle_missing_data(data=return_series_3ff)

        # Calculate the returns of stocks:
        self.return_series_stocks = self.calculate_return_series_stocks(data=self.price_series_stocks)

        # Match the dates (indices) of Stock returns with 3FF returns:
        self.return_series_3ff = self.match_dates(returns_stocks=self.return_series_stocks,
                                                  returns_3ff=self.return_series_3ff)

    def handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Examines the data set whether it contains missing values and handles them.
        :param data: (pd.DataFrame) data set examined for missing values.
        :return: (pd.DataFrame) data set without missing data, (# days, # stocks) <numpy.float64>.
        """

        bool_missing_data = data.isnull().values.any()

        if bool_missing_data:

            print('\nThe data set contains missing data:')

            num_rows = data.shape[0]
            num_cols = data.shape[1]

            # Count columns with missing data:
            num_cols_missing_data = data.isnull().any().sum()
            pct_cols_missing_data = (num_cols_missing_data / num_cols) * 100

            print(f' - Columns with missing data are {num_cols_missing_data} out of {num_cols}, '
                  f'that is, {pct_cols_missing_data:.2f}%.')

            # Count rows per column with missing data:
            num_rows_missing_data = data.isnull().sum()
            pct_rows_missing_data = (num_rows_missing_data / num_rows) * 100

            self.plotting.plot_patterns_and_pct_missing_data(data=data,
                                                             pct_missing_data=pct_rows_missing_data)

            # Detect columns with missing data lower than the Threshold (%):
            mask_cols_missing_data_threshold = pct_rows_missing_data.values < THRESHOLD_PCT_MISSING_DATA

            print(f' - Columns with missing data lower than {THRESHOLD_PCT_MISSING_DATA}% are '
                  f'{np.sum(mask_cols_missing_data_threshold)} out of {num_cols}.')

            # Keep columns with missing data lower than the Threshold:
            data = data.iloc[:, mask_cols_missing_data_threshold]

            # If the remaining missing data is not located at the beginning of the examined period,
            # propagate last valid observation forward to next valid:
            index_threshold = int(num_rows * (THRESHOLD_PCT_MISSING_DATA / 100))
            mask_remaining_cols_missing_data = data.isnull().any().values
            for i, bool_col in enumerate(mask_remaining_cols_missing_data):
                if bool_col:
                    indices_missing_data = np.where(data.iloc[:, i].isnull())
                    if np.max(indices_missing_data) > index_threshold:
                        data.iloc[:, i].fillna(method='ffill', inplace=True)

            # Drop rows with missing data located at the beginning of examined period:
            data = data.dropna()

        else:

            print('\nThe data set does not contain missing data.')

        return data

    @staticmethod
    def calculate_return_series_stocks(data: pd.DataFrame) -> pd.DataFrame:
        """
        Computes percentage changes between the current and a prior element.
        :param data: (pd.DataFrame) price series of stocks of S&P500 index.
        :return: (pd.DataFrame) daily returns of stocks of S&P500 index,
                 (# days, # stocks) <numpy.float64>.
        """

        return_series_stocks = data.pct_change()

        # Drop the first NaN row after calculating returns
        return_series_stocks = return_series_stocks.dropna()

        return return_series_stocks

    @staticmethod
    def match_dates(returns_stocks: pd.DataFrame,
                    returns_3ff: pd.DataFrame) -> pd.DataFrame:
        """
        Matches the dates (indices) of stock returns with 3FF returns.
        :param returns_stocks: (pd.DataFrame) return series of stocks of S&P500 index.
        :param returns_3ff: (pd.DataFrame) return series of 3FF factors.
        :return: (pd.DataFrame) return series of 3FF factors with the same dates (indices)
                 as Stock returns, (# days, 3) <numpy.float64>.
        """

        returns_3ff = returns_3ff.loc[returns_stocks.index]

        return returns_3ff
