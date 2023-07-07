
import pandas as pd
import yfinance as yf
from modules.default import *


class DataCollection:

    def __init__(self):

        # Price series of stocks of Standard & Poor's 500 (S&P500) index:
        symbols_sp500 = self.get_symbols_sp500(url=URL_SP500)
        self.price_series_stocks = self.get_price_series_stocks(symbols=symbols_sp500)

        # Returns of Fama/French 3 (three) Factors:
        self.returns_3ff = self.get_returns_3ff(url=URL_3FF)

    @staticmethod
    def get_symbols_sp500(url: str) -> list:
        """
        Reads the URL and scrapes ticker data.
        :param url: (str) reference of Wikipedia website of stocks of S&P500 index.
        :return: (list) all the stock symbols of S&P500 index, (# symbols,) <str>.
        """

        symbols_sp500 = pd.read_html(url)[0][URL_SP500_SYMBOL_COL]

        # Replace '.' with '-':
        symbols_sp500 = symbols_sp500.apply(lambda x: x.replace('.', '-'))

        # Keep all symbols into a list:
        symbols_sp500 = symbols_sp500.to_list()

        print(f'\nThe number of stocks of S&P500 index is: {len(symbols_sp500)}')

        return symbols_sp500

    @staticmethod
    def get_price_series_stocks(symbols: list) -> pd.DataFrame:
        """
        Downloads the prices series of Adjusted closing prices of stocks included in S&P500 index.
        :param symbols: (list) stock symbols included in S&P500 index.
        :return: (pd.DataFrame) Price series of stocks of S&P500 index, (# days, # stocks) <numpy.float64>.
        """

        print('\nDownloading price series of stocks...')

        # Download the price series for each stock:
        price_series = yf.download(symbols,
                                   start=START_DATE,
                                   end=END_DATE,
                                   progress=True)

        # Keep only the Adjusted closing prices:
        price_series_stocks = price_series[ADJ_CLOSE_COL]

        print(f'\nThe shape of price series dataframe is: {price_series_stocks.shape}')

        return price_series_stocks

    @staticmethod
    def get_returns_3ff(url: str) -> pd.DataFrame:
        """
        Downloads the return series of 3 (three) common risk factors in returns on stock, which are:
        - 1st factor related to the excess return on the market (Rm-Rf),
        - 2nd factor related to firm size (Small Minus Big, "SMB"), and
        - 3rd factor related to book-to-market equity (High Minus Low, "HML").
        :param url: (str) reference of Prof Kenneth R. French's website about Fama/French factors.
        :return: (pd.DataFrame) 3 Fama-French (3FF) factors, (# days, 3) <numpy.float64>.
        """

        return_series_3ff = pd.read_csv(url, skiprows=4, skipfooter=2, engine='python')

        # Exclude the last 'Rf' column:
        return_series_3ff = return_series_3ff.iloc[:, :-1]

        # Rename the columns:
        return_series_3ff.columns = COL_FOR_3FF

        # Convert the 'Date' column to datetime:
        return_series_3ff[COL_FOR_3FF[0]] = pd.to_datetime(return_series_3ff[COL_FOR_3FF[0]], format='%Y%m%d')

        # Set 'Date' as index:
        return_series_3ff.set_index(COL_FOR_3FF[0], inplace=True)

        # Scale returns - percentages not needed:
        return_series_3ff = return_series_3ff / 100

        print(f'\nThe shape of return series dataframe (3FF) is: {return_series_3ff.shape}')

        return return_series_3ff


