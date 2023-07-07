
import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.rolling import RollingOLS
from modules.default import *


def get_directory() -> str:
    """
    Returns the pathname of the python file from which the module
    was loaded, if it was loaded from a file.
    :return: (str) directory that the python file loaded.
    """

    return os.path.dirname(os.path.dirname(__file__))


def spread_and_zscores(selected_stock_price_series: pd.Series,
                       leg_stock_price_series: pd.Series,
                       hyperparameter_set: tuple) -> pd.Series:
    """
    Calculates the rolling z-score values of Spread of pair=(selected stock, leg stock).
    :param selected_stock_price_series: (pd.Series) selected stock's price series of pair.
    :param leg_stock_price_series: (pd.Series) leg stock's price series of pair.
    :param hyperparameter_set: (tuple) hyperparameters of number of top pairs, window size, and thresholds.
    :return: (pd.Series) z-score values of pair's Spread (= selected_stock - gamma * leg_stock).
    """

    window = hyperparameter_set[1]

    # Rolling gamma (cointegration) coefficients to avoid look-ahead biases:
    # intercept is neglected (=0) (Krauss et al. 2016).
    rres = RollingOLS(endog=selected_stock_price_series,  # y - dependent/target (selected stock)
                      exog=leg_stock_price_series,  # X - independent (leg stock)
                      window=window).fit()

    # Keep rolling gamma coefficients:
    rgamma = rres.params.iloc[:, 0]

    # cointegration coef (gamma) should be positive (Puspaningrum et al. 2010)
    if np.min(rgamma) < 0:
        print(f'[ WARNING ] Cointegration coefficient (gamma) is negative.')

    # Rolling spread:
    rspread = selected_stock_price_series - rgamma * leg_stock_price_series

    # Rolling average of spread:
    rmean = rspread.rolling(window).mean()

    # Rolling standard deviation of spread:
    rstd = rspread.rolling(window).std()

    # Rolling z-score values (dimensionless).
    # z-score measures how many standard deviations away we are from the mean.
    # - If z-score=0, it indicates that the data point's value is identical to the mean.
    # - If z-score=2, it indicates that the data point's value is 2 (two) standard deviations from the mean.
    rz_score_spread = (rspread - rmean) / rstd

    # Shift all the values "forward" one day.
    rz_score_spread = rz_score_spread.shift(periods=1)

    # Drop NaN (missing) values:
    rz_score_spread = rz_score_spread.dropna()

    return rz_score_spread


def entry_exit_positions(pair_zscores: pd.Series,
                         hyperparameter_set: tuple) -> tuple[pd.Series, pd.DataFrame]:
    """
    Defines the Entry and Exit positions based on the rolling z-score values and thresholds.
    :param pair_zscores: (pd.Series) rolling z-score values from pair.
    :param hyperparameter_set: (tuple) hyperparameters of number of top pairs, window size, and thresholds.
    :return: (tuple[pd.Series, pd.DataFrame]) trading positions.
    """

    thresholds = [
        hyperparameter_set[2],  # entry to market to short z-score
        hyperparameter_set[3],  # exit market when short z-score
        hyperparameter_set[4],  # entry to market to long z-score
        hyperparameter_set[5]   # exit market when long z-score
    ]

    positions_for_returns = []

    # Create a DataFrame using the date index of z-scores:
    positions_for_plotting = pd.DataFrame(0,
                                          columns=THRESHOLDS_COLS,
                                          index=pair_zscores.index)

    # When we start backtesting, we are out of market:
    mode = EXIT_MODE

    for i in range(len(positions_for_plotting)):

        # Entry based on Threshold (a):
        if mode == EXIT_MODE and pair_zscores[i] > thresholds[0]:
            positions_for_plotting.iloc[i, positions_for_plotting.columns.get_loc(THRESHOLDS_COLS[0])] = 1
            mode = MARKET_MODE[0]

        # Exit based on Threshold (b):
        if mode == MARKET_MODE[0] and pair_zscores[i] < thresholds[1]:
            positions_for_plotting.iloc[i, positions_for_plotting.columns.get_loc(THRESHOLDS_COLS[1])] = 1
            mode = EXIT_MODE

        # Entry based on Threshold (c):
        if mode == EXIT_MODE and pair_zscores[i] < thresholds[2]:
            positions_for_plotting.iloc[i, positions_for_plotting.columns.get_loc(THRESHOLDS_COLS[2])] = 1
            mode = MARKET_MODE[1]

        # Exit based on Threshold (d):
        if mode == MARKET_MODE[1] and pair_zscores[i] > thresholds[3]:
            positions_for_plotting.iloc[i, positions_for_plotting.columns.get_loc(THRESHOLDS_COLS[3])] = 1
            mode = EXIT_MODE

        positions_for_returns.append(mode)

    # convert it to pd.Series along with date index:
    positions_for_returns = pd.Series(positions_for_returns, index=pair_zscores.index)

    return positions_for_returns, positions_for_plotting


def get_pair_returns(selected_stock_price_series: pd.Series,
                     leg_stock_price_series: pd.Series,
                     positions: pd.Series) -> np.ndarray:
    """
    Calculates and concatenates the returns of pair.

    Short position: investor expects price to fall (Pt < Pt-1), e.g. ret_l,t = (Pt-1 / Pt) - 1.
    Long position: investor expects price to go up (Pt > Pt-1), e.g. ret_s,t = (Pt / Pt-1) - 1.

    From ret_l,t equation above, we can get (Pt / Rt-1) = 1 / (ret_l,t + 1). Replacing it in
    the ret_s,t equation, we have:

    ret_s,t = - ret_l,t / (ret_l,t + 1)

    With regards to Pairs trading:
    - Short position on z-score series means short selected stock and long leg stock.
    - Long position on z-score series means long selected stock and short leg stock.

    :param selected_stock_price_series: (pd.Series) selected stock's price series of pair.
    :param leg_stock_price_series: (pd.Series) leg stock's price series of pair.
    :param positions: (pd.Series) trading positions.
    :return: (np.ndarray) pair returns, (# days in market,) <numpy.float64>.
    """

    # Selected stock's Long/Short returns:
    selected_stock_long_returns = selected_stock_price_series.pct_change(periods=1)
    selected_stock_short_returns = - selected_stock_long_returns / (selected_stock_long_returns + 1)

    # Leg stock's Long/Short returns:
    leg_stock_long_returns = leg_stock_price_series.pct_change(periods=1)
    leg_stock_short_returns = -leg_stock_long_returns / (leg_stock_long_returns + 1)

    # Short z-score series -> short selected stock, long leg stock:
    short_zscore_mask = positions[positions == MARKET_MODE[0]].index
    pair_short_returns = selected_stock_short_returns[short_zscore_mask] + leg_stock_long_returns[short_zscore_mask]

    # Long z-score series -> long selected stock, short leg stock:
    long_zscore_mask = positions[positions == MARKET_MODE[1]].index
    pair_long_returns = selected_stock_long_returns[long_zscore_mask] + leg_stock_short_returns[long_zscore_mask]

    # concatenate
    pair_returns = np.concatenate([pair_short_returns.values, pair_long_returns.values])

    return pair_returns


def get_pair_returns_statistics(pair_returns: np.ndarray,
                                num_test_days: int,
                                pair: tuple) -> list:
    """
    Calculates statistics of pair.
    :param pair_returns: (np.ndarray) contains the returns of pair.
    :param num_test_days: (int) number of days of test period.
    :param pair: (tuple) symbols of stocks of pair.
    :return: (list) statistics of pair, (12,) <tuple, int, float>.
    """

    num_trading_days = pair_returns.shape[0]  # in-market days
    pct_trading_days = num_trading_days / num_test_days
    num_positive_ret = np.sum(pair_returns > 0)  # positive returns
    pct_positive_ret = num_positive_ret / num_trading_days
    ret_mean = np.mean(pair_returns)
    ret_std = np.std(pair_returns)
    ret_min = np.min(pair_returns)
    ret_median = np.median(pair_returns)
    ret_max = np.max(pair_returns)
    ret_skew = stats.skew(pair_returns)  # for normal distribution = 0
    ret_kurtosis = stats.kurtosis(pair_returns)  # for normal distribution = 0

    pair_returns_stats = [pair,              # 0
                          num_trading_days,  # 1
                          pct_trading_days,  # 2
                          num_positive_ret,  # 3
                          pct_positive_ret,  # 4
                          ret_mean,          # 5
                          ret_std,           # 6
                          ret_min,           # 7
                          ret_median,        # 8
                          ret_max,           # 9
                          ret_skew,          # 10
                          ret_kurtosis       # 11
                          ]

    return pair_returns_stats


def save_data_csv(data: pd.DataFrame, csv_leaf: str, csv_name: str) -> None:
    """
    Generates a .csv file containing the results.
    :param data: (pd.DataFrame) data to be saved in a .csv format.
    :param csv_leaf: (str) leaf name of directory in which data is saved.
    :param csv_name: (str) csv name of data.
    :return: None.
    """

    directory = get_directory()
    csv_directory = os.path.join(directory, csv_leaf)
    os.makedirs(csv_directory, exist_ok=True)
    path_to_save = os.path.join(csv_directory, csv_name)
    data.to_csv(path_to_save, index=False)
