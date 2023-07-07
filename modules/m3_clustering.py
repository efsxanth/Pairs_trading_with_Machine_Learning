
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from modules.plotting import Plotting
from modules.default import *


class Clustering:

    def __init__(self, return_series_stocks, return_series_3ff):

        # Create the plotting object:
        self.plotting = Plotting()

        training_ret_stocks = return_series_stocks[:SPLIT_DATE]
        training_ret_3ff = return_series_3ff[:SPLIT_DATE]

        # Get coefficients of multiple linear regression (mlr) models:
        coef_3ff = self.get_coef_mlr(x=training_ret_3ff, y=training_ret_stocks)

        # Get clusters using 3FF statistically significant stock coefficients:
        self.stock_clusters = self.get_clusters(coef_3ff=coef_3ff)

    @staticmethod
    def get_coef_mlr(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Multiple Linear Regression models between 3FF returns (three independent variables)
        and Stock returns (target/dependent variable).

            return = a + b1 * (Rm-Rf) + b2 * SMB + b3 * HML

        :param x: (pd.DataFrame): return series of Fama/French 3 (three) factors (3FF).
        :param y: (pd.DataFrame): return series of Stocks of S&P500 index.
        :return: (pd.DataFrame):  statistically significant stock coefficients (b1, b2, b3) from
                 Linear Regression models, (# stocks, 4) <str>, <numpy.float64>.
        """

        x = x.values  # convert it to np.array.

        coef_list = []

        for col in y:  # going through stocks

            # return series per stock and convert it to np.array:
            y_ret_stock = y[col].values

            # Multiple Linear Regression:
            regr = sm.OLS(y_ret_stock, sm.add_constant(x)).fit()

            # p-values of coefficients and constants:
            pvalue_const = regr.pvalues[0]  # p-value of constant (intercept)
            pvalue_b1 = regr.pvalues[1]  # p-value of coef b1 (Rm-Rf)
            pvalue_b2 = regr.pvalues[2]  # p-value of coef b2 (SMB)
            pvalue_b3 = regr.pvalues[3]  # p-value of coef b3 (HML)

            # Keep statistically significant coefficients:
            if (pvalue_const >= P_VALUE) and (pvalue_b1 < P_VALUE) and (pvalue_b2 < P_VALUE) and (pvalue_b3 < P_VALUE):

                b1 = regr.params[1]
                b2 = regr.params[2]
                b3 = regr.params[3]

                coef_list.append([col, b1, b2, b3])

        stock_coef_df = pd.DataFrame(coef_list)
        stock_coef_df.columns = COEF_COLS

        print(f'\nThe shape of statistically significant stock coefficients dataframe: {stock_coef_df.shape}')

        return stock_coef_df

    def get_clusters(self, coef_3ff: pd.DataFrame):
        """
        Finds the clusters of statistically significant 3FF coefficients from Linear Regression between
        3FF factors (three independent variables) and Stock returns (target/dependent variable).
        :param coef_3ff: (pd.DataFrame) statistically significant 3FF coefficients from Linear Regressions.
        :return: (pd.DataFrame) clusters linked to stocks, (# stocks, 2), (symbol, cluster), <str>, <int>.
        """

        # Keep the symbols and convert it to np.array:
        symbols = coef_3ff.iloc[:, 0].values

        # Keep only coef as Features and convert it to np.array:
        x_coef = coef_3ff.iloc[:, 1:].values

        # Standardise coefficients by removing the mean and scaling to unit variance:
        x_coef_scaling = StandardScaler().fit_transform(x_coef)

        if SEARCH_EPS:
            self.plotting.plot_various_eps(coef_scaling=x_coef_scaling)

        # DBSCAN clustering algorithm:
        dbscan = DBSCAN(eps=DBSCAN_SELECT_EPS, min_samples=DBSCAN_SELECT_MIN_SAMPLES)
        dbscan.fit(x_coef_scaling)

        clusters = dbscan.labels_

        clustering_stocks = pd.DataFrame(zip(symbols, clusters), columns=CLUSTERING_COLS)

        print('\nNumber of stocks per cluster. Outliers are indicated as -1.')
        print(clustering_stocks.iloc[:, 1].value_counts())

        mask = clusters != -1  # exclude outliers (-1)

        if PLOT_CLUSTERS:

            self.plotting.plot_clusters(coef=x_coef, clusters=clusters)

            # without outliers (-1):
            self.plotting.plot_clusters(coef=x_coef[mask], clusters=clusters[mask])

        # Exclude outliers (clusters indicated as -1):
        clustering_stocks_no_outliers = clustering_stocks[mask].reset_index(drop=True)

        return clustering_stocks_no_outliers
