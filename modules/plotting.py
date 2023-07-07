
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from modules.utils import get_directory
from modules.default import *


class Plotting:

    def __init__(self):

        directory = get_directory()
        self.directory_plots = os.path.join(directory, PLOTS_LEAF)
        os.makedirs(self.directory_plots, exist_ok=True)

    def plot_patterns_and_pct_missing_data(self, data: pd.DataFrame,
                                           pct_missing_data: pd.Series) -> None:
        """
        Plots patterns and percentages of missing values of stocks.
        :param data: (pd.DataFrame) data set with missing values.
        :param pct_missing_data: (pd.Series) percentages (%) of missing values of rows per column.
        :return: None.
        """

        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(4, 5))

        ax[0].imshow(data.isnull(), interpolation='none', aspect='auto', cmap='gray')
        ax[0].set_ylabel(PLOT_YLABEL_DAY)
        ax[0].set_title(PLOT_TITLE_PATTERNS_MISSING_DATA)

        x_coordinates = np.arange(pct_missing_data.shape[0])

        ax[1].bar(x=x_coordinates, height=pct_missing_data)
        ax[1].axhline(y=THRESHOLD_PCT_MISSING_DATA, color='k', linestyle='--', label=PLOT_THRESHOLD_PCT_LABEL)
        ax[1].set_xlabel(PLOT_XLABEL_STOCKS)
        ax[1].set_ylabel(PLOT_PCT_MISSING_DATA)
        ax[1].set_title(PLOT_PCT_TITLE_MISSING_DATA)
        ax[1].legend()

        fig.tight_layout()  # if bbox_inches of fig.savefig is used, it overlaps subtitles.

        path_to_save = os.path.join(self.directory_plots, PLOT_MISSING_DATA_SVG)
        fig.savefig(path_to_save)
        plt.close()

    def plot_various_eps(self, coef_scaling: np.array) -> None:
        """
        Plots the percentages of outliers identified by DBSCAN for various eps values.
        :param coef_scaling: (np.array) statistically significant coefficients of 3FF from Linear Regression models.
        :return: None.
        """

        outlier_pct = []
        num_stocks = coef_scaling.shape[0]

        eps_samples = np.linspace(0.001, 1, 50)

        for eps in eps_samples:
            dbscan = DBSCAN(eps=eps, min_samples=DBSCAN_SELECT_MIN_SAMPLES)
            dbscan.fit(coef_scaling)
            num_outliers = np.sum(dbscan.labels_ == -1)  # -1 indicates outliers.
            pct_outliers = (num_outliers / num_stocks) * 100
            outlier_pct.append(pct_outliers)

        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.plot(eps_samples, outlier_pct, color='#2c3c43')
        ax.set_xlabel(PLOT_EPS_LABEL)
        ax.set_ylabel(PLOT_OUTLIERS_LABEL)

        fig.tight_layout()

        path_to_save = os.path.join(self.directory_plots, PLOT_EPS_CURVE_SVG)
        fig.savefig(path_to_save)
        plt.close()

    @staticmethod
    def plot_clusters(coef: np.array, clusters: np.array) -> None:
        """
        Plots the clusters with different colour each.
        :param coef: (np.array) statistically significant coefficients of 3FF from Linear Regressions.
        :param clusters: (np.array) clusters detected by DBSCAN algorithm.
        :return: None.
        """

        unique_clusters = np.unique(clusters)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.set_aspect("equal")

        for k in unique_clusters:

            mask = np.where(clusters == k)
            coef_mask = coef[mask]

            x_dim = coef_mask[:, 0]  # Rm-Rf coef
            y_dim = coef_mask[:, 1]  # SMB coef
            z_dim = coef_mask[:, 2]  # HML coef

            ax.scatter(x_dim, y_dim, z_dim, color=PLOT_COLORS_CLUSTERS[k], alpha=1, label=k)

        ax.set_xlabel(COEF_COLS[1])
        ax.set_ylabel(COEF_COLS[2])
        ax.set_zlabel(COEF_COLS[3])

        ax.legend(title=CLUSTERING_COLS[1])

        plt.show()

    def plot_entry_exit_positions(self, pair_zscores: pd.Series,
                                  positions_for_plotting: pd.DataFrame,
                                  case: str,
                                  num_pair: int,
                                  selected_stock_symbol: str,
                                  leg_stock_symbol: str,
                                  hyperparameter_set: tuple) -> None:
        """
        Plots z-score series along with entry and exit positions.
        :param pair_zscores: (pd.Series) z-score series from pair.
        :param positions_for_plotting: (pd.DataFrame) entry and exit positions based on z-scores and thresholds.
        :param case: (str) indicates the source of pairs, e.g. universe, cluster 0, etc.
        :param num_pair: (int) indicates the number of pair, e.g. 1st pair, 2nd pair etc.
        :param selected_stock_symbol: (str) the first stock of pair.
        :param leg_stock_symbol: (str) the second stock of pair.
        :param hyperparameter_set: (tuple) hyperparameters of number of top pairs, window size, and thresholds.
        :return: None.
        """

        # plots regarding pairs and z-scores:
        directory_pair_plots = os.path.join(self.directory_plots, PLOTS_PAIRS_LEAF)
        os.makedirs(directory_pair_plots, exist_ok=True)

        case_directory = os.path.join(directory_pair_plots, case)
        os.makedirs(case_directory, exist_ok=True)

        thresholds = [
            hyperparameter_set[2],  # entry to market to short z-score
            hyperparameter_set[3],  # exit market when short z-score
            hyperparameter_set[4],  # entry to market to long z-score
            hyperparameter_set[5]  # exit market when long z-score
        ]

        plt.figure(figsize=(9, 6))
        plt.plot(pair_zscores, label=PLOT_ZSCORES_LEGENGS[0])

        for i, threshold in enumerate(positions_for_plotting):

            position_indices = positions_for_plotting[positions_for_plotting[threshold] == 1].index
            position_zscores = pair_zscores[position_indices]

            plt.plot(position_indices, position_zscores,
                     marker=PLOT_ZSCORES_MARKERS[i], markersize=PLOT_ZSCORES_MARKERS_SIZE,
                     color=PLOT_ZSCORES_COLOURS[i], linestyle='None', label=PLOT_ZSCORES_LEGENGS[i+1])

        plt.axhline(0, c=PLOT_ZSCORES_COLOURS[-1])
        plt.axhline(thresholds[0], c=PLOT_ZSCORES_COLOURS[0], ls='--')
        plt.axhline(thresholds[1], c=PLOT_ZSCORES_COLOURS[1], ls='--')
        plt.axhline(thresholds[2], c=PLOT_ZSCORES_COLOURS[0], ls='--')
        plt.axhline(thresholds[3], c=PLOT_ZSCORES_COLOURS[1], ls='--')

        plt.xlabel(PLOT_ZSCORES_XLABEL)
        plt.ylabel(PLOT_ZSCORES_YLABEL)
        plt.title(PLOT_ZSCORES_TITLE.format(selected_stock_symbol, leg_stock_symbol))
        plt.legend()

        path_to_save = os.path.join(case_directory, PLOT_ZSCORES_PATH.format(case, num_pair))
        plt.savefig(path_to_save)
        plt.close()
