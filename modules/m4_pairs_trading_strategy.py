
from itertools import combinations
from statsmodels.tsa import stattools
from modules.utils import *
from modules.plotting import Plotting
from modules.default import *


class PairsTradingStrategy:

    def __init__(self, price_series_stocks, clusters):

        plotting = Plotting()

        training_price_series = price_series_stocks[:SPLIT_DATE]
        test_price_series = price_series_stocks[SPLIT_DATE:]

        # Bear in mind it may be a time-consuming process:
        pairs_universe = self.find_pairs_universe(training_price_series=training_price_series)

        pairs_clusters = self.find_pairs_clusters(pairs_universe=pairs_universe, clusters=clusters)

        self.results_baseline_list = []
        self.results_clusters_list = []

        for hyperparameter_set in HYPERPARAMETERS:

            print(f'\n[ Hyperparameters ] {hyperparameter_set}')

            pairs_top = self.keep_top_pairs(pairs_universe=pairs_universe,
                                            pairs_clusters=pairs_clusters,
                                            num_top_pairs=hyperparameter_set[0])

            portfolio_average_returns = self.run_pairs_trading_strategy(pairs_top=pairs_top,
                                                                        test_price_series=test_price_series,
                                                                        hyperparameter_set=hyperparameter_set,
                                                                        plotting=plotting)

            # compare Baseline portfolios with Cluster ones:
            cluster_portfolio_t_tests = self.get_portfolio_t_tests(portfolio_average_returns=portfolio_average_returns)

            self.get_results_baseline(portfolio_average_returns=portfolio_average_returns,
                                      hyperparameter_set=hyperparameter_set)

            self.get_results_clusters(portfolio_average_returns=portfolio_average_returns,
                                      cluster_portfolio_t_tests=cluster_portfolio_t_tests,
                                      hyperparameter_set=hyperparameter_set)

        # convert lists to dataframe
        self.results_baseline_df = pd.DataFrame(self.results_baseline_list)
        self.results_baseline_df.columns = RESULTS_COLS[:-2]
        save_data_csv(data=self.results_baseline_df,
                      csv_leaf=CSV_RESULTS_LEAF,
                      csv_name=CSV_BASELINE_NAME)

        self.results_clusters_df = pd.DataFrame(self.results_clusters_list)
        self.results_clusters_df.columns = RESULTS_COLS
        save_data_csv(data=self.results_clusters_df,
                      csv_leaf=CSV_RESULTS_LEAF,
                      csv_name=CSV_CLUSTERS_NAME)

    @staticmethod
    def find_pairs_universe(training_price_series: pd.DataFrame) -> pd.DataFrame:
        """
        Returns all possible pair combinations from the universe of stocks listed on S&P500 index.

        When we are dealing with two Non-stationary series and are looking for a potential relationship
        between them, we can examine if they are cointegrated.

        To do this, we are going to use the augmented Engle-Granger two-step cointegration test.

        The Null hypothesis is that there is no cointegration, the alternative hypothesis is that
        there is cointegrating relationship. If the p-value is small, below a critical size (e.g.
        p-value <= critical_value), then we can reject the hypothesis that there is no cointegrating relationship.

        - Null hypothesis: there is no cointegration between two variables (p-value > critical_value).
        - Alt hypothesis : there is a cointegrating relationship between two variables (p-value <= critical_value)

        The pairs are sorted based on p-values.

        :return: (pd.DataFrame) cointegrated pairs (statistically significant) from the universe of stocks
                 listed on S&P500 index, (# pairs, 3) <str, str, numpy.float64>.
        """

        combinations_in_total = len([combo for combo in combinations(training_price_series.columns, 2)])

        pairs_list = []

        for i, combo in enumerate(combinations(training_price_series.columns, 2)):

            print(f'pair {i} out of {combinations_in_total} ({(i / combinations_in_total) * 100:.2f}%)')

            # Symbols:
            symbol_selected_stock = combo[0]
            symbol_leg_stock = combo[1]

            # Stock price series:
            selected_stock = training_price_series[symbol_selected_stock].values
            leg_stock = training_price_series[symbol_leg_stock].values

            # Cointegration test:
            coint_t, pvalue, _ = stattools.coint(selected_stock, leg_stock)

            # Keep only the cointegrated pairs:
            if pvalue <= P_VALUE_COINTEGRATION:
                print('Cointegrated pair!')
                pairs_list.append([symbol_selected_stock, symbol_leg_stock, pvalue])

        pairs_df = pd.DataFrame(pairs_list, columns=COINT_COLS)

        # sort by p-values:
        pairs_df_sort = pairs_df.sort_values(COINT_COLS[2]).reset_index(drop=True)

        return pairs_df_sort

    @staticmethod
    def find_pairs_clusters(pairs_universe: pd.DataFrame,
                            clusters: pd.DataFrame) -> dict:
        """
        Finds the cointegrated pairs (statistically significant) of stock clusters generated by
        DBSCAN clustering algorithm using beta coefficients of Fama/French 3-factor model.
        :param pairs_universe: (pd.DataFrame) cointegrated pairs from stocks of universe listed on S&P500 index.
        :param clusters: (pd.DataFrame) stock clusters.
        :return: (dict) cointegrated pairs of stocks included in clusters, (# pairs, 3) <str, str, np.float64>.
        """

        symbols_pair_universe = pairs_universe.iloc[:, 0].values  # convert it to np.array

        unique_clusters = np.unique(clusters.iloc[:, 1])

        pairs_clusters_dict = dict()

        for k in unique_clusters:

            mask = clusters.iloc[:, 1] == k
            cluster = clusters[mask]
            symbols_cluster = list(cluster.iloc[:, 0])  # convert it to list to iterate it.

            indices_list = []

            for symbol in symbols_cluster:

                symbol_indices = np.where(symbols_pair_universe == symbol)[0]  # [0] as np.where returns a tuple.
                indices_list.append(symbol_indices)

            indices_cluster = np.concatenate(indices_list)
            pairs_cluster_df = pairs_universe.iloc[indices_cluster]

            # sort by p-values:
            pairs_cluster_df_sort = pairs_cluster_df.sort_values(COINT_COLS[2]).reset_index(drop=True)

            pairs_clusters_dict[CLUSTER_LABEL + f'_{k}'] = pairs_cluster_df_sort

        return pairs_clusters_dict

    @staticmethod
    def keep_top_pairs(pairs_universe: pd.DataFrame,
                       pairs_clusters: dict,
                       num_top_pairs: int) -> dict:
        """
        Keeps top pairs of stocks of universe and clusters.
        :param pairs_universe: (pd.DataFrame) cointegrated pairs of stocks of universe.
        :param pairs_clusters: (dict) cointegrated pairs of stocks included in clusters.
        :param num_top_pairs: (int) the number of top pairs.
        :return: (dict) top pairs of stocks of universe and clusters, (# top pairs, 2) <str, str>.
        """

        pairs_top_dict = dict()

        pairs_top_universe = pairs_universe[:num_top_pairs]
        pairs_top_universe_symbols = pairs_top_universe.iloc[:, :2].values  # keep symbols
        pairs_top_universe_symbols = list(map(tuple, pairs_top_universe_symbols))  # from 2D array to 1D list of tuples.
        # top pairs from universe are considered as baseline:
        pairs_top_dict[BASELINE] = pairs_top_universe_symbols

        for k in pairs_clusters:

            cluster = pairs_clusters[k]
            pairs_top_cluster = cluster[:num_top_pairs]
            pairs_top_cluster_symbols = pairs_top_cluster.iloc[:, :2].values  # keep symbols
            # from 2D array to 1D list of tuples.
            pairs_top_cluster_symbols = list(map(tuple, pairs_top_cluster_symbols))

            if len(pairs_top_cluster_symbols) >= num_top_pairs:
                pairs_top_dict[k] = pairs_top_cluster_symbols
            else:
                print(f'\nCluster {k} contains less than {num_top_pairs} pairs.')
                continue

        return pairs_top_dict

    @staticmethod
    def run_pairs_trading_strategy(pairs_top: dict,
                                   test_price_series: pd.DataFrame,
                                   hyperparameter_set: tuple,
                                   plotting: Plotting) -> dict:
        """
        Generates a dictionary which contains statistics per portfolio per case (universe, cluster 0, etc.)
        :param pairs_top: (dict) top pairs based on p-value of baseline (universe) and clusters.
        :param test_price_series: (pd.DataFrame) test prices series for each stock.
        :param hyperparameter_set: (tuple) hyperparameters of number of top pairs, window size, and thresholds.
        :param plotting: object from Plotting class.
        :return: (dict) statistics of portfolios per case (universe, cluster 0, etc.).
        """

        print(' - running pairs trading strategy...')

        portfolio_returns_stats = dict()

        for key in pairs_top:

            print(f'\n{key}')

            pair_returns_stats_list = []

            for i, pair in enumerate(pairs_top[key]):

                print(f'pair: {pair}')

                selected_stock_symbol = pair[0]  # <str>
                selected_stock_price_series = test_price_series[selected_stock_symbol]

                leg_stock_symbol = pair[1]  # <str>
                leg_stock_price_series = test_price_series[leg_stock_symbol]

                # z-scores (standardised spread) for pair:
                pair_zscores = spread_and_zscores(selected_stock_price_series=selected_stock_price_series,
                                                  leg_stock_price_series=leg_stock_price_series,
                                                  hyperparameter_set=hyperparameter_set)

                positions = entry_exit_positions(pair_zscores=pair_zscores,
                                                 hyperparameter_set=hyperparameter_set)

                positions_for_returns = positions[0]
                positions_for_plotting = positions[1]

                if PLOT_ZSCORES_BOOL:
                    plotting.plot_entry_exit_positions(pair_zscores=pair_zscores,
                                                       positions_for_plotting=positions_for_plotting,
                                                       case=key,
                                                       num_pair=i,
                                                       selected_stock_symbol=selected_stock_symbol,
                                                       leg_stock_symbol=leg_stock_symbol,
                                                       hyperparameter_set=hyperparameter_set)

                pair_returns = get_pair_returns(selected_stock_price_series=selected_stock_price_series,
                                                leg_stock_price_series=leg_stock_price_series,
                                                positions=positions_for_returns)

                # after rolling calculations, the number of days for trading period is:
                num_test_days = pair_zscores.shape[0]

                pair_returns_stats = get_pair_returns_statistics(pair_returns=pair_returns,
                                                                 num_test_days=num_test_days,
                                                                 pair=pair)

                pair_returns_stats_list.append(pair_returns_stats)

            portfolio_returns_stats[key] = pd.DataFrame(pair_returns_stats_list,
                                                        columns=PORTFOLIO_COLS)

        return portfolio_returns_stats

    @staticmethod
    def get_portfolio_t_tests(portfolio_average_returns: dict) -> dict:
        """
        Calculate t-tests for the means of two independent samples:
         - 1st sample: average daily returns of pairs of Baseline portfolio.
         - 2nd sample: average daily returns of pairs of Cluster portfolio.

         Essentially, we compare the average returns of Baseline portfolios with the average
         returns of Clusters.

        :param portfolio_average_returns: (dict) contains average daily returns of pairs of portfolio coming
               from Baseline and Clusters.
        :return: (dict) t-statistics and p-values of comparisons between Baseline and Cluster portfolios,
                        (# clusters,) <np.float64>.
        """

        baseline_average_returns = portfolio_average_returns[BASELINE][PORTFOLIO_COLS[5]].values

        cluster_keys = [key for key in portfolio_average_returns.keys() if key != BASELINE]

        ttest_dict = dict()

        for cluster in cluster_keys:

            cluster_average_returns = portfolio_average_returns[cluster][PORTFOLIO_COLS[5]].values
            t_stats, pvalue = stats.ttest_ind(a=baseline_average_returns, b=cluster_average_returns, equal_var=False)
            ttest_dict[cluster] = (t_stats, pvalue)

        return ttest_dict

    def get_results_baseline(self, portfolio_average_returns: dict,
                             hyperparameter_set: tuple) -> None:
        """
        Organises and appends the results produced by the current set of hyperparameters (rules) and baseline.
        :param portfolio_average_returns: (dict) contains average daily returns of pairs of portfolio coming
               from Baseline and Clusters.
        :param hyperparameter_set: (tuple) the set of hyperparameters.
        :return: None.
        """

        results_baseline = [BASELINE, hyperparameter_set]
        baseline = portfolio_average_returns[BASELINE]
        baseline = baseline.iloc[:, 1:]  # Exclude 'pair' column
        # TODO: calculate differently std of portfolio instead of averaging.
        baseline_mean_per_col = baseline.mean()  # mean values per column
        results_baseline = results_baseline + list(baseline_mean_per_col)
        self.results_baseline_list.append(results_baseline)

    def get_results_clusters(self, portfolio_average_returns: dict,
                             cluster_portfolio_t_tests: dict,
                             hyperparameter_set: tuple) -> None:
        """
        Organises and appends the results produced by the current set of hyperparameters and clusters.
        :param portfolio_average_returns: (dict) contains average daily returns of pairs of portfolio coming
               from Baseline and Clusters.
        :param cluster_portfolio_t_tests: (dict) contains t-statistics and p-values from comparisons between
               Baseline and Cluster portfolios.
        :param hyperparameter_set: (tuple) the set of hyperparameters (rules).
        :return: None.
        """

        for key in cluster_portfolio_t_tests:

            cluster_results = [key, hyperparameter_set]
            cluster = portfolio_average_returns[key]
            cluster = cluster.iloc[:, 1:]  # Exclude 'pair' column
            # TODO: calculate differently std of portfolio instead of averaging.
            cluster_mean_per_col = cluster.mean()  # mean values per column
            cluster_ttest = cluster_portfolio_t_tests[key]
            cluster_results = cluster_results + list(cluster_mean_per_col) + list(cluster_ttest)
            self.results_clusters_list.append(cluster_results)
