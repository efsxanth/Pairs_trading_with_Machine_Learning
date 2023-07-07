
import pandas as pd
from modules.default import *
from modules.utils import *


class ExtractInsights:

    def __init__(self, results_baseline_df, results_clusters_df):

        # Keep "hp" hyperparameter and t-tests:
        hp_and_ttests = self.get_hyperparameters_and_t_tests(results_clusters_df=results_clusters_df)

        # "bs" baseline, "ss" statistically significant, "port" portfolios:
        bs_ss_port = self.get_baseline_stat_significant_portfolios(hyperparameters_and_t_tests=hp_and_ttests,
                                                                   results_baseline_df=results_baseline_df)
        # "cl" clusters, "ss" statistically significant, "port" portfolios:
        cl_ss_port = self.get_clusters_stat_significant_portfolios(results_clusters_df=results_clusters_df)

        # average performance of statistically significant portfolios
        self.examine_average_performance(baseline_ss_port=bs_ss_port, cluster_ss_port=cl_ss_port)

    @staticmethod
    def get_hyperparameters_and_t_tests(results_clusters_df: pd.DataFrame) -> pd.DataFrame:
        """
        Keeps hyperparameters, t-statistic, and p-value columns of interest used in detecting
        the statistically significant baseline portfolios.
        :param results_clusters_df: (pd.DataFrame) contains the columns of interest.
        :return: (pd.DataFrame) columns of hyperparameters, t-statistic, and p-value.
        """

        hyperparameters_and_t_tests = results_clusters_df[[RESULTS_COLS[1],   # hyperparameters
                                                           RESULTS_COLS[13],  # t-statistic
                                                           RESULTS_COLS[14]   # p-value
                                                           ]]

        return hyperparameters_and_t_tests

    @staticmethod
    def get_baseline_stat_significant_portfolios(hyperparameters_and_t_tests: pd.DataFrame,
                                                 results_baseline_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects the statistically significant baseline portfolios.

        If t-statistic > 0 (positive) and p-value <= 0.05, it means that the average daily return of baseline
        portfolio is both higher than the cluster one and statistically significant.

        It is expected baseline portfolio to be statistically significant when t-statistic > 0 (positive),
        because the numerator of t-stat is (baseline_mean - cluster_mean).

        :param hyperparameters_and_t_tests: (pd.DataFrame) contains hyperparameters, t-stat, and p-values.
        :param results_baseline_df: (pd.DataFrame) contains statistical metrics of baseline portfolios per
                                    hyperparameter set.
        :return: (pd.DataFrame) statistically significant baseline portfolios.
        """

        tstat_col = RESULTS_COLS[13]  # <str>
        pvalue_col = RESULTS_COLS[14]  # <str>

        condition_tstat = hyperparameters_and_t_tests[tstat_col] > 0  # positive
        condition_pvalue = hyperparameters_and_t_tests[pvalue_col] <= P_VALUE

        baseline_stat_significant_mask = condition_tstat & condition_pvalue

        hyperparameters_col = RESULTS_COLS[1]  # <str>
        hyperparameters = hyperparameters_and_t_tests[hyperparameters_col]

        hyperparameters_stat_significant = hyperparameters[baseline_stat_significant_mask]

        # count how many times baseline portfolios are statistically significant per rule (hyperparameter set):
        count_stat_significant_portfolios = hyperparameters_stat_significant.value_counts(sort=False)

        # drop duplicates coming from comparisons between baseline and clusters for
        # the same set of hyperparameters:
        hyperparameters_stat_significant = hyperparameters_stat_significant.drop_duplicates()

        hyperparameters_stat_significant_mask = results_baseline_df[hyperparameters_col].isin(hyperparameters_stat_significant)

        baseline_stat_significant_portfolios = results_baseline_df[hyperparameters_stat_significant_mask].reset_index(drop=True)

        # check whether hyperparameters ("hp") column and index of value counts are equal:
        hp_1 = baseline_stat_significant_portfolios[hyperparameters_col]
        hp_2 = pd.Series(count_stat_significant_portfolios.index)
        equal_bool = hp_1.equals(hp_2)

        if equal_bool:
            # insert the value counts to baseline dataframe:
            baseline_stat_significant_portfolios.insert(2,
                                                        COUNT_STAT_SIGNIFICANT_PORTFOLIOS,
                                                        count_stat_significant_portfolios.values)
        else:
            print('\n[WARNING] The hyperparameters sets are not equal.')

        save_data_csv(data=baseline_stat_significant_portfolios,
                      csv_leaf=CSV_INSIGHTS_LEAF,
                      csv_name=CSV_BASELINE_STAT_SIGNIFICANT_PORTFOLIOS)

        return baseline_stat_significant_portfolios

    @staticmethod
    def get_clusters_stat_significant_portfolios(results_clusters_df: pd.DataFrame) -> dict:
        """
        Detects the statistically significant cluster portfolios.

        If t-statistic < 0 (negative) and p-value <= 0.05, it means that the average daily return of
        cluster portfolio is both higher than the baseline one and statistically significant.

        It is expected cluster portfolio to be statistically significant when t-statistic < 0 (negative),
        because the numerator of t-stat is (baseline_mean - cluster_mean).

        :param results_clusters_df: (pd.DataFrame) contains statistical metrics of cluster portfolios per
                                    hyperparameter set.
        :return: (dict) statistically significant portfolios per cluster.
        """

        case_col = RESULTS_COLS[0]  # <str>
        unique_clusters = results_clusters_df[case_col].unique()

        clusters_stat_significant_portfolios_dict = dict()

        for cluster_index in unique_clusters:

            cluster = results_clusters_df[results_clusters_df[case_col] == cluster_index]

            tstat_col = RESULTS_COLS[13]  # <str>
            pvalue_col = RESULTS_COLS[14]  # <str>

            condition_tstat = cluster[tstat_col] < 0  # negative
            condition_pvalue = cluster[pvalue_col] <= P_VALUE

            cluster_stat_significant_mask = condition_tstat & condition_pvalue

            cluster_stat_significant_portfolios = cluster[cluster_stat_significant_mask]

            csv_name = cluster_index + CSV_STAT_SIGNIFICANT_PORTFOLIOS

            save_data_csv(data=cluster_stat_significant_portfolios,
                          csv_leaf=CSV_INSIGHTS_LEAF,
                          csv_name=csv_name)

            clusters_stat_significant_portfolios_dict[cluster_index] = cluster_stat_significant_portfolios

        return clusters_stat_significant_portfolios_dict

    @staticmethod
    def examine_average_performance(baseline_ss_port: pd.DataFrame,
                                    cluster_ss_port: dict) -> None:
        """
        Examines the average performance of statistically significant portfolios of baseline and clusters.
        :param baseline_ss_port: (pd.DataFrame) metrics of baseline statistically significant portfolios.
        :param cluster_ss_port: (dict) metrics of cluster statistically significant portfolios.
        :return: None.
        """

        average_performances_dict = dict()

        baseline_count_ss_port = baseline_ss_port[COUNT_STAT_SIGNIFICANT_PORTFOLIOS].sum()
        baseline_average_performance = baseline_ss_port.iloc[:, 3:].mean().values
        baseline_average_performance = np.concatenate([np.array([baseline_count_ss_port]),
                                                       baseline_average_performance])
        average_performances_dict[BASELINE] = baseline_average_performance

        for cluster_index in cluster_ss_port:

            cluster = cluster_ss_port[cluster_index]
            cluster_count_ss_port = cluster.shape[0]
            cluster_average_performance = cluster.iloc[:, 2:-2].mean().values
            cluster_average_performance = np.concatenate([np.array([cluster_count_ss_port]),
                                                          cluster_average_performance])
            average_performances_dict[cluster_index] = cluster_average_performance

        # convert dict to dataframe
        average_performances_df = pd.DataFrame().from_dict(average_performances_dict).T
        average_performances_df.columns = [COUNT_STAT_SIGNIFICANT_PORTFOLIOS] + PORTFOLIO_COLS[1:]
        average_performances_df = average_performances_df.reset_index(names=[RESULTS_COLS[0]])

        save_data_csv(data=average_performances_df,
                      csv_leaf=CSV_INSIGHTS_LEAF,
                      csv_name=CSV_AVERAGE_PERFORMANCES)
