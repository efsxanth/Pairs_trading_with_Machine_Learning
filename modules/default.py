
# ---------------------------------------------------------------------------------------------------------------------
#                                              Global
# ---------------------------------------------------------------------------------------------------------------------
SPLIT_DATE = '2017-01-01'  # for training and test sets.
P_VALUE = 0.05  # the level of significance is taken at 5%.

# ---------------------------------------------------------------------------------------------------------------------
#                                              Data Collection
# ---------------------------------------------------------------------------------------------------------------------

# /---------[ Price series of S&P 500 stocks ]---------/

# URL of Wikipedia about the symbols of stocks of S&P500 index:
URL_SP500 = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
URL_SP500_SYMBOL_COL = 'Symbol'

# Yahoo Finance:
START_DATE = '2007-01-01'
END_DATE = '2020-12-31'
ADJ_CLOSE_COL = 'Adj Close'  # Keep only the Adjusted closing prices:

#
# /---------[ Return series of Fama/French 3 (three) factors  ]---------/

# Fama/French 3 Factors [Daily] (.zip extension)
URL_3FF = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'

# Name of .csv file included in the .zip file above:
CSV_NAME_FF = 'F-F_Research_Data_Factors_daily.CSV'
COL_FOR_3FF = ['Date', 'Rm_Rf', 'SMB', 'HML']


# ---------------------------------------------------------------------------------------------------------------------
#                                              Data Preprocessing
# ---------------------------------------------------------------------------------------------------------------------
THRESHOLD_PCT_MISSING_DATA = 5  # indicates 5%.


# ---------------------------------------------------------------------------------------------------------------------
#                                              Clustering
# ---------------------------------------------------------------------------------------------------------------------
COEF_COLS = ['symbol', COL_FOR_3FF[1] + '_coef', COL_FOR_3FF[2] + '_coef', COL_FOR_3FF[3] + '_coef']
CLUSTER_LABEL = 'cluster'
CLUSTERING_COLS = [COEF_COLS[0], CLUSTER_LABEL]
SEARCH_EPS = True
DBSCAN_SELECT_EPS = 0.33  # via elbow method.
DBSCAN_SELECT_MIN_SAMPLES = 5  # 2 * dimensions - 1 (Sander et al, 1998), where dimensions = 3.


# ---------------------------------------------------------------------------------------------------------------------
#                                              Pairs Trading
# ---------------------------------------------------------------------------------------------------------------------
P_VALUE_COINTEGRATION = 0.05
COINT_COLS = ['selected_stock', 'leg_stock', 'pvalue']
PAIR_LABEL = 'pairs'
BASELINE = 'baseline'
NUM_TOP_PAIRS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
WINDOW = [50, 150, 200]  # values used for rolling calculations.
THRESHOLD_ENTRY = 2  # entry to market
THRESHOLD_EXIT = [0, 0.1, 0.2, 0.3, 0.4, 0.5]  # exit market.
HYPERPARAMETERS = [(top, window, THRESHOLD_ENTRY, THRESHOLD_EXIT[i], -THRESHOLD_ENTRY, -THRESHOLD_EXIT[i])
                   for top in NUM_TOP_PAIRS for window in WINDOW for i in range(len(THRESHOLD_EXIT))]
THRESHOLDS_COLS = ['threshold_a', 'threshold_b', 'threshold_c', 'threshold_d']
MARKET_MODE = ['short_zscore', 'long_zscore']
EXIT_MODE = 'off_market'
POSITIONS_COL = 'positions'
PORTFOLIO_COLS = ['pair', 'num_trading_days', 'pct_trading_days', 'num_positive_returns',
                  'pct_positive_returns', 'mean', 'std', 'min', 'median', 'max',
                  'skew', 'kurtosis']
RESULTS_COLS = ['case', 'hyperparameters'] + PORTFOLIO_COLS[1:] + ['t_statistic', 'p_value']
STAT_SIGNIFICANT = ['baseline_stat_significant', 'cluster_stat_significant']
CSV_RESULTS_LEAF = 'results'
CSV_BASELINE_NAME = 'results_baseline.csv'
CSV_CLUSTERS_NAME = 'results_clusters.csv'


# ---------------------------------------------------------------------------------------------------------------------
#                                              Extract Insights
# ---------------------------------------------------------------------------------------------------------------------
COUNT_STAT_SIGNIFICANT_PORTFOLIOS = 'count_stat_significant_portfolios'
CSV_INSIGHTS_LEAF = 'insights'
CSV_STAT_SIGNIFICANT_PORTFOLIOS = '_stat_significant_portfolios.csv'
CSV_BASELINE_STAT_SIGNIFICANT_PORTFOLIOS = BASELINE + CSV_STAT_SIGNIFICANT_PORTFOLIOS
CSV_AVERAGE_PERFORMANCES = 'average_portfolio_performances.csv'


# ---------------------------------------------------------------------------------------------------------------------
#                                              Plotting
# ---------------------------------------------------------------------------------------------------------------------
PLOT_BBOX_INCHES = 'tight'
PLOTS_LEAF = 'plots'
PLOT_XLABEL_STOCKS = 'Stocks'
PLOT_YLABEL_DAY = 'Day (date)'
PLOT_TITLE_PATTERNS_MISSING_DATA = 'Missing data patterns (white color)'
PLOT_THRESHOLD_PCT_LABEL = f'{THRESHOLD_PCT_MISSING_DATA}% threshold'
PLOT_PCT_MISSING_DATA = 'Percentage (%)'
PLOT_PCT_TITLE_MISSING_DATA = '% missing data'
PLOT_MISSING_DATA_SVG = 'missing_data.svg'
PLOT_EPS_CURVE_SVG = 'eps_curve.svg'
PLOTS_PAIRS_LEAF = 'pairs'  # for z-scores
PLOT_EPS_LABEL = 'eps'
PLOT_OUTLIERS_LABEL = 'Outlier percentage (%)'
PLOT_CLUSTERS = True
PLOT_COLORS_CLUSTERS = colors = {
    -1: 'k',  # black
    0: 'tab:blue',
    1: 'tab:orange',
    2: 'tab:green',
    3: 'tab:red',
    4: 'tab:purple',
    5: 'tab:brown',
    6: 'tab:pink',
    7: 'tab:gray',
    8: 'tab:olive',
    9: 'tab:cyan'
}
# z-scores
PLOT_ZSCORES_BOOL = False
PLOT_ZSCORES_MARKERS = ['^', 'v', 'P', 'X']
PLOT_ZSCORES_MARKERS_SIZE = 10
PLOT_ZSCORES_COLOURS = ['g', 'r', 'g', 'r', 'k']
PLOT_ZSCORES_LEGENGS = ['z-score series', 'Short z-score', 'Exit Short', 'Long z-score', 'Exit Long']
PLOT_ZSCORES_XLABEL = 'time'
PLOT_ZSCORES_YLABEL = 'z-score'
PLOT_ZSCORES_TITLE = 'Entry and Exit positions of ({}, {}) pair'
PLOT_ZSCORES_PATH = '{}_pair_{}.png'

