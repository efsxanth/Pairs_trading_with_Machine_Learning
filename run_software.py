
from modules.m1_data_collection import DataCollection
from modules.m2_data_preprocessing import DataPreprocessing
from modules.m3_clustering import Clustering
from modules.m4_pairs_trading_strategy import PairsTradingStrategy
from modules.m5_extract_insights import ExtractInsights


if __name__ == '__main__':

    """
    =====================
         ENTRY POINT
    =====================
    
    Improving Pairs trading strategy by using DBSCAN clustering
    algorithm coupled with 3 (three) Fama/French factors.
    
    """

    # /-----[ Data Collection ] -----/
    data_collection = DataCollection()
    price_series_stocks = data_collection.price_series_stocks  # stocks of S&P500 index
    return_series_3ff = data_collection.returns_3ff  # returns of Fama/French 3 (three) factors

    # /-----[ Data Preprocessing ] -----/
    data_processing = DataPreprocessing(price_series_stocks=price_series_stocks, return_series_3ff=return_series_3ff)
    price_series_stocks = data_processing.price_series_stocks
    return_series_stocks = data_processing.return_series_stocks
    return_series_3ff = data_processing.return_series_3ff

    # /-----[ Clustering ] -----/
    clustering = Clustering(return_series_stocks=return_series_stocks, return_series_3ff=return_series_3ff)
    clusters = clustering.stock_clusters

    # /-----[ Pairs Trading Strategy ] -----/
    pairs_trading_strategy = PairsTradingStrategy(price_series_stocks=price_series_stocks, clusters=clusters)
    results_baseline_df = pairs_trading_strategy.results_baseline_df
    results_clusters_df = pairs_trading_strategy.results_clusters_df

    # /-----[ Extract Insights ] -----/
    extract_insights = ExtractInsights(results_baseline_df=results_baseline_df,
                                       results_clusters_df=results_clusters_df)
