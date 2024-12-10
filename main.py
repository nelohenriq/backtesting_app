import logging
from src.data_collection.data_collection_agent import DataCollectionAgent
from src.data_preprocessing.data_preprocessing_agent import AdvancedDataPreprocessingAgent
from src.fundamental_analysis.fundamental_analysis_agent import AdvancedFundamentalAnalysisAgent
from src.on_chain_analysis.on_chain_analysis_agent import AdvancedOnChainAnalysisAgent
from src.ai_assistance.ai_assistant_agent import AdvancedAIAssistanceAgent
from src.backtesting.backtesting_platform import BacktestingPlatform
from src.utils.utility_agent import UtilityAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Initialize the UtilityAgent
    utility_agent = UtilityAgent()
    logger = utility_agent.get_logger('main')
    engine = utility_agent.get_engine()

    # Define assets and database URL
    assets = ['AAPL', 'MSFT', 'BTC-USD', 'ETH-USD']
    db_url = utility_agent.config['database']['url']

    # Initialize and run DataCollectionAgent
    logger.info("Initializing DataCollectionAgent...")
    data_collection_agent = DataCollectionAgent(assets, db_url)
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    table_name = 'historical_data'
    data_collection_agent.collect_and_store_data(start_date, end_date, table_name)

    # Initialize and run AdvancedDataPreprocessingAgent
    logger.info("Initializing AdvancedDataPreprocessingAgent...")
    data_preprocessing_agent = AdvancedDataPreprocessingAgent(db_url)
    preprocessed_data = data_preprocessing_agent.preprocess_data(table_name, normalization_method='standard')
    preprocessed_table_name = 'preprocessed_data'
    data_preprocessing_agent.store_preprocessed_data(preprocessed_data, preprocessed_table_name)

    # Initialize and run AdvancedFundamentalAnalysisAgent
    logger.info("Initializing AdvancedFundamentalAnalysisAgent...")
    fundamental_analysis_agent = AdvancedFundamentalAnalysisAgent(db_url)
    analysis_results = fundamental_analysis_agent.perform_fundamental_analysis(preprocessed_table_name)
    analysis_table_name = 'fundamental_analysis_results'
    fundamental_analysis_agent.store_analysis_results(analysis_results, analysis_table_name)

    # Initialize and run AdvancedOnChainAnalysisAgent
    logger.info("Initializing AdvancedOnChainAnalysisAgent...")
    on_chain_analysis_agent = AdvancedOnChainAnalysisAgent(db_url)
    on_chain_analysis_results = on_chain_analysis_agent.perform_on_chain_analysis(preprocessed_table_name)
    on_chain_analysis_table_name = 'on_chain_analysis_results'
    on_chain_analysis_agent.store_analysis_results(on_chain_analysis_results, on_chain_analysis_table_name)

    # Initialize and run AdvancedAIAssistanceAgent
    logger.info("Initializing AdvancedAIAssistanceAgent...")
    ai_assistance_agent = AdvancedAIAssistanceAgent(db_url)
    ai_analysis_results = ai_assistance_agent.perform_ai_analysis(preprocessed_table_name)
    ai_analysis_table_name = 'ai_analysis_results'
    ai_assistance_agent.store_analysis_results(ai_analysis_results, ai_analysis_table_name)

    # Initialize and run BacktestingPlatform
    logger.info("Initializing BacktestingPlatform...")
    backtesting_platform = BacktestingPlatform(db_url)
    strategy = 'RF_Signal'
    backtesting_results, performance_metrics = backtesting_platform.perform_backtesting(ai_analysis_table_name, strategy)
    backtesting_table_name = 'backtesting_results'
    backtesting_platform.store_backtesting_results(backtesting_results, backtesting_table_name)

    # Print performance metrics
    logger.info("Performance Metrics:")
    for metric, value in performance_metrics.items():
        logger.info(f"{metric}: {value}")

if __name__ == "__main__":
    main()