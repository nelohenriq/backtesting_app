import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import zscore
from sqlalchemy import create_engine
import logging
from fundamental_analysis.fundamental_analysis_agent import AdvancedFundamentalAnalysisAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedFundamentalAnalysisAgent:
    def __init__(self, db_url):
        """
        Initialize the AdvancedFundamentalAnalysisAgent.

        :param db_url: Database URL (e.g., 'sqlite:///data.db').
        """
        self.db_url = db_url
        self.engine = create_engine(db_url)

    def load_data(self, table_name):
        """
        Load data from the database.

        :param table_name: Name of the table to load data from.
        :return: DataFrame containing the data.
        """
        query = f"SELECT * FROM {table_name}"
        data = pd.read_sql(query, self.engine)
        return data

    def fetch_fundamental_data(self, asset):
        """
        Fetch fundamental data for a given asset.

        :param asset: Asset symbol (e.g., 'AAPL').
        :return: DataFrame containing fundamental data.
        """
        try:
            ticker = yf.Ticker(asset)
            info = ticker.info
            df = pd.DataFrame([info])
            df['Asset'] = asset
            return df

        except Exception as e:
            logging.error(f"Error fetching fundamental data for {asset}: {e}")

            return pd.DataFrame()
    def calculate_financial_ratios(self, data):
        """
        Calculate key financial ratios.

        :param data: DataFrame containing fundamental data.
        :return: DataFrame with calculated financial ratios.
        """
        try:
            required_columns = [
                'currentPrice', 'trailingEps', 'bookValue', 'returnOnEquity',
                'returnOnAssets', 'totalDebt', 'totalStockholderEquity',
                'totalCurrentAssets', 'totalCurrentLiabilities',
                'dividendYield', 'earningsGrowth'
            ]
            
            # Check if all required columns exist
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logging.error(f"Missing required columns: {missing_columns}")
                return data

            data['P/E Ratio'] = data['currentPrice'] / data['trailingEps']
            data['P/B Ratio'] = data['currentPrice'] / data['bookValue']
            data['ROE'] = data['returnOnEquity']
            data['ROA'] = data['returnOnAssets']
            data['Debt/Equity'] = data['totalDebt'] / data['totalStockholderEquity']
            data['Current Ratio'] = data['totalCurrentAssets'] / data['totalCurrentLiabilities']
            data['Dividend Yield'] = data['dividendYield']
            data['PEG Ratio'] = data['P/E Ratio'] / data['earningsGrowth']

            return data

        except Exception as e:
            logging.error(f"Error calculating financial ratios: {e}")
            return data
    def perform_zscore_analysis(self, data):
        """
        Perform Z-score analysis to identify outliers.

        :param data: DataFrame containing financial ratios.
        :return: DataFrame with Z-scores.
        """
        z_scores = data[['P/E Ratio', 'P/B Ratio', 'ROE', 'ROA', 'Debt/Equity', 'Current Ratio', 'Dividend Yield', 'PEG Ratio']].apply(zscore)
        data = pd.concat([data, z_scores], axis=1)

        return data

    def perform_fundamental_analysis(self, table_name):
        """
        Perform fundamental analysis on the data.

        :param table_name: Name of the table to perform analysis on.
        :return: DataFrame with fundamental analysis results.
        """
        logging.info(f"Loading data from {table_name}...")
        data = self.load_data(table_name)

        if data is None:
            logging.warning(f"No data loaded from {table_name}. Returning empty DataFrame.")
            return pd.DataFrame()

        if data.empty:
            logging.warning(f"Data loaded from {table_name} is empty. Returning empty DataFrame.")
            return pd.DataFrame()

        logging.info("Fetching fundamental data...")
        if 'Asset' in data.columns:
            asset_column = data['Asset']
            if asset_column is not None and not asset_column.empty:
                for asset in asset_column.dropna().unique():
                    fundamental_data = self.fetch_fundamental_data(asset)
                    if not fundamental_data.empty:
                        data = data.merge(fundamental_data, on='Asset', how='left')
            else:
                logging.warning("'Asset' column is None or empty. Skipping fundamental data fetching.")
        else:
            logging.warning("No 'Asset' column found in the data. Skipping fundamental data fetching.")

        logging.info("Calculating financial ratios...")
        data = self.calculate_financial_ratios(data)

        logging.info("Performing Z-score analysis...")
        data = self.perform_zscore_analysis(data)

        return data
        
    def store_analysis_results(self, data, table_name):
        """
        Store the analysis results in the database.

        :param data: DataFrame containing analysis results.
        :param table_name: Name of the table to store data.
        """
        data.to_sql(table_name, self.engine, if_exists='replace', index=True)
        logging.info(f"Analysis results stored in {table_name}.")

if __name__ == "__main__":
    # Define database URL
    db_url = 'sqlite:///advanced_historical_data.db'

    # Initialize the AdvancedFundamentalAnalysisAgent
    agent = AdvancedFundamentalAnalysisAgent(db_url)

    # Perform fundamental analysis
    table_name = 'preprocessed_data'
    analysis_results = agent.perform_fundamental_analysis(table_name)

    # Store analysis results
    analysis_table_name = 'fundamental_analysis_results'
    agent.store_analysis_results(analysis_results, analysis_table_name)