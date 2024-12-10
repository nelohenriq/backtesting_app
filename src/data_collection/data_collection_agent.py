import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import schedule
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataCollectionAgent:
    def __init__(self, assets, db_url):
        """
        Initialize the DataCollectionAgent.

        :param assets: List of assets (e.g., ['AAPL', 'BTC-USD']).
        :param db_url: Database URL (e.g., 'sqlite:///data.db').
        """
        self.assets = assets
        self.db_url = db_url
        self.engine = create_engine(db_url)

    def fetch_data(self, asset, start_date, end_date):
        """
        Fetch historical data for a given asset.

        :param asset: Asset symbol (e.g., 'AAPL').
        :param start_date: Start date for data collection (e.g., '2020-01-01').
        :param end_date: End date for data collection (e.g., '2023-01-01').
        :return: DataFrame containing historical data.
        """
        try:
            data = yf.download(asset, start=start_date, end=end_date)
            data['Asset'] = asset
            return data
        except Exception as e:
            logging.error(f"Error fetching data for {asset}: {e}")
            return pd.DataFrame()

    def preprocess_data(self, data):
        """
        Preprocess the collected data.

        :param data: DataFrame containing raw data.
        :return: Preprocessed DataFrame.
        """
        # Handle missing values (e.g., forward fill)
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

        # Add additional features (e.g., moving averages)
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['MA_200'] = data['Close'].rolling(window=200).mean()

        return data

    def store_data(self, data, table_name):
        """
        Store the preprocessed data in the database.

        :param data: Preprocessed DataFrame.
        :param table_name: Name of the table to store data.
        """
        if not data.empty:
            data.to_sql(table_name, self.engine, if_exists='append', index=True)

    def collect_and_store_data(self, start_date, end_date, table_name):
        """
        Collect data for all assets and store it in the database.

        :param start_date: Start date for data collection.
        :param end_date: End date for data collection.
        :param table_name: Name of the table to store data.
        """
        for asset in self.assets:
            logging.info(f"Fetching data for {asset}...")
            raw_data = self.fetch_data(asset, start_date, end_date)
            preprocessed_data = self.preprocess_data(raw_data)
            self.store_data(preprocessed_data, table_name)
            logging.info(f"Data for {asset} stored in {table_name}.")

    def schedule_data_collection(self, start_date, end_date, table_name, interval='1d'):
        """
        Schedule data collection at specified intervals.

        :param start_date: Start date for data collection.
        :param end_date: End date for data collection.
        :param table_name: Name of the table to store data.
        :param interval: Interval for scheduling (e.g., '1d' for daily).
        """
        if interval == '1d':
            schedule.every().day.at("00:00").do(self.collect_and_store_data, start_date, end_date, table_name)
        elif interval == '1h':
            schedule.every().hour.do(self.collect_and_store_data, start_date, end_date, table_name)
        else:
            raise ValueError("Unsupported interval")

        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    # Define assets and database URL
    assets = ['AAPL', 'MSFT', 'BTC-USD', 'ETH-USD']
    db_url = 'sqlite:///historical_data.db'

    # Initialize the DataCollectionAgent
    agent = DataCollectionAgent(assets, db_url)

    # Collect and store data
    start_date = '2020-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    table_name = 'historical_data'

    # Schedule data collection
    agent.schedule_data_collection(start_date, end_date, table_name, interval='1d')