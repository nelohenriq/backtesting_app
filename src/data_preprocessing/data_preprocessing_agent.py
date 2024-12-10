import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sqlalchemy import create_engine
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedDataPreprocessingAgent:
    def __init__(self, db_url):
        """
        Initialize the AdvancedDataPreprocessingAgent.

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

    def handle_missing_values(self, data):
        """
        Handle missing values in the data.

        :param data: DataFrame containing raw data.
        :return: DataFrame with missing values handled.
        """
        # Forward fill and backward fill
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

        # Drop remaining rows with missing values
        data.dropna(inplace=True)

        return data

    def normalize_data(self, data, method='standard'):
        """
        Normalize the data.

        :param data: DataFrame containing data to be normalized.
        :param method: Normalization method ('standard' or 'minmax').
        :return: DataFrame with normalized data.
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported normalization method")

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

        return data

    def feature_engineering(self, data):
        """
        Perform feature engineering on the data.

        :param data: DataFrame containing raw data.
        :return: DataFrame with engineered features.
        """
        # Calculate moving averages
        if 'Close' in data.columns:
            data['MA_50'] = data['Close'].rolling(window=50).mean()
            data['MA_200'] = data['Close'].rolling(window=200).mean()

        # Calculate technical indicators
        if 'Close' in data.columns and 'Volume' in data.columns:
            data['RSI'] = self.calculate_rsi(data['Close'])
            data['MACD'] = self.calculate_macd(data['Close'])
            data['OBV'] = self.calculate_obv(data['Close'], data['Volume'])

        return data

    def calculate_rsi(self, prices, period=14):
        """
        Calculate Relative Strength Index (RSI).

        :param prices: Series containing price data.
        :param period: Period for RSI calculation.
        :return: Series containing RSI values.
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, short_period=12, long_period=26, signal_period=9):
        """
        Calculate Moving Average Convergence Divergence (MACD).

        :param prices: Series containing price data.
        :param short_period: Short period for MACD calculation.
        :param long_period: Long period for MACD calculation.
        :param signal_period: Signal period for MACD calculation.
        :return: Series containing MACD values.
        """
        short_ema = prices.ewm(span=short_period, adjust=False).mean()
        long_ema = prices.ewm(span=long_period, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd - signal

    def calculate_obv(self, prices, volumes):
        """
        Calculate On-Balance Volume (OBV).

        :param prices: Series containing price data.
        :param volumes: Series containing volume data.
        :return: Series containing OBV values.
        """
        obv = np.where(prices > prices.shift(1), volumes, np.where(prices < prices.shift(1), -volumes, 0)).cumsum()
        return obv

    def preprocess_data(self, table_name, normalization_method='standard'):
        """
        Preprocess the data.

        :param table_name: Name of the table to preprocess data from.
        :param normalization_method: Normalization method ('standard' or 'minmax').
        :return: Preprocessed DataFrame.
        """
        logging.info(f"Loading data from {table_name}...")
        data = self.load_data(table_name)

        logging.info("Handling missing values...")
        data = self.handle_missing_values(data)

        logging.info("Performing feature engineering...")
        data = self.feature_engineering(data)

        logging.info(f"Normalizing data using {normalization_method} method...")
        data = self.normalize_data(data, method=normalization_method)

        return data

    def store_preprocessed_data(self, data, table_name):
        """
        Store the preprocessed data in the database.

        :param data: Preprocessed DataFrame.
        :param table_name: Name of the table to store data.
        """
        data.to_sql(table_name, self.engine, if_exists='replace', index=True)
        logging.info(f"Preprocessed data stored in {table_name}.")

if __name__ == "__main__":
    # Define database URL
    db_url = 'sqlite:///advanced_historical_data.db'

    # Initialize the AdvancedDataPreprocessingAgent
    agent = AdvancedDataPreprocessingAgent(db_url)

    # Preprocess data
    table_name = 'historical_data'
    preprocessed_data = agent.preprocess_data(table_name, normalization_method='standard')

    # Store preprocessed data
    preprocessed_table_name = 'preprocessed_data'
    agent.store_preprocessed_data(preprocessed_data, preprocessed_table_name)