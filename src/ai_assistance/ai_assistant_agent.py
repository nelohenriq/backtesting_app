import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sqlalchemy import create_engine
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedAIAssistanceAgent:
    def __init__(self, db_url):
        """
        Initialize the AdvancedAIAssistanceAgent.

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

    def preprocess_data(self, data):
        """
        Preprocess the data for AI models.

        :param data: DataFrame containing raw data.
        :return: Preprocessed DataFrame.
        """
        # Handle missing values
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

        # Normalize data
        scaler = StandardScaler()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

        return data

    def train_random_forest(self, X, y):
        """
        Train a Random Forest model.

        :param X: Features.
        :param y: Target variable.
        :return: Trained Random Forest model.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logging.info(f"Random Forest MSE: {mse}")
        return model

    def train_lstm(self, X, y):
        """
        Train an LSTM model.

        :param X: Features.
        :param y: Target variable.
        :return: Trained LSTM model.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=1, epochs=10)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logging.info(f"LSTM MSE: {mse}")
        return model
    def generate_trading_signals(self, model, X):
        """
        Generate trading signals using the trained model.

        :param model: Trained model.
        :param X: Features.
        :return: Trading signals.
        """
        predictions = model.predict(X)
        signals = np.where(predictions > 0, 1, -1)
        return signals

    def perform_ai_analysis(self, table_name):
        """
        Perform AI analysis on the data.

        :param table_name: Name of the table to perform analysis on.
        :return: DataFrame with AI analysis results.
        """
        logging.info(f"Loading data from {table_name}...")
        data = self.load_data(table_name)

        logging.info("Preprocessing data...")
        data = self.preprocess_data(data)

        # Prepare features and target variable
        X = data.drop(columns=['Close'])
        y = data['Close']

        logging.info("Training Random Forest model...")
        rf_model = self.train_random_forest(X, y)

        logging.info("Training LSTM model...")
        lstm_model = self.train_lstm(X, y)

        logging.info("Generating trading signals...")
        rf_signals = self.generate_trading_signals(rf_model, X)
        lstm_signals = self.generate_trading_signals(lstm_model, X)

        data['RF_Signal'] = rf_signals
        data['LSTM_Signal'] = lstm_signals

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

    # Initialize the AdvancedAIAssistanceAgent
    agent = AdvancedAIAssistanceAgent(db_url)

    # Perform AI analysis
    table_name = 'preprocessed_data'
    analysis_results = agent.perform_ai_analysis(table_name)

    # Store analysis results
    analysis_table_name = 'ai_analysis_results'
    agent.store_analysis_results(analysis_results, analysis_table_name)