import unittest
import pandas as pd
import logging
from fundamental_analysis.fundamental_analysis_agent import AdvancedFundamentalAnalysisAgent

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TestAdvancedFundamentalAnalysisAgent(unittest.TestCase):
    def setUp(self):
        self.db_url = 'sqlite:///test_database.db'
        self.agent = AdvancedFundamentalAnalysisAgent(self.db_url)

    def test_load_data(self):
        # Create a test table with sample data
        test_data = pd.DataFrame({'Asset': ['AAPL', 'GOOGL'], 'Price': [150, 2800]})
        test_data.to_sql('test_table', self.agent.engine, if_exists='replace', index=False)

        # Test the load_data method
        loaded_data = self.agent.load_data('test_table')
        self.assertIsNotNone(loaded_data)
        self.assertEqual(len(loaded_data), 2)
        self.assertIn('Asset', loaded_data.columns)

    def test_fetch_fundamental_data(self):
        asset = 'AAPL'
        fundamental_data = self.agent.fetch_fundamental_data(asset)
        self.assertIsNotNone(fundamental_data)
        self.assertIn('Asset', fundamental_data.columns)
        self.assertIsNotNone(fundamental_data['Asset'])
        self.assertEqual(fundamental_data['Asset'].iloc[0], asset)

    def test_calculate_financial_ratios(self):
        test_data = pd.DataFrame({
            'currentPrice': [150],
            'trailingEps': [10],
            'bookValue': [50],
            'returnOnEquity': [0.15],
            'returnOnAssets': [0.1],
            'totalDebt': [1000],
            'totalStockholderEquity': [2000],
            'totalCurrentAssets': [500],
            'totalCurrentLiabilities': [300],
            'dividendYield': [0.02],
            'earningsGrowth': [0.05]
        })
        
        result = self.agent.calculate_financial_ratios(test_data)
        self.assertIn('P/E Ratio', result.columns)
        self.assertIn('P/B Ratio', result.columns)
        self.assertIn('ROE', result.columns)

    def test_perform_zscore_analysis(self):
        test_data = pd.DataFrame({
            'P/E Ratio': [15, 20, 25],
            'P/B Ratio': [2, 3, 4],
            'ROE': [0.1, 0.15, 0.2],
            'ROA': [0.05, 0.08, 0.1],
            'Debt/Equity': [0.5, 0.7, 0.9],
            'Current Ratio': [1.5, 2, 2.5],
            'Dividend Yield': [0.02, 0.03, 0.04],
            'PEG Ratio': [1, 1.5, 2]
        })
        
        result = self.agent.perform_zscore_analysis(test_data)
        self.assertEqual(len(result.columns), 16)  # Original 8 columns + 8 z-score columns

    def test_perform_fundamental_analysis(self):
        # Create a test table with sample data
        test_data = pd.DataFrame({'Asset': ['AAPL', 'GOOGL'], 'Price': [150, 2800]})
        test_data.to_sql('test_analysis_table', self.agent.engine, if_exists='replace', index=False)

        result = self.agent.perform_fundamental_analysis('test_analysis_table')
        self.assertIsNotNone(result)
        self.assertIn('Asset', result.columns)
        # Add more assertions based on expected columns and data

if __name__ == '__main__':
    unittest.main()