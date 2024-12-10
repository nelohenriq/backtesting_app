import logging
import logging.config
import yaml
from sqlalchemy import create_engine

class UtilityAgent:
    def __init__(self, config_file='config/config.yaml'):
        """
        Initialize the UtilityAgent.

        :param config_file: Path to the configuration file.
        """
        self.config_file = config_file
        self.config = self.load_config()
        self.setup_logging()
        self.engine = self.setup_database()

    def load_config(self):
        """
        Load configuration from the YAML file.

        :return: Configuration dictionary.
        """
        with open(self.config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def setup_logging(self):
        """
        Set up logging based on the configuration.
        """
        logging.config.dictConfig(self.config['logging'])

    def setup_database(self):
        """
        Set up the database connection.

        :return: SQLAlchemy engine.
        """
        db_url = self.config['database']['url']
        engine = create_engine(db_url)
        return engine

    def get_logger(self, name):
        """
        Get a logger instance.

        :param name: Name of the logger.
        :return: Logger instance.
        """
        return logging.getLogger(name)

    def get_engine(self):
        """
        Get the SQLAlchemy engine.

        :return: SQLAlchemy engine.
        """
        return self.engine

if __name__ == "__main__":
    # Initialize the UtilityAgent
    utility_agent = UtilityAgent()

    # Example usage
    logger = utility_agent.get_logger('example_logger')
    logger.info("UtilityAgent initialized successfully.")

    engine = utility_agent.get_engine()
    logger.info(f"Database engine created: {engine}")