import sys
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml

from src.youtubeViewerSentimentAnalysis.logger import logging
from src.youtubeViewerSentimentAnalysis.exception import CustomException


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError as e:
        logging.error('File not found: %s', params_path)
        raise CustomException(e, sys)
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise CustomException(e, sys)
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise CustomException(e, sys)

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logging.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise CustomException(e, sys)
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise CustomException(e, sys)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, duplicates, and empty strings."""
    try:
        # Removing missing values
        df.dropna(inplace=True)
        # Removing duplicates
        df.drop_duplicates(inplace=True)
        # Removing rows with empty strings
        df = df[df['clean_comment'].str.strip() != '']

        logging.debug('Data preprocessing completed: Missing values, duplicates, and empty strings removed.')
        return df
    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise CustomException(e, sys)
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise CustomException(e, sys)

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')

        # Create the data/raw directory if it does not exist
        os.makedirs(raw_data_path, exist_ok=True)

        # Save the train and test data
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logging.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise CustomException(e, sys)

def main():
    try:
        # Load parameters from the params.yaml in the root directory
        params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../params.yaml'))
        test_size = params['data_ingestion']['test_size']

        # Load data from the specified URL
        df = load_data(data_url='https://raw.githubusercontent.com/Gireesh-Kumar-Gowd/YouTube-Viewer-Sentiment-Analysis-End-to-End-MLOps-Pipeline/refs/heads/main/dataset/comment_data.csv')

        # Preprocess the data
        final_df = preprocess_data(df)

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        # Save the split datasets and create the raw folder if it doesn't exist
        save_data(train_data, test_data, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../data'))

    except CustomException as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()