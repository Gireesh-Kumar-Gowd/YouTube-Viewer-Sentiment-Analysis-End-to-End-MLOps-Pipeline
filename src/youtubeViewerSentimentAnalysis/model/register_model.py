# register model

import sys
import json
import mlflow
import os

from src.youtubeViewerSentimentAnalysis.logger import logging
from src.youtubeViewerSentimentAnalysis.exception import CustomException

# Set up MLflow tracking URI
mlflow.set_tracking_uri("http://ec2-100-28-125-79.compute-1.amazonaws.com:5000/")


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError as e:
        logging.error('File not found: %s', file_path)
        raise CustomException(e, sys)
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise CustomException(e, sys)

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)

        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise CustomException(e, sys)

def main():
    try:
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "yt_chrome_plugin_model"
        # model_name = "my_model"
        register_model(model_name, model_info)
    except CustomException as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()