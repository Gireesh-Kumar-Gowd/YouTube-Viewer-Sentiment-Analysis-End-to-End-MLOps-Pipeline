import sys
import os
import pickle
import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.lightgbm
import mlflow.sklearn

os.environ.setdefault('PYTHONUTF8', '1')
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature

from src.youtubeViewerSentimentAnalysis.logger import logging
from src.youtubeViewerSentimentAnalysis.exception import CustomException


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logging.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logging.error('Error loading data from %s: %s', file_path, e)
        raise CustomException(e, sys)


def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logging.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logging.error('Error loading model from %s: %s', model_path, e)
        raise CustomException(e, sys)


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logging.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logging.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise CustomException(e, sys)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logging.error('Error loading parameters from %s: %s', params_path, e)
        raise CustomException(e, sys)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and log classification metrics and confusion matrix."""
    try:
        # Predict and calculate classification metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        logging.debug('Model evaluation completed')

        return report, cm
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise CustomException(e, sys)


def log_confusion_matrix(cm, dataset_name):
    """Log confusion matrix as an artifact."""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Save confusion matrix plot as a file and log it to MLflow
        cm_file_path = f'confusion_matrix_{dataset_name}.png'
        plt.savefig(cm_file_path)
        mlflow.log_artifact(cm_file_path)
        plt.close()
        logging.debug('Confusion matrix logged for %s', dataset_name)
    except Exception as e:
        logging.error('Error occurred while logging the confusion matrix: %s', e)
        raise CustomException(e, sys)


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        # Create a dictionary with the info you want to save
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        # Save the dictionary as a JSON file
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise CustomException(e, sys)


def save_local_model(model, file_path: str) -> None:
    """Persist the trained model locally so it can be uploaded as an MLflow artifact."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.debug('Local model saved to %s', file_path)
    except Exception as e:
        logging.error('Error saving local model to %s: %s', file_path, e)
        raise CustomException(e, sys)


def main():
    mlflow.set_tracking_uri("http://ec2-100-28-125-79.compute-1.amazonaws.com:5000/")

    mlflow.set_experiment('dvc-pipeline-runs')

    with mlflow.start_run() as run:
        try:
            # Load parameters from YAML file
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Load model and vectorizer
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Load test data for signature inference
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

            # Prepare test data
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Create a DataFrame for signature inference (using first few rows as an example)
            input_example = pd.DataFrame(X_test_tfidf.toarray()[:5], columns=vectorizer.get_feature_names_out())  # <--- Added for signature

            # Infer the signature
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))  # <--- Added for signature

            # Log model with signature. LightGBM models are not sklearn-native, so prefer the
            # LightGBM flavor and fall back to sklearn logging with explicit trusted types.
            try:
                mlflow.lightgbm.log_model(
                    model,
                    "lgbm_model",
                    signature=signature,
                    input_example=input_example,
                )
            except Exception as exc:
                logging.warning(
                    'LightGBM model logging via mlflow.lightgbm failed; retrying with sklearn flavor and trusted LightGBM types: %s',
                    exc,
                )
                mlflow.sklearn.log_model(
                    model,
                    "lgbm_model",
                    signature=signature,
                    input_example=input_example,
                    skops_trusted_types=[
                        'collections.OrderedDict',
                        'lightgbm.basic.Booster',
                        'lightgbm.sklearn.LGBMClassifier',
                    ],
                )

            # Save a concrete pickle artifact so the model appears in the run's artifacts folder.
            model_file = os.path.join(root_dir, 'lgbm_model.pkl')
            save_local_model(model, model_file)
            mlflow.log_artifact(model_file)

            # Save model info
            model_path = "lgbm_model.pkl"
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')

            # Log the vectorizer as an artifact
            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Evaluate model and get metrics
            report, cm = evaluate_model(model, X_test_tfidf, y_test)

            # Log classification report metrics for the test data
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            # Log confusion matrix
            log_confusion_matrix(cm, "Test Data")

            # Add important tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

        except CustomException as e:
            logging.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")

if __name__ == '__main__':
    main()   