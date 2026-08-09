import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project_name = "youtube_viewer_sentiment_analysis"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/data/__init__.py",
    f"src/{project_name}/data/data_ingestion.py",
    f"src/{project_name}/data/data_preprocessing.py",
    f"src/{project_name}/model/__init__.py",
    f"src/{project_name}/model/model_building.py",
    f"src/{project_name}/model/model_evaluation.py",
    f"src/{project_name}/model/register_model.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")