import os, sys, base64, json, yaml, joblib
from pathlib import Path
import pandas as pd
from box import ConfigBox
from box.exceptions import BoxValueError
from typing import Any
from ensure import ensure_annotations
from core.logging import logger
from core.exception import CustomException

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise CustomException("yaml file is empty", sys)
    except Exception as e:
        raise CustomException(e, sys)

@ensure_annotations
def save_json(content: dict, path: Path):
    try:
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(content, f, indent=4)
        
        logger.info(f"json file saved at: {path}")
    except Exception as e:
        raise CustomException(e, sys)
    
@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    try:
        with open(path) as f:
            content = json.load(f)
            logger.info(f"json file: {path} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise CustomException("json file is empty", sys)
    except Exception as e:
        raise CustomException(e, sys)

@ensure_annotations
def load_csv_dir(directory: Path) -> pd.DataFrame:
    csv_files = list(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {directory}")
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    logger.info(f"Loaded {len(csv_files)} CSV(s) from {directory} — {len(df)} rows")
    return df
    
@ensure_annotations
def save_bin(data: Any, path: Path):
    try:
        joblib.dump(data, path)
    except Exception as e:
        raise CustomException(e, sys)

@ensure_annotations
def load_bin(path: Path) -> Any:
    try:
        data = joblib.load(path)
        logger.info(f"binary load from {path}")
        return data
    except Exception as e:
        raise CustomException(e, sys)

@ensure_annotations
def get_size(path: Path) -> str:
    try:
        size_in_kb = round(os.path.getsize(path) / 1024)
        return f"~ {size_in_kb} KB"
    except Exception as e:
        raise CustomException(e, sys)

def decode_image(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()

def encode_image_to_base64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')