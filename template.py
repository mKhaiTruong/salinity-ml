import os, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

proj_name = "salinity"
list_of_files = [
    ".github/workflows/.gitkeep",        # CI/CD
    ".github/workflows/main.yaml",
    
    # ------------- NOTEBOOKS ----------------------------
    f"notebooks/01_eda.ipynb",
    f"notebooks/02_training.ipynb",
    f"notebooks/03_export.ipynb",
    f"notebooks/04_evaluation.ipynb",
    
    # ------------- MICROSERVICES ----------------------------
    f"packages/core/pyproject.toml",  
    f"packages/core/src/core/__init__.py",
    f"packages/core/src/core/logging.py",
    f"packages/core/src/core/exception.py",
    f"packages/core/src/core/constants/__init__.py",
    
    f"packages/training/pyproject.toml",
    f"packages/training/src/training/__init__.py",
    
    f"packages/export/pyproject.toml",
    f"packages/export/src/export/__init__.py",
    
    f"packages/evaluation/pyproject.toml",
    f"packages/evaluation/src/evaluation/__init__.py",

    "config/config.yaml",   # paths & settings
    "params.yaml",          # hyperparameters
    ".env",                 # secrets
    "requirements.in",
    "pyproject.toml",               # to package code
    
    # IGNORES
    ".gitignore",
    ".dockerignore",
]

for file in list_of_files:
    file_path = Path(file)
    file_dir ,file_name = os.path.split(file_path)
    
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for file: {file_name}")
    
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, "w") as f:
            pass
            logging.info(f"Creating empty file: {file_path}")
    else:
        logging.info(f"{file_path} already exists")