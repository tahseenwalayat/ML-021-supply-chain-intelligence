import os
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
