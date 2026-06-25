"""
mlflow configuration settings
"""

import logging
import mlflow
from configs import paths


logger = logging.getLogger(__name__)


def setup_mlflow(experiment_name: str):
    """
    Helper function to setup mlflow for the experiments
    """

    mlflow.set_tracking_uri(f"sqlite:///{paths.ROOT_DIR / 'mlflow.db'}")
    mlflow.set_experiment(experiment_name)

    logger.info("MLflow experiment: %s", experiment_name)
    logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
