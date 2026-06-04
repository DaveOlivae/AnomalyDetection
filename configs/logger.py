"""
This module provides a function to set up a logger for the project.
"""

import logging
from pathlib import Path


def setup_logger(log_path: Path) -> logging.Logger:
    """
    Creates and configures a logger instance.
    """

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)

    # evita handlers duplicados
    # se o logger já tiver handlers, retorna ele sem adicionar novos handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)

    # esse handler escreve os logs em um arquivo
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    # esse handler escreve os logs no console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # adiciona os handlers ao logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
