"""Logger setup utility for the tensor_manifold_gvf project.

Provides a standardized logging configuration across all modules.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "tensor_manifold_gvf",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up and return a configured logger.

    Args:
        name: Logger name.
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        log_file: Optional path to a log file. If provided, logs will
                  be written to both console and file.
        format_string: Optional custom format string. If None, uses a
                       default format with timestamp, level, and message.

    Returns:
        Configured logging.Logger instance.
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
