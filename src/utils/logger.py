"""
Logging utilities for Enhanced MARL Two-Tower Recommendation System

- Consistent logger setup across modules
- Supports colored console output and file logging
- Adjustable verbosity (DEBUG/INFO/WARNING/ERROR)
- Optional integration with experiment tracking (WandB, TensorBoard)
"""

import logging
import sys
import os

_LOG_FORMAT = "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default log directory if file logging is enabled
_LOG_DIR = "logs"

def setup_logger(
    name: str = None,
    level: int = logging.INFO,
    log_file: str = None,
    console: bool = True,
    filemode: str = "a",
    propagate: bool = False
) -> logging.Logger:
    """
    Create and configure a logger for the MARL system.

    Args:
        name: Logger name (use __name__ in modules)
        level: Logging level (logging.INFO by default)
        log_file: Optional file path to append logs
        console: Whether to log to console (default True)
        filemode: File open mode ("a"=append, "w"=overwrite)
        propagate: If False, prevents log propagation to root

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    # Remove previous handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console logging
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # File logging
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file) or _LOG_DIR, exist_ok=True)
        fh = logging.FileHandler(log_file, mode=filemode)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

# Shorthand for the main system logger
main_logger = setup_logger("MARLSystem")

def set_global_log_level(level: int):
    """
    Globally set log level for all loggers created so far.

    Args:
        level: logging.INFO, logging.DEBUG, etc.
    """
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(level)

# Utility for logging experiment metrics
def log_metrics(metrics: dict, logger: logging.Logger = None, step: int = None):
    """
    Log experiment metrics to both logger and (optional) external trackers.

    Args:
        metrics: Dictionary of metric names and values
        logger: Logger instance (defaults to main_logger)
        step: Optional training/evaluation step/epoch
    """
    logger = logger or main_logger
    msg = f"Step {step} - " if step is not None else ""
    msg += ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items())
    logger.info(msg)
    # Insert integration with wandb or TensorBoard if needed.

# Example of usage in other modules:
# from utils.logger import setup_logger
# logger = setup_logger(__name__, level=logging.DEBUG)

