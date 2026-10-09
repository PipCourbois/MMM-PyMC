"""
Centralized logging configuration for PyMC Marketing.

This module provides a simple on/off logging system controlled by the config file.
All loggers can be enabled or disabled with a single switch in mmm_config_simu.yml.

Usage:
    from pymc_marketing.logging_config import setup_logging, get_logger

    # Initialize logging (call once at the start)
    setup_logging('mmm_config_simu.yml')

    # Get a logger in any module
    logger = get_logger(__name__)
    logger.info("This is a log message")
"""

import logging
from pathlib import Path
from typing import Optional

import yaml


# Global flag to track if logging is enabled
_LOGGING_ENABLED = True
_LOGGING_LEVEL = logging.INFO
_LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(config_path: Optional[str] = None) -> None:
    """
    Setup logging configuration from YAML config file.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file. If None, uses default settings.

    Examples
    --------
    >>> setup_logging('mmm_config_simu.yml')
    >>> logger = get_logger(__name__)
    >>> logger.info("Logging is configured!")
    """
    global _LOGGING_ENABLED, _LOGGING_LEVEL, _LOGGING_FORMAT, _DATE_FORMAT

    # Default values
    enabled = True
    level = "INFO"
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Load configuration if path provided
    if config_path is not None:
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)

                    if 'logging' in config:
                        logging_config = config['logging']
                        enabled = logging_config.get('enabled', True)
                        level = logging_config.get('level', 'INFO')
                        log_format = logging_config.get('format', log_format)
                        date_format = logging_config.get('date_format', date_format)
        except Exception as e:
            # If config loading fails, use defaults and warn
            print(f"Warning: Could not load logging config from {config_path}: {e}")
            print("Using default logging configuration.")

    # Store configuration globally
    _LOGGING_ENABLED = enabled
    _LOGGING_LEVEL = getattr(logging, level.upper(), logging.INFO)
    _LOGGING_FORMAT = log_format
    _DATE_FORMAT = date_format

    # Configure root logger
    if enabled:
        logging.basicConfig(
            level=_LOGGING_LEVEL,
            format=_LOGGING_FORMAT,
            datefmt=_DATE_FORMAT,
            force=True  # Override any existing configuration
        )
    else:
        # Disable all logging
        logging.basicConfig(
            level=logging.CRITICAL + 1,  # Higher than any standard level
            format=_LOGGING_FORMAT,
            force=True
        )
        # Add NullHandler to root logger to suppress all output
        logging.getLogger().addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.

    The logger will respect the global enabled/disabled state set by setup_logging().

    Parameters
    ----------
    name : str
        Name for the logger, typically __name__ of the calling module.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing data...")
    >>> logger.debug("Detailed debug information")
    """
    logger = logging.getLogger(name)

    if not _LOGGING_ENABLED:
        # Ensure logger is effectively disabled
        logger.setLevel(logging.CRITICAL + 1)
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
    else:
        logger.setLevel(_LOGGING_LEVEL)
        logger.propagate = True

    return logger


def is_logging_enabled() -> bool:
    """
    Check if logging is currently enabled.

    Returns
    -------
    bool
        True if logging is enabled, False otherwise.
    """
    return _LOGGING_ENABLED
