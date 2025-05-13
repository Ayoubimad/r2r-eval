"""
Logging utility module for the RAG evaluation framework.

This module provides consistent logging functionality across the application,
with support for colored output and various log levels.
"""

import logging
import os
from typing import Dict, Optional, Literal
import colorlog


class LoggingConfig:
    """Configuration for logging"""

    def __init__(
        self,
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
        include_timestamps: bool = True,
        log_to_file: bool = False,
        log_file_path: Optional[str] = None,
        log_format: Optional[str] = None,
    ):
        """
        Initialize logging configuration

        Args:
            level: Log level to use
            include_timestamps: Whether to include timestamps in log messages
            log_to_file: Whether to log to a file
            log_file_path: Path to log file (required if log_to_file is True)
            log_format: Custom log format (if None, uses default format)
        """
        self.level = level
        self.include_timestamps = include_timestamps
        self.log_to_file = log_to_file
        self.log_file_path = log_file_path
        self.log_format = log_format

        if self.log_to_file and not self.log_file_path:
            raise ValueError("log_file_path must be specified when log_to_file is True")


def get_log_level(level_name: str) -> int:
    """
    Convert a log level name to a logging level constant

    Args:
        level_name: Name of the log level

    Returns:
        Logging level constant

    Raises:
        ValueError: If level_name is invalid
    """
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    if level_name not in levels:
        valid_levels = ", ".join(levels.keys())
        raise ValueError(
            f"Invalid log level: {level_name}. Valid levels are: {valid_levels}"
        )

    return levels[level_name]


def get_color_scheme() -> Dict[str, str]:
    """
    Get color scheme for different log levels

    Returns:
        Dictionary mapping log level names to color specifications
    """
    return {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    }


def create_console_formatter(
    include_timestamps: bool = True,
) -> colorlog.ColoredFormatter:
    """
    Create a formatter for console output

    Args:
        include_timestamps: Whether to include timestamps in log messages

    Returns:
        Configured formatter
    """
    colors = get_color_scheme()

    if include_timestamps:
        format_str = "%(asctime)s - %(log_color)s%(levelname)s%(reset)s - %(message)s"
    else:
        format_str = "%(log_color)s%(levelname)s%(reset)s - %(message)s"

    return colorlog.ColoredFormatter(
        format_str,
        log_colors=colors,
        reset=True,
        style="%",
    )


def create_file_formatter(include_timestamps: bool = True) -> logging.Formatter:
    """
    Create a formatter for file output

    Args:
        include_timestamps: Whether to include timestamps in log messages

    Returns:
        Configured formatter
    """
    if include_timestamps:
        format_str = "%(asctime)s - %(levelname)s - %(message)s"
    else:
        format_str = "%(levelname)s - %(message)s"

    return logging.Formatter(format_str)


def create_console_handler(level: int) -> logging.Handler:
    """
    Create a handler for console output

    Args:
        level: Logging level constant

    Returns:
        Configured handler
    """
    handler = logging.StreamHandler()
    handler.setLevel(level)
    return handler


def create_file_handler(log_file_path: str, level: int) -> logging.Handler:
    """
    Create a handler for file output

    Args:
        log_file_path: Path to log file
        level: Logging level constant

    Returns:
        Configured handler
    """
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    handler = logging.FileHandler(log_file_path)
    handler.setLevel(level)
    return handler


def setup_logger(name: str, config: Optional[LoggingConfig] = None) -> logging.Logger:
    """
    Configure and return a logger with the given name and configuration

    Args:
        name: Logger name
        config: Logging configuration

    Returns:
        Configured logger
    """
    if config is None:
        config = LoggingConfig()

    logger = logging.getLogger(name)

    level = get_log_level(config.level)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = create_console_handler(level)
        console_formatter = create_console_formatter(config.include_timestamps)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        if config.log_to_file and config.log_file_path:
            file_handler = create_file_handler(config.log_file_path, level)
            file_formatter = create_file_formatter(config.include_timestamps)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger


create_logger = setup_logger
