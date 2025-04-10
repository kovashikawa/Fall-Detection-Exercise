import logging
import os
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """Setup a logger with both file and console handlers."""
    # Create logger
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = logging.getLogger(f"{name}_{timestamp}")
    logger.setLevel(level)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Create and add file handler if log_file is provided
    if log_file:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Add timestamp to log filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/{log_file}_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger
