import logging
import os

def setup_logger(log_dir, log_name="training.log"):
    """Set up a logger to log messages to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)

    return logger

# Example usage:
# logger = setup_logger("logs")
# logger.info("This is an info message")
