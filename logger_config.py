import logging

# Create a logger instance
logger = logging.getLogger("TradingAgentLogger")
logger.setLevel(logging.DEBUG)  # Set the lowest level to capture all logs

# Handler for Log File 1 (General Logs)
file_handler_general = logging.FileHandler("general_logs.log")
file_handler_general.setLevel(logging.INFO)  # Log only INFO and above
file_handler_general.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))

# Handler for Log File 2 (Detailed Logs)
file_handler_detailed = logging.FileHandler("detailed_logs.log")
file_handler_detailed.setLevel(logging.DEBUG)  # Log everything, including DEBUG
file_handler_detailed.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]",
    datefmt="%Y-%m-%d %H:%M:%S"
))

# Add both handlers to the logger
logger.addHandler(file_handler_general)
logger.addHandler(file_handler_detailed)