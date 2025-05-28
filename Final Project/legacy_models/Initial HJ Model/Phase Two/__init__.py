import logging.config
import logging

logging.basicConfig(format='%(levelname)s:%(message)s')

# Create the Logger
loggers = logging.getLogger(__name__)
loggers.setLevel(logging.DEBUG)

# Create the Handler for logging data to a file
logger_handler = logging.FileHandler(
    filename='/Users/bramschork/Documents/connect/Phase Two/connect.log')
logger_handler.setLevel(logging.DEBUG)

# Create a Formatter for formatting the log messages
logger_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

# Add the Formatter to the Handler
logger_handler.setFormatter(logger_formatter)

# Add the Handler to the Logger
loggers.addHandler(logger_handler)
loggers.info('Completed configuring logger()!')
