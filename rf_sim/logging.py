import logging

def setup_logging(debug_mode=False):
    """Configures the root logger for the entire application."""
    log_level = logging.DEBUG if debug_mode else logging.INFO

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler() # Prints to terminal
            # logging.FileHandler("simulation.log") # Uncomment to also save to a file
        ]
    )