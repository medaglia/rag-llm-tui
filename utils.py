import datetime
import logging
from zoneinfo import ZoneInfo

utc = ZoneInfo("UTC")
LOG_PATH = "logs"


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def get_logger(name, level=logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    now = datetime.datetime.now(utc)
    file_path = f"{LOG_PATH}/" + now.strftime("%Y-%m-%d_%H") + ".log"
    log_format = "%(asctime)s - %(levelname)s -  %(name)s - %(message)s"

    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    # logger.propagate = False  # Prevent bubbling up to root logger
    return logger
