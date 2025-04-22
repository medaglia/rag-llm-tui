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


class History:
    """Maintain a history of messages and allow the user to page through it."""

    def __init__(self, size: int = 100):
        self.size = size
        self.history = []
        self.index = 0

    def append(self, message: str):
        """Add a message to the history."""
        self.history.append(message)
        if len(self.history) > self.size:
            self.history.pop(0)

        self.index = len(self.history) - 1

    def prev(self) -> str | None:
        """
        Get the message from history at the index and traverse back if possible.
        If there is no history, return None.
        """
        if len(self.history) == 0:
            return None

        res = self.history[self.index]
        self.index -= 1 if self.index > 0 else 0
        return res

    def next(self) -> str | None:
        """
        Traverse forward in the history and return the message.
        Return None if we're at the most-recent message or there is no history.
        """
        if len(self.history) == 0:
            return None

        # If we're at the end of the history, return None
        if self.index == len(self.history) - 1:
            return None

        self.index += 1
        return self.history[self.index]
