class History:
    """Maintain a history of messages and allow the user to page through it."""

    def __init__(self, size: int = 100):
        self.size = size
        self.history: list[str] = []
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
