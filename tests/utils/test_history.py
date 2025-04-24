from utils import History
from utils.history import MAX_HISTORY_SIZE


class TestHistory:
    def test_init(self):
        history = History()
        assert history.size == MAX_HISTORY_SIZE
        assert history.history == []
        assert history.index == 0

    def test_append(self):
        history = History()
        history.append("test")
        assert history.history == ["test"]
        assert history.index == 0

    def test_max_size(self):
        history = History()
        for i in range(MAX_HISTORY_SIZE + 1):
            history.append(f"test {i}")
        assert len(history.history) == MAX_HISTORY_SIZE
        assert history.index == MAX_HISTORY_SIZE - 1

    def test_prev(self):
        history = History()

        # Can't go back if there is no history
        assert history.prev() is None

        history.append("test1")
        history.append("test2")
        assert history.prev() == "test2"
        assert history.prev() == "test1"

        # Can't go back if we're at the first message
        assert history.prev() == "test1"

    def test_next(self):
        history = History()

        # Can't go forward if there is no history
        assert history.next() is None

        history.append("test1")
        history.append("test2")

        # Can't go forward if we're at the last message
        assert history.next() is None

        history.prev()
        assert history.next() == "test2"
