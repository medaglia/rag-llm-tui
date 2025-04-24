from utils.singleton import Singleton


class TestClass(metaclass=Singleton):
    pass


def test_singleton():
    assert TestClass() is TestClass()
