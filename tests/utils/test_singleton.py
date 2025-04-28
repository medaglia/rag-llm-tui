from utils.singleton import Singleton


class TestClass(metaclass=Singleton):
    pass


class TestSingleton:
    def test_singleton(self):
        assert TestClass() is TestClass()

    def test_clear(self):
        test_class = TestClass()
        TestClass.clear()
        assert TestClass() is not test_class
