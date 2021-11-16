class MockRsync:
    def __init__(self):
        self.params = []

    def __call__(self, *args, **kwargs):
        self.params.append(args[0])


class MockProcess:
    def __init__(self, *args, **kwargs):
        pass

    def wait(self):
        return 0
