class LRScheduler:
    def __init__(self, lr: float):
        self.lr = lr

    def step(self):
        raise NotImplementedError


class SimpleLRScheduler(LRScheduler):
    """
    Scheduler that does nothing.
    """

    def __init__(self, lr: float):
        super().__init__(lr)

    def step(self):
        pass
