class ValidationSaver:

    def __init__(self, loss):
        """loss: boolean indicating if we monitor loss (True) or metric (False)"""
        self.loss = loss
        self.best = 10e9 if loss else 0
        self.fn = lambda x, y: x < y if loss else x > y

    def __call__(self, val_perf, trainer, step):
        if self.fn(val_perf, self.best):
            # => improvement
            self.best = val_perf
            trainer.save_checkpoint(step=step, perf=val_perf, is_best=True)
