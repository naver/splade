import numpy as np


class EarlyStopping:

    def __init__(self, patience, mode):
        """mode: early stopping on loss or metrics ?
        """
        self.patience = patience
        self.counter = 0
        self.best = np.Inf if mode == "loss" else 0
        self.fn = lambda x, y: x < y if mode == "loss" else lambda a, b: a > b
        self.stop = False
        print("-- initialize early stopping with {}, patience={}".format(mode, patience))

    def __call__(self, val_perf, trainer, step):
        if self.fn(val_perf, self.best):
            # => improvement
            self.best = val_perf
            self.counter = 0
            trainer.save_checkpoint(step=step, perf=val_perf, is_best=True)
        else:
            # => no improvement
            self.counter += 1
            if self.counter > self.patience:
                self.stop = True
