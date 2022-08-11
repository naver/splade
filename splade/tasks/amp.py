import torch

# inspired from Colbert repo: https://github.com/stanford-futuredata/ColBERT

PyTorch_over_1_6 = float((torch.__version__.split('.')[1])) >= 6 and float((torch.__version__.split('.')[0])) >= 1


# replace this with  contextlib.nullcontext if python >3.7
# see https://stackoverflow.com/a/45187287
class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass


class MixedPrecisionManager:
    def __init__(self, activated):
        assert (not activated) or PyTorch_over_1_6, "Cannot use AMP for PyTorch version < 1.6"

        print("Using FP16:", activated)
        self.activated = activated
        if self.activated:
            self.scaler = torch.cuda.amp.GradScaler()

    def context(self):
        return torch.cuda.amp.autocast() if self.activated else NullContextManager()

    def backward(self, loss):
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer):
        if self.activated:
            self.scaler.unscale_(optimizer)
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()
