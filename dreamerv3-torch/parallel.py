
import atexit
import os
import sys
import time
import traceback
import enum
from functools import partial as bind


class Parallel:
    def __init__(self, ctor, strategy):
        self.worker = Worker(bind(self._respond, ctor), strategy, state=True)
        self.callables = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            if name not in self.callables:
                self.callables[name] = self.worker(PMessage.CALLABLE, name)()
            if self.callables[name]:
                return bind(self.worker, PMessage.CALL, name)
            else:
                return self.worker(PMessage.READ, name)()
        except AttributeError:
            raise ValueError(name)

    def __len__(self):
        return self.worker(PMessage.CALL, "__len__")()

    def close(self):
        self.worker.close()

    @staticmethod
    def _respond(ctor, state, message, name, *args, **kwargs):
        state = state or ctor
        if message == PMessage.CALLABLE:
            assert not args and not kwargs, (args, kwargs)
            result = callable(getattr(state, name))
        elif message == PMessage.CALL:
            result = getattr(state, name)(*args, **kwargs)
        elif message == PMessage.READ:
            assert not args and not kwargs, (args, kwargs)
            result = getattr(state, name)
        return state, result

