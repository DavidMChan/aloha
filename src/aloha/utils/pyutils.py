import functools
import hashlib
import inspect
import logging
import os
import time
from contextlib import AbstractContextManager
from typing import Any, Generic, List, Optional, Type, TypeVar

import torch


def compute_md5_hash_from_bytes(input_bytes: bytes) -> str:
    """Compute the MD5 hash of a byte array."""
    return str(hashlib.md5(input_bytes).hexdigest())


class chdir(AbstractContextManager):  # noqa: N801
    """Non thread-safe context manager to change the current working directory."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._old_cwd: List[str] = []

    def __enter__(self) -> None:
        self._old_cwd.append(os.getcwd())
        os.chdir(self.path)

    def __exit__(self, *excinfo) -> None:  # type: ignore
        os.chdir(self._old_cwd.pop())


def retry(func):
    """Retry a function if it throws an exception."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        error = None
        for backoff in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Backoff and try again
                time.sleep(backoff)
                logging.warning(
                    f"Encountered error {e} while running {func.__name__}. Retrying {func.__name__} after {backoff} seconds."
                )
                error = e
                continue

        # If we get here, we've exhausted all retries
        logging.error(f"Encountered error {error} while running {func.__name__}. Retries exhaused. Aborting.")
        raise error

    return wrapper


T = TypeVar("T")
S = TypeVar("S")


class _SingletonWrapper(Generic[T]):
    def __init__(self, cls: Type[T]):
        self.__wrapped__ = cls
        self._instance: Optional[T] = None
        functools.update_wrapper(self, cls)

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Returns a single instance of decorated class"""
        if self._instance is None:
            self._instance = self.__wrapped__(*args, **kwargs)
        return self._instance


def singleton(cls: Type[S]) -> _SingletonWrapper[S]:
    return _SingletonWrapper(cls)


def select_device(device: Optional[str] = None) -> Optional[str]:
    if device is None:
        # Check if CUDA is available
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    if isinstance(device, str):
        if device == "cuda":
            # If the user specifies only CUDA, use the GPU with the most free memory
            free_memory = [
                torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                for i in range(torch.cuda.device_count())
            ]
            device = "cuda:" + str(torch.argmax(torch.tensor(free_memory)).item())
        elif device.startswith("cuda:"):
            # If the user specifies a specific GPU, make sure it exists
            if int(device[5:]) >= torch.cuda.device_count():
                raise ValueError(f"Invalid device: {device}")

        return device


def partialclass(cls, *args, **kwds):
    # Remove any kwargs from the **kwds that are not in the __init__ signature
    sig = inspect.signature(cls.__init__)
    for key in list(kwds.keys()):
        if key not in sig.parameters:
            del kwds[key]

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls
