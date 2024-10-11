"""
Shared Memory Manager.

Author: Hongjie Fang
Modified: Xinyu Zhan
"""

import numpy as np
from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister
import time


class SharedMemoryManager:
    """
    Shared Memory Manager.
    """

    def __init__(self, name: str, type: int = 0, shape=(1,), dtype=np.float32, timeout=30):
        """
        Initialization.

        Parameters
        ----------
        - name: the name of the shared memory;
        - type: integer in [0, 1];
            * 0: sender;
            * 1: receiver.
        - shape: optional, default: (1,), the array shape.
        - dtype: optional, default: np.float32, the element type of the array.
        """
        super(SharedMemoryManager, self).__init__()
        self.name = name
        self.type = type
        self.shape = shape
        self.dtype = np.dtype(dtype)

        timestamp_dtype = np.int64
        timestamp_size = np.dtype(timestamp_dtype).itemsize

        total_size = self.dtype.itemsize * np.prod(self.shape) + timestamp_size

        if self.type not in [0, 1]:
            raise AttributeError("Invalid type in shared memory manager.")
        if self.type == 0:
            self.shared_memory = shared_memory.SharedMemory(name=self.name, create=True, size=total_size)
            self.buffer = self.shared_memory.buf
        else:
            _start = time.time()
            _shared_memory = None
            _last_exception = None
            while time.time() - _start <= timeout:
                try:
                    _shared_memory = shared_memory.SharedMemory(name=self.name)
                    break
                except FileNotFoundError as e:
                    _last_exception = e
                    time.sleep(0.1)  # wait 100 ms
            if _shared_memory is None:
                raise _last_exception from None

            self.shared_memory = _shared_memory
            self.buffer = self.shared_memory.buf

        self.data_array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.buffer[:-timestamp_size])
        self.timestamp = np.ndarray((1,), dtype=timestamp_dtype, buffer=self.buffer[-timestamp_size:])

    def execute(self, arr=None, timestamp=None):
        """
        Execute the function.

        Paramters
        ---------
        - arr: np.array object, only used in sender, the array.
        """
        if self.type == 0:
            if arr is None:
                raise AttributeError("Array should be specified in shared memory sender.")
            try:
                self.data_array[:] = arr[:]
                if timestamp is None:
                    timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds
                self.timestamp[0] = timestamp
            except Exception as e:
                raise RuntimeError(f"Error writing to shared memory: {e}")
        else:
            data_copy = np.copy(self.data_array)
            timestamp_copy = self.timestamp[0]
            return data_copy, timestamp_copy

    def close(self):
        self.shared_memory.close()
        if self.type == 0:
            self.shared_memory.unlink()

    def unregister(self):
        unregister(self.shared_memory._name, 'shared_memory')
