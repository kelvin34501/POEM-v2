from __future__ import annotations
import numpy as np
import time

from .shared_memory import SharedMemoryManager
from collections import deque


class SyncUnit:

    def __init__(self, shared_memory_list: list[SharedMemoryManager], fps: float = 30.0) -> None:
        self.shared_memory_list = shared_memory_list
        self.num_recv = len(self.shared_memory_list)
        self.fps_target = fps
        self.frame_interval = 1.0 / self.fps_target

        self.data_list = [None] * self.num_recv
        self.timestamp_list = [None] * self.num_recv
        self.current_timestamp = None

        self.last_timestamp = None
        self.last_exec = time.time()
        self.msg_gap_list = deque(maxlen=1000)

    def current_timestamp_valid(self):
        return 0 not in self.timestamp_list

    def execute(self):
        if self.current_timestamp_valid():
            self.last_timestamp = self.current_timestamp
        self.last_exec = time.time()
        for offset, shared_memory in enumerate(self.shared_memory_list):
            msg, ts = shared_memory.execute()
            self.data_list[offset] = msg
            self.timestamp_list[offset] = ts
        self.current_timestamp = np.round(np.mean(self.timestamp_list))
        if self.last_timestamp is not None and self.current_timestamp_valid():
            self.msg_gap_list.append(self.current_timestamp - self.last_timestamp)
        return self.data_list.copy(), self.current_timestamp

    def wait(self):
        # wait up to fps
        while True:
            gap = time.time() - self.last_exec
            if gap >= self.frame_interval:
                break
            time.sleep(0.001)

    def fps(self):
        if len(self.msg_gap_list) == 0:
            return 0.0
        gap = float(np.mean(self.msg_gap_list))
        if np.abs(gap) < 1e-5:
            gap = 1e-5
        return 1000.0 / gap
