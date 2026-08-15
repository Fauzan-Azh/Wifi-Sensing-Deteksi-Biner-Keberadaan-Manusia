import numpy as np


class CircularBuffer:
    __slots__ = ('_buf', '_size', '_ptr', '_count', '_render_buf')

    def __init__(self, size: int, dtype=np.float32):
        self._buf        = np.zeros(size, dtype=dtype)
        self._size       = size
        self._ptr        = 0
        self._count      = 0
        self._render_buf = np.empty(size, dtype=dtype)

    def append(self, value: float) -> None:
        self._buf[self._ptr] = value
        self._ptr = (self._ptr + 1) % self._size
        if self._count < self._size:
            self._count += 1

    def get_ordered(self) -> np.ndarray:
        if self._count < self._size:
            return self._buf[:self._count].copy()
        np.copyto(self._render_buf, np.roll(self._buf, -self._ptr))
        return self._render_buf
