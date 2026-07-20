import os
import queue
import threading
from datetime import datetime

from pyqtgraph.Qt import QtCore

from config import DUMP_DIR


class DumpWorker(QtCore.QThread):
    dump_started       = QtCore.pyqtSignal(str)
    dump_stopped       = QtCore.pyqtSignal(int)
    dump_count_updated = QtCore.pyqtSignal(int)

    def __init__(self, dump_queue: queue.Queue):
        super().__init__()
        self._queue  = dump_queue
        self._active = False
        self._alive  = True
        self._file   = None
        self._count  = 0
        self._lock   = threading.Lock()

    def start_dump(self) -> None:
        os.makedirs(DUMP_DIR, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(DUMP_DIR, f"csi_dump_{ts}.csv")

        with self._lock:
            if self._file and not self._file.closed:
                self._file.close()
            self._file  = open(filepath, 'w', encoding='utf-8', buffering=1)
            self._file.write("packet_idx,epoch_s,sub30_amplitude,rssi_dbm,raw_line\n")
            self._count  = 0
            self._active = True

        self.dump_started.emit(filepath)

    def stop_dump(self) -> None:
        with self._lock:
            self._active = False
            saved        = self._count
            if self._file and not self._file.closed:
                self._file.flush()
                self._file.close()
                self._file = None

        self.dump_stopped.emit(saved)

    def stop_thread(self) -> None:
        if self._active:
            self.stop_dump()
        self._alive = False
        self.wait(2000)

    def run(self) -> None:
        while self._alive:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue

            with self._lock:
                if not self._active or self._file is None or self._file.closed:
                    continue
                pkt_idx, epoch_s, sub30, rssi, raw = item
                self._file.write(
                    f"{pkt_idx},{epoch_s:.6f},{sub30:.4f},{rssi:.0f},{raw}\n"
                )
                self._count += 1
                if self._count % 100 == 0:
                    self.dump_count_updated.emit(self._count)
