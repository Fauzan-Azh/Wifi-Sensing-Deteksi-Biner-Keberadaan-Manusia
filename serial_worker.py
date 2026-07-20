import os
import time
import queue
from typing import Optional

import serial
import numpy as np
from scipy.signal import savgol_filter
from pyqtgraph.Qt import QtCore
import tensorflow as tf

from config import (
    VARIANCE_WINDOW, TIME_STEPS, INPUT_DIM,
    MODEL_PATH, DATASET_DIR, DATASET_FILES,
    HAMPEL_WINDOW, HAMPEL_THRESHOLD, SG_WINDOW, SG_POLYORDER
)
from preprocessor import RealtimePreprocessor


class SerialWorker(QtCore.QThread):
    connection_ok   = QtCore.pyqtSignal()
    connection_lost = QtCore.pyqtSignal(str)

    def __init__(self, port: str, baud: int,
                 gui_queue: queue.Queue, dump_queue: queue.Queue):
        super().__init__()
        self._port       = port
        self._baud       = baud
        self._gui_queue  = gui_queue
        self._dump_queue = dump_queue
        self._running    = True
        self._text_buf   = ""
        self._pkt_count  = 0

        self._var_win   = np.zeros(VARIANCE_WINDOW, dtype=np.float64)
        self._var_ptr   = 0
        self._var_count = 0

        self._lstm_buf     = np.zeros((TIME_STEPS, INPUT_DIM), dtype=np.float32)
        self._preprocessor = RealtimePreprocessor()

        self._model                  = self._load_model()
        self._min_val, self._max_val = self._compute_scaling()

    @staticmethod
    def _load_model() -> tf.keras.Model:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
        print(f"[INFO] Memuat model: {MODEL_PATH}")
        return tf.keras.models.load_model(MODEL_PATH)

    def _compute_scaling(self):
        """
        Load parameter scaling global (.npy) yang sudah dihitung saat training.
        Mendukung pembacaan dari root directory maupun folder output_model/
        """
        import os
        import numpy as np

        # Jalur Alternatif 1: Di root directory langsung
        min_path_root = "global_min_csi.npy"
        max_path_root = "global_max_csi.npy"

        # Jalur Alternatif 2: Di dalam folder output_model
        min_path_out = os.path.join("output_model", "global_min_csi.npy")
        max_path_out = os.path.join("output_model", "global_max_csi.npy")

        # Cek Jalur 1 (Root) terlebih dahulu
        if os.path.exists(min_path_root) and os.path.exists(max_path_root):
            mn = np.load(min_path_root).astype(np.float32)
            mx = np.load(max_path_root).astype(np.float32)
            # Proteksi pembagian dengan nol
            mx[mx == mn] += 1e-8
            print(f"[INFO] Scaling parameter sukses dimuat dari root directory.")
            return mn, mx

        # Cek Jalur 2 (output_model/) jika Jalur 1 tidak ada
        elif os.path.exists(min_path_out) and os.path.exists(max_path_out):
            mn = np.load(min_path_out).astype(np.float32)
            mx = np.load(max_path_out).astype(np.float32)
            mx[mx == mn] += 1e-8
            print(f"[INFO] Scaling parameter sukses dimuat dari folder output_model/.")
            return mn, mx

        # Jika dua-duanya tidak ketemu (Skenario Darurat / Fallback ke kode lama lu)
        else:
            print("[WARN] File biner scaling .npy tidak ditemukan. Menghitung otomatis dari dataset lama...")
            all_features = []
            for name in DATASET_FILES:
                path = os.path.join(DATASET_DIR, name)
                if not os.path.exists(path): continue
                file_features = []
                with open(path, 'r', errors='ignore') as f:
                    for line in f:
                        amp = self._extract_amplitude(line)
                        if amp is not None and len(amp) == INPUT_DIM:
                            file_features.append(amp)
                if not file_features: continue
                arr = np.array(file_features, dtype=np.float32)
                arr = self._handle_missing_values(arr)
                arr = self._hampel_filter_centered(arr)
                arr = savgol_filter(arr, SG_WINDOW, SG_POLYORDER, axis=0).astype(np.float32)
                all_features.extend(arr)
            
            if not all_features:
                return (np.zeros(INPUT_DIM, dtype=np.float32), np.ones(INPUT_DIM, dtype=np.float32))
            
            X = np.array(all_features, dtype=np.float32)
            mn = np.min(X, axis=0)
            mx = np.max(X, axis=0)
            mx[mx == mn] += 1e-8
            return mn, mx

    @staticmethod
    def _handle_missing_values(arr: np.ndarray) -> np.ndarray:
        import pandas as pd
        df = pd.DataFrame(arr)
        df = df.replace(0, np.nan)
        df = df.interpolate(method='linear', axis=0).bfill().ffill()
        df = df.fillna(0.0) 
        return df.to_numpy(dtype=np.float32)

    @staticmethod
    def _hampel_filter_centered(arr: np.ndarray) -> np.ndarray:
        k = HAMPEL_WINDOW           
        n = len(arr)
        out = arr.copy()
        for col in range(arr.shape[1]):
            for i in range(k, n - k):
                window = arr[i - k : i + k + 1, col]
                median = np.median(window)
                mad    = np.median(np.abs(window - median))
                threshold = HAMPEL_THRESHOLD * 1.4826 * np.maximum(mad, 1e-6)
                if np.abs(arr[i, col] - median) > threshold:
                    out[i, col] = median
        return out

    @staticmethod
    def _extract_amplitude(line: str) -> Optional[np.ndarray]:
        try:
            s = line.find('[')
            e = line.rfind(']')
            if s == -1 or e <= s + 1:
                return None
            raw = np.fromstring(line[s + 1:e], sep=' ', dtype=np.float32)
            if len(raw) < 62 or (len(raw) & 1):
                return None
            return np.hypot(raw[0::2], raw[1::2])
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _extract_rssi(line: str) -> float:
        try:
            return float(line.split(',')[3])
        except (ValueError, IndexError):
            return 0.0

    def _rolling_variance(self, val: float) -> float:
        self._var_win[self._var_ptr % VARIANCE_WINDOW] = val
        self._var_ptr  += 1
        self._var_count = min(self._var_count + 1, VARIANCE_WINDOW)
        if self._var_count < 2:
            return 0.0
        w = (self._var_win if self._var_count >= VARIANCE_WINDOW
             else self._var_win[:self._var_count])
        return float(np.var(w))

    @staticmethod
    def _push(q: queue.Queue, item) -> None:
        try:
            q.put_nowait(item)
        except queue.Full:
            try:   q.get_nowait()
            except queue.Empty: pass
            try:   q.put_nowait(item)
            except queue.Full:  pass

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

    def run(self) -> None:
        ser: Optional[serial.Serial] = None
        try:
            ser = serial.Serial(self._port, self._baud, timeout=0.001)
            ser.reset_input_buffer()
            print(f"[INFO] Serial terhubung: {self._port} @ {self._baud}")
            self.connection_ok.emit()
        except serial.SerialException as e:
            self.connection_lost.emit(str(e))
            return

        while self._running:
            try:
                waiting = ser.in_waiting
                if not waiting:
                    QtCore.QThread.msleep(1)
                    continue

                self._text_buf += ser.read(waiting).decode('utf-8', errors='ignore')

                if len(self._text_buf) > 65_536:
                    nl = self._text_buf.rfind('\n')
                    self._text_buf = self._text_buf[nl + 1:] if nl != -1 else ""
                    continue

                if '\n' not in self._text_buf:
                    continue

                *lines, self._text_buf = self._text_buf.split('\n')

                for line in lines:
                    if 'CSI_DATA' not in line or '[' not in line:
                        continue

                    amp_raw = self._extract_amplitude(line)
                    if amp_raw is None or len(amp_raw) != INPUT_DIM:
                        continue

                    amp_filtered = self._preprocessor.process(amp_raw)
                    rssi       = self._extract_rssi(line)
                    sub30_raw  = float(amp_raw[30])
                    sub30_filt = float(amp_filtered[30])
                    var_val    = self._rolling_variance(sub30_filt)
                    epoch_s    = time.time()

                    # Proses prediksi langsung berjalan sejak paket pertama diterima
                    if self._pkt_count < TIME_STEPS:
                        self._lstm_buf[self._pkt_count] = amp_filtered
                        pred_label = 0
                    else:
                        self._lstm_buf     = np.roll(self._lstm_buf, -1, axis=0)
                        self._lstm_buf[-1] = amp_filtered
                        scaled     = (self._lstm_buf - self._min_val) / (self._max_val - self._min_val)
                        inp        = np.expand_dims(scaled, axis=0)
                        preds      = self._model(inp, training=False).numpy()
                        pred_label = int(np.argmax(preds[0]))

                    self._pkt_count += 1

                    # Kirim data ke GUI Queue
                    self._push(self._gui_queue, (
                        sub30_raw, sub30_filt, rssi, var_val,
                        pred_label, self._pkt_count, epoch_s,
                        amp_filtered
                    ))
                    
                    # Kirim data ke Dump Queue langsung dari paket pertama tanpa penyaringan waktu
                    self._push(self._dump_queue, (
                        self._pkt_count, epoch_s, sub30_filt, rssi, line.strip()
                        ))

            except serial.SerialException as e:
                self.connection_lost.emit(str(e))
                break
            except Exception as e:
                print(f"[WARN] {e}")
                self._text_buf = ""

        if ser and ser.is_open:
            ser.close()