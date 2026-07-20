import numpy as np
from scipy.signal import savgol_filter

from config import INPUT_DIM, HAMPEL_WINDOW, HAMPEL_THRESHOLD, SG_WINDOW, SG_POLYORDER


class RealtimePreprocessor:
    """
    Preprocessing real-time (causal) per paket CSI.

    Urutan filter identik dengan preprocessing.ipynb:
      1. Hampel Filter (window_size=7 → k=3)
      2. Savitzky-Golay (window_length=11, polyorder=2)

    Catatan penting:
    - Training menggunakan Hampel CENTERED (melihat masa depan).
    - Real-time menggunakan Hampel CAUSAL (hanya masa lalu) — tidak bisa look-ahead.
    - Parameter k=3 dan n_sigmas=3 disamakan dengan training.
    - SG window_length=11 disamakan dengan training.
    """

    def __init__(self):
        # Buffer harus cukup untuk window terbesar
        buf_size       = max(HAMPEL_WINDOW * 2 + 1, SG_WINDOW)
        self._raw_buf  = np.zeros((buf_size, INPUT_DIM), dtype=np.float32)
        self._ham_buf  = np.zeros((buf_size, INPUT_DIM), dtype=np.float32)
        self._fill     = 0

    def process(self, amp: np.ndarray) -> np.ndarray:
        """
        Proses satu vektor amplitudo, kembalikan versi yang sudah difilter.
        Jika belum cukup data untuk window minimum, kembalikan nilai asli.
        """
        self._raw_buf  = np.roll(self._raw_buf, -1, axis=0)
        self._raw_buf[-1] = amp
        self._fill     = min(self._fill + 1, len(self._raw_buf))

        ham_out        = self._apply_hampel()

        self._ham_buf  = np.roll(self._ham_buf, -1, axis=0)
        self._ham_buf[-1] = ham_out

        return self._apply_savgol()

    def _apply_hampel(self) -> np.ndarray:
        """
        Hampel filter causal.

        Training: hampel_filter_mad(window_size=7, n_sigmas=3)
          k = window_size // 2 = 3

        Real-time causal menggunakan k=HAMPEL_WINDOW=3 → window = 2*3+1 = 7
        hanya dari masa lalu (tidak bisa centered seperti di training).
        """
        min_pts = HAMPEL_WINDOW * 2 + 1    # = 7, sama dengan window_size training
        if self._fill < min_pts:
            return self._raw_buf[-1].copy()

        window   = self._raw_buf[-min_pts:]
        med      = np.median(window, axis=0)
        mad      = np.median(np.abs(window - med), axis=0)
        sigma    = 1.4826 * np.maximum(mad, 1e-6)

        current  = self._raw_buf[-1].copy()
        outliers = np.abs(current - med) > HAMPEL_THRESHOLD * sigma
        current[outliers] = med[outliers]
        return current

    def _apply_savgol(self) -> np.ndarray:
        """
        Savitzky-Golay filter causal.

        Training: apply_savitzky_golay(window_length=11, polyorder=2)
        Disamakan: SG_WINDOW=11, SG_POLYORDER=2.
        """
        if self._fill < SG_WINDOW:
            return self._ham_buf[-1].copy()

        window   = self._ham_buf[-SG_WINDOW:]
        smoothed = savgol_filter(window, SG_WINDOW, SG_POLYORDER, axis=0)
        return smoothed[-1].astype(np.float32)
