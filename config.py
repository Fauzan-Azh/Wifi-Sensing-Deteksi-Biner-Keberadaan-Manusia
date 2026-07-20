COM_PORT           = 'COM5'
BAUD_RATE          = 115200

WINDOW_SIZE        = 100
VARIANCE_WINDOW    = 15
VARIANCE_THRESHOLD = 30.0

GUI_FPS            = 30
GUI_INTERVAL_MS    = int(1000 / GUI_FPS)

GUI_QUEUE_SIZE     = 500
DUMP_QUEUE_SIZE    = 1000

TIME_STEPS         = 30
INPUT_DIM          = 64

MODEL_PATH         = "output_model/lstm_best_model.keras"
DUMP_DIR           = "dump_sessions"
DATASET_DIR        = "dataset"

# File yang digunakan untuk menghitung parameter scaling.
# HARUS IDENTIK dengan SCENARIOS di preprocessing.ipynb — tidak boleh ada tambahan.
# ruangan_terisi.csv sengaja tidak dimasukkan karena tidak ada di training.
DATASET_FILES      = []

GUARD_SUBCARRIERS  = [0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63]
VALID_SUBCARRIERS  = [i for i in range(64) if i not in GUARD_SUBCARRIERS]

# Parameter Hampel — harus cocok dengan preprocessing.ipynb:
#   hampel_filter_mad(window_size=7, n_sigmas=3)
#   window_size=7 → k = window_size // 2 = 3
HAMPEL_WINDOW      = 3      # setengah window (k), total window = 2*k+1 = 7
HAMPEL_THRESHOLD   = 3.0   # n_sigmas

# Parameter Savitzky-Golay — harus cocok dengan preprocessing.ipynb:
#   apply_savitzky_golay(window_length=11, polyorder=2)
SG_WINDOW          = 11
SG_POLYORDER       = 2
