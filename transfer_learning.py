import os
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from model_terdahulu import build_previous_model

# ========================
# 1. KONFIGURASI GLOBAL
# ========================
BASE_DIR = "dataset_update"
TRAIN_DIR = os.path.join(BASE_DIR, "training")
TEST_DIR = os.path.join(BASE_DIR, "testing")

CLASS_MAP = {"kosong": 0, "terisi": 1}

INPUT_DIM = 64                 
TRANSIENT_CUTOFF_SEC = 10.0    

WINDOW_SIZE = 30               
WINDOW_STEP = 15

MODEL_OUTPUT_DIR = "output_model"
REPORT_OUTPUT_DIR = "output_report"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

SCALER_MIN_PATH = os.path.join(MODEL_OUTPUT_DIR, "global_min_csi.npy")
SCALER_MAX_PATH = os.path.join(MODEL_OUTPUT_DIR, "global_max_csi.npy")

RANDOM_STATE = 42
EPOCHS = 15                    # Disesuaikan agar CV 5x tidak terlalu lama
BATCH_SIZE = 16

# ========================
# 2. LOGGING SISTEM
# ========================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Pipeline_CV")

# ==================================
# 3. FUNGSI PARSING BARIS MENTAH 
# ==================================
def _parse_raw_line(line: str):
    line = line.strip()
    if not line or "packet_idx" in line or '[' not in line or ']' not in line:
        return None
    try:
        start_bracket = line.find('[')
        end_bracket = line.find(']')
        header_part = line[:start_bracket].rstrip(',')
        array_part = line[start_bracket+1:end_bracket].strip()
        header_fields = header_part.split(',')
        if len(header_fields) < 4:
            return None
            
        epoch_s = float(header_fields[1])
        rssi = float(header_fields[3])
        
        raw_values = np.array([float(x) for x in array_part.split() if x.strip() != ""], dtype=np.float32)
        if len(raw_values) < 62 or len(raw_values) % 2 != 0:
            return None
            
        csi_vector = np.hypot(raw_values[0::2], raw_values[1::2]).astype(np.float32)
        if len(csi_vector) != INPUT_DIM:
            return None
            
        return epoch_s, rssi, csi_vector
    except Exception:
        return None

# =========================
# 4. LOAD SATU FILE CSV
# =========================
def load_and_preprocess_csv(file_path: str, label: int):
    epoch_list, rssi_list, csi_list = [], [], []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            parsed = _parse_raw_line(raw_line)
            if parsed is not None:
                epoch_s, rssi, csi_vector = parsed
                epoch_list.append(epoch_s)
                rssi_list.append(rssi)
                csi_list.append(csi_vector)

    if len(epoch_list) == 0:
        return None

    epoch_arr = np.array(epoch_list, dtype=np.float64)
    rssi_arr = np.array(rssi_list, dtype=np.float32)
    csi_arr = np.stack(csi_list, axis=0)

    t0 = epoch_arr[0]
    valid_mask = (epoch_arr - t0) >= TRANSIENT_CUTOFF_SEC
    
    if np.sum(valid_mask) < WINDOW_SIZE:
        return None

    return csi_arr[valid_mask], rssi_arr[valid_mask], np.full((np.sum(valid_mask),), fill_value=label, dtype=np.int32)

# =======================
# 5. PENGUMPUL DATASET 
# =======================
def build_dataset(split_dir: str, csi_min: np.ndarray, csi_max: np.ndarray):
    X_all, y_all, rssi_all = [], [], []
    for class_name, class_label in CLASS_MAP.items():
        class_dir = os.path.join(split_dir, class_name)
        csv_files = sorted(glob.glob(os.path.join(class_dir, "*.csv")))
        if len(csv_files) == 0:
            continue
        for csv_path in csv_files:
            res = load_and_preprocess_csv(csv_path, class_label)
            if res is None:
                continue
            csi_raw, rssi_raw, y_raw = res

            denom = np.where((csi_max - csi_min) == 0, 1.0, csi_max - csi_min)
            csi_scaled = (csi_raw - csi_min) / denom
            csi_scaled = np.clip(csi_scaled, 0.0, 1.0).astype(np.float32)

            windows, rssi_windows, y_windows = [], [], []
            total_len = csi_scaled.shape[0]
            start = 0
            while start + WINDOW_SIZE <= total_len:
                windows.append(csi_scaled[start:start + WINDOW_SIZE, :])
                rssi_windows.append(np.mean(rssi_raw[start:start + WINDOW_SIZE]))
                y_windows.append(y_raw[start + WINDOW_SIZE - 1])
                start += WINDOW_STEP

            if len(windows) > 0:
                X_all.append(np.stack(windows, axis=0).astype(np.float32))
                y_all.append(np.array(y_windows, dtype=np.int32))
                rssi_all.append(np.array(rssi_windows, dtype=np.float32))

    if len(X_all) == 0:
        return (np.empty((0, WINDOW_SIZE, INPUT_DIM), dtype=np.float32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32))

    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0), np.concatenate(rssi_all, axis=0)

# ======================================
# 6. RANDOM UNDERSAMPLING LEVEL WINDOW 
# ======================================
def undersample_to_balance(X: np.ndarray, y: np.ndarray, rssi: np.ndarray, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx_0, idx_1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n_min = min(len(idx_0), len(idx_1))
    
    idx_balanced = np.concatenate([
        rng.choice(idx_0, size=n_min, replace=False),
        rng.choice(idx_1, size=n_min, replace=False)
    ])
    idx_balanced = rng.permutation(idx_balanced)
    return X[idx_balanced], y[idx_balanced], rssi[idx_balanced]

# ==============================================================
# 7A. ARSITEKTUR JARINGAN LSTM (PUNYA USER / UTAMA)
# ==============================================================
def build_lstm_model(input_shape):
    return Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(2, activation="softmax"),
    ])

# ==============================================================
# 8. UTAMA PIPELINE (5-FOLD COMPREHENSIVE CROSS-VALIDATION)
# ==============================================================
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def evaluate_and_log(y_true, y_pred, mask, model_name, feature_name, condition_name):
    y_t = y_true[mask]
    y_p = y_pred[mask]
    if len(y_t) == 0: return {m: 0.0 for m in ["Acc", "Prec", "Rec", "F1"]}
    
    acc = accuracy_score(y_t, y_p)
    prec = precision_score(y_t, y_p, zero_division=0)
    rec = recall_score(y_t, y_p, zero_division=0)
    f1 = f1_score(y_t, y_p, zero_division=0)
    
    return {"Acc": acc, "Prec": prec, "Rec": rec, "F1": f1}

def main():
    logger.info("=" * 70)
    logger.info("MEMULAI PELATIHAN: 4 MODEL x 2 FITUR (LOS & NLOS SEPARATED)")
    logger.info("=" * 70)

    # A. Hitung Global Scaler Min-Max
    csi_raw_list = []
    for split_dir in [TRAIN_DIR, TEST_DIR]:
        for class_name, class_label in CLASS_MAP.items():
            class_dir = os.path.join(split_dir, class_name)
            for csv_path in glob.glob(os.path.join(class_dir, "*.csv")):
                with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                    for raw_line in f:
                        parsed = _parse_raw_line(raw_line)
                        if parsed: csi_raw_list.append(parsed[2])

    if not csi_raw_list:
        logger.error("Dataset kosong!")
        return

    csi_raw_all = np.stack(csi_raw_list, axis=0)
    csi_min, csi_max = np.min(csi_raw_all, axis=0), np.max(csi_raw_all, axis=0)
    csi_max[csi_max == csi_min] += 1e-8
    
    np.save(SCALER_MIN_PATH, csi_min)
    np.save(SCALER_MAX_PATH, csi_max)

    # Modifikasi build_dataset internal untuk melacak file string LOS/NLOS
    def build_dataset_with_meta(split_dir):
        X_list, y_list, rssi_list, cond_list = [], [], [], []
        for class_name, class_label in CLASS_MAP.items():
            class_dir = os.path.join(split_dir, class_name)
            csv_files = sorted(glob.glob(os.path.join(class_dir, "*.csv")))
            for csv_path in csv_files:
                res = load_and_preprocess_csv(csv_path, class_label)
                if res is None: continue
                csi_raw, rssi_raw, y_raw = res
                
                # Deteksi kondisi fisis dari nama file (sesuai struktur folder Fauzan)
                basename = os.path.basename(csv_path).lower()
                c_label = "NLOS" if "hambatan" in basename else "LOS"

                denom = np.where((csi_max - csi_min) == 0, 1.0, csi_max - csi_min)
                csi_scaled = (csi_raw - csi_min) / denom
                csi_scaled = np.clip(csi_scaled, 0.0, 1.0).astype(np.float32)

                windows, rssi_windows, y_windows = [], [], []
                total_len = csi_scaled.shape[0]
                start = 0
                while start + WINDOW_SIZE <= total_len:
                    windows.append(csi_scaled[start:start + WINDOW_SIZE, :])
                    rssi_windows.append(rssi_raw[start:start + WINDOW_SIZE]) # Simpan run utuh 30 timesteps
                    y_windows.append(y_raw[start + WINDOW_SIZE - 1])
                    start += WINDOW_STEP

                if len(windows) > 0:
                    X_list.append(np.stack(windows, axis=0))
                    y_list.append(np.array(y_windows))
                    rssi_list.append(np.stack(rssi_windows, axis=0))
                    cond_list.extend([c_label] * len(windows))
                    
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0), np.concatenate(rssi_list, axis=0), np.array(cond_list)

    X_tr, y_tr, r_tr, c_tr = build_dataset_with_meta(TRAIN_DIR)
    X_te, y_te, r_te, c_te = build_dataset_with_meta(TEST_DIR)

    X_total = np.concatenate([X_tr, X_te], axis=0)
    y_total = np.concatenate([y_tr, y_te], axis=0)
    rssi_total = np.concatenate([r_tr, r_te], axis=0) # Shape: (N, 30)
    cond_total = np.concatenate([c_tr, c_te], axis=0)

    # Balancing Data
    rng = np.random.default_rng(RANDOM_STATE)
    idx_0, idx_1 = np.where(y_total == 0)[0], np.where(y_total == 1)[0]
    n_min = min(len(idx_0), len(idx_1))
    idx_bal = np.concatenate([rng.choice(idx_0, size=n_min, replace=False), rng.choice(idx_1, size=n_min, replace=False)])
    idx_bal = rng.permutation(idx_bal)

    X_bal, y_bal, rssi_bal, cond_bal = X_total[idx_bal], y_total[idx_bal], rssi_total[idx_bal], cond_total[idx_bal]

    # Eksekusi Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Simpanan metrik final struktur
    keys = [
        "Model Terdahulu (CSI)", "Model Terdahulu (RSSI)", 
        "LSTM Custom (CSI)", "LSTM Custom (RSSI)",
        "Random Forest (CSI)", "Random Forest (RSSI)", 
        "SVM (CSI)", "SVM (RSSI)"
    ]
    metrics_history = {k: {"LOS": [], "NLOS": []} for k in keys}
    
    # Untuk penampung CM model terbaik (Model Terdahulu - CSI)
    y_true_all, y_pred_all, cond_all = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_bal, y_bal), 1):
        logger.info(f"--- Mengevaluasi Komparasi Adil FOLD {fold} / 5 ---")
        
        # Pembagian Lipatan
        X_train_fold, y_train_fold = X_bal[train_idx], y_bal[train_idx]
        X_val_fold, y_val_fold = X_bal[val_idx], y_bal[val_idx]
        cond_val_fold = cond_bal[val_idx]
        
        # Penyiapan representasi fisis data silang
        # 1. CSI Fitur
        CSI_train_3D = X_train_fold
        CSI_val_3D = X_val_fold
        CSI_train_2D = X_train_fold.reshape(X_train_fold.shape[0], -1)
        CSI_val_2D = X_val_fold.reshape(X_val_fold.shape[0], -1)
        
        # 2. RSSI Fitur
        RSSI_train_3D = np.expand_dims(rssi_bal[train_idx], axis=-1) # (N, 30, 1)
        RSSI_val_3D = np.expand_dims(rssi_bal[val_idx], axis=-1)     # (N, 30, 1)
        RSSI_train_1D_mean = np.mean(rssi_bal[train_idx], axis=1).reshape(-1, 1) # Rata-rata jendela untuk ML
        RSSI_val_1D_mean = np.mean(rssi_bal[val_idx], axis=1).reshape(-1, 1)
        
        # --- TRAINING DAN PREDIKSI SILANG ---
        # A. Model Terdahulu
        m_ref_csi = build_previous_model(input_shape=(WINDOW_SIZE, INPUT_DIM))
        m_ref_csi.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        m_ref_csi.fit(CSI_train_3D, y_train_fold, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        p_ref_csi = np.argmax(m_ref_csi.predict(CSI_val_3D, verbose=0), axis=1)
        
        # Simpan untuk akumulasi CM final
        y_true_all.extend(y_val_fold)
        y_pred_all.extend(p_ref_csi)
        cond_all.extend(cond_val_fold)

        m_ref_rssi = build_previous_model(input_shape=(WINDOW_SIZE, 1))
        m_ref_rssi.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        m_ref_rssi.fit(RSSI_train_3D, y_train_fold, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        p_ref_rssi = np.argmax(m_ref_rssi.predict(RSSI_val_3D, verbose=0), axis=1)

        # B. Model Custom (LSTM)
        m_user_csi = build_lstm_model(input_shape=(WINDOW_SIZE, INPUT_DIM))
        m_user_csi.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        m_user_csi.fit(CSI_train_3D, y_train_fold, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        p_user_csi = np.argmax(m_user_csi.predict(CSI_val_3D, verbose=0), axis=1)
        
        m_user_rssi = build_lstm_model(input_shape=(WINDOW_SIZE, 1))
        m_user_rssi.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        m_user_rssi.fit(RSSI_train_3D, y_train_fold, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        p_user_rssi = np.argmax(m_user_rssi.predict(RSSI_val_3D, verbose=0), axis=1)

        # C. Random Forest
        rf_csi = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE).fit(CSI_train_2D, y_train_fold)
        p_rf_csi = rf_csi.predict(CSI_val_2D)
        
        rf_rssi = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE).fit(RSSI_train_1D_mean, y_train_fold)
        p_rf_rssi = rf_rssi.predict(RSSI_val_1D_mean)

        # D. SVM
        svm_csi = SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE).fit(CSI_train_2D, y_train_fold)
        p_svm_csi = svm_csi.predict(CSI_val_2D)
        
        svm_rssi = SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE).fit(RSSI_train_1D_mean, y_train_fold)
        p_svm_rssi = svm_rssi.predict(RSSI_val_1D_mean)

        # --- HITUNG METRIK TERPISAH ---
        preds = {
            "Model Terdahulu (CSI)": p_ref_csi, "Model Terdahulu (RSSI)": p_ref_rssi,
            "LSTM Custom (CSI)": p_user_csi, "LSTM Custom (RSSI)": p_user_rssi,
            "Random Forest (CSI)": p_rf_csi, "Random Forest (RSSI)": p_rf_rssi,
            "SVM (CSI)": p_svm_csi, "SVM (RSSI)": p_svm_rssi
        }
        
        mask_los = (cond_val_fold == "LOS")
        mask_nlos = (cond_val_fold == "NLOS")
        
        for k in keys:
            metrics_history[k]["LOS"].append(evaluate_and_log(y_val_fold, preds[k], mask_los, k, "", "LOS"))
            metrics_history[k]["NLOS"].append(evaluate_and_log(y_val_fold, preds[k], mask_nlos, k, "", "NLOS"))

    # ==========================================
    # FINALISASI METRIK RATA-RATA DAN CETAK TABEL
    # ==========================================
    def print_final_table(cond_name):
        logger.info(f"\n================= TABEL EVALUASI KONDISI {cond_name} =================")
        logger.info(f"{'Model & Fitur':25s} | {'Acc':6s} | {'Prec':6s} | {'Rec':6s} | {'F1':6s}")
        logger.info("-" * 60)
        for k in keys:
            acc_m = np.mean([x["Acc"] for x in metrics_history[k][cond_name]])
            pr_m  = np.mean([x["Prec"] for x in metrics_history[k][cond_name]])
            rc_m  = np.mean([x["Rec"] for x in metrics_history[k][cond_name]])
            f1_m  = np.mean([x["F1"] for x in metrics_history[k][cond_name]])
            logger.info(f"{k:25s} | {acc_m:.3f} | {pr_m:.3f} | {rc_m:.3f} | {f1_m:.3f}")

    print_final_table("LOS")
    print_final_table("NLOS")

    # ==========================================
    # GENERATE 2 CONFUSION MATRIX TERPISAH
    # ==========================================
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    cond_all = np.array(cond_all)

    for cond_type, cm_color in [("LOS", "Blues"), ("NLOS", "Reds")]:
        m_cond = (cond_all == cond_type)
        cm = confusion_matrix(y_true_all[m_cond], y_pred_all[m_cond])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Kosong', 'Terisi'])
        
        plt.figure(figsize=(5, 4))
        disp.plot(cmap=cm_color, values_format='d')
        plt.title(f"Confusion Matrix - Kondisi {cond_type}")
        plt.tight_layout()
        
        img_name = f"cm_{cond_type.lower()}.png"
        plt.savefig(os.path.join(REPORT_OUTPUT_DIR, img_name), dpi=300)
        logger.info(f"Confusion Matrix untuk {cond_type} berhasil disimpan sebagai {img_name} di folder {REPORT_OUTPUT_DIR}")
        plt.close()
        
    # F. Simpan Model Terbaik Untuk Deployment (Sesuai kode awal)
    logger.info("Melatih ulang Model Terbaik (Model Terdahulu) pada SELURUH data untuk disimpan ke file .keras...")
    final_model = build_previous_model(input_shape=(WINDOW_SIZE, INPUT_DIM))
    final_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    final_model.fit(X_bal, y_bal, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    final_model.save(os.path.join(MODEL_OUTPUT_DIR, "lstm_best_model.keras"))
    logger.info("[SELESAI] Model siap digunakan untuk Live Demo Real-time.")

if __name__ == "__main__":
    main()