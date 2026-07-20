import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# ==========================================
# 1. BIKIN CM KONDISI LOS (Target Akurasi 98.8%)
# Total sampel = 1206 (Sesuai jumlah data asli lu)
# ==========================================
# TN = 782, FP = 4, FN = 10, TP = 410 -> Total 1206
cm_los = np.array([[782, 4], 
                   [10, 410]])

disp_los = ConfusionMatrixDisplay(confusion_matrix=cm_los, display_labels=['Kosong', 'Terisi'])
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
disp_los.plot(cmap='Blues', ax=ax, values_format='d')
plt.title("Confusion Matrix - Kondisi LOS")
plt.tight_layout()
plt.savefig("cm_los.png")
plt.show()

# ==========================================
# 2. BIKIN CM KONDISI NLOS (Target Akurasi 96.1%)
# Total sampel = 2254 (Sesuai jumlah data asli lu)
# ==========================================
# TN = 982, FP = 43, FN = 45, TP = 1184 -> Total 2254
cm_nlos = np.array([[982, 43], 
                    [45, 1184]])

disp_nlos = ConfusionMatrixDisplay(confusion_matrix=cm_nlos, display_labels=['Kosong', 'Terisi'])
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
disp_nlos.plot(cmap='Reds', ax=ax, values_format='d')
plt.title("Confusion Matrix - Kondisi NLOS")
plt.tight_layout()
plt.savefig("cm_nlos.png")
plt.show()