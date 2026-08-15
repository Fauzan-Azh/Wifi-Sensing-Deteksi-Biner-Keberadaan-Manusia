# Skenario Pengujian Lintas Batas Distribusi Asli (Tanpa Undersampling pada Data Uji)
X_test_raw_pure, y_test_raw_pure, rssi_test_raw_pure = build_dataset_clean_method(TEST_DIR, csi_min, csi_max)

# Prediksi menggunakan model yang sudah dilatih
y_pred_pure = np.argmax(lstm_model.predict(X_test_raw_pure, verbose=0), axis=1)
acc_pure = accuracy_score(y_test_raw_pure, y_pred_pure)

print("\n========================================================")
print(f"AKURASI PADA TEST SET ALAMI (TANPA UNDERSAMPLING): {acc_pure * 100:.2f}%")
print("========================================================")