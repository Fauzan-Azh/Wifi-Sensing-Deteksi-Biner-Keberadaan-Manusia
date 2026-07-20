import numpy as np

try:
    with open("latency_log.txt", "r") as f:
        # Baca semua angka, abaikan baris kosong
        latencies = [float(line.strip()) for line in f.readlines() if line.strip()]
    
    if latencies:
        print("=== HASIL EVALUASI LATENSI REAL-TIME ===")
        print(f"Rata-rata (Mean) : {np.mean(latencies):.2f} ms")
        print(f"Minimum (Min)    : {np.min(latencies):.2f} ms")
        print(f"Maksimum (Max)   : {np.max(latencies):.2f} ms")
        print(f"Std Deviasi      : {np.std(latencies):.2f} ms")
        print("========================================")
    else:
        print("File log kosong.")
except FileNotFoundError:
    print("File latency_log.txt tidak ditemukan.")