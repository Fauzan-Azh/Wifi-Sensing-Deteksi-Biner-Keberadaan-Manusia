import serial
import os
import time

# =================================================================
# KONFIGURASI UTAMA
# =================================================================
COM_PORT = 'COM5'       # Port ESP32 Receiver
BAUD_RATE = 115200      # Kecepatan sinkron dengan firmware

COUNTDOWN_SECONDS = 20  # Waktu bagi kamu untuk berjalan masuk ke posisi
DURATION_SECONDS = 180  # Durasi perekaman otomatis (3 menit)

def main():
    print("=======================================================")
    print("      WIFI SENSING DATA RECORDER (LIVE COUNTER MODE)   ")
    print("=======================================================")
    print("Pilih Skenario Pengambilan Data:")
    print("1. Ruangan Kosong (Profil Referensi Statis)")
    print("2. Orang Diam (Aktivitas Statis / Sitting Quietly)")
    print("3. Orang Bergerak (Aktivitas Dinamis / Walking Around)")
    print("4. Posisi Dekat Sinyal (Objek ~0.5 Meter dari RX)")
    print("5. Posisi Jauh Sinyal (Objek di Sudut Ruangan)")
    print("6. Menembus Hambatan Fisik (Skenario Privasi / NLOS di Luar Kaca)")
    
    pilihan = input("Masukkan pilihan (1-6): ").strip()
    
    if pilihan == '1':
        file_path = "dataset/ruangan_kosong.csv"
    elif pilihan == '2':
        file_path = "dataset/orang_diam.csv"
    elif pilihan == '3':
        file_path = "dataset/orang_bergerak.csv"
    elif pilihan == '4':
        file_path = "dataset/posisi_dekat.csv"
    elif pilihan == '5':
        file_path = "dataset/posisi_jauh.csv"
    elif pilihan == '6':
        file_path = "dataset/hambatan_nlos.csv"
    else:
        print("Pilihan tidak valid! Program dihentikan.")
        return

    os.makedirs('dataset', exist_ok=True)
    
    print(f"\n[INFO] Menghubungkan ke {COM_PORT}...")
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.5)
        print(f"[SUKSES] Berhasil terhubung!")
        print(f"[PENTING] Data akan disimpan ke: {file_path}")
        print("-------------------------------------------------------")
        
        print(f"[READY] Silakan masuk ke posisi skenario. Hitung mundur dimulai:")
        for i in range(COUNTDOWN_SECONDS, 0, -1):
            print(f"⌛ {i} detik lagi...")
            time.sleep(1)
        
        ser.flushInput() 
        print("\n>>>PEREKAMAN DIMULAI SECARA OTOMATIS! <<<")
        
        start_time = time.time()
        packet_count = 0
        
        with open(file_path, 'w', encoding='utf-8') as f:
            while (time.time() - start_time) < DURATION_SECONDS:
                if ser.in_waiting > 0:
                    raw_line = ser.readline()
                    try:
                        line = raw_line.decode('utf-8', errors='ignore').strip()
                    except Exception:
                        continue
                    
                    if "CSI_DATA" in line or "type,role,mac" in line:
                        f.write(line + "\n")
                        packet_count += 1
                
                # LIVE MONITORING: Cetak status paket ke layar SETIAP DETIK
                waktu_berjalan = int(time.time() - start_time)
                if waktu_berjalan % 1 == 0:
                    print(f"⏱️ Waktu: {waktu_berjalan}/{DURATION_SECONDS}s | 📦 Paket Terambil: {packet_count}", end='\r')
                    time.sleep(0.01) # Jeda mikro agar terminal tidak berkedip ekstrem
            
            f.flush()

        print("\n\n>>> 🟢 PEREKAMAN SELESAI SECARA OTOMATIS! <<<")
        print("=======================================================")
        print(f"[SUKSES] Durasi terpenuhi. Total {packet_count} paket tersimpan di {file_path}")
        print("=======================================================")

    except serial.SerialException:
        print(f"\n[ERROR] Gagal membuka {COM_PORT}. Pastikan device dicolok dengan benar.")
    except KeyboardInterrupt:
        print("\n[INFO] Perekaman dihentikan paksa oleh user.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()