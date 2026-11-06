# AI Prediksi Hujan untuk JemuranGuard

Dokumen ini menjelaskan fitur AI yang ditambahkan ke proyek JemuranGuard (Kelompok 07, mata kuliah Internet of Things): model *machine learning* ringan yang memprediksi apakah **hujan akan turun dalam 15 menit ke depan**, berdasarkan data sensor HP A di jemuran.

## Inti Singkat

| | |
|---|---|
| **Input** | Data sensor HP A selama 120 detik terakhir: cahaya (lux), penurunan cahaya (lux_drop), dan goyangan (RMS) |
| **Proses** | Hitung 6 fitur dari data tersebut, lalu masukkan ke model Logistic Regression yang sudah dilatih |
| **Output** | Probabilitas hujan dalam 15 menit ke depan (`rain_prob`), jika ≥ 0,7 status menjadi `WASPADA` dan HP B mendapat notifikasi |

---

## 1. Latar Belakang

Sistem JemuranGuard saat ini memakai aturan tetap (*rule-based*):

```
BAHAYA = (lux_drop > 80) DAN (rms_swing > 2.5)
```

Aturan ini baru bereaksi **setelah** kondisi buruk terjadi (sudah mendung tebal dan jemuran sudah bergoyang kencang). Pemilik di kampus sering baru tahu saat sudah terlambat.

AI prediksi hujan melengkapi aturan tersebut dengan **peringatan dini**: model mempelajari pola perubahan cahaya dan goyangan beberapa menit sebelum hujan, lalu memberi status **WASPADA** lebih awal. Status WASPADA ini juga sudah tercantum di slide presentasi, tetapi belum ada di `jemuran.py`.

---

## 2. Gambaran Alur

```
HP A (sensor cahaya + accelerometer)
        │  lux, lux_drop, rms_swing  (tiap 0,5 detik)
        ▼
  Jendela 120 detik terakhir
        │
        ▼
  Ekstraksi 6 fitur  ──►  Model Logistic Regression  ──►  rain_prob (0..1)
        │
        ▼
  rain_prob ≥ 0,7 ?  ──► status WASPADA  ──► MQTT topic alerts
                                                   │
                                                   ▼
                                  HP B: getar + notifikasi
                                  "Kemungkinan hujan 82% dalam 15 menit"
```

Pelatihan dilakukan di **laptop**, sedangkan prediksi berjalan di **HP A** (edge). Dengan begitu, data mentah tidak perlu dikirim ke cloud dan prediksi tetap jalan walau internet sempat putus.

---

## 3. Input, Proses, Output

### Input

Data dari sensor HP A dalam jendela waktu **120 detik terakhir**. Setiap sampel berisi:

| Data | Arti |
|---|---|
| `ts` | Waktu pengambilan sampel |
| `light_lux` | Intensitas cahaya (lux) |
| `lux_drop` | Penurunan cahaya dari baseline (maks. 120 sampel terakhir) |
| `rms_swing` | Intensitas goyangan dari accelerometer (m/s²) |

### Proses

**a. Ekstraksi fitur.** Dari jendela 120 detik dihitung 6 fitur:

| Fitur | Arti | Kenapa relevan |
|---|---|---|
| `lux_mean` | Rata-rata cahaya | Hari mendung cenderung lebih gelap |
| `lux_slope_per_min` | Laju perubahan cahaya per menit | Nilai negatif besar berarti langit cepat menggelap |
| `lux_drop_max` | Penurunan cahaya terbesar | Mengukur seberapa tebal awan menutup |
| `rms_mean` | Rata-rata goyangan | Angin yang konsisten |
| `rms_max` | Goyangan maksimum | Hembusan angin kencang (gust) |
| `rms_trend_per_min` | Tren goyangan per menit | Angin yang makin kencang sebelum hujan |

**b. Model.** *Logistic Regression* dengan fitur yang sudah distandardisasi:

```
z = intercept + Σ ( koefisien_i × (fitur_i − rata_i) / skala_i )
rain_prob = 1 / (1 + e^(−z))
```

Model ini dipilih karena ringan, mudah dijelaskan (setiap koefisien menunjukkan fitur mana yang berpengaruh), dan **bisa dijalankan dengan Python murni di Termux** tanpa scikit-learn: bobot model disimpan sebagai JSON, lalu dihitung manual.

**c. Pelabelan data.** Setiap sampel latih diberi label:

- `1` jika **ada hujan dalam 15 menit ke depan**
- `0` jika tidak ada

Sampel pada saat hujan sedang berlangsung **dibuang**, karena yang ingin diprediksi adalah *awal* hujan, bukan hujan yang sudah terjadi.

### Output

```json
{
  "device_id": "EDGE_A",
  "light_lux": 112.0,
  "lux_drop": 288.0,
  "rms_swing": 3.41,
  "rain_prob": 0.82,
  "status": "WASPADA",
  "ts": 1730800000.0
}
```

Aturan status gabungan:

| Kondisi | Status |
|---|---|
| Aturan lama terpenuhi (mendung DAN goyang) | `JEMURAN_BAHAYA` (sirine lokal + notifikasi) |
| Tidak BAHAYA dan `rain_prob ≥ 0,7` | `WASPADA` (notifikasi, maks. sekali per 5 menit) |
| Lainnya | `AMAN` |

### 3.1 Ringkasan Input-Proses-Output per Fase

Sistem ini punya dua fase, dan masing-masing punya input-proses-output sendiri.

| | Fase Training (laptop, dilakukan sekali) | Fase Inferensi (HP A, berjalan terus) |
|---|---|---|
| **Input** | File CSV berisi `ts, lux, lux_drop, rms, rain` hasil `collect_data.py` | Jendela 120 detik terakhir berisi `(ts, lux, lux_drop, rms)` dari sensor |
| **Proses** | Membuat sampel latih tiap 10 detik, menghitung 6 fitur, memberi label "hujan < 15 menit", membagi data 70/30 berdasarkan waktu, melatih Logistic Regression | Menghitung 6 fitur yang sama, standardisasi, menghitung `z`, lalu sigmoid, diulang tiap 5 detik |
| **Output** | `rain_model.json` (rata-rata, skala, koefisien, intercept) dan laporan evaluasi | `rain_prob` (0 sampai 1) lalu status `AMAN` atau `WASPADA`, dikirim lewat MQTT |

### 3.2 Contoh Format Data

**Input fase training** (isi file CSV):

```csv
ts,lux,lux_drop,rms,rain
1730800000.5,640.0,12.0,0.41,0
1730800001.0,622.0,30.0,0.55,0
1730800001.5,410.0,230.0,1.92,0
...
1730800905.0,95.0,545.0,3.40,1
```

Kolom `rain` bernilai 0 atau 1 sesuai tombol `r` dan `s` yang kamu tekan saat merekam.

**Input fase inferensi** (di memori HP A, satu baris per sampel sensor):

```python
hist = [(1730800000.5, 640.0, 12.0, 0.41),
        (1730800001.0, 622.0, 30.0, 0.55),
        ...]          # sekitar 240 baris untuk 120 detik
```

### 3.3 Contoh Perhitungan (Fase Inferensi)

Angka di bawah ini **ilustrasi** untuk memperlihatkan alurnya. Nilai asli koefisien akan berbeda, tergantung data yang kalian kumpulkan.

**Langkah 1: hitung 6 fitur dari jendela 120 detik.**

| Fitur | Nilai |
|---|---|
| `lux_mean` | 350 lx |
| `lux_slope_per_min` | -90 lx/menit |
| `lux_drop_max` | 200 lx |
| `rms_mean` | 1,8 m/s² |
| `rms_max` | 3,6 m/s² |
| `rms_trend_per_min` | +0,9 m/s² per menit |

**Langkah 2: standardisasi** dengan rumus `(nilai - rata-rata latih) / skala latih`, lalu kalikan dengan koefisien.

| Fitur | Rata-rata | Skala | Nilai standar | Koefisien | Kontribusi |
|---|---|---|---|---|---|
| `lux_mean` | 600 | 250 | -1,00 | -0,6 | +0,60 |
| `lux_slope_per_min` | -20 | 60 | -1,17 | -0,9 | +1,05 |
| `lux_drop_max` | 60 | 70 | +2,00 | +0,7 | +1,40 |
| `rms_mean` | 1,0 | 0,8 | +1,00 | +0,3 | +0,30 |
| `rms_max` | 2,0 | 1,2 | +1,33 | +0,4 | +0,53 |
| `rms_trend_per_min` | 0,1 | 0,5 | +1,60 | +0,5 | +0,80 |
| | | | | **Jumlah** | **+4,68** |

**Langkah 3: tambahkan intercept dan hitung sigmoid.**

```
z         = 4,68 + (-3,2)        = 1,48
rain_prob = 1 / (1 + e^(-1,48))  = 0,82
```

**Langkah 4: ambil keputusan.** Karena `0,82 ≥ 0,7` dan aturan BAHAYA belum terpenuhi, status menjadi `WASPADA`.

**Output yang dikirim ke MQTT dan diterima HP B:**

```json
{"device_id": "EDGE_A", "light_lux": 112.0, "lux_drop": 288.0,
 "rms_swing": 3.41, "rain_prob": 0.82, "status": "WASPADA", "ts": 1730800900.0}
```

Notifikasi di HP B: **"Waspada hujan: Kemungkinan hujan 82% dalam 15 menit"**.

### 3.4 Cara Membaca Hasilnya

- Koefisien **negatif** pada `lux_mean` dan `lux_slope_per_min` berarti: makin gelap dan makin cepat menggelap, probabilitas hujan naik (karena nilainya dikalikan angka standar yang negatif, hasilnya positif).
- Koefisien **positif** pada `lux_drop_max`, `rms_max`, dan `rms_trend_per_min` berarti: makin besar penurunan cahaya dan makin kencang angin, probabilitas hujan naik.
- Setelah training, `train_model.py` mencetak bobot ini. Kalau tandanya masuk akal secara fisik seperti di atas, itu pertanda baik bahwa model belajar pola yang benar. Kalau tandanya aneh (misalnya angin kencang justru menurunkan probabilitas), periksa lagi kualitas dan label datanya.

---

## 4. Daftar File

| File | Dijalankan di | Fungsi |
|---|---|---|
| `features.py` | Laptop dan HP A | Rumus 6 fitur. Dipakai sama persis saat training dan inferensi supaya tidak ada selisih hasil |
| `collect_data.py` | Laptop | Merekam telemetry dari MQTT ke CSV, sambil memberi label hujan secara manual |
| `train_model.py` | Laptop | Melatih model, mengevaluasi, dan menyimpan `rain_model.json` |
| `rain_model.py` | HP A | Membaca `rain_model.json` dan menghitung probabilitas hujan (Python murni) |

---

## 5. Langkah Penggunaan

### Tahap 1: Kumpulkan data

1. Di HP A jalankan: `python jemuran.py edge`
2. Di laptop jalankan: `python collect_data.py`
3. Beri label lewat keyboard, lalu Enter:
   - `r` : hujan mulai
   - `s` : hujan berhenti
   - `q` : keluar dan simpan

Hasilnya file CSV dengan kolom `ts, lux, lux_drop, rms, rain`.

### Tahap 2: Latih model

Instal dependensi sekali saja:

```bash
pip install pandas numpy scikit-learn paho-mqtt
```

Latih dengan satu atau beberapa file CSV:

```bash
python train_model.py data_a.csv data_b.csv
```

Script akan mencetak *confusion matrix*, *precision/recall*, dan bobot tiap fitur, lalu menyimpan `rain_model.json`.

### Tahap 3: Pasang di HP A

Salin `rain_model.json`, `features.py`, dan `rain_model.py` ke folder yang sama dengan `jemuran.py` di HP A, lalu terapkan patch pada bagian 6 di bawah.

---

## 6. Patch untuk `jemuran.py`

### Mode edge

**Import** (di bagian atas):

```python
from rain_model import RainModel
from features import WINDOW_S
RAIN_TH = 0.7
```

**Inisialisasi** (setelah baris `lux_h, dev_h, last_alert = ...`):

```python
try: model = RainModel()
except Exception as e: model = None; print("[EDGE] model tidak ada:", e)
hist, prob, last_pred, last_warn = deque(), None, 0, 0
```

**Di dalam loop** (setelah baris `status = ... # Fusion`, sebelum `p = json.dumps(...)`):

```python
now = time.time()
hist.append((now, lux, drop, rms))
while hist and now - hist[0][0] > WINDOW_S: hist.popleft()
if model and now - last_pred >= 5:
    prob, last_pred = model.predict(list(hist)), now
if status == "AMAN" and prob is not None and prob >= RAIN_TH:
    status = "WASPADA"
```

Tambahkan `"rain_prob": prob` ke dalam dict pada `json.dumps(...)`.

**Kirim alert WASPADA** (setelah blok `if status == "JEMURAN_BAHAYA" ...`):

```python
if status == "WASPADA" and now - last_warn > 300:
    last_warn = now
    if c.is_connected(): c.publish(T["alerts"], p, qos=1)
```

### Mode receiver

Di `on_message`, setelah blok `alerts` untuk BAHAYA:

```python
elif k == "alerts" and d.get("status") == "WASPADA":
    pr = d.get("rain_prob") or 0
    notif("Waspada hujan", f"Kemungkinan hujan {pr*100:.0f}% dalam 15 menit", 800)
```

> Catatan: dashboard `jemuran.html` mungkin perlu penyesuaian agar menampilkan status WASPADA dan `rain_prob`.

---

## 7. Evaluasi Model

Agar hasil evaluasi jujur, `train_model.py` memakai **pembagian berdasarkan waktu**: 70% data paling awal untuk melatih, 30% terakhir untuk menguji. Pembagian acak tidak dipakai karena sampel yang berdekatan waktunya sangat mirip, sehingga hasil uji akan terlihat bagus secara menipu.

Metrik yang perlu dilaporkan:

| Metrik | Arti dalam konteks ini |
|---|---|
| **Recall** (kelas hujan) | Dari semua kejadian hujan, berapa persen yang berhasil diperingatkan sebelumnya. Penting agar tidak kecolongan |
| **Precision** (kelas hujan) | Dari semua peringatan, berapa persen yang benar-benar diikuti hujan. Penting agar tidak sering salah alarm |
| **Confusion matrix** | Rincian benar/salah per kelas |

Threshold `RAIN_TH` (default 0,7) bisa diatur: **naikkan** jika terlalu sering salah alarm, **turunkan** jika sering terlambat memberi peringatan.

---

## 8. Skenario Pengujian (tambahan untuk slide)

| # | Skenario | Harapan |
|---|---|---|
| 1 | Cahaya normal, HP diam | `AMAN`, `rain_prob` rendah |
| 2 | Cahaya turun bertahap, HP diam | `rain_prob` naik, bisa jadi `WASPADA` |
| 3 | Cahaya normal, HP digoyang | Tidak langsung `WASPADA` (goyangan saja tidak cukup) |
| 4 | Cahaya turun + HP digoyang | `JEMURAN_BAHAYA`, sirine bunyi, HP B getar, ACK terkirim |
| 5 | `WASPADA` berulang dalam 5 menit | Notifikasi tidak spam (jeda 5 menit) |
| 6 | File `rain_model.json` tidak ada | Sistem tetap jalan dengan aturan lama, tanpa crash |

---

## 9. Faktor yang Menentukan Keberhasilan

1. **Jumlah kejadian hujan, bukan jumlah baris data.** Satu hari bisa menghasilkan puluhan ribu baris, tetapi jika hanya ada satu hujan, model tidak punya cukup contoh. Targetkan minimal **5 sampai 10 kejadian hujan**.
2. **Rekam juga hari mendung yang tidak hujan.** Tanpa contoh ini, model belajar "gelap = hujan" dan sering salah alarm.
3. **Posisi HP A harus tetap** selama pengumpulan data dan saat dipakai, karena nilai lux sangat bergantung pada posisi dan arah sensor.
4. **Label manual harus konsisten.** Tekan `r` saat hujan benar-benar mulai terlihat dan `s` saat berhenti.

---

## 10. Batasan

- Model hanya memakai **cahaya dan goyangan**. Sensor yang paling informatif untuk hujan sebenarnya tekanan udara (barometer), tetapi belum dipakai karena kode `jemuran.py` saat ini belum membacanya. Bisa ditambahkan sebagai pengembangan lanjutan jika HP mendukung.
- Model belajar dari **satu lokasi**. Jika HP A dipindahkan atau kondisi sekitar berubah (misalnya ada bangunan baru yang menaungi), model perlu dilatih ulang.
- Hasilnya berupa **probabilitas**, bukan kepastian. Peringatan WASPADA artinya "kemungkinan hujan tinggi", bukan "pasti hujan".
- Prediksi hanya menjawab "hujan dalam 15 menit atau tidak", **tanpa estimasi menit tepatnya**.

---

## 11. Pengembangan Lanjutan

- Tambah sensor **barometer** sebagai fitur ke-7.
- Tambah fitur **jam dalam sehari** (hanya jika data sudah mencakup banyak hari, agar tidak *overfitting*).
- **Threshold adaptif** (z-score bergerak) sebagai pengganti angka tetap `LUX_DROP` dan `RMS_TH`.
- **Chatbot LLM** yang membaca log MQTT dan menjawab pertanyaan pemilik, misalnya "kenapa sirine tadi bunyi?".
- Bandingkan dengan **Random Forest** untuk melihat apakah ada peningkatan akurasi.
