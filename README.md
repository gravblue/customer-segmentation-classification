# Clustering & Klasifikasi Transaksi Bank

Proyek machine learning end-to-end pada dataset transaksi bank, mulai dari segmentasi nasabah secara unsupervised (clustering) hingga klasifikasi supervised terhadap hasil segmentasi tersebut.

## 📊 Dataset

2.512 data transaksi bank dengan atribut yang mencakup detail transaksi, demografi nasabah, dan pola penggunaan (jumlah transaksi, tipe transaksi, channel, usia nasabah, pekerjaan, saldo akun, jumlah percobaan login, dll). Cocok untuk kasus deteksi fraud dan identifikasi anomali.

## 1️⃣ Clustering

**Tujuan:** mengelompokkan nasabah berdasarkan perilaku transaksi dan demografi tanpa label yang ditentukan sebelumnya.

**Alur pengerjaan:**
- **EDA**: heatmap korelasi, histogram distribusi, dan boxplot (nilai transaksi berdasarkan pekerjaan)
- **Pembersihan data**: menangani missing value dan data duplikat, drop kolom id/address/date
- **Pra-pemrosesan**: encoding fitur kategorikal dengan `LabelEncoder`, penanganan outlier dengan metode IQR, feature scaling dengan `StandardScaler`, dan binning usia
- **Penentuan jumlah cluster**: Elbow Method dengan `KElbowVisualizer` (metrik silhouette)
- **Clustering**: `KMeans` (k = 2)
- **Evaluasi**: Silhouette Score: **0,57**
- **Visualisasi**: proyeksi 2D hasil cluster menggunakan PCA
- **Interpretasi**: analisis karakteristik tiap cluster pada data yang masih di-scale maupun yang sudah di-inverse (skala asli)

**Hasil:** dua segmen nasabah yang berbeda, segmen dengan saldo lebih stabil dan durasi transaksi lebih lama, versus segmen dengan pola transaksi lebih cepat dan dinamis.
- **Label 0**: Nasabah Profesional dengan Aktivitas Stabil, dan 
- **Label 1**: Nasabah Muda dengan Pola Transaksi Lebih Dinamis 

## 2️⃣ Klasifikasi

**Tujuan:** membangun model yang dapat mengklasifikasikan nasabah baru ke dalam segmen/cluster yang telah didefinisikan sebelumnya (hasil clustering).

**Alur pengerjaan:**
- Memuat data hasil inverse transform dari tahap clustering, lalu melakukan One-Hot Encoding pada fitur kategorikal
- **Data splitting**: pembagian data latih/uji 80/20 (1.556 data latih / 389 data uji)
- **Model dasar**: `DecisionTreeClassifier`
- **Perbandingan model**: menambahkan `RandomForestClassifier` dan `LogisticRegression`
- **Hyperparameter tuning**: `GridSearchCV` (5-fold CV) pada Random Forest dan Logistic Regression

**Hasil:** model klasifikasi berhasil mempelajari pola pengelompokan yang dihasilkan oleh KMeans dengan akurasi tinggi, sehingga dapat digunakan untuk mengklasifikasikan nasabah baru ke dalam segmen yang telah ditentukan tanpa perlu menjalankan ulang proses clustering.

**Hasil (data uji):**
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Decision Tree | 1.00 | 1.00 | 1.00 | 1.00 |
| Random Forest | 1.00 | 1.00 | 1.00 | 1.00 |
| Logistic Regression | 0.99 | 0.99 | 0.99 | 0.99 |
| Random Forest (tuned) | 1.00 | 1.00 | 1.00 | 1.00 |


## 🛠️ Tools & Library

`pandas` · `numpy` · `scikit-learn` (`KMeans`, `DecisionTreeClassifier`, `RandomForestClassifier`, `LogisticRegression`, `GridSearchCV`) · `yellowbrick` · `matplotlib` · `seaborn` · `joblib`

