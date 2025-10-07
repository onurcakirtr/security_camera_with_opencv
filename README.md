
# Security Cam / Güvenlik Kamerası

---

## English

### 📋 Overview  
Security Cam is a Python application that combines motion detection and multi-face recognition to automatically record and snapshot video when unauthorized movement or unknown faces are detected. It provides both a powerful command-line interface (CLI) and a modern GUI with dark/light themes and EN/TR language support.

### ✨ Features  
- **Motion Detection** via running-average background subtraction  
- **Face Recognition** with `face_recognition` + SVM classifier  
- **Trusted/Unknown** face tagging and color-coded bounding boxes  
- **Automatic Recording** when motion & unknown faces occur  
- **Snapshot Support** at configurable intervals  
- **Multiple Cameras** support (`--cam-num`)  
- **CLI Flags** for all options (e.g. `--min-area`, `--threshold`, `--no-record`)  
- **Modern GUI** (`python cli.py --gui`) with:  
  - Dark / Light theme toggle  
  - English / Türkçe language selector  
  - Browse dialogs for model/data paths   

### 🛠 Installation  
1. Download codes.  
2. Install dependencies:  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### 🚀 Usage

#### Command-Line Interface  
- **Train model:**  
  ```bash
  python cli.py -train --data faces --model model.pkl
  ```  
- **Start camera monitoring:**  
  ```bash
  python cli.py -cam --min-area 500 --threshold 0.7
  ```  
- **Combine options:**  
  ```bash
  python cli.py -cam --cam-num 2 --snapshot --snapshot-interval 10
  ```

#### Graphical User Interface  
```bash
python cli.py --gui
```  

#### GUI Features
1. Select language (EN / TR)  
2. Choose dark or light theme  
3. Adjust parameters with sliders, checkboxes, and Browse buttons  
4. Click **Run** to launch the monitoring process  

---

## Türkçe

### 📋 Genel Bakış  
Güvenlik Kamerası, hareket algılama ve çoklu yüz tanıma özelliklerini birleştirerek izinsiz hareket veya tanınmayan yüz tespitinde otomatik olarak video kaydı ve anlık görüntü (snapshot) alır. Hem güçlü bir komut satırı arayüzü (CLI) hem de modern bir GUI sunar; koyu/açık tema ve İngilizce/Türkçe dil seçenekleri mevcuttur.

### ✨ Özellikler  
- **Hareket Algılama**: Koşan ortalama arka plan çıkarımı  
- **Yüz Tanıma**: `face_recognition` + SVM sınıflandırıcı  
- **Güvenilir/Tanımsız** yüz etiketleme, renkli kutular  
- **Otomatik Kayıt**: Hareket ve tanınmayan yüz tespitinde  
- **Anlık Görüntüler**: Ayarlanabilir aralıklarla snapshot  
- **Çoklu Kamera** desteği (`--cam-num`)  
- **CLI Bayrakları** tüm seçenekler için (örn. `--min-area`, `--threshold`, `--no-record`)  
- **Modern GUI** (`python cli.py --gui`):  
  - Koyu / Açık tema  
  - İngilizce / Türkçe dil seçimi  
  - Model ve veri dizini için Gözat düğmeleri   

### 🛠 Kurulum  
1. Kodları indirin.  
3. Bağımlılıkları yükleyin:  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### 🚀 Kullanım

#### Komut Satırı Arayüzü  
- **Model Eğit:**  
  ```bash
  python cli.py -train --data faces --model model.pkl
  ```  
- **Kamerayı Başlat:**  
  ```bash
  python cli.py -cam --min-area 500 --threshold 0.7
  ```  
- **Seçenekleri Birleştirme:**  
  ```bash
  python cli.py -cam --cam-num 2 --snapshot --snapshot-interval 10
  ```

#### Grafiksel Kullanıcı Arayüzü  
```bash
python cli.py --gui
```  
1. Dil seçin (EN / TR)  
2. Koyu veya açık tema seçin  
3. Kaydırıcılar, onay kutuları ve Gözat düğmeleriyle parametreleri ayarlayın  
4. **Çalıştır** düğmesine basarak izlemeyi başlatın  
