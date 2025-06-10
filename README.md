# 🎙️ Kişi ve Duygu Tespiti — Ses Verisi Üzerinden Makine Öğrenmesi ile Analiz

Bu proje, ses kayıtları üzerinden hem konuşan kişiyi hem de duygusunu tespit etmeyi amaçlayan bir makine öğrenmesi çalışmasıdır. Sadece bir sonuç üretmek değil, aynı zamanda ses özniteliklerinin etkilerini karşılaştırarak **en verimli, en dengeli ve gerçek zamanlı kullanılabilir bir yapı** oluşturmak hedeflenmiştir.

---

## 💡 Projenin Amacı

Ses işleme dünyasında kişi ve duygu tespiti gibi görevlerde kullanılan farklı öznitelik çıkarım tekniklerinin etkinliğini anlamak ve bu teknikleri karşılaştırmalı olarak analiz ederek:

- **Hangi öznitelik seti ne zaman daha iyi performans gösteriyor?**
- **Bu görevler için en iyi kombinasyon nedir?**
- **Bu kombinasyonu optimize ederek gerçek zamanlı sistemlerde kullanılabilir hâle getirebilir miyiz?**

sorularına yanıt arıyoruz.

---

## 🧠 Kullanılan Yöntemler ve Kitaplıklar

- **Python 3**
- **Librosa** – MFCC, Chroma, Zero-Crossing Rate, Spectral Centroid vb.
- **Scikit-learn** – Sınıflandırma (SVM, Random Forest), model değerlendirme
- **NumPy & Pandas** – Veri analizi
- **Matplotlib** – Görselleştirme

---

## 🗂️ Proje Yapısı

Şu anda proje iki farklı `.py` dosyası üzerinden yürütülmektedir:

- `LogPitchRMS.py`: Log-Mel, Pitch ve RMS energy tabanlı analiz ve model eğitimi
- `MFCCTabanlı.py`: MFCC tabanlı analiz ve model eğitimi

Gelecekte bu yapı sadeleştirilecek ve tüm işlem adımlarını içeren **tek bir modüler `.py` dosyası** altında toplanacaktır. Böylece hem yeniden kullanılabilirlik artacak hem de gerçek zamanlı sistemler için uyarlanabilirlik kolaylaşacaktır.

---

## 🎯 Hedefler

Bu projenin temel hedefi, ses öznitelik çıkarımı tekniklerini karşılaştırarak, **kişi tanıma ve duygu tespiti** görevleri için en doğru, hızlı ve dengeli kombinasyonu bulmaktır.

- Doğruluk oranlarını en üst seviyeye çıkarmak
- Gereksiz karmaşıklıktan kaçınmak
- Modelin hız ve hafiflik bakımından optimize edilmesini sağlamak
- Nihayetinde gerçek zamanlı çalışan bir sistem altyapısı kurmak

---

## ⚠️ Lisans

Bu proje sadece geliştirici olan **Ahmet Veysel Altun** tarafından geliştirilmekte ve kullanılmaktadır. Kodlar referans amacıyla görüntülenebilir; ancak:

- Kullanılamaz  
- Kopyalanamaz  
- Dağıtılamaz  
- Akademik veya ticari projelere entegre edilemez  

Detaylar için `LICENSE` dosyasına bakabilirsiniz.

---

## 📌 Not

Bu proje gelişim aşamasındadır ve düzenli olarak güncellenecektir. Takipte kalın!
