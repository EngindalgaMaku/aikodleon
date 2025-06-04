import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python ile Veri Bilimi Projeleri | Python Veri Bilimi | Kodleon',
  description: 'Gerçek dünya veri bilimi projelerini Python ile uygulayın. Müşteri segmentasyonu, öneri sistemleri ve daha fazlası.',
};

const content = `
# Veri Bilimi Projeleri

Bu bölümde, gerçek dünya problemlerine yönelik kapsamlı veri bilimi projelerini inceleyeceğiz. Her proje, veri analizi, model geliştirme ve değerlendirme aşamalarını içermektedir.

## Müşteri Segmentasyonu Projesi

\`\`\`python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

class MusteriSegmentasyonu:
    def __init__(self, veri_yolu):
        self.veri = pd.read_csv(veri_yolu)
        self.scaler = StandardScaler()
        self.model = None
        self.optimal_kume_sayisi = None
        
    def veri_on_isleme(self):
        # Eksik değerleri doldur
        self.veri = self.veri.fillna(self.veri.mean())
        
        # Kategorik değişkenleri dönüştür
        kategorik_kolonlar = self.veri.select_dtypes(include=['object']).columns
        self.veri = pd.get_dummies(self.veri, columns=kategorik_kolonlar)
        
        # Ölçeklendirme
        self.X_olcekli = self.scaler.fit_transform(self.veri)
        
    def optimal_kume_bul(self, max_kume=10):
        sil_skorlari = []
        inertia_degerleri = []
        
        for n in range(2, max_kume + 1):
            kmeans = KMeans(n_clusters=n, random_state=42)
            kmeans.fit(self.X_olcekli)
            
            sil_skor = silhouette_score(self.X_olcekli, kmeans.labels_)
            sil_skorlari.append(sil_skor)
            inertia_degerleri.append(kmeans.inertia_)
            
        # Dirsek yöntemi grafiği
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(2, max_kume + 1), inertia_degerleri, 'bo-')
        plt.xlabel('Küme Sayısı')
        plt.ylabel('Inertia')
        plt.title('Dirsek Yöntemi')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(2, max_kume + 1), sil_skorlari, 'ro-')
        plt.xlabel('Küme Sayısı')
        plt.ylabel('Silhouette Skoru')
        plt.title('Silhouette Analizi')
        
        plt.tight_layout()
        plt.show()
        
        # Optimal küme sayısını belirle
        self.optimal_kume_sayisi = sil_skorlari.index(max(sil_skorlari)) + 2
        print(f"Optimal küme sayısı: {self.optimal_kume_sayisi}")
        
    def kumeleme_yap(self):
        self.model = KMeans(n_clusters=self.optimal_kume_sayisi, random_state=42)
        self.veri['Segment'] = self.model.fit_predict(self.X_olcekli)
        
    def segment_analizi(self):
        # Segment bazında özet istatistikler
        segment_ozet = self.veri.groupby('Segment').mean()
        print("\\nSegment Özeti:")
        print(segment_ozet)
        
        # Segment boyutları
        segment_boyutlari = self.veri['Segment'].value_counts()
        
        plt.figure(figsize=(10, 5))
        segment_boyutlari.plot(kind='bar')
        plt.title('Segment Boyutları')
        plt.xlabel('Segment')
        plt.ylabel('Müşteri Sayısı')
        plt.show()
        
        # Özellik dağılımları
        for kolon in self.veri.select_dtypes(include=['float64', 'int64']).columns:
            if kolon != 'Segment':
                plt.figure(figsize=(10, 5))
                sns.boxplot(x='Segment', y=kolon, data=self.veri)
                plt.title(f'{kolon} Dağılımı')
                plt.show()
                
    def segment_tahmin(self, yeni_musteri):
        # Yeni müşteriyi ölçeklendir
        yeni_musteri_olcekli = self.scaler.transform(yeni_musteri)
        
        # Segment tahmin et
        segment = self.model.predict(yeni_musteri_olcekli)
        return segment[0]

# Kullanım örneği
if __name__ == "__main__":
    # Örnek veri oluştur
    np.random.seed(42)
    musteri_sayisi = 1000
    
    veri = pd.DataFrame({
        'Yillik_Gelir': np.random.normal(50000, 20000, musteri_sayisi),
        'Harcama_Skoru': np.random.normal(50, 15, musteri_sayisi),
        'Sadakat_Puani': np.random.normal(60, 20, musteri_sayisi),
        'Yas': np.random.normal(40, 15, musteri_sayisi)
    })
    
    veri.to_csv('musteri_verileri.csv', index=False)
    
    # Segmentasyon analizi
    segmentasyon = MusteriSegmentasyonu('musteri_verileri.csv')
    segmentasyon.veri_on_isleme()
    segmentasyon.optimal_kume_bul()
    segmentasyon.kumeleme_yap()
    segmentasyon.segment_analizi()
    
    # Yeni müşteri tahmini
    yeni_musteri = pd.DataFrame({
        'Yillik_Gelir': [60000],
        'Harcama_Skoru': [55],
        'Sadakat_Puani': [70],
        'Yas': [35]
    })
    
    segment = segmentasyon.segment_tahmin(yeni_musteri)
    print(f"\\nYeni müşteri segmenti: {segment}")
\`\`\`

## Öneri Sistemi Projesi

\`\`\`python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class OneriSistemi:
    def __init__(self):
        self.kullanici_urun_matrisi = None
        self.benzerlik_matrisi = None
        self.urunler = None
        self.kullanicilar = None
        
    def veri_yukle(self, degerlendirmeler_df):
        # Pivot tablo oluştur
        self.kullanici_urun_matrisi = degerlendirmeler_df.pivot(
            index='kullanici_id',
            columns='urun_id',
            values='puan'
        ).fillna(0)
        
        self.kullanicilar = self.kullanici_urun_matrisi.index
        self.urunler = self.kullanici_urun_matrisi.columns
        
        # Sparse matris oluştur
        sparse_matris = csr_matrix(self.kullanici_urun_matrisi.values)
        
        # Benzerlik matrisini hesapla
        self.benzerlik_matrisi = cosine_similarity(sparse_matris)
        
    def oneri_uret(self, kullanici_id, n_oneri=5):
        # Kullanıcı indeksini bul
        kullanici_idx = list(self.kullanicilar).index(kullanici_id)
        
        # Kullanıcı benzerliklerini al
        kullanici_benzerlikleri = self.benzerlik_matrisi[kullanici_idx]
        
        # En benzer kullanıcıları bul
        benzer_kullanicilar = np.argsort(kullanici_benzerlikleri)[::-1][1:6]
        
        # Önerileri hesapla
        oneriler = np.zeros(len(self.urunler))
        
        for benzer_idx in benzer_kullanicilar:
            benzerlik = kullanici_benzerlikleri[benzer_idx]
            kullanici_puanlari = self.kullanici_urun_matrisi.iloc[benzer_idx].values
            
            # Ağırlıklı puanları hesapla
            oneriler += benzerlik * kullanici_puanlari
            
        # Kullanıcının zaten değerlendirdiği ürünleri filtrele
        kullanici_puanlari = self.kullanici_urun_matrisi.iloc[kullanici_idx].values
        oneriler[kullanici_puanlari > 0] = 0
        
        # En yüksek puanlı ürünleri seç
        en_iyi_urunler = np.argsort(oneriler)[::-1][:n_oneri]
        
        return pd.DataFrame({
            'urun_id': self.urunler[en_iyi_urunler],
            'tahmini_puan': oneriler[en_iyi_urunler]
        })
    
    def benzer_urunler(self, urun_id, n_oneri=5):
        # Ürün benzerlik matrisini hesapla
        urun_benzerlikleri = cosine_similarity(
            self.kullanici_urun_matrisi.T
        )
        
        # Ürün indeksini bul
        urun_idx = list(self.urunler).index(urun_id)
        
        # En benzer ürünleri bul
        benzer_urunler = np.argsort(urun_benzerlikleri[urun_idx])[::-1][1:n_oneri+1]
        
        return pd.DataFrame({
            'urun_id': self.urunler[benzer_urunler],
            'benzerlik_skoru': urun_benzerlikleri[urun_idx][benzer_urunler]
        })

# Kullanım örneği
if __name__ == "__main__":
    # Örnek veri oluştur
    np.random.seed(42)
    
    kullanici_sayisi = 100
    urun_sayisi = 50
    degerlendirme_sayisi = 1000
    
    degerlendirmeler = pd.DataFrame({
        'kullanici_id': np.random.randint(1, kullanici_sayisi + 1, degerlendirme_sayisi),
        'urun_id': np.random.randint(1, urun_sayisi + 1, degerlendirme_sayisi),
        'puan': np.random.randint(1, 6, degerlendirme_sayisi)
    })
    
    # Öneri sistemi oluştur
    oneri_sistemi = OneriSistemi()
    oneri_sistemi.veri_yukle(degerlendirmeler)
    
    # Kullanıcı için öneriler
    kullanici_id = 1
    oneriler = oneri_sistemi.oneri_uret(kullanici_id)
    print(f"\\nKullanıcı {kullanici_id} için öneriler:")
    print(oneriler)
    
    # Benzer ürünler
    urun_id = 1
    benzer_urunler = oneri_sistemi.benzer_urunler(urun_id)
    print(f"\\nÜrün {urun_id} için benzer ürünler:")
    print(benzer_urunler)
\`\`\`

## Zaman Serisi Analizi Projesi

\`\`\`python
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class ZamanSerisiAnalizi:
    def __init__(self, veri, tarih_kolonu, hedef_kolonu):
        self.veri = veri.copy()
        self.veri[tarih_kolonu] = pd.to_datetime(self.veri[tarih_kolonu])
        self.veri.set_index(tarih_kolonu, inplace=True)
        self.hedef_kolonu = hedef_kolonu
        self.model = None
        
    def mevsimsellik_analizi(self):
        # Zaman serisi bileşenlerini ayrıştır
        dekompozisyon = seasonal_decompose(
            self.veri[self.hedef_kolonu],
            period=12  # Aylık veri için
        )
        
        # Grafikleri çiz
        plt.figure(figsize=(12, 10))
        
        plt.subplot(411)
        plt.plot(self.veri[self.hedef_kolonu])
        plt.title('Orijinal Veri')
        
        plt.subplot(412)
        plt.plot(dekompozisyon.trend)
        plt.title('Trend')
        
        plt.subplot(413)
        plt.plot(dekompozisyon.seasonal)
        plt.title('Mevsimsellik')
        
        plt.subplot(414)
        plt.plot(dekompozisyon.resid)
        plt.title('Artıklar')
        
        plt.tight_layout()
        plt.show()
        
        return dekompozisyon
        
    def model_egit(self, egitim_veri, test_veri, order=(1,1,1), seasonal_order=(1,1,1,12)):
        # SARIMA modeli oluştur ve eğit
        self.model = SARIMAX(
            egitim_veri[self.hedef_kolonu],
            order=order,
            seasonal_order=seasonal_order
        )
        
        self.sonuclar = self.model.fit()
        print(self.sonuclar.summary())
        
    def tahmin_yap(self, baslangic, bitis):
        # Tahmin yap
        tahminler = self.sonuclar.predict(
            start=baslangic,
            end=bitis
        )
        return tahminler
        
    def model_degerlendirme(self, gercek_degerler, tahminler):
        mae = mean_absolute_error(gercek_degerler, tahminler)
        rmse = np.sqrt(mean_squared_error(gercek_degerler, tahminler))
        
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        
        # Grafik çiz
        plt.figure(figsize=(12, 6))
        plt.plot(gercek_degerler.index, gercek_degerler, label='Gerçek')
        plt.plot(tahminler.index, tahminler, label='Tahmin')
        plt.title('Gerçek vs Tahmin')
        plt.legend()
        plt.show()
        
    def gelecek_tahmin(self, adim_sayisi=12):
        # Gelecek tahminleri yap
        tahminler = self.sonuclar.forecast(steps=adim_sayisi)
        
        # Grafik çiz
        plt.figure(figsize=(12, 6))
        plt.plot(self.veri.index, self.veri[self.hedef_kolonu], label='Geçmiş')
        plt.plot(tahminler.index, tahminler, label='Tahmin')
        plt.title('Gelecek Tahminleri')
        plt.legend()
        plt.show()
        
        return tahminler

# Kullanım örneği
if __name__ == "__main__":
    # Örnek veri oluştur
    np.random.seed(42)
    
    tarihler = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
    trend = np.linspace(100, 200, len(tarihler))
    mevsimsellik = 20 * np.sin(2 * np.pi * np.arange(len(tarihler)) / 12)
    rastgele = np.random.normal(0, 10, len(tarihler))
    
    veri = pd.DataFrame({
        'tarih': tarihler,
        'satis': trend + mevsimsellik + rastgele
    })
    
    # Analiz
    analiz = ZamanSerisiAnalizi(veri, 'tarih', 'satis')
    
    # Mevsimsellik analizi
    dekompozisyon = analiz.mevsimsellik_analizi()
    
    # Veriyi eğitim ve test olarak böl
    egitim_veri = veri[:'2023-06-30']
    test_veri = veri['2023-07-01':]
    
    # Model eğitimi
    analiz.model_egit(egitim_veri, test_veri)
    
    # Test verisi üzerinde tahmin
    tahminler = analiz.tahmin_yap('2023-07-01', '2023-12-31')
    
    # Model değerlendirme
    analiz.model_degerlendirme(test_veri['satis'], tahminler)
    
    # Gelecek tahminleri
    gelecek_tahminler = analiz.gelecek_tahmin(12)
    print("\\nGelecek 12 ay için tahminler:")
    print(gelecek_tahminler)
\`\`\`

## Alıştırmalar

1. **Müşteri Segmentasyonu**
   - Farklı kümeleme algoritmaları deneyin
   - RFM analizi ekleyin
   - Segment profillerini detaylandırın

2. **Öneri Sistemi**
   - İçerik tabanlı filtreleme ekleyin
   - Hibrit öneri sistemi geliştirin
   - Değerlendirme metriklerini genişletin

3. **Zaman Serisi**
   - Farklı mevsimsellik periyotları deneyin
   - Prophet modelini implemente edin
   - Çoklu değişken analizi yapın

## Sonraki Adımlar

1. [Derin Öğrenme Projeleri](/topics/python/veri-bilimi/derin-ogrenme-projeleri)
2. [MLOps ve DevOps](/topics/python/veri-bilimi/mlops)
3. [İleri Seviye Makine Öğrenmesi](/topics/python/veri-bilimi/ileri-makine-ogrenmesi)

## Faydalı Kaynaklar

- [Scikit-learn Dokümantasyonu](https://scikit-learn.org/stable/)
- [Pandas Dokümantasyonu](https://pandas.pydata.org/docs/)
- [Statsmodels Dokümantasyonu](https://www.statsmodels.org/stable/index.html)
`;

export default function DataScienceProjectsPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild className="mb-6">
            <Link href="/topics/python/veri-bilimi" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Veri Bilimi
            </Link>
          </Button>
        </div>

        <div className="prose prose-lg dark:prose-invert">
          <MarkdownContent content={content} />
        </div>

        <div className="mt-16 text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} Kodleon | Python Eğitim Platformu</p>
        </div>
      </div>
    </div>
  );
} 