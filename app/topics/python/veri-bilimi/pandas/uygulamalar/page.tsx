import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Pandas ile Pratik Uygulamalar | Python Veri Bilimi | Kodleon',
  description: 'Pandas kütüphanesi ile gerçek dünya veri analizi örnekleri ve örnek çalışmalar.',
};

const content = `
# Pandas ile Pratik Uygulamalar

Bu bölümde, Pandas kütüphanesinin gerçek dünya problemlerinde nasıl kullanıldığını öğreneceğiz. Her örnek, veri bilimi sürecinin farklı aşamalarını içermektedir.

## Finansal Veri Analizi

### Borsa Verisi Analizi

\`\`\`python
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Borsa verisi çekme
bist = yf.download('XU100.IS', start='2023-01-01', end='2023-12-31')

# Günlük getiri hesaplama
bist['Günlük_Getiri'] = bist['Close'].pct_change()

# Aylık ortalama getiri
aylik_getiri = bist['Günlük_Getiri'].resample('M').mean()

# Volatilite hesaplama
bist['Volatilite'] = bist['Günlük_Getiri'].rolling(window=21).std()

# Görselleştirme
plt.figure(figsize=(12, 6))
plt.plot(bist.index, bist['Close'])
plt.title('BIST 100 Endeksi')
plt.xlabel('Tarih')
plt.ylabel('Kapanış Fiyatı')
plt.grid(True)
plt.show()

# İstatistiksel özet
print("İstatistiksel Özet:")
print(bist['Günlük_Getiri'].describe())
\`\`\`

### Portföy Analizi

\`\`\`python
# Örnek hisse senetleri
hisseler = ['THYAO.IS', 'EREGL.IS', 'GARAN.IS']
portfoy = pd.DataFrame()

for hisse in hisseler:
    veri = yf.download(hisse, start='2023-01-01', end='2023-12-31')['Close']
    portfoy[hisse] = veri

# Normalize edilmiş değerler
normalize_portfoy = portfoy / portfoy.iloc[0] * 100

# Korelasyon analizi
plt.figure(figsize=(8, 6))
sns.heatmap(portfoy.corr(), annot=True, cmap='coolwarm')
plt.title('Hisse Senetleri Korelasyon Matrisi')
plt.show()
\`\`\`

## Müşteri Analizi

### Müşteri Segmentasyonu

\`\`\`python
# Örnek müşteri verisi
musteri_data = pd.DataFrame({
    'MusteriID': range(1000),
    'Yas': np.random.normal(45, 15, 1000),
    'Gelir': np.random.normal(50000, 20000, 1000),
    'AlisverisFrekansi': np.random.poisson(10, 1000),
    'OrtalamaHarcama': np.random.normal(200, 50, 1000)
})

# Veri ön işleme
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
olcekli_veri = scaler.fit_transform(musteri_data.drop('MusteriID', axis=1))

# K-means kümeleme
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
musteri_data['Segment'] = kmeans.fit_predict(olcekli_veri)

# Segment analizi
segment_ozet = musteri_data.groupby('Segment').agg({
    'Yas': 'mean',
    'Gelir': 'mean',
    'AlisverisFrekansi': 'mean',
    'OrtalamaHarcama': 'mean'
}).round(2)

print("Segment Özeti:")
print(segment_ozet)
\`\`\`

## E-Ticaret Analizi

### Satış Trendleri

\`\`\`python
# Örnek e-ticaret verisi
np.random.seed(42)
tarihler = pd.date_range('2023-01-01', '2023-12-31')
satis_data = pd.DataFrame({
    'Tarih': tarihler,
    'Satis': np.random.normal(1000, 200, len(tarihler)) + \
             np.sin(np.linspace(0, 4*np.pi, len(tarihler))) * 100,
    'Kategori': np.random.choice(['Elektronik', 'Giyim', 'Kitap'], len(tarihler))
})

# Günlük satış analizi
gunluk_satis = satis_data.groupby(['Tarih', 'Kategori'])['Satis'].sum().unstack()

# Hareketli ortalama
plt.figure(figsize=(12, 6))
for kategori in gunluk_satis.columns:
    plt.plot(gunluk_satis.index, 
             gunluk_satis[kategori].rolling(7).mean(), 
             label=kategori)

plt.title('Kategori Bazında 7 Günlük Ortalama Satışlar')
plt.xlabel('Tarih')
plt.ylabel('Satış Miktarı')
plt.legend()
plt.grid(True)
plt.show()
\`\`\`

### Sepet Analizi

\`\`\`python
# Örnek sepet verisi
urunler = ['Laptop', 'Mouse', 'Klavye', 'Monitor', 'Kulaklık']
sepet_data = pd.DataFrame({
    'SiparisID': np.repeat(range(500), 3),
    'Urun': np.random.choice(urunler, 1500)
})

# Ürün birliktelik analizi
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Pivot tablo oluşturma
sepet_pivot = pd.crosstab(sepet_data['SiparisID'], sepet_data['Urun'])

# Apriori algoritması
frequent_itemsets = apriori(sepet_pivot, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

print("Ürün Birliktelik Kuralları:")
print(rules.sort_values('lift', ascending=False).head())
\`\`\`

## Sosyal Medya Analizi

### Hashtag Analizi

\`\`\`python
# Örnek sosyal medya verisi
hashtag_data = pd.DataFrame({
    'Tarih': pd.date_range('2023-01-01', periods=1000, freq='H'),
    'Hashtag': np.random.choice(['#python', '#datascience', '#ai', '#ml'], 1000),
    'Kullanim': np.random.poisson(100, 1000)
})

# Saatlik trend analizi
saatlik_trend = hashtag_data.groupby(['Hashtag', 
                                    hashtag_data['Tarih'].dt.hour])['Kullanim'].mean()
saatlik_trend = saatlik_trend.unstack()

# Heatmap görselleştirme
plt.figure(figsize=(12, 6))
sns.heatmap(saatlik_trend, cmap='YlOrRd', annot=True, fmt='.0f')
plt.title('Hashtag Kullanım Yoğunluğu (Saatlik)')
plt.xlabel('Saat')
plt.ylabel('Hashtag')
plt.show()
\`\`\`

## Sağlık Verisi Analizi

### COVID-19 Veri Analizi

\`\`\`python
# Örnek COVID-19 verisi
covid_data = pd.DataFrame({
    'Tarih': pd.date_range('2023-01-01', '2023-12-31'),
    'YeniVaka': np.random.poisson(1000, 365) * \
                np.exp(-np.linspace(0, 2, 365)),
    'Iyilesen': np.random.poisson(900, 365) * \
                np.exp(-np.linspace(0, 2, 365)),
    'Test': np.random.normal(50000, 5000, 365)
})

# Pozitiflik oranı hesaplama
covid_data['PozitiflikOrani'] = (covid_data['YeniVaka'] / covid_data['Test']) * 100

# 7 günlük ortalama
plt.figure(figsize=(12, 6))
plt.plot(covid_data['Tarih'], 
         covid_data['YeniVaka'].rolling(7).mean(), 
         label='Yeni Vaka')
plt.plot(covid_data['Tarih'], 
         covid_data['Iyilesen'].rolling(7).mean(), 
         label='İyileşen')
plt.title('COVID-19 Vaka Trendi (7 Günlük Ortalama)')
plt.xlabel('Tarih')
plt.ylabel('Vaka Sayısı')
plt.legend()
plt.grid(True)
plt.show()
\`\`\`

## Alıştırmalar

1. **Finansal Analiz**
   - Farklı hisse senetleri için teknik göstergeler hesaplayın
   - Risk-getiri analizi yapın
   - Portföy optimizasyonu gerçekleştirin

2. **Müşteri Analizi**
   - RFM (Recency, Frequency, Monetary) analizi yapın
   - Churn (müşteri kaybı) tahmini geliştirin
   - Müşteri yaşam boyu değeri hesaplayın

3. **E-Ticaret**
   - Sezonsal satış analizi yapın
   - Ürün önerisi sistemi geliştirin
   - Stok optimizasyonu yapın

## Sonraki Adımlar

1. [Makine Öğrenmesi](/topics/python/veri-bilimi/makine-ogrenmesi)
2. [Derin Öğrenme](/topics/python/veri-bilimi/derin-ogrenme)
3. [Büyük Veri](/topics/python/veri-bilimi/buyuk-veri)

## Faydalı Kaynaklar

- [Kaggle Veri Setleri](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Google Dataset Search](https://datasetsearch.research.google.com/)
`;

export default function PandasPracticalApplicationsPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild className="mb-6">
            <Link href="/topics/python/veri-bilimi/pandas" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Pandas
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