import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Pandas ile Veri Analizi | Python Veri Bilimi | Kodleon',
  description: 'Pandas kütüphanesi ile istatistiksel analiz, zaman serisi analizi ve korelasyon analizi tekniklerini öğrenin.',
};

const content = `
# Pandas ile Veri Analizi

Pandas, veri analizi için güçlü araçlar sunar. Bu bölümde, temel istatistiksel analizden ileri düzey zaman serisi analizine kadar çeşitli teknikleri öğreneceğiz.

## Temel İstatistiksel Analiz

### Tanımlayıcı İstatistikler

\`\`\`python
import pandas as pd
import numpy as np

# Örnek veri seti
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
})

# Temel istatistikler
print("Temel istatistikler:\\n", df.describe())

# Özel istatistikler
print("\\nOrtalama:\\n", df.mean())
print("\\nMedyan:\\n", df.median())
print("\\nStandart sapma:\\n", df.std())
print("\\nVaryans:\\n", df.var())
print("\\nMinimum:\\n", df.min())
print("\\nMaksimum:\\n", df.max())
print("\\nÇeyrekler:\\n", df.quantile([0.25, 0.5, 0.75]))
\`\`\`

### Frekans ve Mod Analizi

\`\`\`python
# Kategorik veri örneği
df = pd.DataFrame({
    'Kategori': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'A']
})

# Frekans analizi
print("Frekans dağılımı:\\n", df['Kategori'].value_counts())
print("\\nGöreli frekans:\\n", df['Kategori'].value_counts(normalize=True))

# Mod (en sık görülen değer)
print("\\nMod:\\n", df['Kategori'].mode())
\`\`\`

### Dağılım Analizi

\`\`\`python
# Sayısal veri örneği
df = pd.DataFrame({
    'Değer': np.random.normal(100, 15, 1000)
})

# Dağılım istatistikleri
print("Çarpıklık:", df['Değer'].skew())
print("Basıklık:", df['Değer'].kurtosis())

# Histogram bilgileri
hist_values = np.histogram(df['Değer'], bins=30)
print("\\nHistogram değerleri:\\n", hist_values)
\`\`\`

## Korelasyon Analizi

### Pearson Korelasyonu

\`\`\`python
# Örnek veri
df = pd.DataFrame({
    'X': np.random.normal(0, 1, 100),
    'Y': np.random.normal(0, 1, 100),
    'Z': np.random.normal(0, 1, 100)
})

# X ile Y arasında pozitif korelasyon oluştur
df['Y'] = df['X'] * 0.7 + df['Y'] * 0.3

# Korelasyon matrisi
print("Korelasyon matrisi:\\n", df.corr())

# Tek bir korelasyon
print("\\nX ve Y arasındaki korelasyon:", 
      df['X'].corr(df['Y']))
\`\`\`

### Spearman ve Kendall Korelasyonları

\`\`\`python
# Sıralı veri örneği
df = pd.DataFrame({
    'Sınav1': [70, 85, 90, 65, 75],
    'Sınav2': [75, 80, 85, 70, 80]
})

# Farklı korelasyon yöntemleri
print("Pearson korelasyonu:", 
      df['Sınav1'].corr(df['Sınav2'], method='pearson'))
print("Spearman korelasyonu:", 
      df['Sınav1'].corr(df['Sınav2'], method='spearman'))
print("Kendall korelasyonu:", 
      df['Sınav1'].corr(df['Sınav2'], method='kendall'))
\`\`\`

## Zaman Serisi Analizi

### Zaman Serisi Oluşturma

\`\`\`python
# Tarih aralığı oluşturma
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.random.normal(100, 10, len(dates))

# Zaman serisi oluşturma
ts = pd.Series(values, index=dates)
print("Zaman serisi:\\n", ts.head())

# Temel özellikler
print("\\nZaman indeksi özellikleri:")
print("Yıl:", ts.index.year)
print("Ay:", ts.index.month)
print("Gün:", ts.index.day)
print("Haftanın günü:", ts.index.dayofweek)
\`\`\`

### Yeniden Örnekleme ve Pencere İşlemleri

\`\`\`python
# Yeniden örnekleme (Resampling)
print("Aylık ortalama:\\n", ts.resample('M').mean())
print("\\nHaftalık toplam:\\n", ts.resample('W').sum())

# Hareketli ortalama
print("\\n7 günlük hareketli ortalama:\\n", 
      ts.rolling(window=7).mean().head(10))

# Genişleyen pencere
print("\\nGenişleyen pencere ortalaması:\\n", 
      ts.expanding().mean().head(10))
\`\`\`

### Mevsimsellik ve Trend Analizi

\`\`\`python
# Mevsimsel veri örneği
seasonal_data = pd.Series(
    np.sin(np.linspace(0, 4*np.pi, 365)) * 10 + \
    np.random.normal(0, 1, 365),
    index=pd.date_range('2023-01-01', periods=365)
)

# Trend bileşeni
trend = seasonal_data.rolling(window=30).mean()

# Mevsimsel bileşen
seasonal = seasonal_data - trend

print("Orijinal veri:\\n", seasonal_data.head())
print("\\nTrend:\\n", trend.head())
print("\\nMevsimsel bileşen:\\n", seasonal.head())
\`\`\`

## Pivot Tablolar ve Çapraz Tablolar

### Pivot Tablo Analizi

\`\`\`python
# Örnek satış verisi
sales_data = pd.DataFrame({
    'Tarih': pd.date_range('2023-01-01', periods=100),
    'Ürün': np.random.choice(['A', 'B', 'C'], 100),
    'Bölge': np.random.choice(['Kuzey', 'Güney', 'Doğu', 'Batı'], 100),
    'Satış': np.random.randint(100, 1000, 100)
})

# Pivot tablo oluşturma
pivot = pd.pivot_table(
    sales_data,
    values='Satış',
    index='Bölge',
    columns='Ürün',
    aggfunc=['mean', 'sum', 'count']
)

print("Pivot tablo:\\n", pivot)
\`\`\`

### Çapraz Tablo Analizi

\`\`\`python
# Kategorik veri örneği
survey_data = pd.DataFrame({
    'Cinsiyet': np.random.choice(['Erkek', 'Kadın'], 100),
    'Yaş_Grubu': np.random.choice(['18-25', '26-35', '36-50', '50+'], 100),
    'Tercih': np.random.choice(['Evet', 'Hayır'], 100)
})

# Çapraz tablo oluşturma
cross_tab = pd.crosstab(
    [survey_data['Cinsiyet'], survey_data['Yaş_Grubu']],
    survey_data['Tercih'],
    margins=True
)

print("Çapraz tablo:\\n", cross_tab)
\`\`\`

## İleri Düzey Analiz Teknikleri

### Gruplara Göre İstatistiksel Testler

\`\`\`python
from scipy import stats

# Örnek veri
group_data = pd.DataFrame({
    'Grup': np.repeat(['A', 'B'], 50),
    'Değer': np.concatenate([
        np.random.normal(100, 10, 50),  # Grup A
        np.random.normal(105, 10, 50)   # Grup B
    ])
})

# Gruplar arası t-testi
group_a = group_data[group_data['Grup'] == 'A']['Değer']
group_b = group_data[group_data['Grup'] == 'B']['Değer']
t_stat, p_value = stats.ttest_ind(group_a, group_b)

print("T-test sonuçları:")
print("t-istatistiği:", t_stat)
print("p-değeri:", p_value)
\`\`\`

### Regresyon Analizi

\`\`\`python
from sklearn.linear_model import LinearRegression

# Örnek veri
X = np.random.normal(0, 1, (100, 1))
y = 2 * X + np.random.normal(0, 0.5, (100, 1))

# Regresyon modeli
model = LinearRegression()
model.fit(X, y)

print("Regresyon katsayısı:", model.coef_[0][0])
print("Kesişim:", model.intercept_[0])
print("R-kare skoru:", model.score(X, y))
\`\`\`

## Alıştırmalar

1. **Temel İstatistikler**
   - Bir veri seti üzerinde tanımlayıcı istatistikler hesaplayın
   - Farklı veri türleri için uygun istatistiksel ölçümleri seçin
   - Sonuçları yorumlayın

2. **Korelasyon Analizi**
   - İki değişken arasındaki ilişkiyi inceleyin
   - Farklı korelasyon yöntemlerini karşılaştırın
   - Korelasyon matrisini görselleştirin

3. **Zaman Serisi**
   - Günlük verilerden aylık özetler oluşturun
   - Trend ve mevsimsellik analizi yapın
   - Hareketli ortalamalar hesaplayın

## Sonraki Adımlar

1. [Veri Görselleştirme](/topics/python/veri-bilimi/pandas/veri-gorsellestirme)
2. [Pratik Uygulamalar](/topics/python/veri-bilimi/pandas/uygulamalar)
3. [Makine Öğrenmesi](/topics/python/veri-bilimi/makine-ogrenmesi)

## Faydalı Kaynaklar

- [Pandas İstatistiksel Fonksiyonlar](https://pandas.pydata.org/docs/user_guide/computation.html)
- [Zaman Serisi Analizi](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [SciPy İstatistik Kütüphanesi](https://docs.scipy.org/doc/scipy/reference/stats.html)
`;

export default function PandasDataAnalysisPage() {
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