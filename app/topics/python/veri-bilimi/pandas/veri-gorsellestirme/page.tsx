import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Pandas ile Veri Görselleştirme | Python Veri Bilimi | Kodleon',
  description: 'Pandas, Matplotlib ve Seaborn kütüphaneleri ile veri görselleştirme tekniklerini öğrenin.',
};

const content = `
# Pandas ile Veri Görselleştirme

Veri görselleştirme, veri analizi sürecinin önemli bir parçasıdır. Pandas, Matplotlib ve Seaborn kütüphaneleri ile etkili ve güzel görselleştirmeler oluşturmayı öğreneceğiz.

## Temel Görselleştirmeler

### Çizgi Grafikleri (Line Plots)

\`\`\`python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Stil ayarları
plt.style.use('seaborn')
sns.set_palette("husl")

# Örnek zaman serisi verisi
dates = pd.date_range('2023-01-01', periods=100)
df = pd.DataFrame({
    'Tarih': dates,
    'Değer1': np.random.randn(100).cumsum(),
    'Değer2': np.random.randn(100).cumsum()
})

# Basit çizgi grafiği
plt.figure(figsize=(10, 6))
plt.plot(df['Tarih'], df['Değer1'], label='Seri 1')
plt.plot(df['Tarih'], df['Değer2'], label='Seri 2')
plt.title('Zaman Serisi Grafiği')
plt.xlabel('Tarih')
plt.ylabel('Değer')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pandas ile doğrudan çizim
df.set_index('Tarih').plot(figsize=(10, 6))
plt.title('Pandas ile Çizgi Grafiği')
plt.show()
\`\`\`

### Sütun Grafikleri (Bar Plots)

\`\`\`python
# Örnek kategori verisi
kategori_data = pd.DataFrame({
    'Kategori': ['A', 'B', 'C', 'D', 'E'],
    'Değer': [23, 45, 56, 78, 32]
})

# Dikey sütun grafiği
plt.figure(figsize=(8, 6))
plt.bar(kategori_data['Kategori'], kategori_data['Değer'])
plt.title('Kategorilere Göre Değerler')
plt.xlabel('Kategori')
plt.ylabel('Değer')
plt.show()

# Yatay sütun grafiği
plt.figure(figsize=(8, 6))
plt.barh(kategori_data['Kategori'], kategori_data['Değer'])
plt.title('Yatay Sütun Grafiği')
plt.xlabel('Değer')
plt.ylabel('Kategori')
plt.show()
\`\`\`

### Pasta Grafikleri (Pie Charts)

\`\`\`python
# Pasta grafiği verisi
pasta_data = pd.Series([30, 20, 25, 15, 10], 
                      index=['A', 'B', 'C', 'D', 'E'])

# Pasta grafiği
plt.figure(figsize=(8, 8))
plt.pie(pasta_data, labels=pasta_data.index, autopct='%1.1f%%')
plt.title('Kategorilerin Dağılımı')
plt.show()
\`\`\`

## İstatistiksel Görselleştirmeler

### Histogram ve Yoğunluk Grafikleri

\`\`\`python
# Örnek sayısal veri
sayisal_data = pd.DataFrame({
    'Değer': np.random.normal(100, 15, 1000)
})

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(sayisal_data['Değer'], bins=30, density=True, alpha=0.7)
sns.kdeplot(data=sayisal_data['Değer'], color='red')
plt.title('Histogram ve Yoğunluk Grafiği')
plt.xlabel('Değer')
plt.ylabel('Sıklık')
plt.show()
\`\`\`

### Kutu Grafikleri (Box Plots)

\`\`\`python
# Örnek grup verisi
grup_data = pd.DataFrame({
    'Grup': np.repeat(['A', 'B', 'C'], 100),
    'Değer': np.concatenate([
        np.random.normal(100, 10, 100),
        np.random.normal(90, 15, 100),
        np.random.normal(110, 12, 100)
    ])
})

# Kutu grafiği
plt.figure(figsize=(8, 6))
sns.boxplot(x='Grup', y='Değer', data=grup_data)
plt.title('Gruplara Göre Değer Dağılımı')
plt.show()
\`\`\`

### Violin Plots

\`\`\`python
# Violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(x='Grup', y='Değer', data=grup_data)
plt.title('Gruplara Göre Değer Dağılımı (Violin Plot)')
plt.show()
\`\`\`

## İlişkisel Görselleştirmeler

### Scatter Plots

\`\`\`python
# İlişkisel veri örneği
iliski_data = pd.DataFrame({
    'X': np.random.normal(0, 1, 100),
    'Y': np.random.normal(0, 1, 100)
})
iliski_data['Y'] = iliski_data['X'] * 0.7 + iliski_data['Y'] * 0.3

# Scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(iliski_data['X'], iliski_data['Y'], alpha=0.5)
plt.title('X ve Y Arasındaki İlişki')
plt.xlabel('X Değeri')
plt.ylabel('Y Değeri')
plt.show()

# Seaborn ile regresyon çizgisi
sns.lmplot(x='X', y='Y', data=iliski_data, height=8)
plt.title('Regresyon Çizgisi ile Scatter Plot')
plt.show()
\`\`\`

### Korelasyon Matrisi Görselleştirmesi

\`\`\`python
# Çok değişkenli veri
cok_degiskenli = pd.DataFrame(
    np.random.randn(100, 4),
    columns=['A', 'B', 'C', 'D']
)

# Korelasyon matrisi
plt.figure(figsize=(8, 8))
sns.heatmap(cok_degiskenli.corr(), annot=True, cmap='coolwarm')
plt.title('Korelasyon Matrisi')
plt.show()
\`\`\`

## Gelişmiş Görselleştirmeler

### Facet Plots

\`\`\`python
# Çok kategorili veri
facet_data = pd.DataFrame({
    'x': np.random.normal(0, 1, 300),
    'y': np.random.normal(0, 1, 300),
    'kategori1': np.repeat(['A', 'B', 'C'], 100),
    'kategori2': np.tile(['X', 'Y'], 150)
})

# Facet plot
g = sns.FacetGrid(facet_data, col='kategori1', row='kategori2', height=4)
g.map(plt.scatter, 'x', 'y')
plt.show()
\`\`\`

### Joint Plots

\`\`\`python
# Joint plot
sns.jointplot(
    data=iliski_data,
    x='X', y='Y',
    kind='reg',
    height=8
)
plt.show()
\`\`\`

### Pair Plots

\`\`\`python
# Pair plot
sns.pairplot(cok_degiskenli)
plt.show()
\`\`\`

## Görselleştirme İpuçları

### Stil ve Renk Paletleri

\`\`\`python
# Farklı stil örnekleri
stil_ornekleri = ['seaborn', 'seaborn-darkgrid', 'seaborn-whitegrid', 
                 'seaborn-dark', 'seaborn-white']

for stil in stil_ornekleri:
    plt.style.use(stil)
    plt.figure(figsize=(6, 4))
    plt.plot(np.random.randn(100).cumsum())
    plt.title(f'Stil: {stil}')
    plt.show()

# Renk paleti örnekleri
palet_ornekleri = ['husl', 'muted', 'deep', 'pastel', 'bright']

for palet in palet_ornekleri:
    sns.set_palette(palet)
    plt.figure(figsize=(6, 4))
    for i in range(5):
        plt.plot(np.random.randn(100).cumsum(), label=f'Seri {i+1}')
    plt.title(f'Palet: {palet}')
    plt.legend()
    plt.show()
\`\`\`

### Grafik Özelleştirme

\`\`\`python
# Özelleştirilmiş grafik örneği
plt.figure(figsize=(10, 6))
plt.plot(df['Tarih'], df['Değer1'], 
         color='#2ecc71', 
         linewidth=2, 
         linestyle='--',
         marker='o',
         markersize=6,
         label='Seri 1')

plt.title('Özelleştirilmiş Grafik', fontsize=14, pad=20)
plt.xlabel('Tarih', fontsize=12)
plt.ylabel('Değer', fontsize=12)
plt.grid(True, linestyle=':')
plt.legend(loc='best', frameon=True, shadow=True)
plt.xticks(rotation=45)

# Arka plan ve kenarlık ayarları
plt.gca().set_facecolor('#f8f9fa')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
\`\`\`

## Alıştırmalar

1. **Temel Grafikler**
   - Farklı veri setleri için çizgi grafikleri oluşturun
   - Kategorik veriler için sütun grafikleri oluşturun
   - Dağılım grafikleri çizin

2. **İstatistiksel Grafikler**
   - Bir veri setinin dağılımını görselleştirin
   - Kutu ve violin grafikleri ile grup karşılaştırmaları yapın
   - Korelasyon matrisini görselleştirin

3. **Özelleştirme**
   - Farklı stil ve renk paletlerini deneyin
   - Grafiklere başlık, eksen etiketleri ve lejant ekleyin
   - Grafik boyutlarını ve yerleşimini ayarlayın

## Sonraki Adımlar

1. [Pratik Uygulamalar](/topics/python/veri-bilimi/pandas/uygulamalar)
2. [Makine Öğrenmesi](/topics/python/veri-bilimi/makine-ogrenmesi)
3. [Derin Öğrenme](/topics/python/veri-bilimi/derin-ogrenme)

## Faydalı Kaynaklar

- [Matplotlib Dokümantasyonu](https://matplotlib.org/stable/tutorials/index.html)
- [Seaborn Dokümantasyonu](https://seaborn.pydata.org/tutorial.html)
- [Pandas Görselleştirme](https://pandas.pydata.org/docs/user_guide/visualization.html)
`;

export default function PandasDataVisualizationPage() {
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