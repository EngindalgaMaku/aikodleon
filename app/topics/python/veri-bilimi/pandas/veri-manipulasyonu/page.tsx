import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Pandas ile Veri Manipülasyonu | Python Veri Bilimi | Kodleon',
  description: 'Pandas kütüphanesi ile veri temizleme, dönüştürme, gruplama ve birleştirme işlemlerini öğrenin.',
};

const content = `
# Pandas ile Veri Manipülasyonu

Veri analizi sürecinde en önemli adımlardan biri veri manipülasyonudur. Bu bölümde, Pandas ile veri temizleme, dönüştürme ve yeniden şekillendirme tekniklerini öğreneceğiz.

## Veri Temizleme

### Eksik Veriler (Missing Values)

\`\`\`python
import pandas as pd
import numpy as np

# Örnek veri seti
df = pd.DataFrame({
    'A': [1, np.nan, 3, np.nan, 5],
    'B': [np.nan, 2, 3, 4, 5],
    'C': [1, 2, np.nan, 4, 5]
})

# Eksik verileri tespit etme
print("Eksik veriler:\\n", df.isnull())
print("\\nEksik veri sayısı:\\n", df.isnull().sum())

# Eksik verileri doldurma
print("\\nSabit değerle doldurma:\\n", df.fillna(0))
print("\\nİleri yönlü doldurma:\\n", df.fillna(method='ffill'))
print("\\nGeri yönlü doldurma:\\n", df.fillna(method='bfill'))
print("\\nSütun ortalamasıyla doldurma:\\n", df.fillna(df.mean()))

# Eksik verileri silme
print("\\nEksik veri içeren satırları silme:\\n", df.dropna())
print("\\nTüm değerleri eksik olan satırları silme:\\n", df.dropna(how='all'))
print("\\nBelirli bir sayıdan az geçerli veri içeren satırları silme:\\n", 
      df.dropna(thresh=2))
\`\`\`

### Yinelenen Veriler (Duplicates)

\`\`\`python
# Örnek veri
df = pd.DataFrame({
    'İsim': ['Ali', 'Ayşe', 'Ali', 'Mehmet', 'Ayşe'],
    'Yaş': [25, 30, 25, 35, 30]
})

# Yinelenen verileri tespit etme
print("Yinelenen satırlar:\\n", df.duplicated())

# Yinelenen satırları silme
print("\\nTekil satırlar:\\n", df.drop_duplicates())

# Belirli sütunlara göre yinelenen satırları silme
print("\\nİsme göre tekil satırlar:\\n", 
      df.drop_duplicates(subset=['İsim']))
\`\`\`

### Aykırı Değerler (Outliers)

\`\`\`python
# Örnek veri
df = pd.DataFrame({
    'Değer': [1, 2, 3, 100, 4, 5, -50, 6, 7, 8]
})

# İstatistiksel özet
print("İstatistikler:\\n", df.describe())

# IQR yöntemi ile aykırı değer tespiti
Q1 = df['Değer'].quantile(0.25)
Q3 = df['Değer'].quantile(0.75)
IQR = Q3 - Q1
alt_sinir = Q1 - 1.5 * IQR
ust_sinir = Q3 + 1.5 * IQR

# Aykırı değerleri filtreleme
normal_degerler = df[(df['Değer'] >= alt_sinir) & 
                    (df['Değer'] <= ust_sinir)]
print("\\nAykırı değerler filtrelenmiş:\\n", normal_degerler)
\`\`\`

## Veri Dönüştürme

### Veri Tipi Dönüşümleri

\`\`\`python
# Örnek veri
df = pd.DataFrame({
    'A': ['1', '2', '3'],
    'B': [1.1, 2.2, 3.3],
    'C': ['True', 'False', 'True']
})

# Veri tiplerini görüntüleme
print("Veri tipleri:\\n", df.dtypes)

# Veri tipi dönüşümleri
df['A'] = df['A'].astype(int)
df['B'] = df['B'].astype(float)
df['C'] = df['C'].astype(bool)

print("\\nDönüştürülmüş veri tipleri:\\n", df.dtypes)
\`\`\`

### Kategorik Veri Dönüşümleri

\`\`\`python
# Örnek kategorik veri
df = pd.DataFrame({
    'Kategori': ['A', 'B', 'C', 'A', 'B'],
    'Seviye': ['Düşük', 'Orta', 'Yüksek', 'Orta', 'Düşük']
})

# One-Hot Encoding
print("One-Hot Encoding:\\n", pd.get_dummies(df))

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Seviye_Encoded'] = le.fit_transform(df['Seviye'])
print("\\nLabel Encoding:\\n", df)
\`\`\`

### Veri Normalizasyonu

\`\`\`python
# Örnek sayısal veri
df = pd.DataFrame({
    'Değer': [10, 20, 30, 40, 50]
})

# Min-Max Normalizasyonu
df['Normalized_MinMax'] = (df['Değer'] - df['Değer'].min()) / \
                         (df['Değer'].max() - df['Değer'].min())

# Z-Score Normalizasyonu
df['Normalized_ZScore'] = (df['Değer'] - df['Değer'].mean()) / \
                         df['Değer'].std()

print("Normalizasyon sonuçları:\\n", df)
\`\`\`

## Gruplama ve Birleştirme

### Gruplama İşlemleri

\`\`\`python
# Örnek veri
df = pd.DataFrame({
    'Kategori': ['A', 'B', 'A', 'B', 'A'],
    'Alt_Kategori': ['X', 'X', 'Y', 'Y', 'X'],
    'Değer': [1, 2, 3, 4, 5]
})

# Basit gruplama
print("Kategoriye göre ortalama:\\n", 
      df.groupby('Kategori')['Değer'].mean())

# Çoklu gruplama
print("\\nKategori ve Alt_Kategoriye göre:\\n",
      df.groupby(['Kategori', 'Alt_Kategori']).agg({
          'Değer': ['count', 'mean', 'sum']
      }))

# Gruplama ve pivot
pivot_table = pd.pivot_table(df, 
                            values='Değer',
                            index='Kategori',
                            columns='Alt_Kategori',
                            aggfunc='mean')
print("\\nPivot tablo:\\n", pivot_table)
\`\`\`

### Birleştirme İşlemleri

\`\`\`python
# Örnek veriler
df1 = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'İsim': ['Ali', 'Ayşe', 'Mehmet', 'Zeynep']
})

df2 = pd.DataFrame({
    'ID': [1, 2, 3, 5],
    'Yaş': [25, 30, 35, 40]
})

# Merge ile birleştirme
print("Inner Join:\\n", pd.merge(df1, df2, on='ID'))
print("\\nLeft Join:\\n", pd.merge(df1, df2, on='ID', how='left'))
print("\\nRight Join:\\n", pd.merge(df1, df2, on='ID', how='right'))
print("\\nOuter Join:\\n", pd.merge(df1, df2, on='ID', how='outer'))

# Concat ile birleştirme
print("\\nDikey birleştirme:\\n", pd.concat([df1, df1]))
print("\\nYatay birleştirme:\\n", pd.concat([df1, df2], axis=1))
\`\`\`

## Veri Yeniden Şekillendirme

### Pivot ve Melt

\`\`\`python
# Örnek veri
df = pd.DataFrame({
    'Tarih': ['2023-01', '2023-01', '2023-02', '2023-02'],
    'Ürün': ['A', 'B', 'A', 'B'],
    'Satış': [100, 200, 150, 250]
})

# Pivot
pivot_df = df.pivot(index='Tarih', 
                   columns='Ürün', 
                   values='Satış')
print("Pivot:\\n", pivot_df)

# Melt (Unpivot)
melted_df = pivot_df.reset_index().melt(
    id_vars=['Tarih'],
    var_name='Ürün',
    value_name='Satış'
)
print("\\nMelt:\\n", melted_df)
\`\`\`

### Stack ve Unstack

\`\`\`python
# Çok seviyeli indeks örneği
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8]
},
index=pd.MultiIndex.from_tuples([
    ('X', 1), ('X', 2), ('Y', 1), ('Y', 2)
]))

print("Orijinal:\\n", df)
print("\\nStack:\\n", df.stack())
print("\\nUnstack:\\n", df.unstack())
\`\`\`

## Alıştırmalar

1. **Veri Temizleme**
   - Eksik verileri farklı yöntemlerle doldurun
   - Yinelenen verileri tespit edip temizleyin
   - Aykırı değerleri belirleyip filtreyin

2. **Veri Dönüştürme**
   - Kategorik verileri sayısallaştırın
   - Sayısal verileri normalize edin
   - Veri tiplerini dönüştürün

3. **Gruplama ve Birleştirme**
   - Farklı gruplama işlemleri uygulayın
   - Merge ve concat işlemlerini deneyin
   - Pivot tablolar oluşturun

## Sonraki Adımlar

1. [Veri Analizi](/topics/python/veri-bilimi/pandas/veri-analizi)
2. [Veri Görselleştirme](/topics/python/veri-bilimi/pandas/veri-gorsellestirme)
3. [Pratik Uygulamalar](/topics/python/veri-bilimi/pandas/uygulamalar)

## Faydalı Kaynaklar

- [Pandas Veri Manipülasyonu](https://pandas.pydata.org/docs/user_guide/reshaping.html)
- [Pandas Merge ve Join](https://pandas.pydata.org/docs/user_guide/merging.html)
- [Pandas Gruplama](https://pandas.pydata.org/docs/user_guide/groupby.html)
`;

export default function PandasDataManipulationPage() {
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