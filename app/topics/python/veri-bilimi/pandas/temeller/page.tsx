import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Pandas Temelleri | Python Veri Bilimi | Kodleon',
  description: 'Pandas kütüphanesinin temel veri yapıları olan Series ve DataFrame ile veri analizi temellerini öğrenin.',
};

const content = `
# Pandas Temelleri

Pandas'ın temel veri yapıları ve işlemleri hakkında kapsamlı bir rehber.

## Series Veri Yapısı

Series, Pandas'ın tek boyutlu, etiketli dizi yapısıdır. NumPy dizilerine benzer, ancak etiketli indeksleme özelliği vardır.

### Series Oluşturma

\`\`\`python
import pandas as pd
import numpy as np

# Listeden Series oluşturma
s1 = pd.Series([1, 3, 5, 7, 9])
print("Basit Series:\\n", s1)

# Özel indekslerle Series oluşturma
s2 = pd.Series([1, 3, 5, 7], index=['a', 'b', 'c', 'd'])
print("\\nÖzel indeksli Series:\\n", s2)

# Sözlükten Series oluşturma
s3 = pd.Series({'a': 1, 'b': 3, 'c': 5})
print("\\nSözlükten Series:\\n", s3)

# NumPy dizisinden Series oluşturma
s4 = pd.Series(np.random.randn(5))
print("\\nNumPy dizisinden Series:\\n", s4)
\`\`\`

### Series Özellikleri

\`\`\`python
# Series özellikleri
print("Değerler:", s2.values)
print("İndeks:", s2.index)
print("Boyut:", s2.shape)
print("Veri tipi:", s2.dtype)
print("Boyut sayısı:", s2.ndim)
print("Eleman sayısı:", s2.size)
\`\`\`

### Series İşlemleri

\`\`\`python
# Eleman erişimi
print("İndeks ile erişim:", s2['a'])
print("Pozisyon ile erişim:", s2[0])
print("Dilim ile erişim:\\n", s2['a':'c'])

# Matematiksel işlemler
print("\\nToplam:", s2.sum())
print("Ortalama:", s2.mean())
print("Standart sapma:", s2.std())

# Filtreleme
print("\\n3'ten büyük elemanlar:\\n", s2[s2 > 3])

# Eksik veri kontrolü
print("\\nEksik veri var mı?\\n", s2.isnull())
\`\`\`

## DataFrame Veri Yapısı

DataFrame, Pandas'ın iki boyutlu, etiketli veri yapısıdır. Excel tablosuna benzer bir yapıya sahiptir.

### DataFrame Oluşturma

\`\`\`python
# Sözlükten DataFrame oluşturma
df1 = pd.DataFrame({
    'İsim': ['Ali', 'Ayşe', 'Mehmet', 'Zeynep'],
    'Yaş': [25, 30, 35, 28],
    'Şehir': ['İstanbul', 'Ankara', 'İzmir', 'Bursa']
})
print("Sözlükten DataFrame:\\n", df1)

# NumPy dizisinden DataFrame oluşturma
arr = np.random.randn(3, 4)
df2 = pd.DataFrame(arr, columns=['A', 'B', 'C', 'D'])
print("\\nNumPy dizisinden DataFrame:\\n", df2)

# Liste listesinden DataFrame oluşturma
data = [['Ali', 25], ['Ayşe', 30], ['Mehmet', 35]]
df3 = pd.DataFrame(data, columns=['İsim', 'Yaş'])
print("\\nListe listesinden DataFrame:\\n", df3)
\`\`\`

### DataFrame Özellikleri

\`\`\`python
# DataFrame bilgileri
print("Boyut:", df1.shape)
print("Sütunlar:", df1.columns)
print("İndeksler:", df1.index)
print("Veri tipleri:\\n", df1.dtypes)
print("\\nGenel bilgi:\\n")
df1.info()
print("\\nİstatistiksel özet:\\n")
print(df1.describe())
\`\`\`

### Veri Seçme ve Filtreleme

\`\`\`python
# Sütun seçme
print("Tek sütun:\\n", df1['İsim'])
print("\\nBirden fazla sütun:\\n", df1[['İsim', 'Yaş']])

# Satır seçme
print("\\nİlk 2 satır:\\n", df1.head(2))
print("\\nSon 2 satır:\\n", df1.tail(2))

# loc ve iloc ile seçme
print("\\nloc ile seçme:\\n", df1.loc[0:2, 'İsim':'Yaş'])
print("\\niloc ile seçme:\\n", df1.iloc[0:2, 0:2])

# Koşullu filtreleme
print("\\n30 yaşından büyükler:\\n", df1[df1['Yaş'] > 30])
\`\`\`

## Veri Okuma ve Yazma

Pandas, çeşitli dosya formatlarından veri okuma ve yazma işlemlerini destekler.

### CSV Dosyaları

\`\`\`python
# CSV dosyası okuma
df = pd.read_csv('veriler.csv')

# CSV dosyasına yazma
df.to_csv('yeni_veriler.csv', index=False)
\`\`\`

### Excel Dosyaları

\`\`\`python
# Excel dosyası okuma
df = pd.read_excel('veriler.xlsx')

# Excel dosyasına yazma
df.to_excel('yeni_veriler.xlsx', sheet_name='Sayfa1')
\`\`\`

### Diğer Formatlar

\`\`\`python
# JSON dosyası okuma/yazma
df = pd.read_json('veriler.json')
df.to_json('yeni_veriler.json')

# SQL veritabanından okuma
from sqlalchemy import create_engine
engine = create_engine('sqlite:///veritabani.db')
df = pd.read_sql('SELECT * FROM tablo', engine)
\`\`\`

## Temel İndeksleme

İndeksleme, Pandas'ta veri erişiminin temelidir.

### İndeks İşlemleri

\`\`\`python
# İndeks ayarlama
df.set_index('İsim', inplace=True)

# İndeksi sıfırlama
df.reset_index(inplace=True)

# Çoklu indeks
df.set_index(['İsim', 'Şehir'], inplace=True)
\`\`\`

## Alıştırmalar

1. **Series Alıştırmaları**
   - Farklı veri tipleriyle Series oluşturun
   - Series üzerinde matematiksel işlemler yapın
   - Series filtreleme işlemleri uygulayın

2. **DataFrame Alıştırmaları**
   - Kendi verilerinizle bir DataFrame oluşturun
   - Veri seçme ve filtreleme işlemleri yapın
   - DataFrame'i farklı formatlarda kaydedin

3. **Veri Okuma Alıştırmaları**
   - Bir CSV dosyası oluşturup okuyun
   - Excel dosyasından veri okuyun
   - Okunan verileri manipüle edin

## Sonraki Adımlar

1. [Veri Manipülasyonu](/topics/python/veri-bilimi/pandas/veri-manipulasyonu)
2. [Veri Analizi](/topics/python/veri-bilimi/pandas/veri-analizi)
3. [Veri Görselleştirme](/topics/python/veri-bilimi/pandas/veri-gorsellestirme)

## Faydalı Kaynaklar

- [Pandas Series Dokümantasyonu](https://pandas.pydata.org/docs/reference/series.html)
- [Pandas DataFrame Dokümantasyonu](https://pandas.pydata.org/docs/reference/frame.html)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
`;

export default function PandasBasicsPage() {
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