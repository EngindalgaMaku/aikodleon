import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'Pandas ile Veri Analizi | Python Veri Bilimi | Kodleon',
  description: 'Python Pandas kütüphanesi ile veri analizi, veri manipülasyonu ve veri temizleme tekniklerini öğrenin.',
};

const learningPath = [
  {
    title: "Pandas Temelleri",
    description: "Pandas'ın temel veri yapıları olan Series ve DataFrame'leri öğrenin.",
    topics: ["Series ve DataFrame", "Veri Okuma/Yazma", "Temel İşlemler", "İndeksleme"],
    icon: "📊",
    href: "/topics/python/veri-bilimi/pandas/temeller"
  },
  {
    title: "Veri Manipülasyonu",
    description: "Veri setlerini temizleme, dönüştürme ve yeniden şekillendirme teknikleri.",
    topics: ["Veri Temizleme", "Dönüşümler", "Gruplama", "Birleştirme"],
    icon: "🔄",
    href: "/topics/python/veri-bilimi/pandas/veri-manipulasyonu"
  },
  {
    title: "Veri Analizi",
    description: "Veri setlerini analiz etme, özetleme ve istatistiksel hesaplamalar yapma.",
    topics: ["İstatistiksel Analiz", "Pivot Tablolar", "Zaman Serileri", "Korelasyon"],
    icon: "📈",
    href: "/topics/python/veri-bilimi/pandas/veri-analizi"
  },
  {
    title: "Veri Görselleştirme",
    description: "Pandas ve Matplotlib ile veri görselleştirme teknikleri.",
    topics: ["Temel Grafikler", "İstatistiksel Grafikler", "Zaman Serisi Grafikleri"],
    icon: "📉",
    href: "/topics/python/veri-bilimi/pandas/veri-gorsellestirme"
  },
  {
    title: "Pratik Uygulamalar",
    description: "Gerçek dünya veri setleri üzerinde uygulama örnekleri.",
    topics: ["Veri Keşfi", "Veri Temizleme", "Analiz ve Raporlama"],
    icon: "💡",
    href: "/topics/python/veri-bilimi/pandas/uygulamalar"
  }
];

const content = `
# Pandas ile Veri Analizi

Pandas, Python'da veri manipülasyonu ve analizi için en popüler kütüphanelerden biridir. Yüksek performanslı, kullanımı kolay veri yapıları ve veri analizi araçları sunar.

## Pandas Nedir?

Pandas, veri bilimi ve veri analizi için tasarlanmış güçlü bir Python kütüphanesidir. Özellikle:

- Büyük veri setlerini etkili bir şekilde işleme
- Farklı formatlardaki verileri okuma ve yazma
- Veri temizleme ve hazırlama
- Veri birleştirme ve gruplama
- Zaman serisi analizi
- İstatistiksel hesaplamalar

gibi işlemleri kolaylaştırır.

## Neden Pandas?

1. **Veri Yapıları**
   - Series (1-boyutlu dizi)
   - DataFrame (2-boyutlu tablo)
   - Panel (3-boyutlu dizi)

2. **Veri İşleme Özellikleri**
   - Eksik veri yönetimi
   - Veri birleştirme ve şekillendirme
   - Gruplama ve pivot tablolar
   - Zaman serisi işlemleri

3. **Veri Analizi Araçları**
   - İstatistiksel fonksiyonlar
   - Veri filtreleme ve seçme
   - Veri dönüşümleri
   - Görselleştirme araçları

4. **Veri Formatları Desteği**
   - CSV ve metin dosyaları
   - Excel dosyaları
   - SQL veritabanları
   - JSON ve HTML
   - HDF5 Format

## Temel Kavramlar

\`\`\`python
import pandas as pd
import numpy as np

# Series oluşturma
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print("Series:\\n", s)

# DataFrame oluşturma
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': ['a', 'b', 'c', 'd'],
    'C': [1.0, 2.0, 3.0, 4.0]
})
print("\\nDataFrame:\\n", df)
\`\`\`

## Pandas'ın Avantajları

1. **Veri Analizi**
   - Hızlı ve etkili veri işleme
   - Güçlü veri filtreleme
   - Gelişmiş gruplama işlemleri

2. **Veri Temizleme**
   - Eksik veri yönetimi
   - Yinelenen veri tespiti
   - Veri doğrulama ve dönüştürme

3. **Veri Entegrasyonu**
   - Farklı veri kaynaklarıyla uyum
   - Veri birleştirme ve birleşim
   - Veri dışa aktarma

4. **Performans**
   - Optimize edilmiş C kodu
   - Büyük veri setleri için uygun
   - Hızlı hesaplama yetenekleri

## Öğrenme Yolu

1. **Temel Kavramlar**
   - Series ve DataFrame yapıları
   - Veri okuma ve yazma
   - Temel veri işlemleri

2. **Veri Manipülasyonu**
   - Veri seçme ve filtreleme
   - Veri temizleme
   - Veri dönüştürme

3. **Veri Analizi**
   - İstatistiksel hesaplamalar
   - Gruplama ve agregasyon
   - Pivot tablolar

4. **İleri Düzey Konular**
   - Zaman serisi analizi
   - Kategorik veri işleme
   - Performans optimizasyonu

## Başlarken

Pandas'ı kullanmaya başlamak için:

\`\`\`python
# Pandas'ı yükleyin
pip install pandas

# Pandas'ı içe aktarın
import pandas as pd

# İlk DataFrame'inizi oluşturun
df = pd.DataFrame({
    'İsim': ['Ali', 'Ayşe', 'Mehmet'],
    'Yaş': [25, 30, 35],
    'Şehir': ['İstanbul', 'Ankara', 'İzmir']
})

print(df)
\`\`\`

## Alıştırmalar

1. **Temel İşlemler**
   - Farklı veri tiplerinde Series oluşturun
   - Basit bir DataFrame oluşturun
   - Veri seçme ve filtreleme işlemleri yapın

2. **Veri Okuma**
   - CSV dosyası okuyun
   - Excel dosyası okuyun
   - Verileri farklı formatlarda kaydedin

3. **Veri Manipülasyonu**
   - Eksik verileri temizleyin
   - Sütun ekleme ve çıkarma
   - Veri tiplerini dönüştürün

## Faydalı Kaynaklar

- [Pandas Resmi Dokümantasyonu](https://pandas.pydata.org/docs/)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
`;

export default function PandasPage() {
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

        <div className="grid gap-6 mb-8">
          {learningPath.map((item, index) => (
            <Link key={index} href={item.href}>
              <Card className="p-6 hover:bg-muted cursor-pointer transition-colors">
                <div className="flex items-start gap-4">
                  <div className="text-4xl">{item.icon}</div>
                  <div>
                    <h3 className="text-xl font-semibold mb-2">{item.title}</h3>
                    <p className="text-muted-foreground mb-4">{item.description}</p>
                    <div className="flex flex-wrap gap-2">
                      {item.topics.map((topic, i) => (
                        <span key={i} className="bg-primary/10 text-primary px-2 py-1 rounded text-sm">
                          {topic}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </Card>
            </Link>
          ))}
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