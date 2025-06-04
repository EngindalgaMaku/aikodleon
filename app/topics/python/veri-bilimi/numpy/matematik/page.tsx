import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'NumPy Matematiksel İşlemler | Python Veri Bilimi | Kodleon',
  description: 'NumPy ile temel matematiksel işlemler, trigonometrik fonksiyonlar, istatistiksel hesaplamalar ve daha fazlasını öğrenin.',
};

const content = `
# NumPy Matematiksel İşlemler

NumPy, bilimsel hesaplamalar için güçlü matematiksel işlemler sunar. Bu işlemler hem tek elemanlı hem de çok boyutlu diziler üzerinde hızlı ve verimli bir şekilde gerçekleştirilebilir.

## Temel Aritmetik İşlemler

NumPy dizileri üzerinde temel matematiksel işlemler:

\`\`\`python
import numpy as np

# Örnek diziler
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Temel işlemler
print("Toplama:", a + b)          # [6 8 10 12]
print("Çıkarma:", b - a)          # [4 4 4 4]
print("Çarpma:", a * b)           # [5 12 21 32]
print("Bölme:", b / a)            # [5. 3. 2.33 2.]
print("Üs alma:", a ** 2)         # [1 4 9 16]
print("Kalan:", b % a)            # [0 0 1 0]

# Diziler ile skaler işlemler
print("\\nSkaler toplama:", a + 2)  # [3 4 5 6]
print("Skaler çarpma:", a * 3)    # [3 6 9 12]
print("Skaler üs:", a ** 2)       # [1 4 9 16]

# Universal fonksiyonlar (ufunc)
print("\\nKarekök:", np.sqrt(a))    # [1. 1.41 1.73 2.]
print("Üstel:", np.exp(a))        # [2.72 7.39 20.09 54.60]
print("Logaritma:", np.log(a))    # [0. 0.69 1.10 1.39]
\`\`\`

## Trigonometrik Fonksiyonlar

NumPy'ın trigonometrik fonksiyonları:

\`\`\`python
# Açılar (radyan cinsinden)
angles = np.array([0, np.pi/2, np.pi])

# Temel trigonometrik fonksiyonlar
print("Sinüs:", np.sin(angles))     # [0. 1. 0.]
print("Kosinüs:", np.cos(angles))   # [1. 0. -1.]
print("Tanjant:", np.tan(angles))   # [0. inf 0.]

# Ters trigonometrik fonksiyonlar
values = np.array([-1, 0, 1])
print("\\nArcsin:", np.arcsin(values))  # [-pi/2 0. pi/2]
print("Arccos:", np.arccos(values))  # [pi pi/2 0.]
print("Arctan:", np.arctan(values))  # [-pi/4 0. pi/4]

# Derece-Radyan dönüşümleri
degrees = np.array([0, 90, 180, 270, 360])
print("\\nDerece -> Radyan:", np.deg2rad(degrees))
print("Radyan -> Derece:", np.rad2deg(angles))
\`\`\`

## İstatistiksel İşlemler

Diziler üzerinde istatistiksel hesaplamalar:

\`\`\`python
# Örnek veri
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Temel istatistikler
print("Ortalama:", np.mean(data))           # 5.0
print("Medyan:", np.median(data))           # 5.0
print("Standart sapma:", np.std(data))      # 2.582
print("Varyans:", np.var(data))             # 6.667
print("Minimum:", np.min(data))             # 1
print("Maksimum:", np.max(data))            # 9

# Eksen boyunca istatistikler
print("\\nSatır ortalamaları:", np.mean(data, axis=1))  # [2. 5. 8.]
print("Sütun ortalamaları:", np.mean(data, axis=0))     # [4. 5. 6.]

# Kümülatif işlemler
print("\\nKümülatif toplam:\\n", np.cumsum(data))
print("Kümülatif çarpım:\\n", np.cumprod(data))
\`\`\`

## Yuvarlama İşlemleri

Sayısal değerleri yuvarlama:

\`\`\`python
# Örnek dizi
x = np.array([1.1, 2.5, 3.7, 4.2, 5.9])

# Yuvarlama işlemleri
print("Yukarı yuvarlama:", np.ceil(x))    # [2. 3. 4. 5. 6.]
print("Aşağı yuvarlama:", np.floor(x))    # [1. 2. 3. 4. 5.]
print("En yakına yuvarlama:", np.round(x)) # [1. 2. 4. 4. 6.]
print("Kesirli kısmı atma:", np.trunc(x))  # [1. 2. 3. 4. 5.]

# Ondalık hassasiyet
y = np.array([1.2345, 2.3456, 3.4567])
print("\\n2 ondalık basamak:", np.round(y, decimals=2))  # [1.23 2.35 3.46]
\`\`\`

## Özel Matematiksel İşlemler

Diğer önemli matematiksel işlemler:

\`\`\`python
# Mutlak değer
x = np.array([-1, -2, 3, -4])
print("Mutlak değer:", np.abs(x))  # [1 2 3 4]

# İşaret fonksiyonu
print("İşaret:", np.sign(x))       # [-1 -1 1 -1]

# Mod alma
a = np.array([10, 20, 30])
b = 3
print("\\nMod alma:", np.mod(a, b))  # [1 2 0]

# En büyük ortak bölen (GCD)
x = np.array([12, 18, 24])
y = np.array([15, 27, 36])
print("EBOB:", np.gcd(x, y))       # [3 9 12]

# En küçük ortak kat (LCM)
print("EKOK:", np.lcm(x, y))       # [60 54 72]
\`\`\`

## Alıştırmalar

1. **Temel İşlemler**
   - İki farklı boyutta dizi oluşturun ve broadcasting kurallarını gözlemleyin
   - Diziler üzerinde karmaşık matematiksel ifadeler oluşturun
   - Sonuçları farklı veri tipleriyle karşılaştırın

2. **İstatistiksel Hesaplamalar**
   - Rastgele sayılardan oluşan bir veri seti oluşturun
   - Temel istatistiksel ölçümleri hesaplayın
   - Sonuçları farklı eksenler üzerinde karşılaştırın

3. **Trigonometrik İşlemler**
   - Sinüs ve kosinüs grafiği için veri oluşturun
   - Açı dönüşümlerini uygulayın
   - Ters trigonometrik fonksiyonları kullanın

## Sonraki Adımlar

1. [Lineer Cebir](/topics/python/veri-bilimi/numpy/lineer-cebir)
2. [Rastgele Sayılar](/topics/python/veri-bilimi/numpy/rastgele)
3. [NumPy ile Görüntü İşleme](/topics/python/veri-bilimi/numpy/goruntu-isleme)

## Faydalı Kaynaklar

- [NumPy Matematiksel Fonksiyonlar](https://numpy.org/doc/stable/reference/routines.math.html)
- [NumPy İstatistiksel Fonksiyonlar](https://numpy.org/doc/stable/reference/routines.statistics.html)
- [SciPy İstatistiksel Fonksiyonlar](https://docs.scipy.org/doc/scipy/reference/stats.html)
`;

export default function NumPyMathPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild className="mb-6">
            <Link href="/topics/python/veri-bilimi/numpy" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              NumPy
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