import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'NumPy Temelleri | Python Veri Bilimi | Kodleon',
  description: 'NumPy dizileri, temel özellikler, veri tipleri ve dizi işlemleri hakkında detaylı bilgi edinin.',
};

const content = `
# NumPy Temelleri

NumPy, bilimsel hesaplama için Python'ın temel kütüphanesidir. En önemli özelliği, çok boyutlu diziler (ndarray) ve bu diziler üzerinde yüksek performanslı işlemler yapabilmesidir.

## NumPy Kurulumu

NumPy'ı pip kullanarak kurabilirsiniz:

\`\`\`bash
pip install numpy
\`\`\`

## NumPy Dizileri (ndarray)

NumPy'ın temel veri yapısı ndarray'dir. Python listelerinden farklı olarak, homojen veri tipine sahip ve sabit boyutludur.

### Dizi Oluşturma

\`\`\`python
import numpy as np

# Liste kullanarak dizi oluşturma
arr1 = np.array([1, 2, 3, 4, 5])
print("1D array:", arr1)

# Çok boyutlu dizi oluşturma
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print("2D array:\\n", arr2)

# Özel diziler oluşturma
zeros = np.zeros((3, 3))      # Sıfırlardan oluşan 3x3 matris
ones = np.ones((2, 4))        # Birlerden oluşan 2x4 matris
empty = np.empty((2, 2))      # Boş 2x2 matris
arange = np.arange(0, 10, 2)  # 0'dan 10'a kadar 2'şer artarak
linspace = np.linspace(0, 1, 5) # 0 ile 1 arasında 5 eşit aralıklı sayı

print("\\nZeros:\\n", zeros)
print("\\nOnes:\\n", ones)
print("\\nArange:", arange)
print("\\nLinspace:", linspace)
\`\`\`

### Temel Dizi Özellikleri

\`\`\`python
# Örnek bir dizi oluşturalım
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Dizi özellikleri
print("Boyut (Shape):", arr.shape)       # (2, 3)
print("Boyut sayısı:", arr.ndim)         # 2
print("Eleman sayısı:", arr.size)        # 6
print("Veri tipi:", arr.dtype)           # int64
print("Eleman boyutu:", arr.itemsize)    # 8 (bytes)
\`\`\`

## Veri Tipleri ve Dönüşümler

NumPy, farklı veri tipleri sunar ve bunlar arasında dönüşüm yapabilirsiniz.

\`\`\`python
# Farklı veri tipleriyle dizi oluşturma
int_arr = np.array([1, 2, 3], dtype=np.int32)
float_arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
bool_arr = np.array([True, False, True])

# Veri tipi dönüşümleri
float_to_int = float_arr.astype(np.int32)
int_to_float = int_arr.astype(np.float64)

print("Integer dizi:", int_arr.dtype)
print("Float dizi:", float_arr.dtype)
print("Boolean dizi:", bool_arr.dtype)
print("Float -> Int:", float_to_int)
print("Int -> Float:", int_to_float)
\`\`\`

## Dizi Şekillendirme (Reshape)

Dizilerin boyutlarını ve şeklini değiştirebilirsiniz.

\`\`\`python
# Örnek dizi
arr = np.arange(12)
print("Orijinal dizi:", arr)

# Yeniden şekillendirme
reshaped = arr.reshape(3, 4)  # 3x4 matris
print("\\nYeniden şekillendirilmiş (3,4):\\n", reshaped)

# -1 kullanımı: otomatik boyut hesaplama
auto_reshaped = arr.reshape(2, -1)  # 2 satır, sütun sayısı otomatik
print("\\nOtomatik şekillendirme (2,-1):\\n", auto_reshaped)

# Düzleştirme
flattened = reshaped.flatten()  # 1D diziye dönüştürme
print("\\nDüzleştirilmiş:", flattened)
\`\`\`

## Dizi Kopyalama ve Görünüm

NumPy'da dizileri kopyalarken dikkat edilmesi gereken önemli noktalar vardır.

\`\`\`python
# Orijinal dizi
arr = np.array([1, 2, 3, 4])

# Görünüm (View) - orijinal diziyi etkiler
view = arr.view()
view[0] = 10
print("Görünüm sonrası orijinal:", arr)  # [10, 2, 3, 4]

# Kopya (Copy) - bağımsız yeni dizi
copy = arr.copy()
copy[0] = 100
print("Kopya sonrası orijinal:", arr)    # [10, 2, 3, 4]
print("Kopya:", copy)                     # [100, 2, 3, 4]
\`\`\`

## Alıştırmalar

1. **Temel Dizi İşlemleri**
   - 1'den 20'ye kadar olan sayılardan bir dizi oluşturun
   - Bu diziyi 4x5'lik bir matrise dönüştürün
   - Matrisin boyutlarını, eleman sayısını ve veri tipini yazdırın

2. **Veri Tipi Dönüşümleri**
   - Ondalıklı sayılardan oluşan bir dizi oluşturun
   - Bu diziyi tam sayılara dönüştürün
   - Dönüşüm sırasında veri kaybını gözlemleyin

3. **Özel Diziler**
   - 3x3 boyutunda bir birim matris oluşturun
   - 0 ile 1 arasında rastgele sayılardan oluşan 4x4 matris oluşturun
   - Bu matrisleri birleştirerek yeni bir matris oluşturun

## Sonraki Adımlar

1. [Dizilerde İndeksleme ve Dilimleme](/topics/python/veri-bilimi/numpy/indeksleme)
2. [Matematiksel İşlemler](/topics/python/veri-bilimi/numpy/matematik)
3. [Lineer Cebir](/topics/python/veri-bilimi/numpy/lineer-cebir)

## Faydalı Kaynaklar

- [NumPy Resmi Dokümantasyonu](https://numpy.org/doc/stable/)
- [NumPy Cheat Sheet](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
`;

export default function NumPyBasicsPage() {
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