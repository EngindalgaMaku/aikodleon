import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'NumPy Dizilerde İndeksleme ve Dilimleme | Python Veri Bilimi | Kodleon',
  description: 'NumPy dizilerinde temel ve gelişmiş indeksleme teknikleri, dilimleme ve veri erişimi yöntemlerini öğrenin.',
};

const content = `
# NumPy Dizilerde İndeksleme ve Dilimleme

NumPy dizilerinde veri erişimi ve manipülasyonu için çeşitli indeksleme ve dilimleme yöntemleri bulunur. Bu yöntemler, dizilerin belirli elemanlarına veya alt kümelerine erişmenizi sağlar.

## Temel İndeksleme

Tek ve çok boyutlu dizilerde temel indeksleme işlemleri:

\`\`\`python
import numpy as np

# 1D dizi indeksleme
arr = np.array([1, 2, 3, 4, 5])
print("İlk eleman:", arr[0])           # 1
print("Son eleman:", arr[-1])          # 5
print("İkinci eleman:", arr[1])        # 2

# 2D dizi indeksleme
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print("\\n2D dizi:\\n", arr_2d)
print("İlk satır:", arr_2d[0])         # [1 2 3]
print("İlk satır, ikinci sütun:", arr_2d[0, 1])  # 2
print("Son satır:", arr_2d[-1])        # [7 8 9]

# 3D dizi indeksleme
arr_3d = np.array([[[1, 2], [3, 4]],
                   [[5, 6], [7, 8]]])

print("\\n3D dizi:\\n", arr_3d)
print("İlk matris:\\n", arr_3d[0])     # [[1 2] [3 4]]
print("İlk matris, ilk satır:", arr_3d[0, 0])  # [1 2]
print("İlk matris, ilk satır, ilk eleman:", arr_3d[0, 0, 0])  # 1
\`\`\`

## Dilimleme (Slicing)

Dilimleme, dizinin bir bölümünü seçmek için kullanılır. Sözdizimi: \`start:stop:step\`

\`\`\`python
# 1D dizi dilimleme
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print("İlk 5 eleman:", arr[:5])        # [0 1 2 3 4]
print("Son 5 eleman:", arr[5:])        # [5 6 7 8 9]
print("Ortadaki elemanlar:", arr[2:7]) # [2 3 4 5 6]
print("2'şer atlayarak:", arr[::2])    # [0 2 4 6 8]
print("Tersten:", arr[::-1])           # [9 8 7 6 5 4 3 2 1 0]

# 2D dizi dilimleme
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

print("\\n2D dizi:\\n", arr_2d)
print("İlk iki satır:\\n", arr_2d[:2])
print("İlk iki satır, son iki sütun:\\n", arr_2d[:2, 2:])
print("Tüm satırlar, çift sütunlar:\\n", arr_2d[:, ::2])
\`\`\`

## Boolean İndeksleme

Koşullu seçim için boolean maskeleme kullanılır:

\`\`\`python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Koşullu seçim
print("5'ten büyük elemanlar:", arr[arr > 5])  # [6 7 8 9]
print("Çift sayılar:", arr[arr % 2 == 0])      # [2 4 6 8]
print("3 ile 7 arasındaki sayılar:", arr[(arr >= 3) & (arr <= 7)])  # [3 4 5 6 7]

# 2D dizilerde boolean indeksleme
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print("\\n5'ten büyük elemanlar:\\n", arr_2d[arr_2d > 5])
print("Çift sayılar:\\n", arr_2d[arr_2d % 2 == 0])

# Boolean maske ile değer atama
arr_2d[arr_2d % 2 == 0] = 0  # Çift sayıları 0 yap
print("\\nÇift sayılar 0 yapıldıktan sonra:\\n", arr_2d)
\`\`\`

## Fancy İndeksleme

Dizin dizileri kullanarak indeksleme:

\`\`\`python
arr = np.array([10, 20, 30, 40, 50])

# İndeks dizisi ile seçim
indices = [1, 3, 4]
print("Seçilen elemanlar:", arr[indices])  # [20 40 50]

# 2D dizilerde fancy indeksleme
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

rows = [0, 2]
cols = [0, 2]
print("\\nSeçilen satır ve sütunlar:\\n", arr_2d[rows][:, cols])

# Sıralama indeksleri ile seçim
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
sorted_indices = np.argsort(arr)
print("\\nSıralanmış dizi:", arr[sorted_indices])
\`\`\`

## Görünüm vs. Kopya

İndeksleme ve dilimleme işlemlerinde görünüm ve kopya davranışları:

\`\`\`python
# Dilimleme ile görünüm
arr = np.array([1, 2, 3, 4, 5])
view = arr[1:4]  # Görünüm oluşturur
view[0] = 10     # Orijinal diziyi etkiler
print("Orijinal dizi:", arr)  # [1 10 3 4 5]

# Kopya oluşturma
copy = arr[1:4].copy()  # Kopya oluşturur
copy[0] = 20           # Orijinal diziyi etkilemez
print("Orijinal dizi:", arr)  # [1 10 3 4 5]
print("Kopya:", copy)         # [20 3 4]
\`\`\`

## Alıştırmalar

1. **Temel İndeksleme ve Dilimleme**
   - 3x4 boyutunda bir dizi oluşturun
   - İlk ve son satırını seçin
   - Her satırın çift indeksli sütunlarını seçin

2. **Boolean İndeksleme**
   - 1-100 arası sayılardan bir dizi oluşturun
   - 3'e bölünebilen sayıları seçin
   - 5'e bölünebilen ama 2'ye bölünemeyen sayıları seçin

3. **Fancy İndeksleme**
   - 5x5 rastgele bir matris oluşturun
   - Köşegen elemanlarını seçin
   - Belirli satır ve sütunları birleştirerek yeni bir matris oluşturun

## Sonraki Adımlar

1. [Matematiksel İşlemler](/topics/python/veri-bilimi/numpy/matematik)
2. [Lineer Cebir](/topics/python/veri-bilimi/numpy/lineer-cebir)
3. [Rastgele Sayılar](/topics/python/veri-bilimi/numpy/rastgele)

## Faydalı Kaynaklar

- [NumPy İndeksleme Dokümantasyonu](https://numpy.org/doc/stable/reference/arrays.indexing.html)
- [NumPy Dilimleme Örnekleri](https://numpy.org/doc/stable/user/basics.indexing.html)
- [Python Data Science Handbook - İndeksleme](https://jakevdp.github.io/PythonDataScienceHandbook/02.02-the-basics-of-numpy-arrays.html)
`;

export default function NumPyIndexingPage() {
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