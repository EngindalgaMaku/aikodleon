import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'NumPy ile Lineer Cebir | Python Veri Bilimi | Kodleon',
  description: 'NumPy ile matris işlemleri, özdeğerler, vektör işlemleri ve diğer lineer cebir konularını öğrenin.',
};

const content = `
# NumPy ile Lineer Cebir

NumPy, lineer cebir işlemleri için güçlü araçlar sunar. Bu bölümde matris işlemleri, özdeğerler, vektör işlemleri ve diğer önemli lineer cebir konularını inceleyeceğiz.

## Matris Oluşturma

Temel matris oluşturma yöntemleri:

\`\`\`python
import numpy as np

# Basit matris oluşturma
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Özel matrisler
identity = np.eye(3)              # 3x3 birim matris
zeros = np.zeros((3, 3))         # 3x3 sıfır matris
ones = np.ones((2, 4))           # 2x4 birler matrisi
diagonal = np.diag([1, 2, 3])    # Köşegen matris

print("Birim Matris:\\n", identity)
print("\\nKöşegen Matris:\\n", diagonal)
\`\`\`

## Temel Matris İşlemleri

Matrisler üzerinde temel işlemler:

\`\`\`python
# İki matris tanımlayalım
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matris toplama ve çıkarma
print("Toplam:\\n", A + B)
print("\\nFark:\\n", A - B)

# Matris çarpımı
print("\\nMatris Çarpımı:\\n", np.dot(A, B))  # veya A @ B
print("\\nEleman-bazlı çarpım:\\n", A * B)

# Matris transpozu
print("\\nTranspoz:\\n", A.T)

# Matrisin tersi
inverse = np.linalg.inv(A)
print("\\nTers Matris:\\n", inverse)

# Doğrulama: A * A^(-1) = I
print("\\nDoğrulama (A * A^(-1)):\\n", np.dot(A, inverse))
\`\`\`

## Lineer Denklem Sistemleri

Lineer denklem sistemlerini çözme:

\`\`\`python
# Ax = b sistemini çözelim
# Örnek: 2x + y = 5
#        3x + 2y = 8

A = np.array([[2, 1],
              [3, 2]])
b = np.array([5, 8])

# Çözüm
x = np.linalg.solve(A, b)
print("Çözüm (x, y):", x)

# Doğrulama
print("Doğrulama (Ax = b):", np.allclose(np.dot(A, x), b))
\`\`\`

## Özdeğerler ve Özvektörler

Bir matrisin özdeğerlerini ve özvektörlerini bulma:

\`\`\`python
# Örnek matris
A = np.array([[4, -2],
              [1, 1]])

# Özdeğerler ve özvektörler
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Özdeğerler:", eigenvalues)
print("\\nÖzvektörler:\\n", eigenvectors)

# Doğrulama: Av = λv
for i in range(len(eigenvalues)):
    print(f"\\nÖzdeğer {i+1} için doğrulama:")
    print("Av =", np.dot(A, eigenvectors[:, i]))
    print("λv =", eigenvalues[i] * eigenvectors[:, i])
\`\`\`

## Matris Ayrıştırma

Matrislerin farklı ayrıştırma yöntemleri:

\`\`\`python
# Örnek matris
A = np.array([[1, 2], [3, 4]])

# LU ayrıştırma
from scipy.linalg import lu
P, L, U = lu(A)
print("LU Ayrıştırma:")
print("L:\\n", L)
print("U:\\n", U)

# QR ayrıştırma
Q, R = np.linalg.qr(A)
print("\\nQR Ayrıştırma:")
print("Q:\\n", Q)
print("R:\\n", R)

# Tekil Değer Ayrışımı (SVD)
U, s, Vh = np.linalg.svd(A)
print("\\nSVD:")
print("U:\\n", U)
print("Tekil değerler:", s)
print("V^H:\\n", Vh)
\`\`\`

## Vektör İşlemleri

Vektörler üzerinde temel işlemler:

\`\`\`python
# İki vektör tanımlayalım
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# İç çarpım (dot product)
dot_product = np.dot(v1, v2)
print("İç çarpım:", dot_product)

# Dış çarpım (cross product)
cross_product = np.cross(v1, v2)
print("\\nDış çarpım:", cross_product)

# Vektör normu
norm = np.linalg.norm(v1)
print("\\nVektör normu:", norm)

# Vektörü normalize etme
normalized = v1 / np.linalg.norm(v1)
print("\\nNormalize edilmiş vektör:", normalized)
print("Normalize vektör normu:", np.linalg.norm(normalized))  # ≈ 1.0
\`\`\`

## Matris Özellikleri

Matrislerin önemli özellikleri:

\`\`\`python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Determinant
det = np.linalg.det(A)
print("Determinant:", det)

# İz (trace)
trace = np.trace(A)
print("İz:", trace)

# Rank
rank = np.linalg.matrix_rank(A)
print("Rank:", rank)

# Kondisyon sayısı
cond = np.linalg.cond(A)
print("Kondisyon sayısı:", cond)

# Norm
norm = np.linalg.norm(A)
print("Frobenius normu:", norm)
\`\`\`

## Alıştırmalar

1. **Temel Matris İşlemleri**
   - 3x3 bir matris oluşturun ve tersini alın
   - Matrisin determinantını ve izini hesaplayın
   - Matrisin özdeğerlerini bulun

2. **Lineer Denklem Sistemleri**
   - 3 bilinmeyenli bir denklem sistemi oluşturun
   - Sistemi numpy.linalg.solve ile çözün
   - Çözümü doğrulayın

3. **Matris Ayrıştırma**
   - Bir matris için SVD uygulayın
   - Orijinal matrisi SVD bileşenlerinden yeniden oluşturun
   - QR ayrıştırması uygulayın ve sonuçları doğrulayın

## Sonraki Adımlar

1. [Rastgele Sayılar](/topics/python/veri-bilimi/numpy/rastgele)
2. [NumPy ile Görüntü İşleme](/topics/python/veri-bilimi/numpy/goruntu-isleme)
3. [Pandas ile Veri Analizi](/topics/python/veri-bilimi/pandas)

## Faydalı Kaynaklar

- [NumPy Linear Algebra Dokümantasyonu](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [SciPy Linear Algebra Dokümantasyonu](https://docs.scipy.org/doc/scipy/reference/linalg.html)
- [Linear Algebra and Its Applications](https://www.sciencedirect.com/journal/linear-algebra-and-its-applications)
`;

export default function NumPyLinearAlgebraPage() {
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