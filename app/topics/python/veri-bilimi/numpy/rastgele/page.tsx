import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'NumPy ile Rastgele Sayılar | Python Veri Bilimi | Kodleon',
  description: 'NumPy ile rastgele sayı üretimi, olasılık dağılımları, rastgele örnekleme ve simülasyon tekniklerini öğrenin.',
};

const content = `
# NumPy ile Rastgele Sayılar

NumPy'ın rastgele sayı üretme modülü (numpy.random), veri bilimi ve simülasyon çalışmaları için güçlü araçlar sunar. Bu bölümde rastgele sayı üretimi, farklı dağılımlar ve örnekleme yöntemlerini inceleyeceğiz.

## Temel Rastgele Sayı Üretimi

Basit rastgele sayı üretme yöntemleri:

\`\`\`python
import numpy as np

# Rastgele sayı üreteci için seed belirleme
np.random.seed(42)  # Tekrar üretilebilirlik için

# 0-1 arası tekdüze dağılımlı rastgele sayılar
uniform = np.random.random(5)
print("Tekdüze dağılım (0-1):", uniform)

# Belirli bir aralıkta rastgele tam sayılar
integers = np.random.randint(1, 100, size=5)  # 1-100 arası
print("Rastgele tam sayılar:", integers)

# Belirli bir aralıkta tekdüze dağılımlı sayılar
uniform_range = np.random.uniform(0, 10, size=5)  # 0-10 arası
print("Tekdüze dağılım (0-10):", uniform_range)

# Normal dağılımlı rastgele sayılar
normal = np.random.normal(loc=0, scale=1, size=5)  # Ortalama=0, Std=1
print("Normal dağılım:", normal)
\`\`\`

## Olasılık Dağılımları

Farklı olasılık dağılımlarından örnekleme:

\`\`\`python
# Normal (Gaussian) dağılım
normal = np.random.normal(loc=0, scale=1, size=1000)

# Poisson dağılımı
poisson = np.random.poisson(lam=5, size=1000)  # Lambda=5

# Exponential dağılım
exponential = np.random.exponential(scale=1.0, size=1000)

# Binomial dağılım
binomial = np.random.binomial(n=10, p=0.5, size=1000)

# Beta dağılımı
beta = np.random.beta(a=2, b=5, size=1000)

# Temel istatistikler
print("Normal Dağılım - Ortalama:", np.mean(normal))
print("Poisson Dağılımı - Ortalama:", np.mean(poisson))
print("Exponential Dağılım - Ortalama:", np.mean(exponential))
print("Binomial Dağılım - Ortalama:", np.mean(binomial))
print("Beta Dağılımı - Ortalama:", np.mean(beta))
\`\`\`

## Rastgele Örnekleme

Veri setlerinden rastgele örnekleme yöntemleri:

\`\`\`python
# Örnek veri seti
data = np.arange(100)

# Basit rastgele örnekleme
sample = np.random.choice(data, size=10, replace=False)
print("Rastgele örneklem:", sample)

# Ağırlıklı rastgele örnekleme
weights = np.linspace(1, 10, 100)  # Artan ağırlıklar
weighted_sample = np.random.choice(data, size=10, p=weights/np.sum(weights))
print("Ağırlıklı örneklem:", weighted_sample)

# Karıştırma (Shuffle)
shuffled = data.copy()
np.random.shuffle(shuffled)
print("Karıştırılmış dizi:", shuffled[:10])  # İlk 10 eleman

# Permütasyon
permuted = np.random.permutation(10)
print("Permütasyon:", permuted)
\`\`\`

## Çok Boyutlu Rastgele Diziler

Matris ve tensor şeklinde rastgele sayılar:

\`\`\`python
# 2D rastgele matris (normal dağılım)
matrix_normal = np.random.normal(0, 1, size=(3, 4))
print("2D Normal Dağılım:\\n", matrix_normal)

# 3D rastgele tensor (tekdüze dağılım)
tensor_uniform = np.random.uniform(0, 1, size=(2, 3, 2))
print("\\n3D Tekdüze Dağılım:\\n", tensor_uniform)

# Rastgele binary matris
binary_matrix = np.random.choice([0, 1], size=(4, 4), p=[0.7, 0.3])
print("\\nRastgele Binary Matris:\\n", binary_matrix)
\`\`\`

## Rastgele Sayı Üreteci Kontrolü

Rastgele sayı üretecinin yönetimi:

\`\`\`python
# Seed belirleme
np.random.seed(42)
print("İlk üretim:", np.random.random(3))

# Aynı seed ile tekrar
np.random.seed(42)
print("Aynı seed ile:", np.random.random(3))

# RandomState nesnesi kullanımı
rng = np.random.RandomState(42)
print("RandomState ile:", rng.random(3))

# Generator sınıfı (modern yaklaşım)
rng = np.random.default_rng(42)
print("Generator ile:", rng.random(3))
\`\`\`

## Monte Carlo Simülasyonu

Basit bir Monte Carlo simülasyonu örneği:

\`\`\`python
# Pi sayısını Monte Carlo yöntemi ile tahmin etme
def estimate_pi(n_points):
    # Birim kare içinde rastgele noktalar üret
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    
    # Birim çember içindeki noktaları say
    inside_circle = np.sum(x**2 + y**2 <= 1)
    
    # Pi tahminini hesapla
    pi_estimate = 4 * inside_circle / n_points
    return pi_estimate

# Farklı örnek sayıları ile pi tahmini
for n in [1000, 10000, 100000]:
    pi_estimate = estimate_pi(n)
    print(f"{n} nokta ile Pi tahmini: {pi_estimate}")
\`\`\`

## Alıştırmalar

1. **Temel Rastgele Sayı Üretimi**
   - Farklı dağılımlardan 1000'er örnek üretin
   - Örneklerin histogramlarını çizin
   - Ortalama ve standart sapmalarını hesaplayın

2. **Örnekleme ve Simülasyon**
   - Bir zar atma simülasyonu yapın
   - Yazı-tura atma simülasyonu yapın
   - Sonuçların olasılık dağılımını inceleyin

3. **Monte Carlo Uygulamaları**
   - Belirli bir integrali Monte Carlo yöntemi ile hesaplayın
   - Sonuçları analitik çözümle karşılaştırın
   - Örnek sayısının sonuç üzerindeki etkisini inceleyin

## Sonraki Adımlar

1. [NumPy ile Görüntü İşleme](/topics/python/veri-bilimi/numpy/goruntu-isleme)
2. [Pandas ile Veri Analizi](/topics/python/veri-bilimi/pandas)
3. [Matplotlib ile Veri Görselleştirme](/topics/python/veri-bilimi/matplotlib)

## Faydalı Kaynaklar

- [NumPy Random Dokümantasyonu](https://numpy.org/doc/stable/reference/random/index.html)
- [SciPy İstatistik Dokümantasyonu](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Monte Carlo Simülasyonları](https://en.wikipedia.org/wiki/Monte_Carlo_method)
`;

export default function NumPyRandomPage() {
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