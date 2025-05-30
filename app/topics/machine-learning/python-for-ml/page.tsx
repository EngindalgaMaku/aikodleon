import { Metadata } from 'next';
import { createPageMetadata } from '@/lib/seo';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';

export const metadata: Metadata = createPageMetadata({
  title: 'Makine Öğrenmesi için Python Programlama',
  description: "Makine öğrenmesi projeleriniz için gerekli Python programlama becerilerini ve kütüphanelerini öğrenin.",
  path: '/topics/machine-learning/python-for-ml',
  keywords: ['python makine öğrenmesi', 'python ml', 'numpy', 'pandas', 'scikit-learn', 'makine öğrenmesi kütüphaneleri', 'kodleon', 'türkçe ai eğitimi'],
});

export default function PythonForMlPage() {
  return (
    <div className="container max-w-6xl mx-auto py-12">
      <div className="mb-8">
        <Button asChild variant="ghost" size="sm" className="gap-1">
          <Link href="/topics/machine-learning" aria-label="Makine Öğrenmesi konusuna geri dön">
            <ArrowLeft className="h-4 w-4" aria-hidden="true" />
            Makine Öğrenmesi
          </Link>
        </Button>
      </div>
      <h1 className="text-4xl font-bold mb-6">Makine Öğrenmesi için Python Programlama</h1>
      <p className="text-xl text-muted-foreground mb-8">
        Makine öğrenmesi dünyasına adım atarken Python programlama dili en güçlü aracınız olacaktır. Bu bölümde, neden Python'ın ML için popüler bir seçim olduğunu, temel kavramları ve projeniz için ihtiyaç duyacağınız kritik kütüphaneleri öğreneceksiniz.
      </p>

      <div className="prose prose-lg dark:prose-invert max-w-none">
        <h2>Neden Python?</h2>
        <p>
          Python, basit sözdizimi, geniş kütüphane ekosistemi ve büyük topluluk desteği sayesinde makine öğrenmesi ve veri bilimi alanında standart bir dil haline gelmiştir. Öğrenmesi kolaydır ve hızlı prototipleme imkanı sunar. NumPy, Pandas, Matplotlib gibi veri işleme ve analiz kütüphaneleri ile Scikit-learn, TensorFlow ve PyTorch gibi güçlü ML/derin öğrenme kütüphaneleri, Python'ı bu alanda vazgeçilmez kılar.
        </p>

        <h2>Temel Kütüphaneler</h2>
        <ul>
          <li>
            <strong>NumPy:</strong> Sayısal hesaplamalar için temel paket. Özellikle çok boyutlu diziler (ndarray) ve matrisler üzerinde hızlı ve etkin işlemler yapmak için kullanılır. Vektörizasyon sayesinde döngülerden kaçınılarak performans artışı sağlanır. Makine öğrenmesi algoritmalarının çoğu NumPy dizileri üzerinde çalışır.
          </li>
          <li>
            <strong>Pandas:</strong> Veri manipülasyonu, temizliği ve analizi için kullanılır. `DataFrame` ve `Series` gibi veri yapıları, tablo halindeki (CSV, Excel vb.) verilerle kolayca çalışmayı sağlar. Eksik veri işleme, filtreleme, gruplama gibi işlemler Pandas ile rahatlıkla yapılabilir.
          </li>
          <li>
            <strong>Matplotlib & Seaborn:</strong> Veri görselleştirme için kullanılan popüler kütüphaneler. `Matplotlib` temel grafikler (çizgi grafik, histogram, dağılım grafiği vb.) oluşturmak için esnek bir API sunar. `Seaborn` ise Matplotlib üzerine kurulu, daha estetik ve istatistiksel grafikler oluşturmayı kolaylaştıran üst düzey bir kütüphanedir. Veriyi anlamak ve model sonuçlarını sunmak için görselleştirme kritiktir.
          </li>
          <li>
            <strong>Scikit-learn:</strong> Klasik makine öğrenmesi algoritmaları (regresyon, sınıflandırma, kümeleme, boyut indirgeme, model seçimi, ön işleme vb.) için sektör standardı bir kütüphanedir. Tutarlı API yapısı sayesinde farklı algoritmaları denemek oldukça kolaydır. Orta ölçekli problemler ve başlangıç seviyesi ML uygulamaları için idealdir.
          </li>
          <li>
            <strong>TensorFlow & PyTorch:</strong> Derin öğrenme modelleri (sinir ağları) oluşturmak, eğitmek ve dağıtmak için kullanılan lider kütüphanelerdir. GPU hızlandırma desteği sayesinde büyük veri kümeleri ve karmaşık modeller üzerinde verimli çalışırlar. Araştırma ve büyük ölçekli derin öğrenme projelerinde yaygın olarak kullanılırlar.
          </li>
        </ul>

        <h2>Basit Bir Makine Öğrenmesi Örneği (Scikit-learn)</h2>
        <p>
          İşte Pandas ve Scikit-learn kullanarak basit bir regresyon modelinin nasıl oluşturulabileceğine dair temel bir örnek. Bu örnek, veri yükleme, modeli tanımlama ve eğitme adımlarını göstermektedir. (Çalıştırmak için gerekli kütüphanelerin kurulu olması gerekir: `pip install pandas scikit-learn`)
        </p>
        <pre>
          <code 
            className="language-python"
            dangerouslySetInnerHTML={{
              __html: `import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Örnek veri yükleme (Gerçek projede CSV vb. kullanılabilir)
data = {'Metrekare': [50, 75, 100, 120, 150],
        'OdaSayisi': [1, 1, 2, 2, 3],
        'Fiyat': [150000, 220000, 300000, 350000, 450000]}
df = pd.DataFrame(data)

# Özellikler (X) ve Hedef (y) belirleme
X = df[['Metrekare', 'OdaSayisi']]
y = df['Fiyat']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Doğrusal Regresyon modelini oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Model performansını değerlendirme
mse = mean_squared_error(y_test, y_pred)
print(f"Ortalama Karesel Hata (MSE): {mse:.2f}")

# Yeni bir ev için tahmin
yeni_ev = [[80, 2]] # 80 metrekare, 2 oda
tahmin_fiyat = model.predict(yeni_ev)
print(f"Tahmini fiyat: {tahmin_fiyat[0]:.2f} TL")`
            }}
          />
        </pre>

        <h2>Kurulum ve Ortam Yönetimi</h2>
        <p>
          Python ve gerekli kütüphaneleri kurmanın en yaygın yolu Anaconda veya Miniconda gibi dağıtımları kullanmaktır. Bu dağıtımlar, sanal ortamlar oluşturarak farklı projeleriniz için izole geliştirme ortamları kurmanıza olanak tanır. Sanal ortamlar, farklı projelerdeki kütüphane versiyon çakışmalarını önler.
        </p>
        <p>
          Kurulumdan sonra, Jupyter Notebook veya VS Code gibi popüler IDE'leri kullanarak kod yazmaya ve denemeler yapmaya başlayabilirsiniz.
        </p>

        <h2>Sonraki Adımlar</h2>
        <p>
          Python temellerini ve ML için gerekli kütüphanelere giriş yaptıktan sonra, veri yükleme, ön işleme, model seçimi, eğitim ve değerlendirme gibi konuları öğrenmeye hazır olacaksınız. Her adımı uygulayarak ve pratik yaparak Python ile makine öğrenmesi projeleri geliştirmede uzmanlaşabilirsiniz.
        </p>
      </div>
    </div>
  );
} 