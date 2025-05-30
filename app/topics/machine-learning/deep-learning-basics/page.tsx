import { Metadata } from 'next';
import { createPageMetadata } from '@/lib/seo';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';

export const metadata: Metadata = createPageMetadata({
  title: 'Derin Öğrenme Temelleri',
  description: 'Kodleon'da derin öğrenmenin temel kavramlarını, yapay sinir ağlarının çalışma prensiplerini ve popüler mimarileri öğrenin.',
  path: '/topics/machine-learning/deep-learning-basics',
  keywords: ['derin öğrenme', 'deep learning', 'makine öğrenmesi', 'sinir ağları', 'yapay sinir ağları', 'katmanlar', 'aktivasyon fonksiyonları', 'geri yayılım', 'kodleon', 'türkçe ai eğitimi'],
});

export default function DeepLearningBasicsPage() {
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
      <h1 className="text-4xl font-bold mb-6">Derin Öğrenme Temelleri</h1>
      <p className="text-xl text-muted-foreground mb-8">
        Derin Öğrenme (Deep Learning), birden çok katmandan oluşan yapay sinir ağlarını kullanarak verilerden karmaşık temsilleri öğrenen makine öğrenmesinin bir alt alanıdır. Geleneksel makine öğrenmesi yöntemlerinin aksine, derin öğrenme modelleri, ham veriden (örneğin piksellerden veya metin karakterlerinden) otomatik olarak yüksek seviye özellikleri çıkarabilir. Bu, özellikle görüntü, ses ve metin gibi yapısal olmayan verilerle çalışırken derin öğrenmeyi son derece güçlü kılar.
      </p>

      <div className="prose prose-lg dark:prose-invert max-w-none">
        <h2>Çekirdek Kavramlar</h2>
        <p>Derin öğrenmeyi anlamak için bilinmesi gereken temel kavramlar şunlardır:</p>
        <ol>
          <li>**Yapay Sinir Ağları (Artificial Neural Networks - ANN):** Biyolojik sinir ağlarından esinlenen, bağlı düğümlerden (nöronlardan) oluşan hesaplama modelleridir.</li>
          <li>**Katmanlar (Layers):** Sinir ağındaki nöron gruplarıdır. Genellikle bir girdi katmanı, bir veya daha fazla gizli katman ve bir çıktı katmanı bulunur. Derin öğrenme, çok sayıda gizli katmana sahip ağları ifade eder.</li>
          <li>**Nöronlar (Neurons):** Girdi alır, bunları ağırlıklar ve bir bias ile birleştirir ve bir aktivasyon fonksiyonundan geçirerek bir çıktı üretir.</li>
          <li>**Aktivasyon Fonksiyonları (Activation Functions):** Nöronun çıktısını dönüştüren ve ağa doğrusal olmayanlık katan fonksiyonlardır (ReLU, Sigmoid, Tanh gibi).</li>
          <li>**Ağırlıklar (Weights) ve Biaslar (Biases):** Nöronlar arasındaki bağlantıların gücünü ve nöronun aktivasyon eşiğini belirleyen öğrenilebilir parametrelerdir.</li>
          <li>**Geri Yayılım (Backpropagation):** Ağın ağırlıklarını ve biaslarını, tahmin hatalarını (kayıp fonksiyonu) azaltmak için gradyan inişi kullanarak güncelleyen temel eğitim algoritmasıdır.</li>
          <li>**Kayıp Fonksiyonu (Loss Function):** Modelin tahminlerinin gerçek değerlerden ne kadar saptığını ölçen fonksiyondur.</li>
          <li>**Optimizasyon Algoritmaları:** Ağırlıkları güncellemek için gradyanları kullanan algoritmalardır (SGD, Adam, RMSprop gibi).</li>
        </ol>

        <h3>Eğitim Süreci: Geri Yayılım ve Optimizasyon</h3>
        <p>Bir derin öğrenme modelini eğitmek, modelin ağırlıklarını ve biaslarını ayarlayarak tahminlerinin doğruluğunu artırmak anlamına gelir. Bu süreç genellikle şu adımları içerir:</p>
        <ol>
          <li>**İleri Besleme (Forward Pass):** Girdi verisi ağa verilir ve her katmandan geçerek bir çıktı tahmini üretilir.</li>
          <li>**Kayıp Hesaplama (Loss Calculation):** Üretilen tahmin ile gerçek hedef değer arasındaki hata, bir kayıp fonksiyonu kullanılarak hesaplanır.</li>
          <li>**Geri Yayılım (Backpropagation):** Hesaplanan kayıp, ağda geriye doğru yayılarak her bir ağırlık ve biasın kayba ne kadar katkıda bulunduğu bulunur (gradyanlar hesaplanır).</li>
          <li>**Parametre Güncelleme (Parameter Update):** Hesaplanan gradyanlar ve bir optimizasyon algoritması (SGD, Adam vb.) kullanılarak ağırlıklar ve biaslar güncellenir. Bu adım, kaybı minimize edecek yönde yapılır.</li>
        </ol>
        <p>Bu adımlar, genellikle tüm eğitim veri seti üzerinde birden çok kez (epoch) tekrarlanır. Aktivasyon fonksiyonları, ağın doğrusal olmayan ilişkileri öğrenebilmesi için kritik öneme sahiptir. ReLU (Rectified Linear Unit) günümüzde yaygın olarak kullanılan basit ve etkili bir aktivasyon fonksiyonudur.</p>

        <h2>Popüler Derin Öğrenme Mimarileri</h2>
        <p>Farklı veri türleri ve görevler için optimize edilmiş çeşitli derin öğrenme mimarileri bulunmaktadır:</p>
        <ul>
          <li>**Konvolüsyonel Sinir Ağları (CNN):** Görüntü ve video gibi grid benzeri verilere özel olarak tasarlanmıştır.</li>
          <li>**Tekrarlayan Sinir Ağları (RNN):** Metin ve zaman serileri gibi sıralı verilerle çalışmak için uygundur.</li>
          <li>**Transformerlar:** Özellikle doğal dil işlemede çığır açan, dikkat mekanizmasına dayalı mimarilerdir.</li>
          <li>**Üretken Çekişmeli Ağlar (GAN):** Yeni veri örnekleri (görüntüler, metinler) üretmek için kullanılır.</li>
        </ul>

        <h2>Uygulama Alanları</h2>
        <p>Derin öğrenme, yapay zekanın birçok alanında etkili olmuştur:</p>
        <ul>
          <li>**Görüntü Tanıma ve İşleme:** Nesne algılama, yüz tanıma, tıbbi görüntü analizi.</li>
          <li>**Doğal Dil İşleme:** Makine çevirisi, metin özetleme, duygu analizi, sohbet botları.</li>
          <li>**Konuşma Tanıma ve Sentezleme:** Sesli asistanlar.</li>
          <li>**Otonom Sürüş:** Araç algılama, şerit takibi.</li>
          <li>**Tavsiye Sistemleri:** Kişiselleştirilmiş öneriler sunma.</li>
        </ul>

        <p>Derin öğrenme, büyük veri kümeleri ve artan hesaplama gücü sayesinde, daha önce çözülemeyen birçok karmaşık problemin üstesinden gelmeyi mümkün kılmıştır.</p>
      </div>
    </div>
  );
} 