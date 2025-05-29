import { Metadata } from 'next';
import { createPageMetadata } from '@/lib/seo';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';

export const metadata: Metadata = createPageMetadata({
  title: 'Denetimli Öğrenme',
  description: 'Kodleon'da denetimli öğrenme yöntemlerini ve uygulamalarını keşfedin. Regresyon ve sınıflandırma modellerini öğrenin.',
  path: '/topics/machine-learning/supervised-learning',
  keywords: ['denetimli öğrenme', 'supervised learning', 'makine öğrenmesi', 'regresyon', 'sınıflandırma', 'kodleon', 'türkçe ai eğitimi'],
});

export default function SupervisedLearningPage() {
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
      <h1 className="text-4xl font-bold mb-6">Denetimli Öğrenme</h1>
      <p className="text-xl text-muted-foreground mb-8">
        Denetimli öğrenme, etiketlenmiş veri kümelerini kullanarak modellerin çıktıları tahmin etmeyi veya sınıflandırmayı öğrendiği bir makine öğrenmesi türüdür. Bu yaklaşımda, her eğitim verisi örneği bir girdi (özellikler) ve buna karşılık gelen doğru çıktı (etiket) içerir. Algoritma, girdi ve çıktı arasındaki ilişkiyi öğrenerek yeni, görülmemiş girdiler için doğru çıktıları tahmin etmeyi hedefler.
      </p>

      <div className="prose prose-lg dark:prose-invert max-w-none">
        <h2>Nasıl Çalışır?</h2>
        <p>Denetimli öğrenme süreci genellikle şu adımları içerir:</p>
        <ol>
          <li>**Veri Toplama ve Etiketleme:** Eğitim için ilgili veriler toplanır ve her bir veri örneği doğru çıktıyla etiketlenir.</li>
          <li>**Model Seçimi:** Görevin türüne (regresyon veya sınıflandırma) uygun bir makine öğrenmesi modeli seçilir. Popüler modeller arasında Doğrusal Regresyon, Lojistik Regresyon, Karar Ağaçları, Rastgele Ormanlar, Destek Vektör Makineleri (SVM) ve Sinir Ağları bulunur.</li>
          <li>**Model Eğitimi:** Etiketlenmiş eğitim verileri kullanılarak seçilen model eğitilir. Eğitim sırasında modelin parametreleri, tahmin hatalarını minimize edecek şekilde ayarlanır.</li>
          <li>**Model Değerlendirme:** Eğitilmiş model, eğitimde kullanılmayan ayrı bir test veri kümesi üzerinde değerlendirilir. Performans metrikleri (doğruluk, kesinlik, geri çağırma, F1 skoru, Ortalama Karesel Hata vb.) kullanılarak modelin ne kadar iyi genelleme yaptığı ölçülür.</li>
          <li>**Model Ayarlama:** Model performansı yetersizse, hiperparametreler ayarlanabilir veya farklı modeller denenebilir.</li>
        </ol>

        <h2>Uygulama Alanları</h2>
        <p>Denetimli öğrenme çok çeşitli alanlarda kullanılır:</p>
        <ul>
          <li>**E-posta Filtreleme:** Gelen e-postaların spam olup olmadığını sınıflandırma.</li>
          <li>**Görüntü Tanıma:** Resimdeki nesneleri veya yüzleri tanımlama.</li>
          <li>**Tıbbi Teşhis:** Hastalıkların teşhisi için tıbbi görüntü veya hasta verilerini analiz etme.</li>
          <li>**Fiyat Tahmini:** Konut, hisse senedi veya ürün fiyatları gibi sürekli değerleri tahmin etme (Regresyon).</li>
          <li>**Müşteri Kayıp Oranı Tahmini:** Hangi müşterilerin hizmeti bırakma olasılığının yüksek olduğunu tahmin etme (Sınıflandırma).</li>
        </ul>

        <h2>Temel Algoritmalar</h2>
        <p>Denetimli öğrenmenin bazı temel algoritmaları şunlardır:</p>
        <ul>
          <li>Doğrusal Regresyon</li>
          <li>Lojistik Regresyon</li>
          <li>Karar Ağaçları</li>
          <li>Rastgele Ormanlar</li>
          <li>Destek Vektör Makineleri (SVM)</li>
          <li>K-En Yakın Komşular (KNN)</li>
          <li>Naive Bayes</li>
          <li>Yapay Sinir Ağları (Özellikle Sınıflandırma ve Regresyon için)</li>
        </ul>

        <p>Denetimli öğrenme, makine öğrenmesinin en yaygın kullanılan paradigmalarından biridir ve birçok gerçek dünya probleminin çözümünde temel oluşturur.</p>
      </div>
    </div>
  );
} 