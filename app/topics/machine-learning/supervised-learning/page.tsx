import { Metadata } from 'next';
import { createPageMetadata } from '@/lib/seo';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';

export const metadata: Metadata = createPageMetadata({
  title: 'Denetimli Öğrenme',
  description: "Kodleon'da denetimli öğrenme yöntemlerini ve uygulamalarını keşfedin. Regresyon ve sınıflandırma modellerini öğrenin.",
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
        <h2>Regresyon ve Sınıflandırma</h2>
        <p>Denetimli öğrenme problemleri genellikle iki ana kategoriye ayrılır:</p>
        <ul>
          <li>**Regresyon:** Amaç, sürekli bir çıktı değerini tahmin etmektir. Örneğin, bir evin metrekare cinsinden büyüklüğüne, konumuna ve yaşına bakarak satış fiyatını tahmin etmek bir regresyon problemidir. Çıktı (fiyat) sürekli bir sayıdır.</li>
          <li>**Sınıflandırma:** Amaç, bir veri örneğini belirli kategorilerden birine atamaktır. Örneğin, bir e-postanın içeriğine bakarak "spam" veya "spam değil" olarak etiketlemek bir sınıflandırma problemidir. Çıktı (spam/spam değil) belirli bir kategoridir. Diğer örnekler arasında bir resimdeki nesneyi tanıma (kedi, köpek, kuş vb.) veya bir hastanın semptomlarına göre belirli bir hastalığı teşhis etme yer alır.</li>
        </ul>

        <p>Her iki görev türü için de farklı algoritmalar ve değerlendirme metrikleri kullanılır. Örneğin, regresyonda Ortalama Karesel Hata (MSE) veya Ortalama Mutlak Hata (MAE) gibi metrikler kullanılırken, sınıflandırmada Doğruluk (Accuracy), Kesinlik (Precision), Geri Çağırma (Recall) ve F1 Skoru gibi metrikler yaygındır.</p>

        <h2>Nasıl Çalışır?</h2>
        <p>Denetimli öğrenme süreci genellikle şu adımları içerir:</p>
        <ol>
          <li>**Veri Toplama ve Etiketleme:** Eğitim için ilgili veriler toplanır ve her bir veri örneği doğru çıktıyla etiketlenir.</li>
          <li>**Model Seçimi:** Görevin türüne (regresyon veya sınıflandırma) uygun bir makine öğrenmesi modeli seçilir. Popüler modeller arasında Doğrusal Regresyon, Lojistik Regresyon, Karar Ağaçları, Rastgele Ormanlar, Destek Vektör Makineleri (SVM) ve Sinir Ağları bulunur.</li>
          <li>**Model Eğitimi:** Etiketlenmiş eğitim verileri kullanılarak seçilen model eğitilir. Eğitim sırasında modelin parametreleri, tahmin hatalarını minimize edecek şekilde ayarlanır.</li>
          <li>**Model Değerlendirme:** Eğitilmiş model, eğitimde kullanılmayan ayrı bir test veri kümesi üzerinde değerlendirilir. Performans metrikleri (doğruluk, kesinlik, geri çağırma, F1 skoru, Ortalama Karesel Hata vb.) kullanılarak modelin ne kadar iyi genelleme yaptığı ölçülür.</li>
          <li>**Model Ayarlama:** Model performansı yetersizse, hiperparametreler ayarlanabilir veya farklı modeller denenebilir.</li>
        </ol>

        <h2>Yaygın Zorluklar</h2>
        <p>Denetimli öğrenme modellerini eğitirken dikkat edilmesi gereken bazı yaygın zorluklar şunlardır:</p>
        <ul>
          <li>**Aşırı Uyum (Overfitting):** Modelin eğitim verilerini çok iyi öğrenmesi, ancak yeni ve görülmemiş verilere genelleme yapamaması durumudur. Model eğitim verisindeki gürültüyü bile öğrenir, bu da test verisinde kötü performansa yol açar. Genellikle karmaşık modeller veya yetersiz eğitim verisi olduğunda ortaya çıkar. Aşırı uyumu azaltmak için daha fazla veri toplama, model karmaşıklığını azaltma, düzenlileştirme (regularization) teknikleri veya çapraz doğrulama (cross-validation) kullanılabilir.</li>
          <li>**Yetersiz Uyum (Underfitting):** Modelin eğitim verilerini bile yeterince öğrenememesi, dolayısıyla hem eğitim hem de test verisinde kötü performans göstermesi durumudur. Model verinin temel desenlerini yakalayamaz. Genellikle çok basit modeller veya yetersiz özellik seti kullanıldığında ortaya çıkar. Yetersiz uyumu gidermek için daha karmaşık bir model seçme, daha fazla ilgili özellik ekleme veya eğitim süresini artırma gibi yöntemler denenebilir.</li>
        </ul>

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