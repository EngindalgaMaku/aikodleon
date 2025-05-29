import { Metadata } from 'next';
import { createPageMetadata } from '@/lib/seo';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';

export const metadata: Metadata = createPageMetadata({
  title: 'Denetimsiz Öğrenme',
  description: 'Kodleon'da denetimsiz öğrenme algoritmalarını ve kullanım alanlarını keşfedin. Kümeleme ve boyut indirgeme tekniklerini öğrenin.',
  path: '/topics/machine-learning/unsupervised-learning',
  keywords: ['denetimsiz öğrenme', 'unsupervised learning', 'makine öğrenmesi', 'kümeleme', 'boyut indirgeme', 'ilişkilendirme kuralları', 'kodleon', 'türkçe ai eğitimi'],
});

export default function UnsupervisedLearningPage() {
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
      <h1 className="text-4xl font-bold mb-6">Denetimsiz Öğrenme</h1>
      <p className="text-xl text-muted-foreground mb-8">
        Denetimsiz öğrenme, makine öğrenmesinin, algoritmanın etiketlenmemiş veri kümelerindeki desenleri, yapıları ve ilişkileri bağımsız olarak keşfetmeyi öğrendiği bir dalıdır. Denetimli öğrenmenin aksine, denetimsiz öğrenme algoritmalarına "doğru" cevaplar verilmez; bunun yerine verinin içsel yapısını anlamaya odaklanırlar. Bu yaklaşım, veriyi düzenlemek, sıkıştırmak veya anlamak için kullanılır.
      </p>

      <div className="prose prose-lg dark:prose-invert max-w-none">
        <h2>Temel Görevler ve Teknikler</h2>
        <p>Denetimsiz öğrenmenin başlıca görevleri ve bu görevler için kullanılan teknikler şunlardır:</p>
        <ol>
          <li>**Kümeleme (Clustering):** Veri noktalarını, birbirine benzeyenlerin aynı grupta (kümede) toplandığı gruplara ayırma işlemidir. Popüler algoritmalar arasında K-Means, DBSCAN ve Hiyerarşik Kümeleme bulunur.</li>
          <li>**Boyut İndirgeme (Dimensionality Reduction):** Veri setindeki özellik sayısını azaltma yöntemidir. Bu, veriyi görselleştirmeye, depolama gereksinimlerini azaltmaya ve bazı algoritmaların performansını artırmaya yardımcı olur. Temel Bileşen Analizi (PCA) ve t-SNE sık kullanılan tekniklerdir.</li>
          <li>**İlişkilendirme Kuralları Madenciliği (Association Rule Mining):** Veri setindeki öğeler arasındaki ilginç ilişkileri (kuralları) bulma işlemidir. Genellikle sepet analizi gibi uygulamalarda kullanılır (Örn: 'Ekmek alanlar genellikle süt de alır'). Apriori algoritması bu alanda popülerdir.</li>
          <li>**Anormallik Tespiti (Anomaly Detection):** Veri setindeki beklenen desenlerden önemli ölçüde sapan veri noktalarını (anormallikleri) belirleme işlemidir. Kredi kartı sahtekarlığı tespiti gibi alanlarda kullanılır.</li>
        </ol>

        <h2>Uygulama Alanları</h2>
        <p>Denetimsiz öğrenme çeşitli alanlarda geniş kullanım bulur:</p>
        <ul>
          <li>**Müşteri Segmentasyonu:** Müşterileri satın alma davranışlarına veya demografik özelliklerine göre gruplama.</li>
          <li>**Anomali Tespiti:** Siber güvenlikte anormal ağ trafiğini veya finansal işlemlerde sahtekarlığı tespit etme.</li>
          <li>**Tavsiye Sistemleri:** Kullanıcıların geçmiş etkileşimlerine dayanarak benzer öğeleri önerme.</li>
          <li>**Veri Görselleştirme:** Çok boyutlu veriyi daha kolay anlaşılır 2D veya 3D formatlara indirme.</li>
          <li>**Genetik Veri Analizi:** Gen ifadelerindeki desenleri veya genetik varyasyonları kümeleme.</li>
        </ul>

        <h2>Temel Algoritmalar</h2>
        <p>Bazı önemli denetimsiz öğrenme algoritmaları şunlardır:</p>
        <ul>
          <li>K-Means Kümeleme</li>
          <li>DBSCAN</li>
          <li>Hiyerarşik Kümeleme</li>
          <li>Temel Bileşen Analizi (PCA)</li>
          <li>t-SNE</li>
          <li>Apriori Algoritması</li>
          <li>Tekil Değer Ayrışımı (SVD)</li>
        </ul>

        <p>Denetimsiz öğrenme, verinin yapısını anlamak ve gizli içgörüleri ortaya çıkarmak için güçlü bir yaklaşımdır.</p>
      </div>
    </div>
  );
} 