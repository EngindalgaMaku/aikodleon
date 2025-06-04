import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'NumPy ile Bilimsel Hesaplama | Python Veri Bilimi | Kodleon',
  description: 'NumPy kütüphanesi ile çok boyutlu diziler, matematiksel işlemler ve bilimsel hesaplama tekniklerini öğrenin.',
};

const content = `
# NumPy ile Bilimsel Hesaplama

NumPy (Numerical Python), Python'da bilimsel hesaplama için temel kütüphanedir. Çok boyutlu diziler ve matematiksel işlemler için yüksek performanslı araçlar sunar.
`;

const learningPath = [
  {
    title: '1. NumPy Temelleri',
    description: 'NumPy\'ın temel yapı taşları olan diziler ve temel işlemleri öğrenin.',
    topics: [
      'NumPy dizileri (ndarray)',
      'Dizi oluşturma yöntemleri',
      'Temel dizi özellikleri',
      'Veri tipleri ve dönüşümler',
      'Dizi şekillendirme (reshape)',
    ],
    icon: '📊',
    href: '/topics/python/veri-bilimi/numpy/temeller'
  },
  {
    title: '2. Dizilerde İndeksleme ve Dilimleme',
    description: 'NumPy dizilerinde veri erişimi ve manipülasyonu tekniklerini keşfedin.',
    topics: [
      'Temel indeksleme',
      'Gelişmiş dilimleme',
      'Boolean indeksleme',
      'Fancy indeksleme',
      'Görünüm vs. kopya',
    ],
    icon: '✂️',
    href: '/topics/python/veri-bilimi/numpy/indeksleme'
  },
  {
    title: '3. Matematiksel İşlemler',
    description: 'NumPy ile matematiksel ve istatistiksel hesaplamalar yapın.',
    topics: [
      'Temel aritmetik işlemler',
      'Evrensel fonksiyonlar (ufunc)',
      'İstatistiksel işlemler',
      'Trigonometrik fonksiyonlar',
      'Agregasyon işlemleri',
    ],
    icon: '🔢',
    href: '/topics/python/veri-bilimi/numpy/matematik'
  },
  {
    title: '4. Lineer Cebir',
    description: 'NumPy ile matris işlemleri ve lineer cebir uygulamalarını öğrenin.',
    topics: [
      'Matris işlemleri',
      'Determinant ve iz',
      'Özdeğer ve özvektörler',
      'Matris ayrıştırma',
      'Lineer denklem sistemleri',
    ],
    icon: '📐',
    href: '/topics/python/veri-bilimi/numpy/lineer-cebir'
  },
  {
    title: '5. Rastgele Sayılar',
    description: 'NumPy ile rastgele sayı üretimi ve olasılık dağılımlarını keşfedin.',
    topics: [
      'Rastgele sayı üreteci',
      'Olasılık dağılımları',
      'Permütasyon ve kombinasyon',
      'Örnekleme',
      'Tohum (seed) ayarlama',
    ],
    icon: '🎲',
    href: '/topics/python/veri-bilimi/numpy/rastgele'
  },
  {
    title: '6. İleri Düzey Konular',
    description: 'NumPy\'ın ileri düzey özelliklerini ve optimizasyon tekniklerini öğrenin.',
    topics: [
      'Yapısal diziler',
      'Bellek yönetimi',
      'Performans optimizasyonu',
      'Paralel işleme',
      'C/C++ entegrasyonu',
    ],
    icon: '🚀',
    href: '/topics/python/veri-bilimi/numpy/ileri-duzey'
  }
];

export default function NumPyPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild className="mb-6">
            <Link href="/topics/python/veri-bilimi" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Veri Bilimi
            </Link>
          </Button>
        </div>

        <div className="prose prose-lg dark:prose-invert mb-8">
          <MarkdownContent content={content} />
        </div>

        <h2 className="text-2xl font-bold mb-6">Öğrenme Yolu</h2>
        
        <div className="grid gap-6 md:grid-cols-2">
          {learningPath.map((topic, index) => (
            <Card key={index} className="p-6 hover:bg-accent transition-colors cursor-pointer">
              <Link href={topic.href}>
                <div className="flex items-start space-x-4">
                  <div className="text-4xl">{topic.icon}</div>
                  <div className="space-y-2">
                    <h3 className="font-bold">{topic.title}</h3>
                    <p className="text-sm text-muted-foreground">{topic.description}</p>
                    <ul className="text-sm space-y-1 list-disc list-inside text-muted-foreground">
                      {topic.topics.map((t, i) => (
                        <li key={i}>{t}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </Link>
            </Card>
          ))}
        </div>

        <div className="mt-16 text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} Kodleon | Python Eğitim Platformu</p>
        </div>
      </div>
    </div>
  );
} 