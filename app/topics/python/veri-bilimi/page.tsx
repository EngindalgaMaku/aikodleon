import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'Python ile Veri Bilimi | Kodleon',
  description: 'Python programlama dili ile veri bilimi, veri analizi, görselleştirme ve makine öğrenmesi konularını öğrenin.',
};

const content = `
# Python ile Veri Bilimi

Python, veri bilimi ve makine öğrenmesi alanında en çok tercih edilen programlama dillerinden biridir. Zengin kütüphane ekosistemi ve kolay öğrenilebilir yapısı ile veri analizi, görselleştirme ve model geliştirme süreçlerini kolaylaştırır.
`;

const learningPath = [
  {
    title: '1. Numpy ile Bilimsel Hesaplama',
    description: 'Numpy kütüphanesi ile çok boyutlu diziler ve matematiksel işlemleri öğrenin.',
    topics: [
      'Çok boyutlu diziler',
      'Matematiksel işlemler',
      'Dizilerde indeksleme ve dilimleme',
      'Lineer cebir işlemleri',
      'Rastgele sayı üretimi',
    ],
    icon: '🔢',
    href: '/topics/python/veri-bilimi/numpy'
  },
  {
    title: '2. Pandas ile Veri Analizi',
    description: 'Pandas kütüphanesi ile veri analizi ve manipülasyonu tekniklerini keşfedin.',
    topics: [
      'DataFrame ve Series',
      'Veri okuma ve yazma',
      'Veri filtreleme ve gruplama',
      'Veri birleştirme ve dönüştürme',
      'Zaman serisi analizi',
    ],
    icon: '📊',
    href: '/topics/python/veri-bilimi/pandas'
  },
  {
    title: '3. Veri Görselleştirme',
    description: 'Matplotlib, Seaborn ve Plotly ile etkili veri görselleştirme tekniklerini öğrenin.',
    topics: [
      'Matplotlib temelleri',
      'Seaborn ile istatistiksel görselleştirme',
      'Plotly ile interaktif grafikler',
      'Görselleştirme en iyi pratikleri',
      'Dashboard oluşturma',
    ],
    icon: '📈',
    href: '/topics/python/veri-bilimi/veri-gorsellestirme'
  },
  {
    title: '4. Makine Öğrenmesi Temelleri',
    description: 'Scikit-learn ile temel makine öğrenmesi konseptlerini ve modellerini keşfedin.',
    topics: [
      'Denetimli öğrenme',
      'Denetimsiz öğrenme',
      'Model değerlendirme',
      'Model optimizasyonu',
      'Özellik mühendisliği',
    ],
    icon: '🤖',
    href: '/topics/python/veri-bilimi/makine-ogrenmesi'
  },
  {
    title: '5. Derin Öğrenme',
    description: 'TensorFlow ve PyTorch ile derin öğrenme modellerini öğrenin.',
    topics: [
      'Yapay sinir ağları',
      'Evrişimli sinir ağları (CNN)',
      'Tekrarlayan sinir ağları (RNN)',
      'Transfer öğrenme',
      'Model dağıtımı',
    ],
    icon: '🧠',
    href: '/topics/python/veri-bilimi/derin-ogrenme'
  },
  {
    title: '6. Büyük Veri İşleme',
    description: 'Büyük veri teknolojileri ve dağıtık hesaplama sistemlerini keşfedin.',
    topics: [
      'Apache Spark',
      'Dask',
      'Veri akış işleme',
      'Dağıtık sistemler',
      'Performans optimizasyonu',
    ],
    icon: '📦',
    href: '/topics/python/veri-bilimi/buyuk-veri'
  }
];

export default function DataSciencePage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild className="mb-6">
            <Link href="/topics/python" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Python
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