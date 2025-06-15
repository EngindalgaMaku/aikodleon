import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardFooter, CardTitle, CardDescription } from '@/components/ui/card';
import CodeExamplesClient from './CodeExamplesClient';

export const metadata: Metadata = {
  title: 'Yapay Zeka Kod Örnekleri | Kodleon',
  description: 'Yapay zeka, makine öğrenmesi ve derin öğrenme alanlarında pratik Python kod örnekleri ve projeler.',
  openGraph: {
    title: 'Yapay Zeka Kod Örnekleri | Kodleon',
    description: 'Yapay zeka, makine öğrenmesi ve derin öğrenme alanlarında pratik Python kod örnekleri ve projeler.',
    images: [{ url: '/images/code-examples.jpg' }],
  },
};

interface CodeExample {
  id: string;
  title: string;
  description: string;
  category: string;
  level: 'Başlangıç' | 'Orta' | 'İleri';
  image: string;
}

const codeExamples: CodeExample[] = [
  {
    id: 'temel-sinir-agi',
    title: 'Temel Yapay Sinir Ağı Uygulaması',
    description: 'NumPy kullanarak sıfırdan basit bir yapay sinir ağı oluşturma ve eğitme.',
    category: 'Derin Öğrenme',
    level: 'Başlangıç',
    image: '/images/code-examples/neural-network.jpg',
  },
  {
    id: 'resim-siniflandirma',
    title: 'CNN ile Görüntü Sınıflandırma',
    description: 'TensorFlow ve Keras kullanarak evrişimli sinir ağı (CNN) ile görüntü sınıflandırma.',
    category: 'Bilgisayarlı Görü',
    level: 'Orta',
    image: '/images/code-examples/image-classification.jpg',
  },
  {
    id: 'nlp-duygu-analizi',
    title: 'NLP ile Duygu Analizi',
    description: 'NLTK ve scikit-learn kullanarak metin tabanlı duygu analizi uygulaması.',
    category: 'Doğal Dil İşleme',
    level: 'Orta',
    image: '/images/code-examples/sentiment-analysis.jpg',
  },
  {
    id: 'reinforcement-learning',
    title: 'Pekiştirmeli Öğrenme ile Oyun AI',
    description: 'OpenAI Gym kullanarak basit bir oyun için pekiştirmeli öğrenme ajanı geliştirme.',
    category: 'Pekiştirmeli Öğrenme',
    level: 'İleri',
    image: '/images/code-examples/reinforcement-learning.jpg',
  },
  {
    id: 'anomali-tespiti',
    title: 'Anomali Tespiti Algoritması',
    description: 'Denetimsiz öğrenme ile veri setindeki anormallikleri tespit etme.',
    category: 'Makine Öğrenmesi',
    level: 'Orta',
    image: '/images/code-examples/anomaly-detection.jpg',
  },
  {
    id: 'zaman-serisi-tahmini',
    title: 'LSTM ile Zaman Serisi Tahmini',
    description: 'Uzun-Kısa Vadeli Bellek (LSTM) ağları ile zaman serisi verilerinde tahmin yapma.',
    category: 'Derin Öğrenme',
    level: 'İleri',
    image: '/images/code-examples/time-series.jpg',
  },
  {
    id: 'transformers-metin-uretimi',
    title: 'Transformers ile Metin Üretimi',
    description: 'Hugging Face Transformers kütüphanesi kullanarak GPT-2 ile metin üretme.',
    category: 'Doğal Dil İşleme',
    level: 'Orta',
    image: '/images/code-examples/text-generation.jpg',
  },
  {
    id: 'gans-goruntu-sentezi',
    title: 'GANs ile Görüntü Sentezi',
    description: 'PyTorch ve GANs kullanarak sentetik görüntüler (el yazısı rakamlar) oluşturma.',
    category: 'Bilgisayarlı Görü',
    level: 'İleri',
    image: '/images/code-examples/gans.jpg',
  },
  {
    id: 'onerici-sistemler',
    title: 'Önerici Sistemler',
    description: 'Scikit-learn ile işbirlikçi filtreleme tabanlı basit bir film öneri sistemi.',
    category: 'Makine Öğrenmesi',
    level: 'İleri',
    image: '/images/code-examples/recommender-system.jpg',
  },
];

const allCategories = ['Tümü', ...Array.from(new Set(codeExamples.map(ex => ex.category)))];

export default function CodeExamplesPage() {
  return (
    <div className="container max-w-6xl mx-auto py-12 px-4">
      <div className="max-w-3xl mx-auto text-center mb-12">
        <h1 className="text-4xl font-bold tracking-tight mb-4">Yapay Zeka Kod Örnekleri</h1>
        <p className="text-xl text-muted-foreground">
          Yapay zeka ve makine öğrenmesi alanlarında pratik uygulamalar ve kod örnekleri.
        </p>
        <p className="text-sm text-muted-foreground mt-2">
          Başlangıç seviyesinden ileri düzeye kadar Python tabanlı örnekler
        </p>
      </div>
      
      <CodeExamplesClient codeExamples={codeExamples} />
      
      {/* Katkıda Bulunma Bölümü */}
      <div className="mt-16 p-6 bg-muted rounded-xl">
        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold">Katkıda Bulunmak İster misiniz?</h2>
          <p className="text-muted-foreground mt-2">
            Kendi kod örneğinizi göndererek topluluğumuza katkıda bulunabilirsiniz.
          </p>
        </div>
        
        <div className="flex justify-center">
          <Button asChild variant="default" size="lg">
            <Link href="/iletisim">
              Bizimle İletişime Geçin
              <ArrowRight className="h-4 w-4 ml-2" />
            </Link>
          </Button>
        </div>
      </div>
      
      <div className="mt-16 text-center text-sm text-gray-500 dark:text-gray-400">
        <p>© {new Date().getFullYear()} Kodleon | Yapay Zeka Eğitim Platformu</p>
        <p className="mt-2">Tüm kod örnekleri MIT lisansı altında paylaşılmaktadır.</p>
      </div>
    </div>
  );
} 