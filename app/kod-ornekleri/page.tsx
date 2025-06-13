import { Metadata } from 'next';
import Link from 'next/link';
import Image from 'next/image';
import { ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';

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
];

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
      
      {/* Filtreler */}
      <div className="flex flex-wrap gap-2 mb-8 justify-center">
        <Button variant="outline" className="rounded-full">Tümü</Button>
        <Button variant="outline" className="rounded-full">Derin Öğrenme</Button>
        <Button variant="outline" className="rounded-full">Makine Öğrenmesi</Button>
        <Button variant="outline" className="rounded-full">Doğal Dil İşleme</Button>
        <Button variant="outline" className="rounded-full">Bilgisayarlı Görü</Button>
        <Button variant="outline" className="rounded-full">Pekiştirmeli Öğrenme</Button>
      </div>
      
      {/* Kod Örnekleri Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {codeExamples.map((example) => (
          <Card key={example.id} className="overflow-hidden flex flex-col h-full">
            <div className="relative h-48">
              <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-purple-500/20" />
              <div className="absolute top-4 left-4 flex gap-2">
                <span className="px-3 py-1 rounded-full text-xs font-medium bg-background/80 backdrop-blur-sm">
                  {example.category}
                </span>
                <span className={`px-3 py-1 rounded-full text-xs font-medium backdrop-blur-sm ${
                  example.level === 'Başlangıç' ? 'bg-green-500/20 text-green-700 dark:text-green-300' :
                  example.level === 'Orta' ? 'bg-yellow-500/20 text-yellow-700 dark:text-yellow-300' :
                  'bg-red-500/20 text-red-700 dark:text-red-300'
                }`}>
                  {example.level}
                </span>
              </div>
            </div>
            <CardHeader>
              <CardTitle>{example.title}</CardTitle>
              <CardDescription>{example.description}</CardDescription>
            </CardHeader>
            <CardContent className="flex-grow">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span>Python</span>
                <span>•</span>
                <span>Jupyter Notebook</span>
              </div>
            </CardContent>
            <CardFooter>
              <Button asChild variant="default" className="w-full">
                <Link href={`/kod-ornekleri/${example.id}`}>
                  Kodu İncele
                  <ArrowRight className="h-4 w-4 ml-2" />
                </Link>
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>
      
      {/* Yakında Eklenecek Başlığı */}
      <div className="mt-16 mb-8 text-center">
        <h2 className="text-2xl font-bold">Yakında Eklenecek Örnekler</h2>
        <p className="text-muted-foreground mt-2">
          Aşağıdaki konularda yeni kod örnekleri hazırlıyoruz. Takipte kalın!
        </p>
      </div>
      
      {/* Yakında Eklenecek Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="bg-muted/50">
          <CardHeader>
            <CardTitle>Transformers ile Metin Üretimi</CardTitle>
            <CardDescription>Hugging Face transformers kütüphanesi kullanarak metin üretimi ve dil modelleme.</CardDescription>
          </CardHeader>
          <CardFooter>
            <Button variant="outline" disabled className="w-full">Yakında</Button>
          </CardFooter>
        </Card>
        
        <Card className="bg-muted/50">
          <CardHeader>
            <CardTitle>GANs ile Görüntü Sentezi</CardTitle>
            <CardDescription>Üretici Çekişmeli Ağlar (GANs) ile gerçekçi görüntüler oluşturma.</CardDescription>
          </CardHeader>
          <CardFooter>
            <Button variant="outline" disabled className="w-full">Yakında</Button>
          </CardFooter>
        </Card>
        
        <Card className="bg-muted/50">
          <CardHeader>
            <CardTitle>Önerici Sistemler</CardTitle>
            <CardDescription>İşbirlikçi filtreleme ve içerik tabanlı önerici sistemlerin uygulaması.</CardDescription>
          </CardHeader>
          <CardFooter>
            <Button variant="outline" disabled className="w-full">Yakında</Button>
          </CardFooter>
        </Card>
      </div>
      
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