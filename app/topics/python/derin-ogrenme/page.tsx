import { Metadata } from 'next';
import Link from 'next/link';
import Image from 'next/image';
import { ArrowRight, Brain, Code2, Network, Cpu, Layers, BarChart } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python ile Derin Öğrenme | Kodleon',
  description: 'Python programlama dilinde derin öğrenme, yapay sinir ağları, CNN, RNN ve daha fazlasını öğrenin.',
};

const content = `
# Python ile Derin Öğrenme

Derin öğrenme, yapay zekanın en güçlü ve popüler alt alanlarından biridir. Bu eğitim serisinde, Python kullanarak derin öğrenme modellerini nasıl oluşturacağınızı, eğiteceğinizi ve uygulayacağınızı öğreneceksiniz.

## Neden Derin Öğrenme?

- **Güçlü Özellik Çıkarımı**: Veriden otomatik olarak özellik çıkarımı yapabilme
- **Yüksek Performans**: Karmaşık problemlerde üstün başarı
- **Esneklik**: Farklı problem türlerine uyarlanabilirlik
- **Geniş Uygulama Alanı**: Görüntü işleme, doğal dil işleme, ses tanıma ve daha fazlası
- **Sürekli Gelişim**: Aktif araştırma ve yeni yaklaşımlar

## Öğrenme Yolculuğunuz

Bu eğitim serisi, temel kavramlardan ileri düzey uygulamalara kadar kapsamlı bir yol haritası sunar:

1. Yapay Sinir Ağlarının Temelleri
2. Derin Öğrenme Frameworkleri
3. Evrişimli Sinir Ağları (CNN)
4. Tekrarlayan Sinir Ağları (RNN)
5. Modern Mimari ve Yaklaşımlar
6. Pratik Uygulamalar ve Projeler
`;

const sections = [
  {
    title: "1. Yapay Sinir Ağları Temelleri",
    description: "Yapay sinir ağlarının temel kavramlarını ve çalışma prensiplerini öğrenin.",
    icon: <Brain className="h-6 w-6" />,
    href: "/topics/python/derin-ogrenme/yapay-sinir-aglari-temelleri",
    topics: [
      "Perceptron ve Yapay Nöronlar",
      "Aktivasyon Fonksiyonları",
      "İleri Yayılım (Forward Propagation)",
      "Geri Yayılım (Backpropagation)",
      "Optimizasyon Algoritmaları"
    ]
  },
  {
    title: "2. Derin Öğrenme Frameworkleri",
    description: "PyTorch ve TensorFlow gibi popüler derin öğrenme kütüphanelerini keşfedin.",
    icon: <Code2 className="h-6 w-6" />,
    href: "/topics/python/derin-ogrenme/derin-ogrenme-frameworkleri",
    topics: [
      { text: "PyTorch Temelleri", href: "/topics/python/pytorch-dersleri/01-pytorch-kurulumu-ve-tensorlere-giris" },
      { text: "TensorFlow ve Keras" },
      { text: "Model Oluşturma" },
      { text: "Veri Yükleme ve İşleme" },
      { text: "Model Eğitimi ve Değerlendirme" }
    ]
  },
  {
    title: "3. Evrişimli Sinir Ağları (CNN)",
    description: "Görüntü işleme ve bilgisayarlı görü için CNN modellerini öğrenin.",
    icon: <Network className="h-6 w-6" />,
    href: "/topics/python/derin-ogrenme/evrisimli-sinir-aglari",
    topics: [
      "Evrişim Katmanları",
      "Havuzlama İşlemleri",
      "Transfer Öğrenme",
      "Nesne Tespiti",
      "Görüntü Segmentasyonu"
    ]
  },
  {
    title: "4. Tekrarlayan Sinir Ağları (RNN)",
    description: "Sıralı veri işleme ve zaman serisi analizi için RNN modellerini keşfedin.",
    icon: <Cpu className="h-6 w-6" />,
    href: "/topics/python/derin-ogrenme/tekrarlayan-sinir-aglari",
    topics: [
      "RNN Mimarisi",
      "LSTM ve GRU",
      "Seq2Seq Modeller",
      "Attention Mekanizması",
      "Transformers"
    ]
  },
  {
    title: "5. Modern Mimari ve Yaklaşımlar",
    description: "Güncel derin öğrenme mimarilerini ve tekniklerini öğrenin.",
    icon: <Layers className="h-6 w-6" />,
    href: "/topics/python/derin-ogrenme/modern-mimariler",
    topics: [
      "Transformer Mimarisi",
      "GANs (Üretici Çekişmeli Ağlar)",
      "AutoEncoder'lar",
      "Self-Supervised Learning",
      "Few-Shot Learning"
    ]
  },
  {
    title: "6. Pratik Uygulamalar",
    description: "Gerçek dünya problemlerine derin öğrenme çözümleri geliştirin.",
    icon: <BarChart className="h-6 w-6" />,
    href: "/topics/python/derin-ogrenme/pratik-uygulamalar",
    topics: [
      "Görüntü Sınıflandırma",
      "Doğal Dil İşleme",
      "Ses Tanıma",
      "Öneri Sistemleri",
      "Anomali Tespiti"
    ]
  }
];

export default function DeepLearningPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="prose prose-lg dark:prose-invert mb-8">
          <MarkdownContent content={content} />
        </div>
        
        {/* Concept Image */}
        <div className="my-8">
          <Image
            src="/images/python_derin_ogrenme.jpg"
            alt="Derin Öğrenme"
            width={800}
            height={400}
            className="rounded-lg shadow-lg object-cover"
            priority
          />
        </div>

        {/* Interactive Learning Path */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Öğrenme Yolu</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {sections.map((section, index) => (
              <Card key={index} className="group hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg bg-primary/10">
                      {section.icon}
                    </div>
                    <CardTitle>{section.title}</CardTitle>
                  </div>
                  <CardDescription>{section.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                    {section.topics.map((topic, i) => (
                      <li key={i}>
                        {typeof topic === 'string' ? topic : (
                          <Link href={topic.href || '#'} className="text-primary hover:underline">
                            {topic.text}
                          </Link>
                        )}
                      </li>
                    ))}
                  </ul>
                </CardContent>
                <CardFooter>
                  <Button asChild className="w-full group">
                    <Link href={section.href}>
                      Derse Git
                      <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
                    </Link>
                  </Button>
                </CardFooter>
              </Card>
            ))}
          </div>
        </div>

        {/* Additional Resources */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Ek Kaynaklar</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Online Kurslar</CardTitle>
                <CardDescription>İnteraktif öğrenme platformları</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                  <li>Coursera Deep Learning</li>
                  <li>Fast.ai Kursu</li>
                  <li>Stanford CS231n</li>
                </ul>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Araçlar ve Kütüphaneler</CardTitle>
                <CardDescription>Geliştirme için gerekli araçlar</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                  <li>PyTorch</li>
                  <li>TensorFlow</li>
                  <li>Keras</li>
                </ul>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Topluluk ve Kaynaklar</CardTitle>
                <CardDescription>Öğrenme materyalleri</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                  <li>GitHub Projeleri</li>
                  <li>Araştırma Makaleleri</li>
                  <li>Blog Yazıları</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Prerequisites */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Ön Gereksinimler</h2>
          <div className="prose prose-lg dark:prose-invert">
            <ul>
              <li>Python programlama dili temel bilgisi</li>
              <li>NumPy ve Pandas kütüphaneleri</li>
              <li>Temel matematik ve istatistik bilgisi</li>
              <li>Lineer cebir temelleri</li>
              <li>Makine öğrenmesi kavramları</li>
            </ul>
          </div>
        </div>

        <div className="mt-16 text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} Kodleon | Yapay Zeka Eğitim Platformu</p>
        </div>
      </div>
    </div>
  );
} 