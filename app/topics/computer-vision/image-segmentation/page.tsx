import { Metadata } from 'next';
import Link from 'next/link';
import Image from 'next/image';
import { ArrowLeft, ScanLine, Palette, Shapes, Microscope, Tractor, Route } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';

export const metadata: Metadata = {
  title: 'Görüntü Segmentasyonu: Pikselleri Anlamlandırma | Kodleon CV Dersleri',
  description: 'Görüntü segmentasyonunun ne olduğunu, anlamsal, örnek ve panoptik segmentasyon türlerini, U-Net gibi popüler modelleri ve uygulama alanlarını öğrenin.',
  keywords: 'görüntü segmentasyonu, image segmentation, anlamsal segmentasyon, örnek segmentasyon, panoptik segmentasyon, u-net, mask r-cnn, bilgisayarlı görü, piksel düzeyinde sınıflandırma',
  alternates: {
    canonical: 'https://kodleon.com/topics/computer-vision/image-segmentation',
  },
  openGraph: {
    title: 'Görüntü Segmentasyonu: Her Pikselin Bir Anlamı Var | Kodleon',
    description: 'Bir görüntüyü anlamlı bölgelere ayırarak her pikseli sınıflandıran görüntü segmentasyonunu, türlerini, önemli algoritmalarını ve kullanım alanlarını keşfedin.',
    url: 'https://kodleon.com/topics/computer-vision/image-segmentation',
    images: [
      {
        url: '/images/og/topics/computer-vision/image-segmentation-og.png', // Bu görselin oluşturulması/var olması gerekli
        width: 1200,
        height: 630,
        alt: 'Kodleon Görüntü Segmentasyonu Eğitimi'
      }
    ]
  }
};

const segmentationTypes = [
  {
    name: "Anlamsal Segmentasyon (Semantic Segmentation)",
    description: "Görüntüdeki her pikseli, ait olduğu nesne veya bölge sınıfına göre etiketler (örn: tüm arabalar aynı renkle, tüm yayalar farklı bir renkle). Farklı örnekleri ayırt etmez.",
    icon: <Palette className="h-8 w-8 text-sky-500" />
  },
  {
    name: "Örnek Segmentasyonu (Instance Segmentation)",
    description: "Anlamsal segmentasyona ek olarak, aynı sınıfa ait farklı nesne örneklerini de ayırt eder (örn: her bir araba farklı bir renkle işaretlenir).",
    icon: <Shapes className="h-8 w-8 text-lime-500" />
  },
  {
    name: "Panoptik Segmentasyon (Panoptic Segmentation)",
    description: "Anlamsal ve örnek segmentasyonunu birleştirir. Görüntüdeki her pikseli hem bir sınıfa hem de (eğer sayılabilir bir nesne ise) bir örneğe atar.",
    icon: <ScanLine className="h-8 w-8 text-fuchsia-500" />
  }
];

const algorithms = [
  { name: "U-Net", description: "Özellikle biyomedikal görüntü segmentasyonu için geliştirilmiş, kodlayıcı-kod çözücü mimarisine sahip popüler bir CNN modelidir.", year: "2015" },
  { name: "Mask R-CNN", description: "Faster R-CNN'e bir maske tahmin dalı ekleyerek nesne tespitiyle birlikte örnek segmentasyonu da yapar.", year: "2017" },
  { name: "DeepLab Ailesi", description: "Atrous (seyreltilmiş) evrişimler ve koşullu rastgele alanlar (CRF) gibi teknikler kullanarak anlamsal segmentasyonda yüksek doğruluk elde eder.", year: "2016+" },
  { name: "Fully Convolutional Networks (FCN)", description: "Sadece evrişim katmanlarından oluşan ve piksel düzeyinde tahmin yapabilen ilk derin öğrenme modellerindendir.", year: "2015" }
];

const applications = [
  { title: "Tıbbi Görüntüleme", description: "Organların, tümörlerin, lezyonların ve diğer anatomik yapıların sınırlarının hassas bir şekilde belirlenmesi.", icon: <Microscope className="h-8 w-8 text-red-500" /> },
  { title: "Otonom Araçlar", description: "Yolun, şeritlerin, yayaların, araçların ve diğer engellerin hassas bir şekilde ayrıştırılması.", icon: <Route className="h-8 w-8 text-blue-500" /> },
  { title: "Uydu Görüntü Analizi", description: "Arazi örtüsü sınıflandırması (orman, su, şehir), bina tespiti, yol ağlarının çıkarılması.", icon: <Shapes className="h-8 w-8 text-green-500" /> }, // Reused icon, consider a more specific one like Globe or Map
  { title: "Robotik ve Artırılmış Gerçeklik", description: "Çevre haritalama, nesnelerle etkileşim için hassas konumlandırma, sanal nesnelerin gerçek dünyaya yerleştirilmesi.", icon: <ScanLine className="h-8 w-8 text-purple-500" /> }, // Reused icon
  { title: "Tarım", description: "Ekin alanlarının, bitki türlerinin veya hastalıklı bölgelerin hassas tespiti.", icon: <Tractor className="h-8 w-8 text-yellow-600" /> }
];

export default function ImageSegmentationPage() {
  return (
    <div className="bg-background text-foreground">
      <section className="relative py-16 md:py-24 bg-gradient-to-br from-teal-500/5 via-transparent to-cyan-500/5">
        <div className="container max-w-5xl mx-auto px-4 text-center">
          <div className="inline-flex items-center gap-2 mb-4">
            <Button asChild variant="ghost" size="sm" className="gap-1 text-primary hover:text-primary/80">
              <Link href="/topics/computer-vision">
                <ArrowLeft className="h-4 w-4" />
                Bilgisayarlı Görü Ana Sayfası
              </Link>
            </Button>
          </div>
          <Palette className="h-16 w-16 text-primary mx-auto mb-6" />
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
            Görüntü Segmentasyonu
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto">
            Bir görüntüyü sadece nesneleriyle değil, her bir pikselin neye ait olduğunu anlayarak daha derinlemesine analiz etmek! Görüntü segmentasyonu, pikselleri anlamlı bölgelere ayırarak bize bu gücü verir.
          </p>
        </div>
      </section>

      <section className="py-12 md:py-20">
        <div className="container max-w-4xl mx-auto px-4">
          <article className="prose prose-lg dark:prose-invert max-w-none">
            
            <h2>Görüntü Segmentasyonu Nedir?</h2>
            <p>
              Görüntü segmentasyonu, bir dijital görüntüyü birden fazla segmente (piksel kümelerine, süper piksellere veya görüntü nesnelerine) bölme işlemidir. Amaç, bir görüntünün temsilini daha anlamlı ve analiz edilmesi daha kolay bir şeye basitleştirmek ve/veya değiştirmektir. Her piksele, ait olduğu nesne veya bölgeye karşılık gelen bir etiket atanır. Bu, görüntü sınıflandırmanın ("Bu görüntüde ne var?") ve nesne tespitinin ("Bu görüntüde hangi nesneler nerede?") ötesine geçerek, "Bu görüntüdeki her bir piksel hangi nesneye veya bölgeye ait?" sorusunu yanıtlar.
            </p>

            <h2>Segmentasyon Türleri</h2>
            <p>Görüntü segmentasyonunun başlıca türleri şunlardır:</p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 my-6">
              {segmentationTypes.map(type => (
                <Card key={type.name} className="flex flex-col text-center bg-secondary/30 p-6">
                  <div className="mx-auto mb-4 p-3 bg-primary/10 rounded-full">{type.icon}</div>
                  <CardTitle className="text-lg mb-2">{type.name}</CardTitle>
                  <CardDescription className="text-sm text-muted-foreground flex-grow">{type.description}</CardDescription>
                </Card>
              ))}
            </div>

            <Separator className="my-8" />

            <h2>Popüler Algoritmalar ve Mimariler</h2>
            <p>Görüntü segmentasyonu için, özellikle derin öğrenme tabanlı birçok etkili model geliştirilmiştir:</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-6">
              {algorithms.map(algo => (
                <Card key={algo.name} className="bg-secondary/30">
                  <CardHeader>
                    <CardTitle className="text-xl flex justify-between items-center">
                      {algo.name} 
                      <span className="text-sm font-normal text-muted-foreground bg-primary/10 px-2 py-0.5 rounded">
                        {algo.year}
                      </span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">{algo.description}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
            
            <Separator className="my-8" />

            <h2>Kullanım Alanları</h2>
            <p>Görüntü segmentasyonu, piksel düzeyinde hassas bilgi gerektiren birçok alanda kritik öneme sahiptir:</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 my-6">
              {applications.map(app => (
                <Card key={app.title} className="flex flex-col items-center text-center p-6 hover:shadow-lg transition-shadow">
                   <div className="p-3 bg-primary/10 rounded-full mb-3">
                    {app.icon}
                  </div>
                  <CardTitle className="text-lg mb-2">{app.title}</CardTitle>
                  <CardContent className="text-sm text-muted-foreground p-0">
                    {app.description}
                  </CardContent>
                </Card>
              ))}
            </div>

            <Separator className="my-8" />

            <h2>Zorluklar</h2>
            <ul>
              <li><strong>Detay ve Hassasiyet:</strong> Nesne sınırlarının doğru bir şekilde belirlenmesi zor olabilir, özellikle ince yapılar veya karmaşık kenarlar söz konusu olduğunda.</li>
              <li><strong>Veri Etiketleme:</strong> Piksel düzeyinde etiketleme, diğer görevlere göre çok daha zaman alıcı ve maliyetlidir.</li>
              <li><strong>Tıkanıklık ve Benzer Görünümlü Nesneler:</strong> Birbirine yakın veya benzer dokulara sahip nesneleri ayırmak zor olabilir.</li>
              <li><strong>Hesaplama Yoğunluğu:</strong> Özellikle yüksek çözünürlüklü görüntüler için piksel düzeyinde tahminler yapmak hesaplama açısından maliyetli olabilir.</li>
            </ul>

            <Separator className="my-8" />

            <h2>Geleceği</h2>
            <p>
              Görüntü segmentasyonu alanındaki araştırmalar, daha az etiketli veriyle (weakly-supervised, self-supervised segmentation) çalışan, daha hızlı ve daha doğru modeller üzerine yoğunlaşmaktadır. Video segmentasyonu, 3D nokta bulutu segmentasyonu gibi alanlar da aktif olarak geliştirilmektedir. Panoptik segmentasyon gibi birleşik yaklaşımlar, daha bütüncül bir sahne anlayışı sunma potansiyeline sahiptir.
            </p>

          </article>

          <div className="mt-12 text-center">
            <Button asChild>
              <Link href="/topics/computer-vision">
                <ArrowLeft className="mr-2 h-4 w-4" /> Bilgisayarlı Görü Ana Konusuna Dön
              </Link>
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
} 