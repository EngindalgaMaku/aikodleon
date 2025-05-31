import { Metadata } from 'next';
import Link from 'next/link';
import Image from 'next/image';
import { ArrowLeft, SearchCode, Crop, BoxSelect, ScanEye, TrafficCone, ShieldAlert, Factory } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';

export const metadata: Metadata = {
  title: 'Nesne Tespiti: Görüntülerdeki Nesneleri Bulma | Kodleon CV Dersleri',
  description: 'Nesne tespitinin ne olduğunu, R-CNN, YOLO, SSD gibi popüler algoritmalarını ve otonom sürüşten güvenliğe kadar uygulama alanlarını Kodleon\'da öğrenin.',
  keywords: 'nesne tespiti, object detection, r-cnn, yolo, ssd, bilgisayarlı görü, görüntü işleme, yapay zeka, makine öğrenmesi, nesne tanıma, sınırlayıcı kutu',
  alternates: {
    canonical: 'https://kodleon.com/topics/computer-vision/object-detection',
  },
  openGraph: {
    title: 'Nesne Tespiti: Görüntülerdeki "Ne" ve "Nerede" | Kodleon',
    description: 'Görüntülerdeki birden fazla nesneyi sadece tanımakla kalmayıp konumlarını da belirleyen nesne tespiti teknolojisini derinlemesine inceleyin.',
    url: 'https://kodleon.com/topics/computer-vision/object-detection',
    images: [
      {
        url: '/images/og/topics/computer-vision/object-detection-og.png', // Bu görselin oluşturulması/var olması gerekli
        width: 1200,
        height: 630,
        alt: 'Kodleon Nesne Tespiti Eğitimi'
      }
    ]
  }
};

const algorithms = [
  {
    name: "R-CNN Ailesi (R-CNN, Fast R-CNN, Faster R-CNN)",
    description: "Bölge tabanlı evrişimli sinir ağları. Önce potansiyel nesne bölgeleri önerir, sonra bu bölgeleri sınıflandırır. Doğrulukları yüksek ancak yavaş olabilirler.",
    type: "Bölge Tabanlı (Two-Stage)"
  },
  {
    name: "YOLO (You Only Look Once)",
    description: "Görüntüye tek bir kez bakarak nesne tespiti yapar. Sınırlayıcı kutuları ve sınıf olasılıklarını aynı anda tahmin eder. Çok hızlıdır, gerçek zamanlı uygulamalar için idealdir.",
    type: "Tek Aşamalı (One-Stage)"
  },
  {
    name: "SSD (Single Shot MultiBox Detector)",
    description: "YOLO'ya benzer şekilde tek bir ağ geçişiyle çalışır. Farklı ölçeklerdeki nesneleri tespit etmek için farklı katmanlardan özellik haritaları kullanır. Hız ve doğruluk arasında iyi bir denge sunar.",
    type: "Tek Aşamalı (One-Stage)"
  },
  {
    name: "RetinaNet",
    description: "Tek aşamalı dedektörlerdeki sınıf dengesizliği sorununu çözmek için \"Focal Loss\" kavramını tanıtmıştır. Yüksek doğruluk ve hız sunar.",
    type: "Tek Aşamalı (One-Stage)"
  },
  {
    name: "DETR (Detection Transformer)",
    description: "Nesne tespitini bir küme tahmini problemi olarak ele alır ve Transformer mimarisini kullanır. Bölge önerisi veya non-maximum suppression gibi adımlara ihtiyaç duymaz.",
    type: "Transformer Tabanlı"
  }
];

const applications = [
  { title: "Otonom Sürüş", description: "Araçların, yayaların, trafik işaretlerinin ve şeritlerin tespiti.", icon: <TrafficCone className="h-8 w-8 text-orange-500" /> },
  { title: "Güvenlik ve Gözetim", description: "İnsanların, şüpheli nesnelerin veya belirli olayların (kavga, hırsızlık) tespiti.", icon: <ShieldAlert className="h-8 w-8 text-red-500" /> },
  { title: "Perakendecilik", description: "Müşteri sayımı, raf düzeni analizi, ürün takibi.", icon: <ScanEye className="h-8 w-8 text-blue-500" /> },
  { title: "Endüstriyel Otomasyon", description: "Üretim hattında hatalı parçaların tespiti, robotik kollar için nesne konumlandırma.", icon: <Factory className="h-8 w-8 text-gray-500" /> },
  { title: "Tıbbi Görüntüleme", description: "Organların, tümörlerin veya diğer anormalliklerin tespiti ve konumlandırılması.", icon: <Crop className="h-8 w-8 text-green-500" /> },
  { title: "Tarım", description: "Yabani otların, zararlıların veya olgunlaşmış ürünlerin tespiti.", icon: <BoxSelect className="h-8 w-8 text-teal-500" /> }
];

export default function ObjectDetectionPage() {
  return (
    <div className="bg-background text-foreground">
      <section className="relative py-16 md:py-24 bg-gradient-to-br from-blue-500/5 via-transparent to-green-500/5">
        <div className="container max-w-5xl mx-auto px-4 text-center">
          <div className="inline-flex items-center gap-2 mb-4">
            <Button asChild variant="ghost" size="sm" className="gap-1 text-primary hover:text-primary/80">
              <Link href="/topics/computer-vision">
                <ArrowLeft className="h-4 w-4" />
                Bilgisayarlı Görü Ana Sayfası
              </Link>
            </Button>
          </div>
          <SearchCode className="h-16 w-16 text-primary mx-auto mb-6" />
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
            Nesne Tespiti
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto">
            Görüntülerdeki nesneleri sadece sınıflandırmakla kalmayıp, aynı zamanda nerede olduklarını da belirlemek! Nesne tespiti, bilgisayarlı görünün bu heyecan verici görevini üstlenir ve çevremizdeki dünyayı makinelerin daha derinlemesine anlamasını sağlar.
          </p>
        </div>
      </section>

      <section className="py-12 md:py-20">
        <div className="container max-w-4xl mx-auto px-4">
          <article className="prose prose-lg dark:prose-invert max-w-none">
            
            <h2>Nesne Tespiti Nedir?</h2>
            <p>
              Nesne tespiti (Object Detection), bir görüntü veya video karesindeki bir veya daha fazla nesnenin varlığını, sınıfını (örneğin, kedi, araba, insan) ve konumunu (genellikle bir sınırlayıcı kutu - bounding box ile) belirleme görevidir. Görüntü sınıflandırma sadece "Bu görüntüde ne var?" sorusuna cevap verirken, nesne tespiti "Bu görüntüde hangi nesneler var ve tam olarak neredeler?" sorularını yanıtlar.
            </p>
            <p>
              Bu, her nesne için hem bir sınıf etiketi hem de o nesneyi çevreleyen koordinatları tahmin etmeyi içerir. Karmaşıklığı nedeniyle, genellikle görüntü sınıflandırmadan daha zorlu bir görev olarak kabul edilir.
            </p>

            <Separator className="my-8" />

            <h2>Temel Yaklaşımlar</h2>
            <p>Nesne tespiti algoritmaları genel olarak iki ana kategoriye ayrılabilir:</p>
            <ol>
              <li><strong>Bölge Tabanlı (Two-Stage Detectors):</strong> Bu yaklaşımda önce görüntüde potansiyel nesneleri içerebilecek "bölge önerileri" (region proposals) oluşturulur. Ardından bu önerilen bölgeler bir sınıflandırıcıya gönderilerek nesne olup olmadığı ve hangi sınıfa ait olduğu belirlenir. R-CNN, Fast R-CNN ve Faster R-CNN bu kategoriye örnektir. Genellikle daha yüksek doğruluk sunarlar ancak daha yavaştırlar.</li>
              <li><strong>Tek Aşamalı (One-Stage Detectors):</strong> Bu yaklaşımda ise nesne sınıflandırması ve konumlandırması tek bir ağ geçişiyle, eş zamanlı olarak yapılır. YOLO ve SSD bu kategorinin popüler örnekleridir. Genellikle daha hızlıdırlar ve gerçek zamanlı uygulamalar için daha uygundurlar, ancak doğrulukları iki aşamalı dedektörlere göre biraz daha düşük olabilir.</li>
            </ol>

            <Separator className="my-8" />

            <h2>Popüler Algoritmalar ve Mimariler</h2>
            <p>Nesne tespiti alanında çığır açan ve yaygın olarak kullanılan bazı önemli algoritmalar şunlardır:</p>
            <div className="space-y-6 my-6">
              {algorithms.map(algo => (
                <Card key={algo.name} className="bg-secondary/30">
                  <CardHeader>
                    <CardTitle className="text-xl">{algo.name}</CardTitle>
                    <CardDescription className="text-sm pt-1">Yaklaşım Türü: {algo.type}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">{algo.description}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
            
            <Separator className="my-8" />

            <h2>Kullanım Alanları</h2>
            <p>Nesne tespiti, teknolojinin birçok farklı alanında devrim yaratmaktadır:</p>
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

            <h2>Zorluklar ve Dikkat Edilmesi Gerekenler</h2>
            <ul>
              <li><strong>Ölçek Değişimi:</strong> Nesneler görüntülerde farklı boyutlarda görünebilir.</li>
              <li><strong>Tıkanıklık (Occlusion):</strong> Nesneler birbirlerini veya başka nesneler tarafından kısmen örtülebilir.</li>
              <li><strong>Deformasyon:</strong> Esnek nesneler farklı şekillerde görünebilir.</li>
              <li><strong>Aydınlatma Koşulları:</strong> Farklı ışıklandırma koşulları nesnelerin görünümünü önemli ölçüde değiştirebilir.</li>
              <li><strong>Sınıf İçi Varyasyon:</strong> Aynı sınıfa ait nesneler arasında büyük farklılıklar olabilir (örn: farklı köpek türleri).</li>
              <li><strong>Sınırlayıcı Kutu Hassasiyeti:</strong> Nesneyi tam olarak saran doğru bir sınırlayıcı kutu elde etmek zor olabilir.</li>
              <li><strong>Gerçek Zamanlı Performans İhtiyacı:</strong> Birçok uygulama, yüksek hızda ve düşük gecikmeyle çalışan tespit sistemleri gerektirir.</li>
            </ul>

            <Separator className="my-8" />

            <h2>Geleceği</h2>
            <p>
              Nesne tespiti araştırmaları, daha doğru, daha hızlı ve daha az veriyle çalışabilen modeller geliştirmeye odaklanmıştır. Transformer tabanlı modeller (DETR gibi) ve kendi kendine öğrenme (self-supervised learning) yaklaşımları bu alandaki yeni ve heyecan verici gelişmeler arasındadır. Ayrıca, 3D nesne tespiti ve video içerisinde nesne takibi gibi daha karmaşık görevlere yönelik çalışmalar da hızla ilerlemektedir.
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