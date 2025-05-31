"use client";

import Link from "next/link";
import Image from "next/image";
import { ArrowLeft, ArrowRight, Brain, Layers, Cpu, Zap, Lightbulb, CheckCircle, Eye, Ear, Repeat, ThumbsUp, ThumbsDown, Network, FunctionSquare, Settings2, BookOpen, Code, AlertCircle, Car, Heart } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { ReactNode } from "react";

interface ContentListItem {
  type: "list-item";
  item: string;
  icon?: ReactNode;
}

interface ContentSubHeading {
  type: "sub-heading";
  text: string;
  icon: ReactNode;
}

type ContentItem = string | ContentListItem | ContentSubHeading;

interface Section {
  id: string;
  title: string;
  icon: ReactNode;
  content: ContentItem[];
}

const pageTitle = "Derin Öğrenme Temelleri";
const pageDescription = "Yapay sinir ağlarının katmanlı yapısını, temel mimarilerini (CNN, RNN) ve derin öğrenmenin makine öğrenmesindeki devrimsel etkilerini keşfedin.";
const pageKeywords = "derin öğrenme, yapay sinir ağları, CNN, RNN, geri yayılım, aktivasyon fonksiyonları, ileri yayılım, katmanlar, nöronlar, kodleon";
const pageUrl = "https://kodleon.com/topics/machine-learning/deep-learning-basics";
const imageUrl = "https://images.pexels.com/photos/2599244/pexels-photo-2599244.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2";

// Moved icon definitions before usage
// const Car = (props: any) => <Code {...props} /> // Placeholder, will be removed if lucide-react has Car
// const Heart = (props: any) => <Code {...props} /> // Placeholder, will be removed if lucide-react has Heart

const sections: Section[] = [
  {
    id: "yapay-sinir-aglari",
    title: "Yapay Sinir Ağları (YSA)",
    icon: <Brain className="w-8 h-8 text-pink-500" />,
    content: [
      "Derin öğrenmenin temelini yapay sinir ağları oluşturur. İnsan beynindeki nöronların çalışma şeklinden esinlenilmiştir.",
      { type: "list-item", item: "**Nöronlar (Neurons):** Ağın temel hesaplama birimleridir. Girdileri alır, bir işlem uygular ve bir çıktı üretir.", icon: <Cpu className="w-5 h-5 text-pink-500 mr-2" /> },
      { type: "list-item", item: "**Katmanlar (Layers):** Nöronlar katmanlar halinde düzenlenir: Giriş Katmanı (Input Layer), Gizli Katman(lar) (Hidden Layer(s)) ve Çıkış Katmanı (Output Layer). Derin öğrenme, birden fazla gizli katmanın kullanılmasını ifade eder.", icon: <Layers className="w-5 h-5 text-pink-500 mr-2" /> },
      { type: "list-item", item: "**Ağırlıklar (Weights) ve Biaslar (Biases):** Her bağlantının bir ağırlığı vardır ve her nöronun bir bias değeri olabilir. Modelin öğrendiği parametrelerdir.", icon: <Settings2 className="w-5 h-5 text-pink-500 mr-2" /> },
      { type: "list-item", item: "**Aktivasyon Fonksiyonları (Activation Functions):** Nöronun çıktısını belirler ve ağa doğrusal olmayanlık katar. (Örn: Sigmoid, ReLU, Tanh).", icon: <FunctionSquare className="w-5 h-5 text-pink-500 mr-2" /> },
    ]
  },
  {
    id: "ogrenme-sureci",
    title: "Öğrenme Süreci",
    icon: <Repeat className="w-8 h-8 text-indigo-500" />,
    content: [
      "YSA'lar etiketli verilerle (genellikle denetimli öğrenme) eğitilir ve bir kayıp fonksiyonunu minimize etmeye çalışır.",
      { type: "list-item", item: "**İleri Yayılım (Forward Propagation):** Girdiler ağ üzerinden ileri doğru hareket eder ve bir tahmin üretilir.", icon: <ArrowRight className="w-5 h-5 text-indigo-500 mr-2" /> },
      { type: "list-item", item: "**Kayıp Fonksiyonu (Loss Function):** Modelin tahminleri ile gerçek değerler arasındaki farkı ölçer.", icon: <ThumbsDown className="w-5 h-5 text-indigo-500 mr-2" /> },
      { type: "list-item", item: "**Geri Yayılım (Backpropagation):** Kayıp, ağ üzerinden geriye doğru yayılır ve ağırlıklar ile biaslar bu kayba göre güncellenir (genellikle Gradyan İnişi optimizasyon algoritması ile).", icon: <ArrowLeft className="w-5 h-5 text-indigo-500 mr-2" /> },
    ]
  },
  {
    id: "temel-mimari",
    title: "Temel Derin Öğrenme Mimarileri",
    icon: <Network className="w-8 h-8 text-teal-500" />,
    content: [
      "Farklı görev türleri için özelleşmiş birçok YSA mimarisi bulunmaktadır:",
      { type: "list-item", item: "**Evrişimli Sinir Ağları (Convolutional Neural Networks - CNNs):** Özellikle görüntü tanıma, nesne tespiti gibi bilgisayarlı görü görevlerinde başarılıdır. Evrişim (convolution), havuzlama (pooling) gibi özel katmanlar içerir.", icon: <Eye className="w-5 h-5 text-teal-500 mr-2" /> },
      { type: "list-item", item: "**Tekrarlayan Sinir Ağları (Recurrent Neural Networks - RNNs):** Sıralı verileri (örn: zaman serileri, metin, konuşma) işlemek için tasarlanmıştır. Gizli durumları sayesinde geçmiş bilgileri hatırlayabilirler. LSTM ve GRU gibi gelişmiş varyantları bulunur.", icon: <Ear className="w-5 h-5 text-teal-500 mr-2" /> },
      { type: "list-item", item: "**Transformerlar:** Özellikle Doğal Dil İşleme (NLP) alanında devrim yaratmıştır. Dikkat (attention) mekanizmalarını kullanarak uzun mesafeli bağımlılıkları etkili bir şekilde modelleyebilirler.", icon: <BookOpen className="w-5 h-5 text-teal-500 mr-2" /> }
    ]
  },
  {
    id: "uygulama-alanlari",
    title: "Uygulama Alanları",
    icon: <Zap className="w-8 h-8 text-orange-500" />,
    content: [
      "Derin öğrenme günümüzde birçok alanda devrim yaratmaktadır:",
      { type: "list-item", item: "Bilgisayarlı Görü (Görüntü sınıflandırma, nesne tespiti, yüz tanıma)", icon: <Eye className="w-5 h-5 text-orange-500 mr-2" /> },
      { type: "list-item", item: "Doğal Dil İşleme (Makine çevirisi, duygu analizi, metin üretimi)", icon: <BookOpen className="w-5 h-5 text-orange-500 mr-2" /> },
      { type: "list-item", item: "Konuşma Tanıma ve Üretme", icon: <Ear className="w-5 h-5 text-orange-500 mr-2" /> },
      { type: "list-item", item: "Otonom Araçlar", icon: <Car className="w-5 h-5 text-orange-500 mr-2" /> }, // Assuming Car icon is available, if not, use a generic one like Code or Settings2
      { type: "list-item", item: "Sağlık (Hastalık teşhisi, ilaç keşfi)", icon: <Heart className="w-5 h-5 text-orange-500 mr-2" /> }, // Assuming Heart icon, if not, use a generic one
      { type: "list-item", item: "Öneri Sistemleri", icon: <ThumbsUp className="w-5 h-5 text-orange-500 mr-2" /> }
    ]
  },
  {
    id: "avantajlar-ve-limitler",
    title: "Avantajları ve Limitleri",
    icon: <ThumbsUp className="w-8 h-8 text-green-500" />,
    content: [
      { type: "sub-heading", text: "Avantajları:", icon: <CheckCircle className="w-6 h-6 text-green-500 mr-2"/> },
      { type: "list-item", item: "Büyük ve karmaşık veri kümelerinden otomatik olarak özellik öğrenebilme."}, 
      { type: "list-item", item: "Birçok alanda insan performansına yakın veya onu aşan sonuçlar elde etme."}, 
      { type: "list-item", item: "Esnek mimariler sayesinde farklı problem türlerine uyarlanabilme."},
      { type: "sub-heading", text: "Limitleri:", icon: <AlertCircle className="w-6 h-6 text-red-500 mr-2 mt-4" /> },
      { type: "list-item", item: "Büyük miktarda etiketli veriye ihtiyaç duyma (genellikle)."}, 
      { type: "list-item", item: "Yüksek hesaplama gücü gerektirme (eğitim süreci uzun olabilir)."}, 
      { type: "list-item", item: "'Kara kutu' (black box) doğası nedeniyle kararlarının yorumlanması zor olabilir."},
      { type: "list-item", item: "Aşırı uyum (overfitting) riski ve dikkatli hiperparametre ayarı gerektirmesi."}
    ]
  }
];

// Helper for car and heart icons if not directly available in lucide-react, replace with actual or alternative icons.
// const Car = (props: any) => <Code {...props} /> // Placeholder
// const Heart = (props: any) => <Code {...props} /> // Placeholder

export default function DeepLearningBasicsPage() {
  return (
    <div className="container max-w-6xl mx-auto py-12 px-4 md:px-6">
      <section className="mb-12">
        <div className="flex items-center mb-4">
          <Button asChild variant="ghost" size="sm" className="gap-1 text-muted-foreground hover:text-primary">
            <Link href="/topics/machine-learning" aria-label="Makine Öğrenmesi konusuna geri dön">
              <ArrowLeft className="h-4 w-4" />
              Makine Öğrenmesi
            </Link>
          </Button>
        </div>
        <div className="relative h-[300px] md:h-[350px] rounded-lg overflow-hidden mb-8 shadow-xl">
          <Image
            src={imageUrl}
            alt={`${pageTitle} Kapak Fotoğrafı`}
            fill
            className="object-cover"
            priority
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent flex flex-col justify-end p-6 md:p-8">
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-3 leading-tight drop-shadow-md">{pageTitle}</h1>
            <p className="text-lg md:text-xl text-gray-100 drop-shadow-sm">
              {pageDescription}
            </p>
          </div>
        </div>
      </section>

      <section className="mb-16 prose prose-lg dark:prose-invert max-w-none">
        <h2 id="giris" className="text-3xl font-semibold border-b pb-2 mb-6">Giriş: Derin Öğrenme Nedir?</h2>
        <p>
          Derin Öğrenme (Deep Learning), makine öğrenmesinin bir alt alanıdır ve büyük miktarda veriden karmaşık desenleri ve hiyerarşik özellikleri öğrenmek için tasarlanmış yapay sinir ağlarını (özellikle çok katmanlı olanları) kullanır. İnsan beyninin bilgi işleme şeklinden ilham alan bu yaklaşım, son yıllarda görüntü tanıma, doğal dil işleme, konuşma tanıma ve otonom sürüş gibi birçok alanda çığır açan başarılara imza atmıştır.
        </p>
        <p>
          Geleneksel makine öğrenmesi yöntemlerinde genellikle özellik mühendisliği (feature engineering) adı verilen, uzman bilgisi gerektiren ve zaman alıcı bir süreçle veriden anlamlı özellikler çıkarılır. Derin öğrenme modelleri ise bu özellikleri otomatik olarak ve katmanlı bir yapıda (düşük seviyeli özelliklerden yüksek seviyeli özelliklere doğru) öğrenme yeteneğine sahiptir. Bu, onları özellikle karmaşık ve yüksek boyutlu verilerle çalışmak için güçlü kılar.
        </p>
      </section>

      <Separator className="my-12" />

      <section className="mb-16">
        <h2 className="text-3xl font-bold mb-10 text-center">Derin Öğrenmenin Temel Taşları</h2>
        <div className="space-y-8">
          {sections.map((section) => (
            <Card key={section.id} className="shadow-lg hover:shadow-xl transition-shadow duration-300 overflow-hidden">
              <CardHeader className="bg-muted/50">
                <div className="flex items-start gap-4">
                  <div className="p-3 bg-primary/10 rounded-lg flex-shrink-0">
                    {section.icon}
                  </div>
                  <div>
                    <CardTitle id={section.id} className="text-2xl font-semibold mb-1">{section.title}</CardTitle>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="pt-6 text-base">
                {section.content.map((item, index) => {
                  if (typeof item === 'string') {
                    return <p key={index} className="mb-3" dangerouslySetInnerHTML={{ __html: item.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />;
                  } else if (item.type === 'sub-heading') {
                    return (
                      <h4 key={index} className={`flex items-center font-semibold text-lg mb-2 ${index > 0 ? 'mt-4' : ''}`}>
                        {item.icon}
                        {item.text}
                      </h4>
                    );
                  } else if (item.type === 'list-item') {
                    return (
                      <div key={index} className="flex items-start mb-2 ml-2">
                        {item.icon || <CheckCircle className="w-5 h-5 text-green-500 mr-2 mt-1 flex-shrink-0" />}
                        <p dangerouslySetInnerHTML={{ __html: item.item.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
                      </div>
                    );
                  }
                  return null;
                })}
              </CardContent>
            </Card>
          ))}
        </div>
      </section>
      
      <Separator className="my-16" />

      <section className="mb-12 text-center bg-muted p-8 rounded-lg shadow-inner">
        <h2 className="text-3xl font-bold mb-6">Derin Öğrenme ile Verinin Derinliklerine Yolculuk</h2>
        <p className="text-lg text-muted-foreground mb-8 max-w-3xl mx-auto">
          Derin öğrenme, yapay zekanın en heyecan verici ve hızla gelişen alanlarından biridir. Temel kavramları ve mimarileri anlayarak, bu güçlü teknolojinin sunduğu sonsuz olasılıkları keşfetmeye başlayabilirsiniz.
        </p>
        <div className="flex flex-wrap justify-center items-center gap-4">
            <Button asChild size="lg" className="bg-primary hover:bg-primary/90 text-primary-foreground">
                <Link href="/topics/machine-learning">
                  <Lightbulb className="mr-2 h-5 w-5" /> Makine Öğrenmesi Ana Konusuna Dön
                </Link>
            </Button>
            <Button asChild variant="outline" size="lg">
                <Link href="/topics">
                  <Zap className="mr-2 h-5 w-5" /> Tüm Yapay Zeka Konularını Keşfet
                </Link>
            </Button>
        </div>
      </section>

      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": pageTitle,
            "description": pageDescription,
            "keywords": pageKeywords,
            "image": imageUrl,
            "author": {
              "@type": "Organization",
              "name": "Kodleon",
              "url": "https://kodleon.com"
            },
            "publisher": {
              "@type": "Organization",
              "name": "Kodleon",
              "logo": {
                "@type": "ImageObject",
                "url": "https://kodleon.com/logo.png" 
              }
            },
            "datePublished": "2023-10-30", 
            "dateModified": new Date().toISOString().split('T')[0],
            "mainEntityOfPage": {
              "@type": "WebPage",
              "@id": pageUrl
            }
          })
        }}
      />
    </div>
  );
} 