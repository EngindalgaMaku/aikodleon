import Link from "next/link";
import Image from "next/image";
import { ArrowLeft, ArrowRight, Brain, CheckCircle2, Ship as Chip, Database, Eye, FileText, Lightbulb, Rocket, Shapes, Users } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

// Topic data mapping
const topicsData: Record<string, any> = {
  "machine-learning": {
    title: "Makine Öğrenmesi",
    description: "Algoritmaların veri kullanarak nasıl öğrendiğini ve tahminlerde bulunduğunu keşfedin.",
    icon: <Database className="h-8 w-8 text-chart-1" />,
    imageUrl: "https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    longDescription: "Makine öğrenmesi, bilgisayarların açık programlamaya gerek kalmadan öğrenmesini sağlayan yapay zeka uygulamalarıdır. Algoritmaların verilerden öğrenerek tahminlerde bulunmasını ve kararlar vermesini sağlar. Denetimli öğrenme, denetimsiz öğrenme ve pekiştirmeli öğrenme gibi farklı yaklaşımlar içerir.",
    subtopics: [
      {
        title: "Denetimli Öğrenme",
        description: "Etiketli verilerle modellerin nasıl eğitildiğini ve tahminlerde bulunduğunu öğrenin.",
        imageUrl: "https://images.pexels.com/photos/577585/pexels-photo-577585.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      },
      {
        title: "Denetimsiz Öğrenme",
        description: "Etiketlenmemiş verilerden kalıpları ve yapıları nasıl keşfedeceğinizi anlayın.",
        imageUrl: "https://images.pexels.com/photos/373543/pexels-photo-373543.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      },
      {
        title: "Pekiştirmeli Öğrenme",
        description: "Deneme yanılma yoluyla ajanların çevreleriyle nasıl etkileşime girdiğini ve öğrendiğini keşfedin.",
        imageUrl: "https://images.pexels.com/photos/6153354/pexels-photo-6153354.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      },
      {
        title: "Derin Öğrenme Temelleri",
        description: "Derin öğrenmenin temel kavramlarını ve yapay sinir ağlarının çalışma prensiplerini öğrenin.",
        imageUrl: "https://images.pexels.com/photos/2599244/pexels-photo-2599244.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      }
    ],
    skills: ["Veri Analizi", "Python", "Algoritma Tasarımı", "Model Değerlendirme", "Veri Ön İşleme"],
    resources: [
      { title: "Makine Öğrenmesi Temelleri", type: "Kurs", link: "#" },
      { title: "Scikit-Learn ile Uygulamalı ML", type: "Pratik", link: "#" },
      { title: "Makine Öğrenmesi Algoritmaları", type: "E-Kitap", link: "#" }
    ]
  },
  "nlp": {
    title: "Doğal Dil İşleme",
    description: "Makinelerin insan dilini nasıl anlayıp işlediğini ve ürettiğini öğrenin.",
    icon: <FileText className="h-8 w-8 text-chart-2" />,
    imageUrl: "https://images.pexels.com/photos/7412095/pexels-photo-7412095.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    longDescription: "Doğal Dil İşleme (NLP), bilgisayarların insan dilini anlama, işleme ve üretme yeteneğidir. Metin sınıflandırma, duygu analizi, makine çevirisi ve soru cevaplama gibi uygulamaları içerir. Büyük dil modelleri (LLM'ler) ile gelişen bir alandır.",
    subtopics: [
      {
        title: "Metin Analizi",
        description: "Metinleri işleme, temizleme ve yapılandırma teknikleri.",
        imageUrl: "https://images.pexels.com/photos/267669/pexels-photo-267669.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      },
      {
        title: "Dil Modelleri",
        description: "BERT, GPT ve diğer büyük dil modellerinin çalışma prensipleri.",
        imageUrl: "https://images.pexels.com/photos/1181271/pexels-photo-1181271.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      },
      {
        title: "Duygu Analizi",
        description: "Metinlerden duygu ve görüşleri çıkarma yöntemleri.",
        imageUrl: "https://images.pexels.com/photos/590022/pexels-photo-590022.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      },
      {
        title: "Makine Çevirisi",
        description: "Diller arası otomatik çeviri sistemlerinin çalışma prensipleri.",
        imageUrl: "https://images.pexels.com/photos/267669/pexels-photo-267669.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      }
    ],
    skills: ["Metin İşleme", "Vektör Temsilleri", "Dil Modellemesi", "Transformer Mimarileri", "Semantik Analiz"],
    resources: [
      { title: "NLP Temelleri", type: "Kurs", link: "#" },
      { title: "Dil Modelleriyle Çalışma", type: "Atölye", link: "#" },
      { title: "Duygu Analizi Projesi", type: "Pratik", link: "#" }
    ]
  },
  "computer-vision": {
    title: "Bilgisayarlı Görü",
    description: "Bilgisayarların görüntüleri nasıl algıladığını ve işlediğini anlayın.",
    icon: <Eye className="h-8 w-8 text-chart-3" />,
    imageUrl: "https://images.pexels.com/photos/8438922/pexels-photo-8438922.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    longDescription: "Bilgisayarlı görü, makinelerin dijital görüntüleri veya videoları anlama ve işleme yeteneğidir. Görüntü sınıflandırma, nesne algılama, yüz tanıma ve görüntü segmentasyonu gibi uygulamaları içerir. Konvolüsyonel sinir ağları (CNN'ler) gibi derin öğrenme tekniklerini kullanır.",
    subtopics: [
      {
        title: "Görüntü Sınıflandırma",
        description: "Görüntüleri kategorilere ayırma teknikleri.",
        imageUrl: "https://images.pexels.com/photos/60504/security-protection-anti-virus-software-60504.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      },
      {
        title: "Nesne Tespiti",
        description: "Görüntülerdeki nesneleri tespit etme ve konumlandırma.",
        imageUrl: "https://images.pexels.com/photos/762679/pexels-photo-762679.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      },
      {
        title: "Yüz Tanıma",
        description: "Yüz tanıma sistemlerinin çalışma prensipleri ve uygulamaları.",
        imageUrl: "https://images.pexels.com/photos/6203795/pexels-photo-6203795.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      },
      {
        title: "Görüntü Segmentasyonu",
        description: "Görüntüleri anlamlı bölgelere ayırma teknikleri.",
        imageUrl: "https://images.pexels.com/photos/2599244/pexels-photo-2599244.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      }
    ],
    skills: ["OpenCV", "Konvolüsyonel Sinir Ağları", "Görüntü İşleme", "Öznitelik Çıkarımı", "PyTorch/TensorFlow"],
    resources: [
      { title: "Bilgisayarlı Görü Temelleri", type: "Kurs", link: "#" },
      { title: "Nesne Algılama Projesi", type: "Pratik", link: "#" },
      { title: "Derin Öğrenme ile Görüntü İşleme", type: "E-Kitap", link: "#" }
    ]
  },
  "generative-ai": {
    title: "Üretken AI",
    description: "Metin, görüntü ve ses üretebilen yapay zeka modellerini keşfedin.",
    icon: <Lightbulb className="h-8 w-8 text-chart-4" />,
    imageUrl: "https://images.pexels.com/photos/8386434/pexels-photo-8386434.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    longDescription: "Üretken AI, yeni içerik oluşturabilen yapay zeka sistemlerini ifade eder. Generative Adversarial Networks (GAN'lar), Variational Autoencoders (VAE'ler) ve diffusion modelleri gibi tekniklerle metin, görüntü, ses ve video üretebilir. ChatGPT, DALL-E ve Midjourney gibi araçlar bu alandaki önemli örneklerdir.",
    subtopics: [
      {
        title: "Üretken Çekişmeli Ağlar (GAN)",
        description: "GAN'ların yapısı ve gerçekçi içerik üretme yöntemleri.",
        imageUrl: "https://images.pexels.com/photos/7567434/pexels-photo-7567434.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      },
      {
        title: "Diffusion Modelleri",
        description: "Diffusion modellerinin çalışma prensipleri ve uygulamaları.",
        imageUrl: "https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      },
      {
        title: "Büyük Dil Modelleri",
        description: "Metin üreten yapay zeka modellerinin yapısı ve eğitimi.",
        imageUrl: "https://images.pexels.com/photos/3861969/pexels-photo-3861969.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      },
      {
        title: "Text-to-Image Modelleri",
        description: "Metinden görüntü üreten modellerin çalışma prensipleri.",
        imageUrl: "https://images.pexels.com/photos/8566460/pexels-photo-8566460.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      }
    ],
    skills: ["Derin Öğrenme", "Model Mimarisi", "Fine-tuning", "Prompt Engineering", "Yaratıcı AI Uygulamaları"],
    resources: [
      { title: "Üretken AI Temelleri", type: "Kurs", link: "#" },
      { title: "GAN ile Görüntü Üretimi", type: "Pratik", link: "#" },
      { title: "LLM'lerle Çalışma", type: "Atölye", link: "#" }
    ]
  },
  "neural-networks": {
    title: "Sinir Ağları",
    description: "Beynin çalışma prensibinden esinlenen yapay sinir ağları hakkında bilgi edinin.",
    icon: <Brain className="h-8 w-8 text-chart-5" />,
    imageUrl: "https://images.pexels.com/photos/8386421/pexels-photo-8386421.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    longDescription: "Yapay sinir ağları, insan beyninin nöral yapısından esinlenen hesaplama sistemleridir. Katmanlar halinde düzenlenmiş nöronlardan oluşur ve karmaşık örüntüleri öğrenebilir. Derin öğrenmenin temelini oluşturur ve görüntü tanıma, dil anlama ve karar verme gibi birçok AI uygulamasında kullanılır.",
    subtopics: [
      {
        title: "Temel Sinir Ağı Mimarileri",
        description: "Temel yapay sinir ağı yapıları ve çalışma prensipleri.",
        imageUrl: "https://images.pexels.com/photos/1181271/pexels-photo-1181271.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      },
      {
        title: "Konvolüsyonel Sinir Ağları (CNN)",
        description: "Görüntü işlemede kullanılan CNN'lerin yapısı ve uygulamaları.",
        imageUrl: "https://images.pexels.com/photos/60504/security-protection-anti-virus-software-60504.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      },
      {
        title: "Tekrarlayan Sinir Ağları (RNN)",
        description: "Dizileri işleyen RNN'lerin yapısı ve kullanım alanları.",
        imageUrl: "https://images.pexels.com/photos/577585/pexels-photo-577585.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      },
      {
        title: "Transformerlar",
        description: "Modern NLP'nin temelini oluşturan transformer mimarisi.",
        imageUrl: "https://images.pexels.com/photos/267669/pexels-photo-267669.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
      }
    ],
    skills: ["Backpropagation", "Aktivasyon Fonksiyonları", "Hiperparametre Ayarlama", "Gradient Descent", "Model Optimizasyonu"],
    resources: [
      { title: "Sinir Ağları Temelleri", type: "Kurs", link: "#" },
      { title: "PyTorch ile Neural Networks", type: "Pratik", link: "#" },
      { title: "Derin Öğrenme Mimarileri", type: "E-Kitap", link: "#" }
    ]
  }
};

// List of all topics for related topics section
const allTopicSlugs = [
  "machine-learning", 
  "nlp", 
  "computer-vision", 
  "generative-ai", 
  "neural-networks", 
  "ai-ethics"
];

export default function TopicPage({ params }: { params: { slug: string } }) {
  const { slug } = params;
  const topic = topicsData[slug] || {
    title: "Konu Bulunamadı",
    description: "İstediğiniz konu şu anda mevcut değil.",
    imageUrl: "https://images.pexels.com/photos/3861969/pexels-photo-3861969.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    subtopics: [],
    skills: [],
    resources: []
  };
  
  // Get 3 related topics (excluding current one)
  const relatedTopicSlugs = allTopicSlugs
    .filter(s => s !== slug)
    .sort(() => 0.5 - Math.random())
    .slice(0, 3);
  
  const relatedTopics = relatedTopicSlugs.map(s => ({
    slug: s,
    ...topicsData[s]
  }));

  return (
    <div>
      {/* Hero section */}
      <section className="relative">
        <div className="relative h-[300px] md:h-[400px]">
          <Image 
            src={topic.imageUrl}
            alt={topic.title}
            fill
            className="object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-background via-background/80 to-transparent" />
        </div>
        <div className="container relative -mt-32 pb-12">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 mb-4">
              <Button asChild variant="ghost" size="sm" className="gap-1">
                <Link href="/topics">
                  <ArrowLeft className="h-4 w-4" />
                  Tüm Konular
                </Link>
              </Button>
            </div>
            <div className="flex items-center gap-4 mb-6">
              <div className="p-3 rounded-full bg-primary/10 backdrop-blur-sm">
                {topic.icon}
              </div>
              <h1 className="text-4xl font-bold">{topic.title}</h1>
            </div>
            <p className="text-xl text-muted-foreground">
              {topic.description}
            </p>
          </div>
        </div>
      </section>
      
      {/* Main content */}
      <section className="container py-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
          <div className="lg:col-span-2">
            <div className="prose prose-lg dark:prose-invert max-w-none">
              <h2>Genel Bakış</h2>
              <p>{topic.longDescription}</p>
              
              <h2>Alt Konular</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 not-prose">
                {topic.subtopics?.map((subtopic: any, index: number) => (
                  <Card key={index} className="overflow-hidden">
                    <div className="relative h-40">
                      <Image 
                        src={subtopic.imageUrl}
                        alt={subtopic.title}
                        fill
                        className="object-cover"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent" />
                    </div>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg">{subtopic.title}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground">{subtopic.description}</p>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </div>
          
          <div>
            <div className="bg-muted rounded-lg p-6 sticky top-24">
              <h3 className="text-xl font-medium mb-4">Bu Konuda Kazanacağınız Beceriler</h3>
              <ul className="space-y-3 mb-6">
                {topic.skills?.map((skill: string, index: number) => (
                  <li key={index} className="flex items-start gap-2">
                    <CheckCircle2 className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                    <span>{skill}</span>
                  </li>
                ))}
              </ul>
              
              <Separator className="my-6" />
              
              <h3 className="text-xl font-medium mb-4">Önerilen Kaynaklar</h3>
              <ul className="space-y-4">
                {topic.resources?.map((resource: any, index: number) => (
                  <li key={index}>
                    <Link 
                      href={resource.link} 
                      className="flex items-center justify-between p-3 bg-background rounded-md hover:bg-secondary transition-colors"
                    >
                      <span className="font-medium">{resource.title}</span>
                      <span className="text-sm text-muted-foreground">{resource.type}</span>
                    </Link>
                  </li>
                ))}
              </ul>
              
              <Button className="w-full mt-6 rounded-full">Derse Kaydol</Button>
            </div>
          </div>
        </div>
      </section>
      
      {/* Related topics */}
      <section className="bg-muted py-16">
        <div className="container">
          <h2 className="text-2xl font-bold mb-8">İlgili Konular</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {relatedTopics.map((relatedTopic, index) => (
              <Card key={index} className="overflow-hidden transition-all duration-300 hover:shadow-lg hover:-translate-y-1">
                <div className="relative h-48">
                  <Image 
                    src={relatedTopic.imageUrl}
                    alt={relatedTopic.title}
                    fill
                    className="object-cover"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent" />
                  <div className="absolute bottom-4 left-4 p-2 rounded-full bg-background/80 backdrop-blur-sm">
                    {relatedTopic.icon}
                  </div>
                </div>
                <CardHeader>
                  <CardTitle>{relatedTopic.title}</CardTitle>
                  <CardDescription>{relatedTopic.description}</CardDescription>
                </CardHeader>
                <CardFooter>
                  <Button asChild variant="ghost" className="gap-1 ml-auto">
                    <Link href={`/topics/${relatedTopic.slug}`}>
                      Konuyu İncele
                      <ArrowRight className="h-4 w-4" />
                    </Link>
                  </Button>
                </CardFooter>
              </Card>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}