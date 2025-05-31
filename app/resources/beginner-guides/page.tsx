import Link from "next/link";
import { ArrowRight, ExternalLink, BookMarked, Code2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Metadata } from 'next';
import { Badge } from "@/components/ui/badge"; // For tags like 'Türkçe', 'İngilizce', 'Kod Örneği'

export const metadata: Metadata = {
  title: 'Python ile Yapay Zekaya Başlangıç Rehberleri | Kodleon AI Kaynakları',
  description: 'Yapay zeka ve makine öğrenmesine Python ile ilk adımlarınızı atın! Kodleon\'da yeni başlayanlar için en iyi Türkçe ve İngilizce kılavuzları, temel kavramları ve pratik kod örneklerini bulun.',
  keywords: 'Python yapay zeka başlangıç, AI öğrenme rehberi, makine öğrenmesi temelleri, yapay zeka nedir, Python AI dersleri, başlangıç için yapay zeka, AI temel kavramlar, Kodleon başlangıç kılavuzları',
  openGraph: {
    title: 'Python ile Yapay Zekaya Kolay Başlangıç | Kodleon Kılavuzları',
    description: 'Yapay zekaya Python ile başlamak için ihtiyacınız olan tüm temel bilgiler, adım adım rehberler ve örnekler Kodleon\'da.',
    url: 'https://kodleon.com/resources/beginner-guides',
    images: [
      {
        url: '/images/beginner-guides-og.png',
        width: 1200,
        height: 630,
        alt: 'Kodleon Yapay Zeka Başlangıç Seviyesi Kılavuzları'
      }
    ]
  }
};

const mainResources = [
  {
    title: "Python ile Yapay Zekaya Giriş",
    source: "Medium - Tirendaz Akademi",
    description: "Yapay zekanın ne olduğunu, uygulama alanlarını, alt disiplinlerini ve yapay zeka için kullanılan Python kütüphanelerini (Scikit-Learn, TensorFlow, PyTorch vb.) kapsamlı bir şekilde anlatan bir başlangıç rehberi.",
    href: "https://tirendazakademi.medium.com/python-ile-yapay-zekaya-gi%CC%87ri%CC%87%C5%9F-b592817c080f",
    tags: ["Türkçe", "Genel Bakış", "Kütüphaneler"],
    icon: <BookMarked className="h-5 w-5 mr-2" />
  },
  {
    title: "Python Programlama Dili ile Yapay Zeka Başlangıç: Detaylı Adımlar",
    source: "mesutpek.com.tr",
    description: "Python ile yapay zekaya nasıl başlanacağına dair adımlar: Temel kavramlar, araç kurulumu (Python, PyCharm, Jupyter), kütüphane öğrenimi (NumPy, Pandas, Matplotlib, Scikit-learn, TensorFlow) ve pratik örnekler.",
    href: "https://mesutpek.com.tr/python-programlama-dili-ile-yapay-zeka-baslangic-detayli-adimlar/",
    tags: ["Türkçe", "Adım Adım Rehber", "Araçlar"],
    icon: <BookMarked className="h-5 w-5 mr-2" />
  },
];

const supplementaryResources = [
  {
    title: "Artificial Intelligence Python Code Example: A Beginner's Guide",
    source: "Medium - UATeam",
    description: "Python'da basit bir yapay zeka modeli (ev fiyat tahmini) oluşturmayı adım adım gösteren pratik bir kod örneği. Kütüphane kurulumu, veri hazırlama, model eğitme ve değerlendirme aşamalarını içerir.",
    href: "https://medium.com/@aleksej.gudkov/artificial-intelligence-python-code-example-a-beginners-guide-04f3b8291c43",
    tags: ["İngilizce", "Kod Örneği", "Pratik Uygulama"],
    icon: <Code2 className="h-5 w-5 mr-2" />
  },
  {
    title: "Starting to Artificial Intelligence (Python Libraries And Their Uses)",
    source: "Medium - Mlearning.ai (Judeniz ༊)",
    description: "Neden Python'un yapay zeka için iyi bir seçim olduğunu açıklar ve temel Python kütüphanelerini (NumPy, Pandas, TensorFlow, Keras, PyTorch, Scikit-learn) ve kullanım alanlarını tanıtır.",
    href: "https://medium.com/mlearning-ai/starting-to-artificial-intelligence-3bc06bbfe911",
    tags: ["İngilizce", "Kütüphaneler", "Neden Python?"],
    icon: <BookMarked className="h-5 w-5 mr-2" />
  },
];

export default function BeginnerGuidesPage() {
  return (
    <div className="bg-background text-foreground">
      {/* Hero Section */}
      <section className="relative py-16 md:py-20 bg-muted/30" aria-labelledby="beginner-guides-hero-heading">
        <div 
          className="absolute inset-0 opacity-10 dark:opacity-5"
          style={{
            backgroundImage: 'url("/placeholder-dots.svg")', 
            backgroundSize: '30px 30px'
          }}
          aria-hidden="true"
        />
        <div className="container max-w-6xl mx-auto px-4 relative z-10">
          <div className="max-w-3xl mx-auto text-center">
            <Link href="/resources" className="text-sm text-primary hover:underline mb-2 inline-block">&larr; Tüm Kaynak Kategorileri</Link>
            <h1 
              id="beginner-guides-hero-heading" 
              className="text-3xl md:text-4xl font-bold tracking-tight mb-4"
            >
              Başlangıç Seviyesi Kılavuzlar
            </h1>
            <p className="text-lg md:text-xl text-muted-foreground">
              Python programlama dili ile yapay zeka ve makine öğrenmesi dünyasına adım atmak için özenle seçilmiş başlangıç seviyesi rehberler, makaleler ve pratik kod örnekleri.
            </p>
          </div>
        </div>
      </section>

      {/* Main Resources Section */}
      <section className="py-12 md:py-16" aria-labelledby="main-guides-heading">
        <div className="container max-w-4xl mx-auto px-4">
          <h2 id="main-guides-heading" className="text-2xl md:text-3xl font-semibold tracking-tight mb-8 text-center">
            Ana Kılavuzlar (Türkçe)
          </h2>
          <div className="space-y-6">
            {mainResources.map((resource) => (
              <Card key={resource.title} className="transition-shadow hover:shadow-lg">
                <CardHeader>
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <CardTitle className="text-xl mb-1">{resource.title}</CardTitle>
                      <p className="text-sm text-muted-foreground">Kaynak: {resource.source}</p>
                    </div>
                    <Button asChild variant="outline" size="sm" className="flex-shrink-0 ml-auto">
                      <Link href={resource.href} target="_blank" rel="noopener noreferrer">
                        Kaynağa Git
                        <ExternalLink className="ml-2 h-4 w-4" />
                      </Link>
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <CardDescription className="mb-3">{resource.description}</CardDescription>
                  <div className="flex flex-wrap gap-2">
                    {resource.tags.map(tag => <Badge key={tag} variant="secondary">{tag}</Badge>)}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Supplementary Resources Section */}
      <section className="py-12 md:py-16 bg-muted/30" aria-labelledby="supplementary-guides-heading">
        <div className="container max-w-4xl mx-auto px-4">
          <h2 id="supplementary-guides-heading" className="text-2xl md:text-3xl font-semibold tracking-tight mb-8 text-center">
            Ek Kaynaklar ve İleri Okumalar (İngilizce)
          </h2>
          <div className="space-y-6">
            {supplementaryResources.map((resource) => (
              <Card key={resource.title} className="transition-shadow hover:shadow-lg">
                <CardHeader>
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <CardTitle className="text-xl mb-1">{resource.title}</CardTitle>
                      <p className="text-sm text-muted-foreground">Kaynak: {resource.source}</p>
                    </div>
                    <Button asChild variant="outline" size="sm" className="flex-shrink-0 ml-auto">
                      <Link href={resource.href} target="_blank" rel="noopener noreferrer">
                        Kaynağa Git
                        <ExternalLink className="ml-2 h-4 w-4" />
                      </Link>
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <CardDescription className="mb-3">{resource.description}</CardDescription>
                  <div className="flex flex-wrap gap-2">
                    {resource.tags.map(tag => <Badge key={tag} variant="secondary">{tag}</Badge>)}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      <section className="py-12 text-center">
        <Button asChild variant="outline">
          <Link href="/resources">
            <ArrowRight className="mr-2 h-4 w-4 transform rotate-180" />
            Diğer Kaynak Kategorilerine Dön
          </Link>
        </Button>
      </section>
    </div>
  );
} 