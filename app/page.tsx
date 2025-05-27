import Link from "next/link";
import Image from "next/image";
import { ArrowRight, Brain, Database, Eye, FileText, Lightbulb, Shapes, Users } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Kodleon | Türkiye\'nin Lider Yapay Zeka Eğitim Platformu',
  description: 'Yapay zeka dünyasındaki en son gelişmeleri öğrenin ve geleceğin teknolojilerini şekillendiren becerileri Kodleon ile kazanın.',
  keywords: 'yapay zeka eğitimi, kodleon, makine öğrenmesi kursu, doğal dil işleme, bilgisayarlı görü, Türkçe AI eğitimi, yapay zeka dersleri',
  alternates: {
    canonical: 'https://kodleon.com',
  },
  openGraph: {
    type: "website",
    locale: "tr_TR",
    url: "https://kodleon.com",
    title: "Kodleon | Türkiye'nin Lider Yapay Zeka Eğitim Platformu",
    description: "Yapay zeka dünyasındaki en son gelişmeleri öğrenin ve geleceğin teknolojilerini Kodleon ile keşfedin.",
    images: [
      {
        url: "https://kodleon.com/og-image.jpg",
        width: 1200,
        height: 630,
        alt: "Kodleon Yapay Zeka Eğitim Platformu"
      }
    ],
  },
};

const topics = [
  {
    title: "Makine Öğrenmesi",
    description: "Algoritmaların veri kullanarak nasıl öğrendiğini ve tahminlerde bulunduğunu keşfedin.",
    icon: <Database className="h-8 w-8 text-chart-1" aria-hidden="true" />,
    href: "/topics/machine-learning",
    imageUrl: "https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
  },
  {
    title: "Doğal Dil İşleme",
    description: "Makinelerin insan dilini nasıl anlayıp işlediğini ve ürettiğini öğrenin.",
    icon: <FileText className="h-8 w-8 text-chart-2" aria-hidden="true" />,
    href: "/topics/nlp",
    imageUrl: "https://images.pexels.com/photos/7412095/pexels-photo-7412095.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
  },
  {
    title: "Bilgisayarlı Görü",
    description: "Bilgisayarların görüntüleri nasıl algıladığını ve işlediğini anlayın.",
    icon: <Eye className="h-8 w-8 text-chart-3" aria-hidden="true" />,
    href: "/topics/computer-vision",
    imageUrl: "https://images.pexels.com/photos/8438922/pexels-photo-8438922.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
  },
  {
    title: "Üretken AI",
    description: "Metin, görüntü ve ses üretebilen yapay zeka modellerini keşfedin.",
    icon: <Lightbulb className="h-8 w-8 text-chart-4" aria-hidden="true" />,
    href: "/topics/generative-ai",
    imageUrl: "https://images.pexels.com/photos/8386434/pexels-photo-8386434.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
  },
  {
    title: "Sinir Ağları",
    description: "Beynin çalışma prensibinden esinlenen yapay sinir ağları hakkında bilgi edinin.",
    icon: <Brain className="h-8 w-8 text-chart-5" aria-hidden="true" />,
    href: "/topics/neural-networks",
    imageUrl: "https://images.pexels.com/photos/8386421/pexels-photo-8386421.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
  },
  {
    title: "AI Etiği",
    description: "Yapay zekanın etik kullanımı ve toplumsal etkileri üzerine tartışmalar.",
    icon: <Users className="h-8 w-8 text-chart-1" aria-hidden="true" />,
    href: "/topics/ai-ethics",
    imageUrl: "https://images.pexels.com/photos/8386422/pexels-photo-8386422.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
  },
];

export default function Home() {
  return (
    <div className="flex flex-col">
      {/* Hero section */}
      <section className="relative py-20 md:py-32 overflow-hidden" aria-labelledby="hero-heading">
        <div className="absolute inset-0 bg-gradient-to-r from-primary/10 to-secondary/10 dark:from-primary/5 dark:to-secondary/5" />
        <div 
          className="absolute inset-0 opacity-30 dark:opacity-20"
          style={{
            backgroundImage: 'url("data:image/svg+xml,%3Csvg width="20" height="20" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="%239C92AC" fill-opacity="0.2"%3E%3Cpath d="M0 0h20L0 20z"/%3E%3C/g%3E%3C/svg%3E")',
            backgroundSize: '20px 20px'
          }}
          aria-hidden="true"
        />
        <div className="container max-w-6xl mx-auto relative z-10">
          <div className="max-w-3xl mx-auto text-center">
            <h1 id="hero-heading" className="text-4xl md:text-6xl font-bold tracking-tight mb-6 bg-clip-text text-transparent bg-gradient-to-r from-primary to-chart-2">
              Yapay Zeka ile Geleceği Keşfedin
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground mb-8">
              Yapay zeka dünyasındaki en son gelişmeleri öğrenin ve geleceğin teknolojilerini şekillendiren becerileri kazanın.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button asChild size="lg" className="rounded-full">
                <Link href="/topics" aria-label="Yapay zeka konularını keşfedin">Konuları Keşfet</Link>
              </Button>
              <Button asChild size="lg" variant="outline" className="rounded-full">
                <Link href="/about" aria-label="Kodleon hakkında bilgi alın">Hakkımızda</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>
      
      {/* Features section */}
      <section className="py-16 bg-muted/50" aria-labelledby="topics-heading">
        <div className="container max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 id="topics-heading" className="text-3xl font-bold tracking-tight mb-4">Yapay Zeka Eğitim Konuları</h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Yapay zeka teknolojilerinin farklı alanlarını keşfedin ve uzmanlaşmak istediğiniz konuları seçin.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {topics.map((topic, index) => (
              <Card key={index} className="overflow-hidden transition-all duration-300 hover:shadow-lg hover:-translate-y-1">
                <div className="relative h-48">
                  <Image 
                    src={topic.imageUrl}
                    alt={`${topic.title} - Kodleon yapay zeka eğitim içeriği görseli`}
                    fill
                    className="object-cover"
                    loading={index < 3 ? "eager" : "lazy"}
                    sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent" aria-hidden="true" />
                  <div className="absolute bottom-4 left-4 p-2 rounded-full bg-background/80 backdrop-blur-sm">
                    {topic.icon}
                  </div>
                </div>
                <CardHeader>
                  <CardTitle>{topic.title}</CardTitle>
                  <CardDescription>{topic.description}</CardDescription>
                </CardHeader>
                <CardFooter>
                  <Button asChild variant="ghost" className="gap-1 ml-auto">
                    <Link href={topic.href} aria-label={`${topic.title} konusunu daha detaylı inceleyin`}>
                      Daha Fazla
                      <ArrowRight className="h-4 w-4" aria-hidden="true" />
                    </Link>
                  </Button>
                </CardFooter>
              </Card>
            ))}
          </div>
        </div>
      </section>
      
      {/* Benefits section */}
      <section className="py-16" aria-labelledby="benefits-heading">
        <div className="container max-w-6xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div>
              <h2 id="benefits-heading" className="text-3xl font-bold tracking-tight mb-6">Neden Yapay Zeka Öğrenmelisiniz?</h2>
              <div className="space-y-6">
                <div className="flex gap-4">
                  <div className="bg-primary/10 rounded-full p-3 h-fit">
                    <Lightbulb className="h-6 w-6 text-primary" aria-hidden="true" />
                  </div>
                  <div>
                    <h3 className="text-xl font-medium mb-2">Geleceğe Hazırlanın</h3>
                    <p className="text-muted-foreground">
                      Yapay zeka, tüm sektörleri dönüştürüyor. Bugünden becerilerinizi geliştirerek geleceğe hazır olun.
                    </p>
                  </div>
                </div>
                
                <div className="flex gap-4">
                  <div className="bg-primary/10 rounded-full p-3 h-fit">
                    <Shapes className="h-6 w-6 text-primary" aria-hidden="true" />
                  </div>
                  <div>
                    <h3 className="text-xl font-medium mb-2">Problem Çözme Becerileri</h3>
                    <p className="text-muted-foreground">
                      Yapay zeka çalışmak, karmaşık problemleri çözme ve analitik düşünme becerilerinizi geliştirir.
                    </p>
                  </div>
                </div>
                
                <div className="flex gap-4">
                  <div className="bg-primary/10 rounded-full p-3 h-fit">
                    <Users className="h-6 w-6 text-primary" aria-hidden="true" />
                  </div>
                  <div>
                    <h3 className="text-xl font-medium mb-2">Yeni Fırsatlar</h3>
                    <p className="text-muted-foreground">
                      Yapay zeka alanında uzmanlaşarak yüksek talep gören bir kariyer yolunda ilerleyin ve yeni fırsatlar yakalayın.
                    </p>
                  </div>
                </div>
              </div>
              
              <Button asChild className="mt-8 rounded-full">
                <Link href="/topics" aria-label="Yapay zeka öğrenmeye hemen başlayın">Öğrenmeye Başlayın</Link>
              </Button>
            </div>
            
            <div className="relative aspect-square rounded-2xl overflow-hidden">
              <Image 
                src="https://images.pexels.com/photos/3861969/pexels-photo-3861969.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
                alt="Yapay zeka eğitimi alan profesyoneller - Kodleon öğrenim ortamı"
                fill
                className="object-cover"
                priority={true}
                sizes="(max-width: 1024px) 100vw, 50vw"
              />
            </div>
          </div>
        </div>
      </section>
      
      {/* CTA section */}
      <section className="py-16 bg-primary text-primary-foreground" aria-labelledby="cta-heading">
        <div className="container max-w-6xl mx-auto">
          <div className="max-w-3xl mx-auto text-center">
            <h2 id="cta-heading" className="text-3xl font-bold tracking-tight mb-4">
              Yapay Zeka Yolculuğunuza Bugün Başlayın
            </h2>
            <p className="text-xl mb-8 text-primary-foreground/80">
              Geleceğin teknolojilerini şekillendiren becerileri kazanarak kariyerinizde bir adım öne geçin.
            </p>
            <Button asChild size="lg" variant="secondary" className="rounded-full">
              <Link href="/topics" aria-label="Tüm yapay zeka konularını inceleyin">
                Tüm Konuları Görüntüle
                <ArrowRight className="ml-2 h-4 w-4" aria-hidden="true" />
              </Link>
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
}