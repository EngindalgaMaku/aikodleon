import Link from "next/link";
import Image from "next/image";
import { 
  ArrowRight, 
  BookOpen, // General, Academic
  FileText, // Articles
  Video, // Courses (or PlayCircle)
  Download, // Tools (or Github, Code2)
  GraduationCap, // Beginner Guides
  Newspaper, // Articles, Blogs
  MonitorPlay, // Online Courses
  Github, // OS Tools
  Database, // Datasets
  Rocket, // Projects
  FlaskConical, // Academic/Research
  Lightbulb, // Could be for Beginner Guides or Projects
  Users // Research Groups
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Ücretsiz Yapay Zeka Kaynakları: Rehberler, Kurslar, Araçlar | Kodleon',
  description: 'Kodleon Yapay Zeka Kaynak Merkezi: Başlangıç seviyesi rehberler, derinlemesine makaleler, ücretsiz online kurslar, AI araçları, veri setleri ve ilham verici projelerle öğrenme yolculuğunuzu destekleyin.',
  keywords: 'yapay zeka kaynakları, ücretsiz AI eğitimi, AI rehberleri, yapay zeka kursları, AI araçları, makine öğrenmesi veri setleri, AI projeleri, Türkçe yapay zeka kaynakları, Kodleon kaynaklar, derin öğrenme makaleleri, Python AI',
  openGraph: {
    title: 'Kodleon Yapay Zeka Kaynak Merkezi | Kapsamlı ve Ücretsiz',
    description: 'Yapay zeka öğreniminizi bir üst seviyeye taşıyacak rehberler, kurslar, makaleler, araçlar ve daha fazlasını Kodleon\'da keşfedin.',
    url: 'https://kodleon.com/resources',
    images: [
      {
        url: '/images/resources-og.png',
        width: 1200,
        height: 630,
        alt: 'Kodleon Yapay Zeka Kaynakları Merkezi'
      }
    ]
  }
};

const resourceCategories = [
  {
    title: "Başlangıç Seviyesi Kılavuzlar",
    description: "Yapay zeka dünyasına ilk adımlarınızı atmanız için temel kavramları ve başlangıç rehberlerini keşfedin.",
    icon: <GraduationCap className="h-8 w-8 text-primary" />,
    href: "/resources/beginner-guides",
    imageUrl: "https://images.pexels.com/photos/3747505/pexels-photo-3747505.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2" // Example image
  },
  {
    title: "Derinlemesine Makaleler ve Bloglar",
    description: "Yapay zekanın çeşitli alt dalları, son gelişmeler ve uzman analizleri üzerine derinlemesine yazılar.",
    icon: <Newspaper className="h-8 w-8 text-primary" />,
    href: "/resources/in-depth-articles",
    imageUrl: "https://images.pexels.com/photos/590016/pexels-photo-590016.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
  },
  {
    title: "Ücretsiz Online Kurslar ve Platformlar",
    description: "Dünyanın önde gelen platformlarından ve üniversitelerden ücretsiz yapay zeka kurslarına erişin.",
    icon: <MonitorPlay className="h-8 w-8 text-primary" />,
    href: "/resources/online-courses",
    imageUrl: "https://images.pexels.com/photos/3861958/pexels-photo-3861958.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
  },
  {
    title: "Açık Kaynak AI Araçları ve Kütüphaneler",
    description: "Projelerinizde kullanabileceğiniz popüler açık kaynak yapay zeka araçlarını ve kütüphanelerini tanıyın.",
    icon: <Github className="h-8 w-8 text-primary" />,
    href: "/resources/ai-tools",
    imageUrl: "https://images.pexels.com/photos/1181677/pexels-photo-1181677.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
  },
  {
    title: "Kullanıma Hazır Veri Setleri",
    description: "Makine öğrenmesi modellerinizi eğitmek ve test etmek için çeşitli alanlardaki veri setlerini bulun.",
    icon: <Database className="h-8 w-8 text-primary" />,
    href: "/resources/datasets",
    imageUrl: "https://images.pexels.com/photos/669615/pexels-photo-669615.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
  },
  {
    title: "İlham Verici Yapay Zeka Projeleri",
    description: "Yapay zekanın gerçek dünya problemlerine nasıl çözümler sunduğunu gösteren ilham verici projeler.",
    icon: <Rocket className="h-8 w-8 text-primary" />,
    href: "/resources/inspiring-projects",
    imageUrl: "https://images.pexels.com/photos/73910/mars-mars-rover-space-travel-robot-73910.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
  },
  {
    title: "Akademik Yayınlar ve Araştırma Grupları",
    description: "Yapay zeka alanındaki en son bilimsel gelişmeleri takip edebileceğiniz yayınlar ve araştırma grupları.",
    icon: <BookOpen className="h-8 w-8 text-primary" />, // or FlaskConical
    href: "/resources/academic-research",
    imageUrl: "https://images.pexels.com/photos/265076/pexels-photo-265076.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
  }
];

export default function ResourcesPage() {
  return (
    <div className="bg-background text-foreground">
      <section className="relative py-16 md:py-24" aria-labelledby="resources-hero-heading">
        <div 
          className="absolute inset-0 opacity-10 dark:opacity-5"
          style={{
            backgroundImage: 'url("/placeholder-dots.svg")', // Replace with a subtle background pattern if desired
            backgroundSize: '30px 30px'
          }}
          aria-hidden="true"
        />
        <div className="container max-w-6xl mx-auto px-4 relative z-10">
          <div className="max-w-3xl mx-auto text-center mb-12 md:mb-16">
            <h1 
              id="resources-hero-heading" 
              className="text-4xl md:text-5xl font-bold tracking-tight mb-6 
                         bg-clip-text text-transparent bg-gradient-to-r from-primary via-pink-500 to-orange-500 
                         animate-gradient-xy"
            >
              Yapay Zeka Kaynak Merkezi
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground">
              Öğrenme yolculuğunuzu hızlandıracak, en güncel ve kapsamlı yapay zeka kaynaklarını bir arada bulun. Kılavuzlar, kurslar, araçlar ve daha fazlası sizi bekliyor.
            </p>
          </div>
        </div>
      </section>

      <section className="py-12 md:py-16" aria-labelledby="resource-categories-heading">
        <div className="container max-w-6xl mx-auto px-4">
          <div className="text-left mb-10 md:mb-12">
            <h2 id="resource-categories-heading" className="text-3xl md:text-4xl font-bold tracking-tight text-foreground">
              Kaynak Kategorileri
            </h2>
            <p className="mt-3 text-lg text-muted-foreground max-w-2xl">
              İhtiyaçlarınıza uygun kaynakları kolayca bulabilmeniz için kategorilere ayırdık.
            </p>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
            {resourceCategories.map((category) => (
              <Card 
                key={category.title} 
                className="overflow-hidden transition-all duration-300 hover:shadow-xl hover:-translate-y-1.5 border-border hover:border-primary/50 bg-card flex flex-col group"
              >
                <div className="relative h-52 w-full">
                  <Image 
                    src={category.imageUrl}
                    alt={`${category.title} için kategori görseli`}
                    fill
                    className="object-cover transition-transform duration-300 group-hover:scale-105"
                    sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/40 to-transparent" aria-hidden="true" />
                  <div className="absolute top-4 left-4 p-3 rounded-full bg-background/80 backdrop-blur-sm border border-border shadow-md">
                    {category.icon}
                  </div>
                </div>
                <CardHeader className="pb-3">
                  <CardTitle className="text-xl group-hover:text-primary transition-colors">{category.title}</CardTitle>
                </CardHeader>
                <CardContent className="flex-grow">
                  <CardDescription>{category.description}</CardDescription>
                </CardContent>
                <CardFooter className="mt-auto pt-4 border-t border-border/60">
                  <Button asChild variant="default" size="sm" className="w-full group-hover:bg-primary/90 transition-colors">
                    <Link href={category.href} aria-label={`${category.title} kategorisindeki kaynakları incele`}>
                      Tümünü Gör
                      <ArrowRight className="ml-2 h-4 w-4" />
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