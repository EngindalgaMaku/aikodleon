import Link from "next/link";
import Image from "next/image";
import {
  ArrowRight, Brain, Database, Eye, FileText, Lightbulb, Shapes, Users, Zap, Award, UsersRound, BookCopy, Target as TargetIcon, Briefcase, MessageSquareHeart, Code2, Rss
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Metadata } from 'next';
import FeaturedBlogCarousel from "@/components/FeaturedBlogCarousel";

export const metadata: Metadata = {
  title: 'Kodleon | Türkiye\'nin Lider Yapay Zeka Eğitim Platformu',
  description: 'Kodleon ile yapay zeka dünyasındaki en son gelişmeleri öğrenin, geleceğin teknolojilerini şekillendiren AI becerileri kazanın ve kariyerinize yön verin. Uzman eğitmenler, kapsamlı içerik ve uygulamalı projelerle uzmanlaşın.',
  keywords: 'yapay zeka eğitimi, AI kursları, kodleon, makine öğrenmesi, doğal dil işleme, bilgisayarlı görü, derin öğrenme, Türkçe yapay zeka, online AI eğitimi, yapay zeka projeleri, AI sertifikası, veri bilimi eğitimi, yapay zeka uzmanlığı',
  alternates: {
    canonical: 'https://kodleon.com',
  },
  openGraph: {
    type: "website",
    locale: "tr_TR",
    url: "https://kodleon.com",
    title: "Kodleon | Türkiye'nin Lider Yapay Zeka Eğitim Platformu",
    description: "Kodleon ile yapay zeka dünyasındaki en son gelişmeleri öğrenin, geleceğin teknolojilerini şekillendiren AI becerileri kazanın ve kariyerinize yön verin.",
    images: [
      {
        url: "/images/og-image.png",
        width: 1200,
        height: 630,
        alt: "Kodleon Yapay Zeka Eğitim Platformu | Geleceği Kodlayın"
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

const whyKodleonFeatures = [
  {
    title: "Uzman Eğitmen Kadrosu",
    description: "Sektör deneyimli, alanında uzman eğitmenlerden en güncel bilgileri öğrenin.",
    icon: <Award className="h-10 w-10 text-primary" />
  },
  {
    title: "Kapsamlı ve Güncel İçerik",
    description: "Yapay zekanın temellerinden ileri düzey konulara kadar geniş bir yelpazede, sürekli güncellenen dersler.",
    icon: <BookCopy className="h-10 w-10 text-primary" />
  },
  {
    title: "Uygulamalı Projeler ve Gerçek Dünya Senaryoları",
    description: "Teorik bilgiyi pratiğe dökebileceğiniz, portfolyonuzu güçlendirecek projeler üzerinde çalışın.",
    icon: <Briefcase className="h-10 w-10 text-primary" />
  },
  {
    title: "Aktif Topluluk Desteği",
    description: "Öğrenme yolculuğunuzda yalnız değilsiniz! Diğer öğrenciler ve eğitmenlerle etkileşimde bulunun.",
    icon: <UsersRound className="h-10 w-10 text-primary" />
  },
  {
    title: "Tamamen Türkçe Kaynaklar",
    description: "Yapay zeka gibi karmaşık bir konuyu ana dilinizde, anlaşılır ve kaliteli kaynaklarla öğrenin.",
    icon: <MessageSquareHeart className="h-10 w-10 text-primary" />
  },
  {
    title: "Esnek Öğrenme Modeli",
    description: "Kendi hızınızda, istediğiniz zaman ve yerden erişebileceğiniz online ders materyalleri.",
    icon: <Zap className="h-10 w-10 text-primary" />
  }
];

const whatYouCanAchieve = [
  {
    title: "Akıllı Uygulamalar Geliştirin",
    description: "Kullanıcı deneyimini kişiselleştiren, veriye dayalı kararlar alan uygulamalar oluşturun.",
    icon: <Lightbulb className="h-8 w-8 text-green-500" />
  },
  {
    title: "Veriden Değer Yaratın",
    description: "Büyük veri kümelerini analiz ederek anlamlı içgörüler çıkarın ve iş süreçlerini optimize edin.",
    icon: <Database className="h-8 w-8 text-blue-500" />
  },
  {
    title: "Otomasyon Çözümleri Üretin",
    description: "Tekrarlayan görevleri otomatize ederek verimliliği artırın ve insan hatasını azaltın.",
    icon: <Zap className="h-8 w-8 text-purple-500" />
  },
  {
    title: "Yeni Nesil Teknolojilere Liderlik Edin",
    description: "Yapay zeka alanındaki uzmanlığınızla geleceğin teknoloji trendlerini şekillendirin.",
    icon: <TargetIcon className="h-8 w-8 text-red-500" />
  }
];

const latestBlogPosts = [
  {
    title: "Ekranın Ötesinde: Cisimleşmiş Yapay Zeka Dünyamızı Anlamayı ve Şekillendirmeyi Nasıl Öğreniyor?",
    snippet: "On yıllardır Yapay Zeka, büyük ölçüde ekranlarımızın arkasında var oldu; karmaşık hesaplamalarda ustalaştı, anlayışlı metinler üretti ve çarpıcı görseller yarattı. Ancak yeni bir sınır ortaya çıkıyor: etrafımızdaki fiziksel dünyayı algılayabilen, anlayabilen ve onunla etkileşime girebilen Yapay Zeka.",
    imageUrl: "https://images.pexels.com/photos/7661169/pexels-photo-7661169.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    href: "/blog/embodied-ai-future",
    date: new Date(2025, 4, 31).toLocaleDateString('tr-TR', { year: 'numeric', month: 'long', day: 'numeric' }),
    category: "AI Gelişmeleri"
  },
  {
    title: "AI Kod Asistanları Arenası: Cursor vs. Windsurf (Codeium) ve Diğerleri",
    snippet: "Yazılım geliştirme dünyası yapay zeka ile hızla dönüşüyor. Bu dönüşümün en önemli oyuncularından biri de AI kod asistanları. Peki, hangi araç size en uygun? Gelin, popüler seçenekleri mercek altına alalım.",
    imageUrl: "https://images.pexels.com/photos/546819/pexels-photo-546819.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    href: "/blog/ai-kod-asistanlari-karsilastirmasi",
    date: new Date(2025, 4, 31).toLocaleDateString('tr-TR', { year: 'numeric', month: 'long', day: 'numeric' }),
    category: "AI Araçları"
  },
  {
    title: "Yapay Zeka ile Hareketli Hayaller: Veo ve Flow ile Video Üretiminde Yeni Bir Çağ",
    snippet: "Google'ın Veo'su ve Meta'nın Flow'u gibi yeni nesil yapay zeka modelleri, metin ve görsellerden etkileyici videolar ve animasyonlar oluşturarak yaratıcılığın sınırlarını zorluyor. Bu teknolojiler nasıl çalışıyor ve gelecekte bizi neler bekliyor?",
    imageUrl: "https://images.pexels.com/photos/9026285/pexels-photo-9026285.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    href: "/blog/ai-video-uretimi-veo-flow",
    date: new Date().toLocaleDateString('tr-TR', { year: 'numeric', month: 'long', day: 'numeric' }),
    category: "Üretken AI"
  }
];

export default function Home() {
  return (
    <div className="flex flex-col bg-background text-foreground">
      {/* Hero section */}
      <section className="relative py-20 md:py-32 overflow-hidden" aria-labelledby="hero-heading">
        <div className="absolute inset-0 bg-gradient-to-r from-primary/10 via-transparent to-secondary/10 dark:from-primary/5 dark:via-transparent dark:to-secondary/5" />
        <div 
          className="absolute inset-0 opacity-20 dark:opacity-10"
          style={{
            backgroundImage: 'url("data:image/svg+xml,%3Csvg width="30" height="30" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="%239C92AC" fill-opacity="0.1"%3E%3Cpath d="M0 10 L10 0 L20 10 L10 20 Z"/%3E%3C/g%3E%3C/svg%3E")',
            backgroundSize: '30px 30px'
          }}
          aria-hidden="true"
        />
        <div className="container max-w-6xl mx-auto relative z-10 px-4">
          <div className="max-w-3xl mx-auto text-center mb-12 md:mb-16">
            <h1 
              id="hero-heading" 
              className="text-4xl md:text-6xl font-bold tracking-tight mb-6 
                         bg-clip-text text-transparent bg-gradient-to-r from-primary via-pink-500 to-orange-500 
                         animate-gradient-xy"
            >
              Yapay Zeka ile Geleceği Kodlayın
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground mb-10">
              Türkiye'nin lider yapay zeka eğitim platformu Kodleon ile en güncel AI becerilerini kazanın, potansiyelinizi keşfedin ve geleceğin teknolojilerine yön verin.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button asChild size="lg" className="rounded-full shadow-lg hover:shadow-xl transition-shadow">
                <Link href="/topics" aria-label="Yapay zeka konularını keşfedin">AI Konularını Keşfet</Link>
              </Button>
              <Button asChild size="lg" variant="outline" className="rounded-full border-border hover:border-primary/70 transition-colors">
                <Link href="/blog" aria-label="Blog yazılarını keşfedin">
                  <Rss className="mr-2 h-5 w-5" />
                  Blog'u Keşfet
                </Link>
              </Button>
            </div>
          </div>

          {/* Featured Blog Carousel */}
          <FeaturedBlogCarousel posts={latestBlogPosts} />
        </div>
      </section>
      
      {/* Topics section (Existing - can be further enhanced later) */}
      <section className="py-16 md:py-20 bg-muted/30" aria-labelledby="topics-heading">
        <div className="container max-w-6xl mx-auto px-4">
          <div className="text-center mb-12 md:mb-16">
            <h2 id="topics-heading" className="text-3xl md:text-4xl font-bold tracking-tight mb-4 text-foreground">Yapay Zeka Eğitim Konuları</h2>
            <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto">
              Makine öğrenmesinden derin öğrenmeye, doğal dil işlemeden bilgisayarlı görüye kadar geniş bir yelpazede uzmanlaşın.
            </p>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
            {topics.map((topic, index) => (
              <Card key={index} className="overflow-hidden transition-all duration-300 hover:shadow-xl hover:-translate-y-1.5 border-border hover:border-primary/50 bg-card flex flex-col">
                <div className="relative h-52 w-full">
                  <Image 
                    src={topic.imageUrl}
                    alt={`${topic.title} - Kodleon yapay zeka eğitim içeriği görseli`}
                    fill
                    className="object-cover transition-transform duration-300 group-hover:scale-105"
                    loading={index < 3 ? "eager" : "lazy"}
                    sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/40 to-transparent" aria-hidden="true" />
                  <div className="absolute bottom-4 left-4 p-3 rounded-full bg-background/80 backdrop-blur-sm border border-border shadow-md">
                    {topic.icon}
                  </div>
                </div>
                <CardHeader className="pb-2">
                  <CardTitle className="text-xl group-hover:text-primary transition-colors">{topic.title}</CardTitle>
                </CardHeader>
                <CardContent className="flex-grow">
                  <CardDescription>{topic.description}</CardDescription>
                </CardContent>
                <CardFooter className="mt-auto pt-3">
                  <Button asChild variant="ghost" className="gap-1.5 ml-auto text-primary hover:text-primary/80">
                    <Link href={topic.href} aria-label={`${topic.title} konusunu daha detaylı inceleyin`}>
                      Konuyu İncele
                      <ArrowRight className="h-4 w-4" aria-hidden="true" />
                    </Link>
                  </Button>
                </CardFooter>
              </Card>
            ))}
          </div>
        </div>
      </section>
      
      {/* Latest Blog Posts Section */}
      <section className="py-16 md:py-20 bg-background" aria-labelledby="latest-blog-heading">
        <div className="container max-w-6xl mx-auto px-4">
          <div className="text-center mb-12 md:mb-16">
            <div className="inline-block p-3 mb-4 bg-primary/10 rounded-full border border-primary/20">
                <Rss className="h-10 w-10 text-primary" />
            </div>
            <h2 id="latest-blog-heading" className="text-3xl md:text-4xl font-bold tracking-tight mb-4 text-foreground">Blog'dan Son Gelişmeler</h2>
            <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto">
              Yapay zeka dünyasındaki en son trendleri, derinlemesine analizleri ve uzman görüşlerini blogumuzda keşfedin.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {latestBlogPosts.map((post) => (
              <Card key={post.title} className="bg-card border-border overflow-hidden flex flex-col group transition-all duration-300 hover:shadow-xl hover:-translate-y-1.5">
                <div className="relative h-56 w-full">
                  <Image 
                    src={post.imageUrl}
                    alt={`${post.title} - Kodleon Blog Yazısı Görseli`}
                    fill
                    className="object-cover transition-transform duration-300 group-hover:scale-105"
                    sizes="(max-width: 768px) 100vw, (max-width: 1024px) 50vw, 33vw"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/40 to-transparent" aria-hidden="true" />
                  <div className="absolute top-4 right-4 bg-primary/80 text-primary-foreground text-xs font-semibold px-2.5 py-1 rounded-full backdrop-blur-sm">
                    {post.category}
                  </div>
                </div>
                <CardHeader className="flex-grow">
                  <CardTitle className="text-xl mb-1.5 group-hover:text-primary transition-colors">
                    <Link href={post.href} aria-label={`${post.title} blog yazısını oku`}>{post.title}</Link>
                  </CardTitle>
                  <p className="text-xs text-muted-foreground">Yayınlanma Tarihi: {post.date}</p>
                </CardHeader>
                <CardContent className="flex-grow">
                  <CardDescription className="text-sm leading-relaxed line-clamp-3">{post.snippet}</CardDescription>
                </CardContent>
                <CardFooter className="pt-4 mt-auto border-t border-border/60">
                  <Button asChild variant="ghost" size="sm" className="gap-1.5 ml-auto text-primary hover:text-primary/80">
                    <Link href={post.href} aria-label={`${post.title} blog yazısını oku`}>
                      Devamını Oku
                      <ArrowRight className="h-4 w-4" />
                    </Link>
                  </Button>
                </CardFooter>
              </Card>
            ))}
          </div>
          <div className="text-center mt-12">
            <Button asChild size="lg" variant="outline" className="rounded-full border-border hover:border-primary/70 transition-colors">
              <Link href="/blog" aria-label="Tüm blog yazılarını gör">Tüm Blog Yazılarını Gör</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Why Kodleon? Section */}
      <section className="py-16 md:py-20 bg-muted/30" aria-labelledby="why-kodleon-heading">
        <div className="container max-w-6xl mx-auto px-4">
          <div className="text-center mb-12 md:mb-16">
            <h2 id="why-kodleon-heading" className="text-3xl md:text-4xl font-bold tracking-tight mb-4 text-foreground">Neden Kodleon ile Yapay Zeka?</h2>
            <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto">
              Yapay zeka öğrenme yolculuğunuzda Kodleon, size en iyi deneyimi sunmak için tasarlandı.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {whyKodleonFeatures.map((feature) => (
              <div 
                key={feature.title} 
                className="flex flex-col items-center text-center p-6 bg-card border border-border rounded-lg shadow-lg 
                           hover:shadow-xl hover:scale-105 transition-all duration-300 ease-in-out"
              >
                <div className="p-4 bg-primary/10 rounded-full mb-5 border border-primary/20">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold mb-2 text-foreground">{feature.title}</h3>
                <p className="text-muted-foreground text-sm leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* What Can You Achieve? Section */}
      <section className="py-16 md:py-20 bg-background" aria-labelledby="achieve-heading">
        <div className="container max-w-6xl mx-auto px-4">
          <div className="text-center mb-12 md:mb-16">
            <h2 id="achieve-heading" className="text-3xl md:text-4xl font-bold tracking-tight mb-4 text-foreground">Kodleon ile Neler Başarabilirsiniz?</h2>
            <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto">
              Yapay zeka becerileriyle donanarak çeşitli alanlarda fark yaratın ve yenilikçi çözümler üretin.
            </p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {whatYouCanAchieve.map((item) => (
              <Card 
                key={item.title} 
                className="bg-card border-border p-6 flex flex-col items-center text-center 
                           hover:shadow-xl hover:border-primary/50 hover:-translate-y-1 transition-all duration-300 ease-in-out"
              >
                <div className="p-3 bg-primary/10 rounded-full mb-4 border-primary/20">
                  {item.icon}
                </div>
                <h3 className="text-lg font-semibold mb-2 text-foreground">{item.title}</h3>
                <p className="text-sm text-muted-foreground flex-grow">{item.description}</p>
              </Card>
            ))}
          </div>
        </div>
      </section>
      
      {/* Benefits section (Existing - Refined for flow) */}
      <section className="py-16 md:py-20 bg-muted/30" aria-labelledby="benefits-heading">
        <div className="container max-w-6xl mx-auto px-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div className="order-2 lg:order-1">
              <h2 id="benefits-heading" className="text-3xl md:text-4xl font-bold tracking-tight mb-8 text-foreground">Yapay Zeka: Sadece Bir Teknoloji Değil, Geleceğin Kendisi</h2>
              <div className="space-y-6">
                <div className="flex items-start gap-4 p-4 bg-card border border-border rounded-lg hover:shadow-md transition-shadow">
                  <div className="bg-primary/10 rounded-full p-3 flex-shrink-0 mt-1">
                    <Lightbulb className="h-6 w-6 text-primary" aria-hidden="true" />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold mb-1">Geleceğe Yön Verin</h3>
                    <p className="text-muted-foreground text-sm">
                      Yapay zeka, sağlık, finans, eğitim ve daha birçok sektörü kökten dönüştürüyor. Bu devrimin bir parçası olun.
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start gap-4 p-4 bg-card border border-border rounded-lg hover:shadow-md transition-shadow">
                  <div className="bg-primary/10 rounded-full p-3 flex-shrink-0 mt-1">
                    <Shapes className="h-6 w-6 text-primary" aria-hidden="true" />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold mb-1">Analitik Düşünce ve Problem Çözme</h3>
                    <p className="text-muted-foreground text-sm">
                      AI prensiplerini öğrenmek, karmaşık problemleri analiz etme ve veri odaklı çözümler üretme yeteneğinizi geliştirir.
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start gap-4 p-4 bg-card border border-border rounded-lg hover:shadow-md transition-shadow">
                  <div className="bg-primary/10 rounded-full p-3 flex-shrink-0 mt-1">
                    <Users className="h-6 w-6 text-primary" aria-hidden="true" />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold mb-1">Kariyerinizde Zirveye Ulaşın</h3>
                    <p className="text-muted-foreground text-sm">
                      Yapay zeka uzmanlığı, günümüzün ve geleceğin en çok aranan becerilerinden biri. Kariyerinizde yeni kapılar aralayın.
                    </p>
                  </div>
                </div>
              </div>
              
              <Button asChild size="lg" className="mt-10 rounded-full shadow-md hover:shadow-lg transition-shadow">
                <Link href="/topics" aria-label="Yapay zeka öğrenmeye hemen başlayın">Öğrenmeye Hemen Başla</Link>
              </Button>
            </div>
            
            <div className="relative aspect-video lg:aspect-[4/3.5] rounded-2xl overflow-hidden shadow-xl order-1 lg:order-2 group">
              <Image 
                src="https://images.pexels.com/photos/3861969/pexels-photo-3861969.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
                alt="Yapay zeka eğitimi alan profesyoneller - Kodleon öğrenim ortamı"
                fill
                className="object-cover transition-transform duration-500 group-hover:scale-105"
                sizes="(max-width: 1024px) 100vw, 50vw"
              />
               <div className="absolute inset-0 bg-gradient-to-br from-primary/20 via-transparent to-secondary/20 opacity-70 group-hover:opacity-50 transition-opacity duration-500" /> 
            </div>
          </div>
        </div>
      </section>
      
      {/* CTA section (Existing - slight visual refinement) */}
      <section className="py-16 md:py-24 bg-gradient-to-r from-primary via-purple-600 to-pink-600 text-primary-foreground" aria-labelledby="cta-heading">
        <div className="container max-w-6xl mx-auto px-4">
          <div className="max-w-3xl mx-auto text-center">
            <h2 id="cta-heading" className="text-3xl md:text-4xl font-bold tracking-tight mb-6">
              Yapay Zeka Yolculuğunuza Bugün Kodleon ile Adım Atın!
            </h2>
            <p className="text-xl mb-10 text-primary-foreground/90 leading-relaxed">
              Geleceğin en heyecan verici teknolojisini şekillendiren becerileri kazanarak kariyerinizde ve projelerinizde fark yaratın. Kapsamlı eğitimlerimizle potansiyelinizi en üst düzeye çıkarın.
            </p>
            <Button asChild size="lg" variant="secondary" className="rounded-full text-lg py-3 px-8 md:py-4 md:px-10 shadow-lg hover:shadow-xl transition-all duration-300 ease-in-out transform hover:scale-105">
              <Link href="/topics" aria-label="Tüm yapay zeka konularını inceleyin">
                Tüm AI Konularını Görüntüle
                <ArrowRight className="ml-2.5 h-5 w-5" aria-hidden="true" />
              </Link>
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
}