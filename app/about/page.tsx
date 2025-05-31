import { Metadata } from 'next';
// import { createPageMetadata } from '@/lib/seo'; // Assuming you have this utility - keeping it commented if not used directly or defined elsewhere
import Image from "next/image";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Brain, BookOpen, Award, Users, Lightbulb, Rocket, Target as TargetIcon, CheckCircle, TrendingUp, Sparkles, ShieldCheck, Compass, UsersRound } from "lucide-react"; // Added more icons
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"; // Added Card components

// Placeholder for createPageMetadata if not available or you define metadata directly
const createPageMetadata = (data: any) => data; 

export const metadata: Metadata = createPageMetadata({
  title: 'Hakkımızda | Kodleon - Yapay Zeka Geleceğinizi Şekillendirin', // Enhanced title
  description: 'Kodleon yapay zeka eğitim platformunun vizyonunu, misyonunu ve değerlerini keşfedin. Neden Kodleon ile AI öğrenmeniz gerektiğini öğrenin ve ekibimizle tanışın.',
  path: '/about',
  keywords: ['kodleon hakkında', 'yapay zeka eğitim platformu', 'misyon', 'vizyon', 'ekip', 'türkiye ai eğitimi', 'neden kodleon', 'ai öğrenmek'],
  openGraph: { // Added OpenGraph for better sharing
    type: "website",
    locale: "tr_TR",
    url: "https://kodleon.com/about",
    title: "Kodleon Hakkında - Yapay Zeka Eğitimi Vizyonumuz",
    description: "Türkiye'nin lider AI eğitim platformu Kodleon'un misyonunu, vizyonunu ve topluluğa katkılarını öğrenin.",
    images: [
      {
        url: "https://kodleon.com/og-about.jpg", // Replace with an actual relevant image URL
        width: 1200,
        height: 630,
        alt: "Kodleon Hakkımızda"
      }
    ],
  },
});

const whyKodleonFeatures = [
  {
    title: "Güncel ve Kapsamlı İçerikler",
    description: "Sektördeki en son trendleri ve temel konuları içeren, sürekli güncellenen eğitimler.",
    icon: <TrendingUp className="h-8 w-8 text-primary" />
  },
  {
    title: "Uygulamalı Yaklaşım",
    description: "Teorik bilgiyi gerçek dünya projeleriyle pekiştirerek pratik beceriler kazanın.",
    icon: <Sparkles className="h-8 w-8 text-primary" />
  },
  {
    title: "Tamamen Türkçe Kaynaklar",
    description: "Yapay zeka gibi karmaşık bir konuyu ana dilinizde, anlaşılır ve kaliteli kaynaklarla öğrenin.",
    icon: <BookOpen className="h-8 w-8 text-primary" />
  },
  {
    title: "Erişilebilir ve Esnek Öğrenme",
    description: "Kendi hızınızda, istediğiniz zaman ve yerden erişebileceğiniz online ders materyalleri.",
    icon: <ShieldCheck className="h-8 w-8 text-primary" />
  },
  {
    title: "Uzman Eğitmen Desteği",
    description: "Alanında deneyimli eğitmenlerden rehberlik alın ve sorularınıza yanıt bulun.",
    icon: <Award className="h-8 w-8 text-primary" />
  },
  {
    title: "Güçlü Topluluk",
    description: "Öğrenenler ve eğitmenlerden oluşan aktif bir toplulukla etkileşimde bulunun, motive kalın.",
    icon: <UsersRound className="h-8 w-8 text-primary" />
  }
];

export default function AboutPage() {
  return (
    <div className="bg-background text-foreground">
      {/* Hero Section */}
      <section className="py-16 md:py-24 bg-muted/30">
        <div className="container max-w-5xl mx-auto px-4 md:px-6 text-center">
          <div className="inline-block p-3 mb-6 bg-primary/10 rounded-full border border-primary/20">
            <Brain className="h-12 w-12 text-primary" />
          </div>
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-foreground mb-6">
            Kodleon: Yapay Zeka ile Geleceğinizi Şekillendirin
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            Kodleon, Türkiye'deki bireylerin ve kurumların yapay zeka devrimine öncülük etmelerini sağlamak amacıyla kurulmuş, kapsamlı ve erişilebilir bir Türkçe AI eğitim platformudur.
          </p>
        </div>
      </section>

      {/* Vision and Mission Section */}
      <section className="py-16 md:py-20">
        <div className="container max-w-5xl mx-auto px-4 md:px-6 grid md:grid-cols-2 gap-10 md:gap-16 items-start">
          <div className="space-y-4 p-6 bg-card border border-border rounded-lg shadow-sm hover:shadow-lg transition-shadow">
            <div className="flex items-center gap-3">
              <TargetIcon className="h-10 w-10 text-blue-500" />
              <h2 className="text-3xl font-semibold text-foreground">Vizyonumuz</h2>
            </div>
            <p className="text-muted-foreground text-base leading-relaxed">
              Herkes için erişilebilir, yenilikçi ve yüksek kaliteli yapay zeka eğitimi sunarak Türkiye'nin teknolojik geleceğine yön veren lider bir platform olmak ve AI alanında global bir etki yaratmak.
            </p>
          </div>
          <div className="space-y-4 p-6 bg-card border border-border rounded-lg shadow-sm hover:shadow-lg transition-shadow">
            <div className="flex items-center gap-3">
              <Compass className="h-10 w-10 text-green-500" />
              <h2 className="text-3xl font-semibold text-foreground">Misyonumuz</h2>
            </div>
            <p className="text-muted-foreground text-base leading-relaxed">
              En güncel AI bilgilerini ve pratik becerilerini, anlaşılır Türkçe içeriklerle sunmak; merak uyandıran, etkileşimli bir öğrenme topluluğu oluşturmak ve Türkiye'de yapay zeka okuryazarlığını en üst düzeye çıkarmak.
            </p>
          </div>
        </div>
      </section>

      {/* Why Kodleon Section */}
      <section className="py-16 md:py-20 bg-muted/30" aria-labelledby="why-kodleon-heading">
        <div className="container max-w-5xl mx-auto px-4 md:px-6">
          <div className="text-center mb-12 md:mb-16">
            <div className="inline-block p-3 mb-4 bg-primary/10 rounded-full border border-primary/20">
              <CheckCircle className="h-10 w-10 text-primary" />
            </div>
            <h2 id="why-kodleon-heading" className="text-3xl md:text-4xl font-bold tracking-tight mb-4 text-foreground">Neden Kodleon ile AI Öğrenmelisiniz?</h2>
            <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto">
              Yapay zeka yolculuğunuzda Kodleon'u tercih etmeniz için birçok iyi neden var. İşte onlardan bazıları:
            </p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
            {whyKodleonFeatures.map((feature) => (
              <Card key={feature.title} className="bg-card border-border text-center p-6 flex flex-col items-center transform transition-all duration-300 hover:shadow-xl hover:-translate-y-2">
                <div className="p-4 bg-primary/10 rounded-full mb-5 border border-primary/20">
                  {feature.icon}
                </div>
                <CardTitle className="text-xl font-semibold mb-2 text-foreground">{feature.title}</CardTitle>
                <CardContent className="text-sm text-muted-foreground flex-grow">
                  <p>{feature.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Team Section - Kept concise for now */}
      <section className="py-16 md:py-20">
        <div className="container max-w-3xl mx-auto px-4 md:px-6 text-center">
          <div className="inline-block p-3 mb-4 bg-primary/10 rounded-full border border-primary/20">
            <Users className="h-10 w-10 text-primary" />
          </div>
          <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-4 text-foreground">Tutkulu Ekibimiz</h2>
          <p className="text-lg md:text-xl text-muted-foreground leading-relaxed">
            Kodleon, yapay zeka, yazılım geliştirme ve eğitim alanlarında derin uzmanlığa ve tutkuya sahip bir ekip tarafından yönetilmektedir. Amacımız, size en kaliteli öğrenme deneyimini sunmak ve AI hedeflerinize ulaşmanızda size destek olmaktır. Sürekli araştırıyor, geliştiriyor ve en iyi uygulamaları takip ediyoruz.
          </p>
        </div>
      </section>
      
      {/* Call to Action Section */}
      <section className="py-16 md:py-24 bg-gradient-to-r from-primary via-purple-600 to-pink-600 text-primary-foreground">
        <div className="container max-w-4xl mx-auto px-4 md:px-6 text-center">
          <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-6">
            Yapay Zeka Maceranıza Bugün Başlayın!
          </h2>
          <p className="text-xl mb-10 text-primary-foreground/90 leading-relaxed max-w-2xl mx-auto">
            Kodleon ile geleceğin teknolojilerini öğrenmeye hazır mısınız? Kapsamlı eğitimlerimizi keşfedin ve AI dünyasında yerinizi alın.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button asChild size="lg" variant="secondary" className="rounded-full text-lg py-3 px-8 shadow-md hover:shadow-lg transition-transform hover:scale-105">
              <Link href="/topics" aria-label="Yapay zeka konularını keşfedin">Tüm Konuları Keşfet</Link>
            </Button>
            <Button asChild size="lg" variant="outline" className="bg-transparent text-primary-foreground border-primary-foreground/50 hover:bg-primary-foreground/10 hover:border-primary-foreground rounded-full text-lg py-3 px-8 shadow-md hover:shadow-lg transition-transform hover:scale-105">
              <Link href="/contact" aria-label="Bizimle iletişime geçin">Bize Ulaşın</Link>
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
}