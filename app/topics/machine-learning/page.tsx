import Link from "next/link";
import Image from "next/image";
import {
  ArrowLeft, ArrowRight, Brain, CheckCircle2, Database, Code, ChartBar, Network, BookOpen,
  Sigma, BrainCircuit, ClipboardCheck, GraduationCap, FlaskConical, Lightbulb, Users, FileText, Eye // Added new icons
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

// Placeholder for related topics data - adjust as needed
const relatedTopicsData = {
  "nlp": {
    title: "Doğal Dil İşleme",
    description: "Makinelerin insan dilini nasıl anlayıp işlediğini ve ürettiğini öğrenin.",
    icon: <FileText className="h-8 w-8 text-chart-2" />,
    imageUrl: "https://images.pexels.com/photos/7412095/pexels-photo-7412095.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    slug: "nlp"
  },
  "computer-vision": {
    title: "Bilgisayarlı Görü",
    description: "Bilgisayarların görüntüleri nasıl algıladığını ve işlediğini anlayın.",
    icon: <Eye className="h-8 w-8 text-chart-3" />,
    imageUrl: "https://images.pexels.com/photos/8438922/pexels-photo-8438922.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    slug: "computer-vision"
  },
  "ai-ethics": {
    title: "AI Etiği",
    description: "Yapay zekanın etik kullanımı ve toplumsal etkileri üzerine tartışmalar.",
    icon: <Users className="h-8 w-8 text-chart-1" />,
    imageUrl: "https://images.pexels.com/photos/8386422/pexels-photo-8386422.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    slug: "ai-ethics"
  }
};
const relatedTopicSlugs = ["nlp", "computer-vision", "ai-ethics"];


const learningJourneySteps = [
  {
    title: "Temel Matematik ve İstatistik",
    description: "Lineer cebir, kalkülüs ve olasılık gibi temel matematiksel kavramlar ile istatistiksel analiz ve hipotez testi temelleri.",
    icon: <Sigma className="w-10 h-10 text-blue-500" />
  },
  {
    title: "Python Programlama",
    description: "Veri yapıları, kontrol akışı, fonksiyonlar ve NumPy, Pandas gibi temel veri bilimi kütüphaneleri.",
    icon: <Code className="w-10 h-10 text-green-500" />
  },
  {
    title: "Veri Analizi ve Görselleştirme",
    description: "Veri temizleme, dönüştürme, keşifsel veri analizi (EDA) ve Matplotlib, Seaborn gibi araçlarla etkili görselleştirmeler.",
    icon: <ChartBar className="w-10 h-10 text-yellow-500" />
  },
  {
    title: "Temel ML Algoritmaları",
    description: "Regresyon, sınıflandırma, kümeleme gibi temel algoritma türlerini ve çalışma prensiplerini anlama.",
    icon: <BrainCircuit className="w-10 h-10 text-purple-500" />
  },
  {
    title: "Model Geliştirme ve Değerlendirme",
    description: "Veri bölme, model eğitimi, hiperparametre ayarı, çapraz doğrulama ve performans metrikleri ile model değerlendirme.",
    icon: <ClipboardCheck className="w-10 h-10 text-red-500" />
  },
  {
    title: "İleri Düzey Konular ve Uzmanlaşma",
    description: "Derin öğrenme, doğal dil işleme, bilgisayarlı görü gibi alanlarda uzmanlaşma veya MLOps gibi konulara yönelme.",
    icon: <GraduationCap className="w-10 h-10 text-indigo-500" />
  }
];

const machineLearningSkills = [
  "Veri Analizi ve Ön İşleme",
  "Python ile ML Uygulamaları (Scikit-learn, Pandas, NumPy)",
  "Model Seçimi ve Değerlendirme Metrikleri",
  "Denetimli Öğrenme (Regresyon, Sınıflandırma)",
  "Denetimsiz Öğrenme (Kümeleme, Boyut İndirgeme)",
  "Temel Derin Öğrenme Kavramları",
  "Hiperparametre Optimizasyonu",
  "Makine Öğrenmesi İş Akışları (Pipelines)"
];

const machineLearningResources = [
  { title: "Python for Data Science Handbook", type: "E-Kitap", link: "#" },
  { title: "Scikit-learn Dökümantasyonu", type: "Döküman", link: "#" },
  { title: "Kaggle Kursları: Intro to ML", type: "Kurs", link: "#" },
  { title: "StatQuest with Josh Starmer", type: "Video", link: "#" }
];

export default function MachineLearningPage() {
  const pageTitle = "Makine Öğrenmesi";
  const pageDescription = "Algoritmaların veri kullanarak nasıl öğrendiğini ve tahminlerde bulunduğunu keşfedin. Kodleon ile makine öğrenmesi dünyasına adım atın.";
  const heroImageUrl = "https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2";

  return (
    <div className="bg-background text-foreground">
      {/* Hero section */}
      <section className="relative">
        <div className="relative h-[300px] md:h-[450px]">
          <Image
            src={heroImageUrl}
            alt={`${pageTitle} - Kodleon Yapay Zeka Platformu`}
            fill
            className="object-cover"
            priority
          />
          <div className="absolute inset-0 bg-gradient-to-t from-background via-background/80 to-transparent" />
        </div>
        <div className="container max-w-6xl mx-auto relative -mt-32 md:-mt-40 pb-12 px-4 md:px-6">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 mb-4">
              <Button asChild variant="ghost" size="sm" className="gap-1 text-muted-foreground hover:text-primary">
                <Link href="/topics" aria-label="Tüm yapay zeka konularına dön">
                  <ArrowLeft className="h-4 w-4" aria-hidden="true" />
                  Tüm Konular
                </Link>
              </Button>
            </div>
            <div className="flex items-center gap-4 mb-6">
              <div className="p-4 rounded-full bg-primary/10 backdrop-blur-sm border border-primary/20 shadow-lg">
                <Database className="h-10 w-10 text-primary" />
              </div>
              <h1 id="topic-title" className="text-4xl md:text-5xl font-bold tracking-tight text-foreground">
                {pageTitle}
              </h1>
            </div>
            <p className="text-xl text-muted-foreground leading-relaxed">
              {pageDescription}
            </p>
          </div>
        </div>
      </section>

      {/* Main content */}
      <section className="container max-w-6xl mx-auto py-12 md:py-16 px-4 md:px-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-10 lg:gap-12">
          <div className="lg:col-span-2 space-y-12">
            <article className="prose prose-lg dark:prose-invert max-w-none">
              <h2 id="overview-heading">Genel Bakış</h2>
              <p>
                Makine öğrenmesi (ML), bilgisayar sistemlerinin açıkça programlanmadan verilerden öğrenmesini ve bu öğrenme yoluyla belirli görevleri yerine getirmesini sağlayan yapay zekanın dinamik bir dalıdır. ML algoritmaları, büyük veri kümelerindeki desenleri ve ilişkileri tanımlayarak çalışır, böylece bilinmeyen veriler hakkında tahminlerde bulunabilir veya stratejik kararlar alabilirler.
              </p>
              <p>
                Bu alan, denetimli öğrenme (etiketli verilerle eğitim), denetimsiz öğrenme (etiketlenmemiş verilerden desen keşfi) ve pekiştirmeli öğrenme (deneme-yanılma yoluyla öğrenme) gibi çeşitli paradigmaları içerir. Makine öğrenmesi sadece bir algoritma seti olmanın ötesinde; problem tanımlama, veri toplama ve ön işleme, model seçimi, eğitim, titiz değerlendirme ve etkili dağıtım gibi adımları içeren iteratif bir süreçtir. Tavsiye sistemlerinden otonom araçlara, tıbbi teşhisten finansal analizlere ve doğal dil işlemeden bilgisayarlı görüye kadar çok geniş bir uygulama yelpazesine sahiptir. Bu alanda uzmanlaşmak, günümüzün ve geleceğin teknoloji dünyasında çığır açan çözümler geliştirme ve önemli bir yer edinme anlamına gelir.
              </p>
            </article>

            <section aria-labelledby="subtopics-heading">
              <h2 id="subtopics-heading" className="text-3xl font-bold mb-8 text-foreground border-b pb-3">Alt Konular</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {[
                  { title: "Denetimli Öğrenme", description: "Etiketli verilerle modellerin nasıl eğitildiğini ve tahminlerde bulunduğunu öğrenin.", imageUrl: "https://images.pexels.com/photos/577585/pexels-photo-577585.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2", href: "/topics/machine-learning/supervised-learning" },
                  { title: "Denetimsiz Öğrenme", description: "Etiketlenmemiş verilerden kalıpları ve yapıları nasıl keşfedeceğinizi anlayın.", imageUrl: "https://images.pexels.com/photos/373543/pexels-photo-373543.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2", href: "/topics/machine-learning/unsupervised-learning" },
                  { title: "Pekiştirmeli Öğrenme", description: "Deneme yanılma yoluyla ajanların çevreleriyle nasıl etkileşime girdiğini ve öğrendiğini keşfedin.", imageUrl: "https://images.pexels.com/photos/6153354/pexels-photo-6153354.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2", href: "/topics/machine-learning/reinforcement-learning" },
                  { title: "Derin Öğrenme Temelleri", description: "Derin öğrenmenin temel kavramlarını ve yapay sinir ağlarının çalışma prensiplerini öğrenin.", imageUrl: "https://images.pexels.com/photos/2599244/pexels-photo-2599244.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2", href: "/topics/machine-learning/deep-learning-basics" }
                ].map((subtopic, index) => (
                  <Link href={subtopic.href} key={index} className="block no-underline group">
                    <Card className="overflow-hidden transition-all duration-300 hover:shadow-xl hover:-translate-y-1 border-border hover:border-primary/50 bg-card">
                      <div className="relative h-48">
                        <Image
                          src={subtopic.imageUrl}
                          alt={`${subtopic.title} - ${pageTitle} alt konusu`}
                          fill
                          className="object-cover transition-transform duration-300 group-hover:scale-105"
                          sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                          loading={index < 2 ? "eager" : "lazy"}
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/40 to-transparent" />
                      </div>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-xl group-hover:text-primary transition-colors">{subtopic.title}</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-sm text-muted-foreground">{subtopic.description}</p>
                      </CardContent>
                    </Card>
                  </Link>
                ))}
              </div>
            </section>

            <section aria-labelledby="learning-journey-heading">
              <h2 id="learning-journey-heading" className="text-3xl font-bold mb-8 text-foreground border-b pb-3">Makine Öğrenmesi Yolculuğunuz</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {learningJourneySteps.map((step, index) => {
                  const isPythonProgramming = step.title === "Python Programlama";
                  const isBasicAlgorithms = step.title === "Temel ML Algoritmaları";
                  const linkHref = isPythonProgramming
                    ? "/topics/machine-learning/python-for-ml"
                    : isBasicAlgorithms
                    ? "/topics/machine-learning/basic-ml-algorithms"
                    : undefined;

                  if (linkHref) {
                    return (
                      <Link href={linkHref} key={index} className="block no-underline group">
                        <Card className="flex flex-col items-center p-6 text-center hover:shadow-lg transition-shadow duration-300 border-border hover:border-primary/30 bg-card h-full">
                          <div className="p-4 bg-primary/10 rounded-full mb-4 border border-primary/20">
                            {step.icon}
                          </div>
                          <CardTitle className="text-xl mb-2 text-foreground group-hover:text-primary transition-colors">{step.title}</CardTitle>
                          <CardDescription className="text-muted-foreground flex-grow">{step.description}</CardDescription>
                        </Card>
                      </Link>
                    );
                  }

                  return (
                    <Card key={index} className="flex flex-col items-center p-6 text-center hover:shadow-lg transition-shadow duration-300 border-border hover:border-primary/30 bg-card h-full">
                      <div className="p-4 bg-primary/10 rounded-full mb-4 border border-primary/20">
                        {step.icon}
                      </div>
                      <CardTitle className="text-xl mb-2 text-foreground">{step.title}</CardTitle>
                      <CardDescription className="text-muted-foreground flex-grow">{step.description}</CardDescription>
                    </Card>
                  );
                })}
              </div>
            </section>
          </div>

          <aside className="lg:col-span-1">
            <div className="bg-muted/50 rounded-lg p-6 sticky top-24 border border-border shadow-sm">
              <h3 className="text-xl font-semibold mb-6 text-foreground border-b pb-3">Temel Beceriler ve Kaynaklar</h3>
              <div className="space-y-3 mb-6">
                {machineLearningSkills.map((skill, index) => (
                  <div key={index} className="flex items-start gap-3">
                    <CheckCircle2 className="h-5 w-5 text-primary mt-1 flex-shrink-0" aria-hidden="true" />
                    <span className="text-foreground">{skill}</span>
                  </div>
                ))}
              </div>

              <Separator className="my-6" />

              <h3 className="text-lg font-semibold mb-4 text-foreground">Önerilen Kaynaklar</h3>
              <ul className="space-y-3 mb-6">
                {machineLearningResources.map((resource, index) => (
                  <li key={index}>
                    <Link
                      href={resource.link}
                      target="_blank" rel="noopener noreferrer"
                      className="flex items-center justify-between p-3 bg-background rounded-md hover:bg-secondary border border-border hover:border-primary/30 transition-all group"
                      aria-label={`${resource.title} kaynağını incele - ${resource.type}`}
                    >
                      <span className="font-medium text-foreground group-hover:text-primary transition-colors">{resource.title}</span>
                      <span className="text-xs text-muted-foreground bg-primary/10 px-2 py-1 rounded-full group-hover:bg-primary group-hover:text-primary-foreground transition-colors">{resource.type}</span>
                    </Link>
                  </li>
                ))}
              </ul>

              <Separator className="my-6" />
              <h3 className="text-lg font-semibold mb-4 text-foreground">Makine Öğrenmesi Simülatörü</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Makine öğrenmesi algoritmalarını interaktif bir ortamda deneyimleyin ve temel kavramları uygulamalı olarak pekiştirin.
              </p>
              <Button asChild className="w-full bg-green-600 hover:bg-green-700 text-white">
                <Link href="https://ml.kodleon.com" target="_blank" rel="noopener noreferrer">
                  <FlaskConical className="mr-2 h-4 w-4" /> Simülatöre Git
                </Link>
              </Button>

              <Separator className="my-8" />
              <Button asChild variant="default" className="w-full rounded-full bg-primary hover:bg-primary/90 text-primary-foreground">
                <Link href="/contact">
                  <Lightbulb className="mr-2 h-5 w-5" /> Projeleriniz İçin İletişime Geçin
                </Link>
              </Button>
            </div>
          </aside>
        </div>
      </section>

      {/* Related topics */}
      <section className="bg-muted/30 py-16 md:py-20 border-t border-border" aria-labelledby="related-topics-heading">
        <div className="container max-w-6xl mx-auto px-4 md:px-6">
          <h2 id="related-topics-heading" className="text-3xl font-bold mb-10 text-center text-foreground">İlgili Diğer Konular</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8">
            {relatedTopicSlugs.map((slug) => {
              const relatedTopic = relatedTopicsData[slug as keyof typeof relatedTopicsData];
              if (!relatedTopic) return null;
              return (
                <Link href={`/topics/${relatedTopic.slug}`} key={relatedTopic.slug} className="block no-underline group">
                  <Card className="overflow-hidden transition-all duration-300 hover:shadow-xl hover:-translate-y-1.5 border-border hover:border-primary/50 bg-card h-full flex flex-col">
                    <div className="relative h-52">
                      <Image
                        src={relatedTopic.imageUrl}
                        alt={`${relatedTopic.title} - İlgili yapay zeka konusu`}
                        fill
                        className="object-cover transition-transform duration-300 group-hover:scale-105"
                        sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                        loading="lazy"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/40 to-transparent" />
                      <div className="absolute bottom-4 left-4 p-3 rounded-full bg-background/80 backdrop-blur-sm border border-border shadow-md">
                        {relatedTopic.icon}
                      </div>
                    </div>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-xl group-hover:text-primary transition-colors">{relatedTopic.title}</CardTitle>
                    </CardHeader>
                    <CardContent className="flex-grow">
                      <p className="text-sm text-muted-foreground">{relatedTopic.description}</p>
                    </CardContent>
                    <CardFooter>
                      <Button asChild variant="ghost" className="gap-1.5 ml-auto text-primary hover:text-primary/80">
                        <span aria-label={`${relatedTopic.title} konusunu keşfedin`}>
                          Konuyu İncele
                          <ArrowRight className="h-4 w-4" aria-hidden="true" />
                        </span>
                      </Button>
                    </CardFooter>
                  </Card>
                </Link>
              );
            })}
          </div>
        </div>
      </section>

      {/* Structured data for SEO */}
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            "@context": "https://schema.org",
            "@type": "Course", // Can be Article, WebPage, etc.
            "name": `${pageTitle} Eğitimi | Kodleon`,
            "description": pageDescription,
            "provider": {
              "@type": "Organization",
              "name": "Kodleon",
              "sameAs": "https://kodleon.com"
            },
            "image": heroImageUrl,
            "url": `https://kodleon.com/topics/machine-learning`,
            "educationalLevel": "Beginner to Advanced",
            "keywords": "makine öğrenmesi, yapay zeka, python, veri bilimi, kodleon, makine öğrenmesi eğitimi, machine learning",
            "teaches": machineLearningSkills.join(", "),
            "hasCourseInstance": {
              "@type": "CourseInstance",
              "courseMode": "online",
              "inLanguage": "tr"
            }
          })
        }}
      />
    </div>
  );
}