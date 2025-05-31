import { Metadata } from 'next';
import Link from 'next/link';
import Image from 'next/image';
import {
  ArrowLeft, Lightbulb, LayoutGrid, Target, Sigma, Users, BrainCircuit, Share2, ListTree, GitCompareArrows, CheckSquare, SearchCode, Puzzle, Code2
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
// import { createPageMetadata } from '@/lib/seo'; // Assuming you have this utility

// Placeholder for createPageMetadata if not available
const createPageMetadata = (data: any) => data; 

export const metadata: Metadata = createPageMetadata({
  title: 'Temel Makine Öğrenmesi Algoritmaları',
  description: "Regresyon, sınıflandırma ve kümeleme gibi temel makine öğrenmesi algoritmalarının çalışma prensiplerini ve kullanım alanlarını keşfedin.",
  path: '/topics/machine-learning/basic-ml-algorithms',
  keywords: ['makine öğrenmesi algoritmaları', 'regresyon', 'sınıflandırma', 'kümeleme', 'temel ml', 'kodleon', 'supervised learning', 'unsupervised learning', 'linear regression', 'logistic regression', 'k-means', 'decision trees'],
  imageUrl: 'https://images.pexels.com/photos/3861958/pexels-photo-3861958.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2' 
});

const algorithmCategories = [
  {
    id: "supervised-learning",
    title: "Denetimli Öğrenme (Supervised Learning)",
    icon: <Target className="w-8 h-8 text-sky-500" />,
    description: "Etiketli veriler kullanarak modelin girdiler ve çıktılar arasındaki ilişkiyi öğrenmesini hedefler. Model, bu öğrenilen ilişkiyi kullanarak yeni, görünmeyen verilere tahminlerde bulunur.",
    algorithms: [
      {
        name: "Regresyon Algoritmaları (Regression)",
        details: "Sayısal bir hedef değişkeni (örneğin, fiyat, sıcaklık) tahmin etmek için kullanılır.",
        icon: <Sigma className="w-6 h-6 text-green-500" />,
        examples: [
          { name: "Lineer Regresyon", use: "Ev fiyatlarını metrekareye göre tahmin etme." },
          { name: "Polinomsal Regresyon", use: "Doğrusal olmayan ilişkileri modelleme." }
        ]
      },
      {
        name: "Sınıflandırma Algoritmaları (Classification)",
        details: "Bir girdiyi önceden tanımlanmış kategorilerden birine (örneğin, spam/spam değil, kedi/köpek) atamak için kullanılır.",
        icon: <LayoutGrid className="w-6 h-6 text-indigo-500" />,
        examples: [
          { name: "Lojistik Regresyon", use: "Bir e-postanın spam olup olmadığını belirleme." },
          { name: "K-En Yakın Komşu (KNN)", use: "Benzer özelliklere sahip müşterileri gruplama." },
          { name: "Karar Ağaçları (Decision Trees)", use: "Kredi başvurularının onaylanıp onaylanmayacağına karar verme." },
          { name: "Destek Vektör Makineleri (SVM)", use: "Görüntü sınıflandırma." }
        ]
      }
    ]
  },
  {
    id: "unsupervised-learning",
    title: "Denetimsiz Öğrenme (Unsupervised Learning)",
    icon: <Users className="w-8 h-8 text-amber-500" />,
    description: "Etiketlenmemiş verilerdeki gizli kalıpları, yapıları veya ilişkileri bulmayı amaçlar. Veriyi keşfetmek ve anlamlandırmak için kullanılır.",
    algorithms: [
      {
        name: "Kümeleme Algoritmaları (Clustering)",
        details: "Veri noktalarını benzerliklerine göre gruplara (kümelere) ayırır.",
        icon: <Share2 className="w-6 h-6 text-red-500" />,
        examples: [
          { name: "K-Means Kümeleme", use: "Müşteri segmentasyonu, benzer belgeleri gruplama." },
          { name: "Hiyerarşik Kümeleme", use: "Türlerin evrimsel ilişkilerini inceleme." }
        ]
      },
      {
        name: "Boyut İndirgeme (Dimensionality Reduction)",
        details: "Veri kümesindeki özellik (değişken) sayısını azaltırken önemli bilgileri korumayı hedefler. Görselleştirme ve model performansını artırmak için kullanılır.",
        icon: <ListTree className="w-6 h-6 text-teal-500" />,
        examples: [
          { name: "Temel Bileşen Analizi (PCA)", use: "Yüksek boyutlu veriyi görselleştirme, gürültü azaltma." }
        ]
      }
    ]
  }
];

export default function BasicMlAlgorithmsPage() {
  const pageTitle = "Temel Makine Öğrenmesi Algoritmaları";
  const heroImageUrl = "https://images.pexels.com/photos/3861958/pexels-photo-3861958.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2";

  return (
    <div className="bg-background text-foreground">
      {/* Hero Section */}
      <section className="relative bg-gradient-to-b from-muted/30 to-background">
        <div className="absolute inset-0 opacity-10">
          <Image
            src={heroImageUrl}
            alt="Temel ML Algoritmaları Arka Planı"
            fill
            className="object-cover"
            priority
          />
        </div>
        <div className="container max-w-6xl mx-auto py-16 md:py-24 px-4 md:px-6 relative z-10">
          <div className="mb-8">
            <Button asChild variant="ghost" size="sm" className="gap-1.5 text-muted-foreground hover:text-primary">
              <Link href="/topics/machine-learning" aria-label="Makine Öğrenmesi konusuna geri dön">
                <ArrowLeft className="h-4 w-4" aria-hidden="true" />
                Makine Öğrenmesi Ana Konusu
              </Link>
            </Button>
          </div>
          <div className="text-center">
            <div className="inline-block p-3 mb-6 bg-primary/10 rounded-full border border-primary/20">
                <BrainCircuit className="h-12 w-12 text-primary" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-foreground mb-6">
              {pageTitle}
            </h1>
            <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
              Makine öğrenmesinin temelini oluşturan regresyon, sınıflandırma ve kümeleme gibi algoritmaların nasıl çalıştığını ve nerelerde kullanıldığını keşfedin.
            </p>
          </div>
        </div>
      </section>

      {/* Main Content Area */}
      <div className="container max-w-6xl mx-auto py-12 md:py-16 px-4 md:px-6">
        <div className="prose prose-lg dark:prose-invert max-w-none mb-12">
          <h2 id="introduction" className="flex items-center text-3xl font-bold text-foreground"><Lightbulb className="h-8 w-8 text-yellow-500 mr-3" />Giriş: Makine Öğrenmesi Algoritmaları Nedir?</h2>
          <p>
            Makine öğrenmesi algoritmaları, bilgisayar sistemlerinin verilerden öğrenmesini sağlayan matematiksel ve istatistiksel yöntemlerdir. Açıkça programlanmak yerine, bu algoritmalar veri kümelerindeki desenleri tanıyarak ve bu desenlere dayanarak tahminlerde bulunarak veya kararlar alarak 'öğrenirler'. Temelde, bir problemi çözmek için veriyi analiz eden ve bu analiz sonucunda bir model oluşturan talimat setleridir. Bu modeller daha sonra yeni, görünmeyen veriler üzerinde genellemeler yapmak için kullanılır.
          </p>
          <p>
            Farklı problem türleri ve veri yapıları için çeşitli algoritmalar mevcuttur. En yaygın kategoriler denetimli öğrenme, denetimsiz öğrenme ve pekiştirmeli öğrenmedir. Bu sayfada, en temel ve yaygın kullanılan denetimli ve denetimsiz öğrenme algoritmalarına odaklanacağız.
          </p>
        </div>

        {algorithmCategories.map(category => (
          <section key={category.id} aria-labelledby={category.id} className="mb-16">
            <div className="flex items-center gap-4 mb-6 pb-3 border-b border-border">
                {category.icon}
                <h2 id={category.id} className="text-3xl font-semibold text-foreground">{category.title}</h2>
            </div>
            <p className="text-muted-foreground mb-8 text-base leading-relaxed">{category.description}</p>
            <div className="space-y-8">
              {category.algorithms.map(algoType => (
                <Card key={algoType.name} className="shadow-lg border-border hover:border-primary/30 transition-all bg-card">
                  <CardHeader className="bg-muted/30">
                    <div className="flex items-center gap-3">
                      {algoType.icon}
                      <CardTitle className="text-xl font-semibold">{algoType.name}</CardTitle>
                    </div>
                  </CardHeader>
                  <CardContent className="pt-5 text-base space-y-3">
                    <p className="text-muted-foreground">{algoType.details}</p>
                    <h4 className="font-semibold text-foreground">Yaygın Örnekler:</h4>
                    <ul className="list-disc list-inside space-y-1.5 text-muted-foreground">
                      {algoType.examples.map(ex => (
                        <li key={ex.name}><strong>{ex.name}:</strong> {ex.use}</li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              ))}
            </div>
          </section>
        ))}

        <Separator className="my-12 md:my-16" />

        <section aria-labelledby="choosing-algorithm-heading" className="mb-12 md:mb-16">
            <div className="flex items-center gap-4 mb-6 pb-3 border-b border-border">
                <Puzzle className="w-8 h-8 text-purple-500" />
                <h2 id="choosing-algorithm-heading" className="text-3xl font-semibold text-foreground">Doğru Algoritmayı Seçmek</h2>
            </div>
            <p className="text-muted-foreground mb-6 text-base leading-relaxed">
            Bir makine öğrenmesi problemi için en iyi algoritmayı seçmek her zaman basit değildir ve genellikle deneme yanılma gerektirir. Ancak, karar verme sürecinize yardımcı olabilecek bazı faktörler şunlardır:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {[
                    { title: "Problemin Türü", desc: "Regresyon mu, sınıflandırma mı, kümeleme mi?", icon: <GitCompareArrows className="w-7 h-7 text-primary"/> },
                    { title: "Veri Setinin Boyutu ve Yapısı", desc: "Özellik sayısı, örnek sayısı, veri türleri.", icon: <SearchCode className="w-7 h-7 text-primary"/> },
                    { title: "Modelin Yorumlanabilirliği", desc: "Modelin kararlarını ne kadar kolay anlayabiliyoruz?", icon: <CheckSquare className="w-7 h-7 text-primary"/> },
                    { title: "Eğitim Hızı ve Hesaplama Maliyeti", desc: "Modelin eğitilmesi ne kadar sürer ve ne kadar kaynak gerektirir?", icon: <BrainCircuit className="w-7 h-7 text-primary"/> },
                    { title: "Doğruluk ve Performans", desc: "Modelin ne kadar doğru tahminler yapması bekleniyor?", icon: <Target className="w-7 h-7 text-primary"/> },
                    { title: "Verideki Örüntüler", desc: "Veri lineer mi, non-lineer mi? Önyargı var mı?", icon: <ListTree className="w-7 h-7 text-primary"/> }
                ].map(factor => (
                    <Card key={factor.title} className="bg-card border-border">
                        <CardHeader className="flex flex-row items-center gap-3 pb-3">
                            {factor.icon} <CardTitle className="text-lg">{factor.title}</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <p className="text-sm text-muted-foreground">{factor.desc}</p>
                        </CardContent>
                    </Card>
                ))}
            </div>
             <p className="text-muted-foreground mt-6 text-base leading-relaxed">
            Genellikle, basit bir modelle başlamak (örneğin, lineer regresyon veya lojistik regresyon) ve ardından gerektiğinde daha karmaşık modellere geçmek iyi bir stratejidir.
            </p>
        </section>

        <Separator className="my-12 md:my-16" />

        <section className="text-center">
            <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-4">Algoritmalarla Pratik Yapmaya Hazır mısınız?</h2>
            <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Bu temel algoritmaları anlamak, makine öğrenmesi yolculuğunuzda sağlam bir temel oluşturur. Şimdi bu bilgileri kullanarak Python ile pratik yapmaya ve kendi modellerinizi oluşturmaya başlayabilirsiniz!
            </p>
            <div className="flex flex-wrap justify-center items-center gap-4">
                <Button asChild size="lg" className="bg-primary hover:bg-primary/90 text-primary-foreground">
                <Link href="/topics/machine-learning/python-for-ml">
                    <Code2 className="mr-2 h-5 w-5" /> Python ile ML Kütüphaneleri
                </Link>
                </Button>
                <Button asChild variant="outline" size="lg">
                <Link href="/topics">
                    <Lightbulb className="mr-2 h-5 w-5" /> Tüm Yapay Zeka Konuları
                </Link>
                </Button>
            </div>
        </section>
      </div>
    </div>
  );
} 