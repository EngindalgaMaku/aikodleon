import { Metadata } from 'next';
import { createPageMetadata } from '@/lib/seo';
import Link from 'next/link';
import Image from 'next/image';
import { ArrowLeft, Target, Puzzle, Zap, Lightbulb, Cog, Bot, BookOpen, Gamepad2, AlertCircle, TrendingUp, Brain, Dna, SlidersHorizontal, Award, CheckCircle, Goal, Route, GitFork, HelpCircle, ListChecks } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

const pageTitle = "Pekiştirmeli Öğrenme";
const pageDescription = "Bir ajanın deneme-yanılma yoluyla bir ortamda en iyi kararları nasıl verdiğini ve toplam ödülünü nasıl maksimize ettiğini keşfedin. Markov Karar Süreçleri, Q-learning ve Derin Pekiştirmeli Öğrenme gibi temel kavramları ve algoritmaları inceleyin.";
const pageKeywords = "pekiştirmeli öğrenme, reinforcement learning, RL, ajan, ortam, ödül, politika, değer fonksiyonu, MDP, Q-learning, DQN, yapay zeka, kodleon";
const pageUrl = "https://kodleon.com/topics/machine-learning/reinforcement-learning";
const imageUrl = "https://images.pexels.com/photos/6153354/pexels-photo-6153354.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2";

export const metadata: Metadata = createPageMetadata({
  title: pageTitle,
  description: pageDescription,
  path: '/topics/machine-learning/reinforcement-learning',
  keywords: pageKeywords.split(", "),
});

const sections = [
  {
    id: "temel-kavramlar",
    title: "Temel Kavramlar",
    icon: <Puzzle className="w-8 h-8 text-blue-500" />,
    content: [
      "Pekiştirmeli öğrenmenin anlaşılması için bazı temel kavramlar önemlidir:",
      { type: "list-item", item: "**Ajan (Agent):** Öğrenen ve eylemler gerçekleştiren varlıktır.", icon: <Bot className="w-5 h-5 text-blue-500 mr-2" /> },
      { type: "list-item", item: "**Ortam (Environment):** Ajanın etkileşimde bulunduğu dış dünyadır.", icon: <Gamepad2 className="w-5 h-5 text-blue-500 mr-2" /> },
      { type: "list-item", item: "**Durum (State):** Ortamın belirli bir zamandaki anlık konfigürasyonudur.", icon: <HelpCircle className="w-5 h-5 text-blue-500 mr-2" /> },
      { type: "list-item", item: "**Eylem (Action):** Ajanın belirli bir durumda alabileceği kararlardır.", icon: <Route className="w-5 h-5 text-blue-500 mr-2" /> },
      { type: "list-item", item: "**Ödül (Reward):** Ajanın bir eylem sonucunda ortamdan aldığı geri bildirimdir; genellikle sayısal bir değerdir.", icon: <Award className="w-5 h-5 text-blue-500 mr-2" /> },
      { type: "list-item", item: "**Politika (Policy):** Ajanın belirli bir durumda hangi eylemi seçeceğini belirleyen stratejidir.", icon: <Goal className="w-5 h-5 text-blue-500 mr-2" /> },
      { type: "list-item", item: "**Değer Fonksiyonu (Value Function):** Belirli bir durumdan başlayarak veya belirli bir eylemi belirli bir durumda gerçekleştirerek elde edilmesi beklenen gelecekteki toplam ödül miktarını tahmin eder.", icon: <TrendingUp className="w-5 h-5 text-blue-500 mr-2" /> },
    ]
  },
  {
    id: "markov-karar-surecleri",
    title: "Markov Karar Süreçleri (MDPs)",
    icon: <GitFork className="w-8 h-8 text-green-500" />,
    content: [
      "Pekiştirmeli öğrenmenin matematiksel çerçevesi genellikle **Markov Karar Süreçleri (MDPs)** ile tanımlanır. Bir MDP, bir durum kümesi (S), bir eylem kümesi (A), durum geçiş olasılıkları, ödül fonksiyonu ve bir indirim faktöründen oluşur. MDP, bir ajanın dinamik bir ortamda ardışık kararlar alması gereken durumları modellemek için kullanılır. Ajanın mevcut durumu, gelecekteki durumunu ve alabileceği ödülü belirlemek için yeterlidir (Markov özelliği)."
    ]
  },
  {
    id: "algoritmalar-ve-yaklasimlar",
    title: "Algoritmalar ve Yaklaşımlar",
    icon: <SlidersHorizontal className="w-8 h-8 text-purple-500" />,
    content: [
      "Pekiştirmeli öğrenmede çeşitli algoritmalar kullanılır:",
      { type: "list-item", item: "**Q-Learning:** Değer tabanlı bir algoritma olup, her durum-eylem çifti için bir Q-değeri öğrenir.", icon: <ListChecks className="w-5 h-5 text-purple-500 mr-2" /> },
      { type: "list-item", item: "**SARSA:** Q-learning'e benzer, ancak bir sonraki eylemi de dikkate alarak Q-değerlerini günceller.", icon: <ListChecks className="w-5 h-5 text-purple-500 mr-2" /> },
      { type: "list-item", item: "**Derin Q Ağları (DQN):** Q-learning'i derin sinir ağları ile birleştirerek karmaşık durum alanlarına sahip problemlerin çözümüne olanak tanır.", icon: <Brain className="w-5 h-5 text-purple-500 mr-2" /> },
      { type: "list-item", item: "**Politika Gradyanları (Policy Gradients):** Doğrudan politikayı optimize eden algoritmalardır.", icon: <ListChecks className="w-5 h-5 text-purple-500 mr-2" /> },
      { type: "list-item", item: "**Aktör-Kritik (Actor-Critic) Metotlar:** Hem politikayı hem de değer fonksiyonunu öğrenen melez yaklaşımlardır.", icon: <ListChecks className="w-5 h-5 text-purple-500 mr-2" /> },
    ]
  },
  {
    id: "yaygin-zorluklar",
    title: "Yaygın Zorluklar",
    icon: <AlertCircle className="w-8 h-8 text-yellow-500" />,
    content: [
      "Pekiştirmeli öğrenmenin pratik uygulamalarında karşılaşılan bazı zorluklar şunlardır:",
      { type: "list-item", item: "**Ödül Gecikmesi (Reward Delay):** Bir eylemin sonucunun (ödül veya ceza) hemen değil, belirli bir süre sonra ortaya çıkması durumudur. Ajanın, hangi eylemlerin gecikmiş de olsa olumlu sonuçlara yol açtığını doğru bir şekilde ilişkilendirmesi zor olabilir. Bu durum, ajanın öğrenme sürecini yavaşlatabilir veya suboptimal politikalar öğrenmesine neden olabilir.", icon: <AlertCircle className="w-5 h-5 text-yellow-500 mr-2" /> },
      { type: "list-item", item: "**Keşif ve Sömürü Dengesi (Exploration vs. Exploitation):** Ajanın ya en yüksek ödülü getirdiği bilinen eylemleri tekrarlayarak mevcut bilgisini sömürmesi (exploitation) ya da daha iyi ödüller bulma umuduyla yeni eylemleri denemesi (exploration) arasındaki dengeyi bulma sorunudur. Etkili bir öğrenme için her ikisinin de uygun oranlarda yapılması gerekir.", icon: <AlertCircle className="w-5 h-5 text-yellow-500 mr-2" /> },
    ]
  },
  {
    id: "uygulama-alanlari",
    title: "Uygulama Alanları",
    icon: <Gamepad2 className="w-8 h-8 text-red-500" />,
    content: [
      "Pekiştirmeli öğrenme özellikle şu alanlarda etkilidir:",
      { type: "list-item", item: "**Oyun Oynama:** Satranç, Go ve video oyunları gibi karmaşık oyunlarda insanüstü performans sergileme (DeepMind'ın AlphaGo'su gibi)." , icon: <Gamepad2 className="w-5 h-5 text-red-500 mr-2" />},
      { type: "list-item", item: "**Robotik:** Robotların karmaşık hareketleri ve görevleri öğrenmesi.", icon: <Bot className="w-5 h-5 text-red-500 mr-2" /> },
      { type: "list-item", item: "**Otonom Sistemler:** Kendi kendine giden araçlar ve dronlar.", icon: <Cog className="w-5 h-5 text-red-500 mr-2" /> },
      { type: "list-item", item: "**Kaynak Yönetimi:** Enerji şebekeleri veya veri merkezlerinde kaynak dağılımını optimize etme.", icon: <Dna className="w-5 h-5 text-red-500 mr-2" /> },
      { type: "list-item", item: "**Finans:** Ticaret stratejileri geliştirme.", icon: <TrendingUp className="w-5 h-5 text-red-500 mr-2" /> },
    ]
  }
];

export default function ReinforcementLearningPage() {
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
        <h2 id="giris" className="text-3xl font-semibold border-b pb-2 mb-6">Giriş: Pekiştirmeli Öğrenme Nedir?</h2>
        <p>
          Pekiştirmeli Öğrenme (RL), bir yazılım ajanının bir ortamda en iyi kararları nasıl alacağını deneme-yanılma yoluyla öğrendiği bir makine öğrenmesi türüdür. Etiketli veri kümelerine ihtiyaç duymak yerine, RL ajanı eylemler gerçekleştirir ve bu eylemlerin sonuçlarına göre ödüller veya cezalar alır. Ajanın temel amacı, zaman içinde toplam ödülünü maksimize edecek bir strateji (politika) geliştirmektir.
        </p>
        <p>
          Bu öğrenme süreci, insanların ve hayvanların yeni beceriler öğrenme şekline çok benzer. Örneğin, bir çocuğun yürümeyi öğrenmesi gibi; düşer (ceza), kalkar, dener (eylem) ve sonunda dengede durmayı (ödül) başarır. RL, robotların karmaşık görevleri öğrenmesinden oyunlarda insanüstü performans sergileyen yapay zeka sistemlerine kadar geniş bir yelpazede uygulama bulur.
        </p>
      </section>

      <Separator className="my-12" />

      <section className="mb-16">
        <h2 className="text-3xl font-bold mb-10 text-center">Pekiştirmeli Öğrenmenin Temel Bileşenleri ve Yaklaşımları</h2>
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
                  } else if (item.type === 'list-item') {
                    return (
                      <div key={index} className="flex items-start mb-2">
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
        <h2 className="text-3xl font-bold mb-6">Pekiştirmeli Öğrenme ile Geleceği Şekillendirin</h2>
        <p className="text-lg text-muted-foreground mb-8 max-w-3xl mx-auto">
          Pekiştirmeli öğrenme, dinamik ve karmaşık ortamlarda karar verme yeteneğiyle yapay zekanın sınırlarını zorluyor. Bu heyecan verici alanı keşfederek, geleceğin akıllı sistemlerinin geliştirilmesine katkıda bulunabilirsiniz.
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
            "datePublished": "2023-10-29", 
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