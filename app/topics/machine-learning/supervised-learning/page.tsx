import { Metadata } from 'next';
import { createPageMetadata } from '@/lib/seo';
import Link from 'next/link';
import Image from 'next/image';
import { ArrowLeft, Target, CheckSquare, Settings, AlertTriangle, AppWindow, Cpu, Lightbulb, Zap, Brain, BookOpen, Briefcase, Users, BarChart, Shuffle, PieChart, Scale, CheckCircle, Code } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { ReactNode } from 'react';

interface ContentListItem {
  type: "list-item";
  item: string;
  icon?: ReactNode;
}

interface ContentOrderedListItem {
  type: "ordered-list-item";
  item: string;
}

type ContentItem = string | ContentListItem | ContentOrderedListItem;

interface Section {
  id: string;
  title: string;
  icon: ReactNode;
  content: ContentItem[];
}

const pageTitle = "Denetimli Öğrenme";
const pageDescription = "Etiketlenmiş verilerle modellerin nasıl eğitildiğini ve gelecekteki veriler için doğru tahminler yapmayı nasıl öğrendiğini keşfedin. Regresyon ve sınıflandırma gibi temel kavramları ve popüler algoritmaları inceleyin.";
const pageKeywords = "denetimli öğrenme, regresyon, sınıflandırma, makine öğrenmesi, yapay zeka, veri bilimi, model eğitimi, doğruluk, kesinlik, kodleon";
const pageUrl = "https://kodleon.com/topics/machine-learning/supervised-learning";
const imageUrl = "https://images.pexels.com/photos/577585/pexels-photo-577585.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"; // Replaced placeholder

export const metadata: Metadata = createPageMetadata({
  title: pageTitle,
  description: pageDescription,
  path: '/topics/machine-learning/supervised-learning',
  keywords: pageKeywords.split(", "),
});

const sections: Section[] = [
  {
    id: "regresyon-ve-siniflandirma",
    title: "Regresyon ve Sınıflandırma",
    icon: <Target className="w-8 h-8 text-blue-500" />,
    content: [
      "Denetimli öğrenme problemleri genellikle iki ana kategoriye ayrılır:",
      {
        type: "list-item",
        item: "**Regresyon:** Amaç, sürekli bir çıktı değerini tahmin etmektir. Örneğin, bir evin metrekare cinsinden büyüklüğüne, konumuna ve yaşına bakarak satış fiyatını tahmin etmek bir regresyon problemidir. Çıktı (fiyat) sürekli bir sayıdır.",
        icon: <BarChart className="w-5 h-5 text-blue-500 mr-2" />
      },
      {
        type: "list-item",
        item: "**Sınıflandırma:** Amaç, bir veri örneğini belirli kategorilerden birine atamaktır. Örneğin, bir e-postanın içeriğine bakarak \"spam\" veya \"spam değil\" olarak etiketlemek bir sınıflandırma problemidir. Çıktı (spam/spam değil) belirli bir kategoridir. Diğer örnekler arasında bir resimdeki nesneyi tanıma (kedi, köpek, kuş vb.) veya bir hastanın semptomlarına göre belirli bir hastalığı teşhis etme yer alır.",
        icon: <PieChart className="w-5 h-5 text-blue-500 mr-2" />
      },
      "Her iki görev türü için de farklı algoritmalar ve değerlendirme metrikleri kullanılır. Örneğin, regresyonda Ortalama Karesel Hata (MSE) veya Ortalama Mutlak Hata (MAE) gibi metrikler kullanılırken, sınıflandırmada Doğruluk (Accuracy), Kesinlik (Precision), Geri Çağırma (Recall) ve F1 Skoru gibi metrikler yaygındır."
    ]
  },
  {
    id: "nasil-calisir",
    title: "Nasıl Çalışır?",
    icon: <Settings className="w-8 h-8 text-green-500" />,
    content: [
      "Denetimli öğrenme süreci genellikle şu adımları içerir:",
      { type: "ordered-list-item", item: "**Veri Toplama ve Etiketleme:** Eğitim için ilgili veriler toplanır ve her bir veri örneği doğru çıktıyla etiketlenir." },
      { type: "ordered-list-item", item: "**Model Seçimi:** Görevin türüne (regresyon veya sınıflandırma) uygun bir makine öğrenmesi modeli seçilir. Popüler modeller arasında Doğrusal Regresyon, Lojistik Regresyon, Karar Ağaçları, Rastgele Ormanlar, Destek Vektör Makineleri (SVM) ve Sinir Ağları bulunur." },
      { type: "ordered-list-item", item: "**Model Eğitimi:** Etiketlenmiş eğitim verileri kullanılarak seçilen model eğitilir. Eğitim sırasında modelin parametreleri, tahmin hatalarını minimize edecek şekilde ayarlanır." },
      { type: "ordered-list-item", item: "**Model Değerlendirme:** Eğitilmiş model, eğitimde kullanılmayan ayrı bir test veri kümesi üzerinde değerlendirilir. Performans metrikleri (doğruluk, kesinlik, geri çağırma, F1 skoru, Ortalama Karesel Hata vb.) kullanılarak modelin ne kadar iyi genelleme yaptığı ölçülür." },
      { type: "ordered-list-item", item: "**Model Ayarlama:** Model performansı yetersizse, hiperparametreler ayarlanabilir veya farklı modeller denenebilir." }
    ]
  },
  {
    id: "yaygin-zorluklar",
    title: "Yaygın Zorluklar",
    icon: <AlertTriangle className="w-8 h-8 text-yellow-500" />,
    content: [
      "Denetimli öğrenme modellerini eğitirken dikkat edilmesi gereken bazı yaygın zorluklar şunlardır:",
      {
        type: "list-item",
        item: "**Aşırı Uyum (Overfitting):** Modelin eğitim verilerini çok iyi öğrenmesi, ancak yeni ve görülmemiş verilere genelleme yapamaması durumudur. Model eğitim verisindeki gürültüyü bile öğrenir, bu da test verisinde kötü performansa yol açar. Genellikle karmaşık modeller veya yetersiz eğitim verisi olduğunda ortaya çıkar. Aşırı uyumu azaltmak için daha fazla veri toplama, model karmaşıklığını azaltma, düzenlileştirme (regularization) teknikleri veya çapraz doğrulama (cross-validation) kullanılabilir.",
        icon: <Shuffle className="w-5 h-5 text-yellow-500 mr-2" />
      },
      {
        type: "list-item",
        item: "**Yetersiz Uyum (Underfitting):** Modelin eğitim verilerini bile yeterince öğrenememesi, dolayısıyla hem eğitim hem de test verisinde kötü performans göstermesi durumudur. Model verinin temel desenlerini yakalayamaz. Genellikle çok basit modeller veya yetersiz özellik seti kullanıldığında ortaya çıkar. Yetersiz uyumu gidermek için daha karmaşık bir model seçme, daha fazla ilgili özellik ekleme veya eğitim süresini artırma gibi yöntemler denenebilir.",
        icon: <Scale className="w-5 h-5 text-yellow-500 mr-2" />
      }
    ]
  },
  {
    id: "uygulama-alanlari",
    title: "Uygulama Alanları",
    icon: <AppWindow className="w-8 h-8 text-purple-500" />,
    content: [
      "Denetimli öğrenme çok çeşitli alanlarda kullanılır:",
      { type: "list-item", item: "**E-posta Filtreleme:** Gelen e-postaların spam olup olmadığını sınıflandırma.", icon: <CheckSquare className="w-5 h-5 text-purple-500 mr-2" /> },
      { type: "list-item", item: "**Görüntü Tanıma:** Resimdeki nesneleri veya yüzleri tanımlama.", icon: <Users className="w-5 h-5 text-purple-500 mr-2" /> },
      { type: "list-item", item: "**Tıbbi Teşhis:** Hastalıkların teşhisi için tıbbi görüntü veya hasta verilerini analiz etme.", icon: <Briefcase className="w-5 h-5 text-purple-500 mr-2" /> },
      { type: "list-item", item: "**Fiyat Tahmini:** Konut, hisse senedi veya ürün fiyatları gibi sürekli değerleri tahmin etme (Regresyon).", icon: <BarChart className="w-5 h-5 text-purple-500 mr-2" /> },
      { type: "list-item", item: "**Müşteri Kayıp Oranı Tahmini:** Hangi müşterilerin hizmeti bırakma olasılığının yüksek olduğunu tahmin etme (Sınıflandırma).", icon: <Users className="w-5 h-5 text-purple-500 mr-2" /> },
    ]
  },
  {
    id: "temel-algoritmalar",
    title: "Temel Algoritmalar",
    icon: <Cpu className="w-8 h-8 text-red-500" />,
    content: [
      "Denetimli öğrenmenin bazı temel algoritmaları şunlardır:",
      { type: "list-item", item: "Doğrusal Regresyon", icon: <Code className="w-5 h-5 text-red-500 mr-2" /> },
      { type: "list-item", item: "Lojistik Regresyon", icon: <Code className="w-5 h-5 text-red-500 mr-2" /> },
      { type: "list-item", item: "Karar Ağaçları", icon: <Code className="w-5 h-5 text-red-500 mr-2" /> },
      { type: "list-item", item: "Rastgele Ormanlar", icon: <Code className="w-5 h-5 text-red-500 mr-2" /> },
      { type: "list-item", item: "Destek Vektör Makineleri (SVM)", icon: <Code className="w-5 h-5 text-red-500 mr-2" /> },
      { type: "list-item", item: "K-En Yakın Komşular (KNN)", icon: <Code className="w-5 h-5 text-red-500 mr-2" /> },
      { type: "list-item", item: "Naive Bayes", icon: <Code className="w-5 h-5 text-red-500 mr-2" /> },
      { type: "list-item", item: "Yapay Sinir Ağları (Özellikle Sınıflandırma ve Regresyon için)", icon: <Brain className="w-5 h-5 text-red-500 mr-2" /> }
    ]
  }
];

export default function SupervisedLearningPage() {
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
        <h2 id="giris" className="text-3xl font-semibold border-b pb-2 mb-6">Giriş: Denetimli Öğrenme Nedir?</h2>
        <p>
          Denetimli öğrenme, makine öğrenmesinin temel taşlarından biridir ve adından da anlaşılacağı gibi, öğrenme süreci bir "öğretmen" veya "denetçi" gözetiminde gerçekleşir. Bu "öğretmen", modele doğru cevapları içeren etiketlenmiş veriler sunar. Modelin amacı, verilen girdiler (özellikler) ile bu girdilere karşılık gelen çıktılar (etiketler) arasındaki ilişkiyi öğrenmektir. Bu sayede model, daha önce hiç görmediği yeni verilere doğru tahminler veya sınıflandırmalar yapabilir hale gelir.
        </p>
        <p>
          Örneğin, bir e-posta sınıflandırma modelini eğitirken, binlerce e-postayı "spam" veya "spam değil" olarak etiketleyerek modele sunarız. Model, bu etiketli verilerden hangi kelimelerin, ifadelerin veya özelliklerin spam e-postaları karakterize ettiğini öğrenir. Eğitim tamamlandıktan sonra, yeni bir e-posta geldiğinde model bu öğrendiklerini kullanarak e-postanın spam olup olmadığını tahmin edebilir.
        </p>
      </section>

      <Separator className="my-12" />

      <section className="mb-16">
        <h2 className="text-3xl font-bold mb-10 text-center">Denetimli Öğrenmenin Temel Kavramları</h2>
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
                  } else if (item.type === 'ordered-list-item') {
                     return (
                      <div key={index} className="flex items-start mb-2">
                         <span className="mr-2 font-semibold text-primary">{index - (section.content.findIndex(el => typeof el !== 'string')) + 1}.</span>
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
        <h2 className="text-3xl font-bold mb-6">Denetimli Öğrenme ile Akıllı Çözümler Geliştirin</h2>
        <p className="text-lg text-muted-foreground mb-8 max-w-3xl mx-auto">
          Denetimli öğrenme, verilerinizden değer yaratarak birçok alanda akıllı uygulamalar geliştirmenize olanak tanır. Bu temel prensipleri ve algoritmaları öğrenerek, veri odaklı karar verme süreçlerinizi güçlendirebilirsiniz.
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
                "url": "https://kodleon.com/logo.png" // Assume you have a logo here
              }
            },
            "datePublished": "2023-10-27", // Example date
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