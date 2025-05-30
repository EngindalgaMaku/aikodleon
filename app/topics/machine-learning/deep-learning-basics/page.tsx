"use client";

import Link from "next/link";
import Image from "next/image";
import { ArrowLeft, Brain, Zap, Layers3, FunctionSquare, TrendingDown, Shuffle, Cpu, Lightbulb, BookOpen, Eye, Users, FileText, Rocket, Database, Shapes, CheckCircle2, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

const keyConcepts = [
  {
    title: "Yapay Sinir Ağları (ANN)",
    description: "İnsan beyninden esinlenerek oluşturulan, birbirine bağlı işlem birimlerinden (nöronlardan) oluşan hesaplama modelleridir.",
    icon: <Brain className="w-8 h-8 text-blue-500" />,
  },
  {
    title: "Nöronlar (Hücreler)",
    description: "Bir sinir ağının temel yapı taşıdır. Girdileri alır, işler ve bir çıktı üretir.",
    icon: <Zap className="w-8 h-8 text-yellow-500" />,
  },
  {
    title: "Katmanlar (Layers)",
    description: "Nöronların organize olduğu gruplardır: Girdi, Gizli ve Çıktı katmanları.",
    icon: <Layers3 className="w-8 h-8 text-green-500" />,
  },
  {
    title: "Aktivasyon Fonksiyonları",
    description: "Nöronun çıktısını belirleyen, genellikle doğrusal olmayan fonksiyonlardır (örn: Sigmoid, ReLU, Tanh).",
    icon: <FunctionSquare className="w-8 h-8 text-purple-500" />,
  },
  {
    title: "İleri Yayılım (Forward Propagation)",
    description: "Verinin sinir ağı katmanları boyunca girdiden çıktıya doğru hareket etmesi sürecidir.",
    icon: <TrendingDown className="w-8 h-8 text-indigo-500 transform scale-y-[-1]" />,
  },
  {
    title: "Kayıp Fonksiyonu (Loss Function)",
    description: "Modelin tahminleri ile gerçek değerler arasındaki farkı (hatayı) ölçer.",
    icon: <TrendingDown className="w-8 h-8 text-red-500" />,
  },
  {
    title: "Geri Yayılım (Backpropagation)",
    description: "Kayıp fonksiyonundaki hatayı ağ boyunca geriye doğru yayarak ağırlıkların güncellenmesini sağlar.",
    icon: <Shuffle className="w-8 h-8 text-pink-500" />,
  },
  {
    title: "Optimizasyon Algoritmaları",
    description: "Kayıp fonksiyonunu minimize etmek için modelin ağırlıklarını ayarlar (örn: Gradient Descent, Adam).",
    icon: <Cpu className="w-8 h-8 text-teal-500" />,
  },
];

export default function DeepLearningBasicsPage() {
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
            src="https://images.pexels.com/photos/2599244/pexels-photo-2599244.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
            alt="Derin Öğrenme Temelleri Kapak Fotoğrafı"
            fill
            className="object-cover"
            priority
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent flex flex-col justify-end p-6 md:p-8">
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-3 leading-tight drop-shadow-md">Derin Öğrenme Temelleri</h1>
            <p className="text-lg md:text-xl text-gray-100 drop-shadow-sm">
              Yapay sinir ağlarının büyüleyici dünyasına adım atın ve makinelerin nasıl öğrendiğini keşfedin.
            </p>
          </div>
        </div>
      </section>

      <section className="mb-16 prose prose-lg dark:prose-invert max-w-none">
        <h2 id="giris" className="text-3xl font-semibold border-b pb-2 mb-4">Giriş: Derin Öğrenme Nedir?</h2>
        <p>
          Derin Öğrenme (Deep Learning), çok katmanlı yapay sinir ağlarını kullanarak karmaşık problemleri çözmeyi amaçlayan bir makine öğrenmesi alt alanıdır. İnsan beyninin bilgi işleme şeklinden esinlenir ve büyük miktarda veriden özellikleri otomatik olarak öğrenme yeteneğine sahiptir. Bu sayede görüntü tanıma, doğal dil işleme, ses tanıma gibi birçok alanda devrim niteliğinde başarılara imza atmıştır.
        </p>
        <p>
          Temelde, derin öğrenme modelleri, veriyi hiyerarşik bir şekilde işleyen ve her katmanda daha karmaşık temsiller öğrenen katmanlı mimarilerden oluşur. "Derin" ifadesi, bu katmanların sayısının (genellikle gizli katmanların) fazla olmasını ifade eder. Bu katmanlı yapı, modelin basit özelliklerden başlayarak giderek daha soyut ve karmaşık özellikler öğrenmesini sağlar.
        </p>
      </section>
      
      <Separator className="my-12" />

      <section className="mb-16">
        <h2 id="temel-kavramlar" className="text-3xl font-bold mb-10 text-center">Derin Öğrenmenin Temel Taşları</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {keyConcepts.map((concept) => (
            <Card key={concept.title} className="shadow-lg hover:shadow-xl transition-all duration-300 ease-in-out transform hover:-translate-y-1 flex flex-col bg-card">
              <CardHeader className="flex flex-row items-center gap-4 pb-3">
                <div className="p-3 bg-primary/10 rounded-full flex-shrink-0">
                  {concept.icon}
                </div>
                <CardTitle className="text-lg font-semibold mt-0">{concept.title}</CardTitle>
              </CardHeader>
              <CardContent className="flex-grow pt-0">
                <p className="text-sm text-muted-foreground">{concept.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      <Separator className="my-12" />

      <section className="mb-16">
        <h2 id="sinir-agi-mimarisi" className="text-3xl font-bold mb-10 text-center">Yapay Sinir Ağı Mimarisi: Katmanların Dansı</h2>
        <div className="grid md:grid-cols-5 gap-8 items-start">
          <div className="md:col-span-3 prose prose-lg dark:prose-invert max-w-none">
            <p>
              Bir yapay sinir ağı, birbiriyle bağlantılı nöronlardan oluşan katmanlar halinde düzenlenir. Bu yapı, bilginin ağ üzerinden akmasını ve işlenmesini sağlar. Temel katman türleri şunlardır:
            </p>
            <ul className="space-y-3">
              <li><strong className="text-primary">Girdi Katmanı (Input Layer):</strong> Modelin dış dünyadan aldığı ilk veriyi temsil eder. Bu, bir resmin piksel değerleri, bir metindeki kelimelerin sayısal temsilleri veya bir sensörden gelen ölçümler olabilir.</li>
              <li><strong className="text-primary">Gizli Katmanlar (Hidden Layers):</strong> Girdi ve çıktı katmanları arasında yer alan bu katmanlar, asıl "öğrenmenin" gerçekleştiği yerdir. Veri üzerinde karmaşık doğrusal olmayan dönüşümler yaparak ve önemli özellikleri çıkararak modelin soyut temsiller oluşturmasını sağlarlar. Derin öğrenme adını, genellikle bu gizli katmanların çokluğundan alır.</li>
              <li><strong className="text-primary">Çıktı Katmanı (Output Layer):</strong> Modelin sonuca ulaştığı katmandır. Yapılan göreve bağlı olarak, bu katman bir sınıflandırma sonucu (örneğin, 'kedi' veya 'köpek'), bir regresyon değeri (örneğin, bir evin fiyatı) veya üretilmiş bir metin olabilir.</li>
            </ul>
            <p>
              Her katmandaki nöronlar, bir önceki katmandaki nöronlardan gelen sinyalleri alır. Bu sinyaller, bağlantıların 'ağırlıkları' ile çarpılır, bir 'bias' değeri eklenir ve toplam, bir 'aktivasyon fonksiyonundan' geçirilerek nöronun kendi çıktısı oluşturulur. Bu çıktı daha sonra bir sonraki katmana iletilir.
            </p>
          </div>
          <div className="md:col-span-2 bg-muted p-6 rounded-xl flex flex-col items-center justify-center min-h-[350px] shadow-lg sticky top-24">
            <Layers3 className="w-28 h-28 text-primary mb-6 opacity-80" />
            <h3 className="text-xl font-semibold mb-3 text-center">Katmanlı Yapı</h3>
            <p className="text-muted-foreground text-center text-sm mb-2">
              Girdi → Gizli Katman(lar) → Çıktı
            </p>
            <Image src="/images/neural-network-layers.svg" alt="Sinir Ağı Katmanları Diyagramı" width={250} height={150} className="mt-4" /> 
            <p className="text-xs text-muted-foreground mt-2 text-center">
              <em>Basit bir sinir ağı katman diyagramı (örnek görsel)</em>
            </p>
          </div>
        </div>
      </section>

      <Separator className="my-12" />

      <section className="mb-16">
        <h2 id="ogrenme-sureci" className="text-3xl font-bold mb-10 text-center">Öğrenme Süreci: Veriden Bilgeliğe Yolculuk</h2>
        <div className="space-y-10">
          <Card className="shadow-lg overflow-hidden hover:shadow-xl transition-shadow duration-300">
            <CardHeader className="bg-indigo-500/10">
              <CardTitle className="text-2xl font-semibold flex items-center gap-3 text-indigo-700 dark:text-indigo-400">
                <TrendingDown className="w-7 h-7 transform scale-y-[-1]" />
                1. İleri Yayılım (Forward Propagation)
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-6 prose dark:prose-invert max-w-none">
              <p>
                Öğrenme yolculuğu, verinin ağa sunulmasıyla başlar. Girdi verisi, ilk katmandan başlayarak ağ boyunca ileri doğru akar. Her katmanda, nöronlar girdileri alır, ağırlıklarla çarpar, bir bias ekler ve bir aktivasyon fonksiyonundan geçirerek çıktılarını üretir. Bu çıktılar bir sonraki katmanın girdisi olur. Bu süreç, çıktı katmanına kadar devam eder ve model nihai bir tahminde bulunur. Örneğin, bir kedi resmini girdi olarak verdiğimizde, çıktı katmanı 'kedi' sınıfı için yüksek bir olasılık değeri üretebilir.
              </p>
            </CardContent>
          </Card>

          <Card className="shadow-lg overflow-hidden hover:shadow-xl transition-shadow duration-300">
            <CardHeader className="bg-red-500/10">
              <CardTitle className="text-2xl font-semibold flex items-center gap-3 text-red-700 dark:text-red-400">
                <TrendingDown className="w-7 h-7" />
                2. Kayıp Hesaplanması (Loss Calculation)
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-6 prose dark:prose-invert max-w-none">
              <p>
                Modelin yaptığı tahmin ne kadar doğru? İşte bu sorunun cevabını kayıp fonksiyonu verir. Modelin ürettiği tahmin ile gerçek (beklenen) değer arasındaki farkı sayısal olarak ifade eder. Örneğin, model 'köpek' derken gerçekte resim 'kedi' ise, kayıp fonksiyonu bu hatayı ölçer. Kullanılan probleme göre farklı kayıp fonksiyonları (örn: Mean Squared Error, Cross-Entropy) tercih edilir. Amaç her zaman bu kayıp değerini olabildiğince düşürmektir.
              </p>
            </CardContent>
          </Card>

          <Card className="shadow-lg overflow-hidden hover:shadow-xl transition-shadow duration-300">
            <CardHeader className="bg-pink-500/10">
              <CardTitle className="text-2xl font-semibold flex items-center gap-3 text-pink-700 dark:text-pink-400">
                <Shuffle className="w-7 h-7" />
                3. Geri Yayılım (Backpropagation) ve Optimizasyon
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-6 prose dark:prose-invert max-w-none">
              <p>
                Modelin hatasını (kaybını) öğrendikten sonra, bu hatayı azaltmak için ağırlıkları güncellememiz gerekir. İşte burada geri yayılım devreye girer. Hesaplanan kayıp, ağ boyunca çıktı katmanından girdi katmanına doğru geriye doğru yayılır. Bu süreçte, her bir ağırlığın ve bias'ın kayba olan katkısı (yani gradyanları) hesaplanır. Ardından, bir optimizasyon algoritması (en yaygını Gradient Descent ve varyantları olan Adam, RMSprop vb.) bu gradyanları kullanarak ağırlıkları ve bias'ları yavaş yavaş ayarlar. Bu ayarlama, kayıp fonksiyonunu minimize edecek yönde yapılır. Bu ileri yayılım, kayıp hesaplama, geri yayılım ve ağırlık güncelleme döngüsü, model tatmin edici bir performansa ulaşana kadar binlerce veya milyonlarca kez tekrarlanır.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>
    
      <Separator className="my-16" />

      <section className="mb-16">
          <h2 id="mimari-turleri" className="text-3xl font-bold mb-10 text-center">Yaygın Derin Öğrenme Mimarileri</h2>
          <p className="text-lg text-muted-foreground text-center mb-10 max-w-3xl mx-auto">
              Derin öğrenme, farklı türdeki problemleri çözmek için özelleşmiş çeşitli mimarilere sahiptir. İşte en sık karşılaşılanlardan bazıları:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <Card className="shadow-lg hover:shadow-xl transition-shadow duration-300">
                  <CardHeader>
                      <div className="flex items-center gap-3 mb-2">
                          <Eye className="w-10 h-10 text-sky-500" />
                          <CardTitle className="text-2xl font-semibold">Evrişimli Sinir Ağları (CNN)</CardTitle>
                      </div>
                      <CardDescription>Görüntü tanıma, nesne tespiti ve görüntü segmentasyonu gibi görsel görevlerde uzmandır.</CardDescription>
                  </CardHeader>
                  <CardContent>
                      <p className="text-sm text-muted-foreground mb-4">Özellikle pikseller arasındaki mekansal ilişkileri yakalamak için tasarlanmışlardır. Filtreler (kernels) kullanarak görüntülerden özellik haritaları çıkarırlar.</p>
                      <Button variant="outline" asChild><Link href="/topics/neural-networks/convolutional-neural-networks">CNN Detayları (Yakında)</Link></Button>
                  </CardContent>
              </Card>
              <Card className="shadow-lg hover:shadow-xl transition-shadow duration-300">
                  <CardHeader>
                      <div className="flex items-center gap-3 mb-2">
                          <FileText className="w-10 h-10 text-emerald-500" />
                          <CardTitle className="text-2xl font-semibold">Tekrarlayan Sinir Ağları (RNN)</CardTitle>
                      </div>
                      <CardDescription>Doğal dil işleme, zaman serisi analizi ve konuşma tanıma gibi sıralı verilerle çalışmak için idealdir.</CardDescription>
                  </CardHeader>
                  <CardContent>
                      <p className="text-sm text-muted-foreground mb-4">Geçmiş bilgileri hatırlayabilen ve sıralı bağımlılıkları modelleyebilen geri besleme döngülerine sahiptirler. LSTM ve GRU gibi gelişmiş varyantları bulunur.</p>
                      <Button variant="outline" asChild><Link href="/topics/neural-networks/recurrent-neural-networks">RNN Detayları (Yakında)</Link></Button>
                  </CardContent>
              </Card>
          </div>
      </section>

      <Separator className="my-16" />

      <section className="mb-12 text-center bg-muted p-8 rounded-lg shadow-inner">
        <h2 className="text-3xl font-bold mb-6">Derin Öğrenme Yolculuğunuzda Sıradaki Adımlar</h2>
        <p className="text-lg text-muted-foreground mb-8 max-w-3xl mx-auto">
          Derin öğrenmenin temellerini kavradınız! Bu heyecan verici alanda bilginizi daha da derinleştirmek için keşfedebileceğiniz birçok yol var. Pratik uygulamalar yaparak, farklı mimarileri inceleyerek ve en son araştırmaları takip ederek kendinizi geliştirebilirsiniz.
        </p>
        <div className="flex flex-wrap justify-center items-center gap-4">
            <Button asChild size="lg" className="bg-primary hover:bg-primary/90 text-primary-foreground">
                <Link href="/topics/machine-learning">
                  <Brain className="mr-2 h-5 w-5" /> Makine Öğrenmesi Ana Konusuna Dön
                </Link>
            </Button>
            <Button asChild variant="outline" size="lg">
                <Link href="/topics">
                  <BookOpen className="mr-2 h-5 w-5" /> Tüm Yapay Zeka Konularını Keşfet
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
            "headline": "Derin Öğrenme Temelleri",
            "description": "Derin öğrenmenin temel kavramlarını, yapay sinir ağlarının çalışma prensiplerini ve öğrenme süreçlerini keşfedin.",
            "keywords": "derin öğrenme, yapay sinir ağları, nöronlar, aktivasyon fonksiyonları, geri yayılım, derin öğrenme temelleri, kodleon, makine öğrenmesi",
            "image": "https://images.pexels.com/photos/2599244/pexels-photo-2599244.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
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
            "datePublished": "2023-10-27",
            "dateModified": new Date().toISOString().split('T')[0],
            "mainEntityOfPage": {
              "@type": "WebPage",
              "@id": "https://kodleon.com/topics/machine-learning/deep-learning-basics"
            }
          })
        }}
      />
    </div>
  );
} 