"use client";

import Link from "next/link";
import Image from "next/image";
import { ArrowLeft, BarChart3, Cable, CheckCircle, CircleDashed, Dot, GitFork, Group, Layers2, Lightbulb, Scale, Settings2, Shapes, Shuffle, Users, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

const pageTitle = "Denetimsiz Öğrenme";
const pageDescription = "Etiketlenmemiş verilerden desenleri, yapıları ve ilişkileri bağımsız olarak keşfeden makine öğrenmesi dalını derinlemesine inceleyin.";
const pageKeywords = "denetimsiz öğrenme, kümeleme, boyut indirgeme, k-means, pca, dbscan, apriori, unsupervised learning, kodleon";
const pageUrl = "https://kodleon.com/topics/machine-learning/unsupervised-learning";
const imageUrl = "https://images.pexels.com/photos/373543/pexels-photo-373543.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2";

const keyAlgorithms = [
  {
    title: "K-Means Kümeleme",
    description: "Veri noktalarını, önceden belirlenmiş 'K' sayıda kümeye, her bir noktanın kendi küme merkezine (centroid) en yakın olacak şekilde atar.",
    icon: <CircleDashed className="w-8 h-8 text-purple-500" />,
    details: [
      "Basit ve hızlı bir algoritmadır.",
      "Küme sayısının (K) önceden bilinmesi gerekir.",
      "Farklı boyut ve yoğunluktaki kümelere duyarlıdır.",
      "Başlangıç merkezlerinin seçimine göre sonuç değişebilir."
    ]
  },
  {
    title: "DBSCAN (Density-Based Spatial Clustering of Applications with Noise)",
    description: "Yoğunluk tabanlı bir kümeleme algoritmasıdır. Keyfi şekillerdeki kümeleri bulabilir ve gürültü noktalarını (outliers) ayırabilir.",
    icon: <Shapes className="w-8 h-8 text-teal-500" />,
    details: [
      "Küme sayısını önceden belirlemeyi gerektirmez.",
      "Farklı yoğunluktaki kümelere daha az duyarlıdır.",
      "Parametreleri (epsilon ve minPts) belirlemek zor olabilir."
    ]
  },
  {
    title: "Hiyerarşik Kümeleme",
    description: "Veri noktalarını ya birleştirerek (agglomerative) ya da bölerek (divisive) bir küme hiyerarşisi (dendrogram) oluşturur.",
    icon: <GitFork className="w-8 h-8 text-orange-500" />,
    details: [
      "Küme sayısını önceden belirlemek gerekmez; dendrogramdan istenen sayıda küme seçilebilir.",
      "Görselleştirme için dendrogram kullanışlıdır.",
      "Büyük veri kümeleri için hesaplama maliyeti yüksek olabilir."
    ]
  },
  {
    title: "Temel Bileşen Analizi (PCA - Principal Component Analysis)",
    description: "Veri kümesindeki varyansı en üst düzeye çıkaran yeni, daha düşük boyutlu bir özellik uzayı (temel bileşenler) bularak boyut indirgeme yapar.",
    icon: <BarChart3 className="w-8 h-8 text-sky-500" />,
    details: [
      "Veri görselleştirme ve gürültü azaltma için etkilidir.",
      "Doğrusal bir tekniktir; doğrusal olmayan ilişkileri yakalayamayabilir.",
      "Bileşenler daha az yorumlanabilir olabilir."
    ]
  },
  {
    title: "t-SNE (t-distributed Stochastic Neighbor Embedding)",
    description: "Yüksek boyutlu verilerin düşük boyutlu bir alanda (genellikle 2D veya 3D) görselleştirilmesi için kullanılan, doğrusal olmayan bir boyut indirgeme tekniğidir.",
    icon: <Layers2 className="w-8 h-8 text-rose-500" />,
    details: [
      "Özellikle yüksek boyutlu verilerdeki yerel yapıları korumada iyidir.",
      "Hesaplama maliyeti yüksektir, büyük veri kümeleri için yavaş olabilir.",
      "Sonuçlar parametre ayarlarına duyarlıdır."
    ]
  },
  {
    title: "Apriori Algoritması (İlişkilendirme Kuralı Madenciliği)",
    description: "Büyük işlem veritabanlarında sık geçen öğe kümelerini (frequent itemsets) bulmak ve bunlardan güçlü ilişkilendirme kuralları çıkarmak için kullanılır.",
    icon: <Cable className="w-8 h-8 text-lime-500" />,
    details: [
      "Pazar sepeti analizi gibi uygulamalarda popülerdir.",
      "Çok sayıda öğe olduğunda hesaplama açısından pahalı olabilir.",
      "Minimum destek ve güven eşiklerinin belirlenmesi gerekir."
    ]
  }
];

export default function UnsupervisedLearningPage() {
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
        <h2 id="giris" className="text-3xl font-semibold border-b pb-2 mb-4">Giriş: Denetimsiz Öğrenme Nedir?</h2>
        <p>
          Denetimsiz öğrenme, makine öğrenmesinin büyüleyici bir dalıdır çünkü algoritmalara etiketlenmemiş veri kümelerindeki gizli desenleri, yapıları ve ilişkileri kendi başlarına keşfetme yeteneği verir. Denetimli öğrenmenin aksine, burada modele "doğru" cevaplar sağlanmaz; bunun yerine, algoritma verinin içsel yapısını anlamaya, veriyi daha anlamlı bir şekilde düzenlemeye, sıkıştırmaya veya özetlemeye odaklanır.
        </p>
        <p>
          Bu yaklaşım, özellikle elimizde etiketli veri olmadığında veya verinin doğasını anlamak istediğimizde son derece değerlidir. Müşteri segmentasyonu, anomali tespiti, konu modelleme ve veri görselleştirme gibi birçok pratik uygulaması bulunmaktadır.
        </p>
      </section>

      <Separator className="my-12" />

      <section className="mb-16">
        <h2 id="temel-gorevler" className="text-3xl font-bold mb-10 text-center">Temel Görevler ve Algoritmalar</h2>
        <p className="text-lg text-muted-foreground text-center mb-10 max-w-3xl mx-auto">
          Denetimsiz öğrenme, çeşitli görevleri yerine getirmek için farklı algoritmalar kullanır. İşte en yaygın görevler ve bu görevler için kullanılan popüler algoritmalar:
        </p>
        <div className="space-y-12">
          {keyAlgorithms.map((algo) => (
            <Card key={algo.title} className="shadow-lg hover:shadow-xl transition-shadow duration-300 overflow-hidden">
              <CardHeader className="bg-muted/50">
                <div className="flex items-start gap-4">
                  <div className="p-3 bg-primary/10 rounded-lg flex-shrink-0">
                    {algo.icon}
                  </div>
                  <div>
                    <CardTitle className="text-2xl font-semibold mb-1">{algo.title}</CardTitle>
                    <CardDescription className="text-md">{algo.description}</CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="pt-6">
                {algo.details && algo.details.length > 0 && (
                  <div className="space-y-3">
                    <h4 className="font-semibold text-lg mb-2">Önemli Noktalar:</h4>
                    <ul className="list-none space-y-2 pl-0">
                      {algo.details.map((detail, index) => (
                        <li key={index} className="flex items-start">
                          <Dot className="h-5 w-5 text-primary mt-0.5 mr-2 flex-shrink-0" /> 
                          <span>{detail}</span>
                        </li>
                      ))}
        </ul>
                  </div>
                )}
                 {/* Placeholder for a simple diagram or visual hint if applicable */}
                {algo.title.includes("K-Means") && (
                  <div className="mt-4 p-4 bg-primary/5 rounded-md text-center">
                    <Group className="w-16 h-16 text-purple-500 mx-auto mb-2 opacity-70" />
                    <p className="text-sm text-muted-foreground"><i>K-Means: Veri noktalarını K adet kümeye ayırır.</i></p>
                  </div>
                )}
                {algo.title.includes("PCA") && (
                  <div className="mt-4 p-4 bg-primary/5 rounded-md text-center">
                    <Settings2 className="w-16 h-16 text-sky-500 mx-auto mb-2 opacity-70" />
                    <p className="text-sm text-muted-foreground"><i>PCA: Verinin boyutunu azaltırken en çok bilgiyi korur.</i></p>
                  </div>
                )}
              </CardContent>
              {/* Potential Footer for links to more details or examples */}
            </Card>
          ))}
        </div>
      </section>

      <Separator className="my-16" />

      <section className="mb-12 text-center bg-muted p-8 rounded-lg shadow-inner">
        <h2 className="text-3xl font-bold mb-6">Denetimsiz Öğrenme ile Verinin Gizemini Çözün</h2>
        <p className="text-lg text-muted-foreground mb-8 max-w-3xl mx-auto">
          Denetimsiz öğrenme teknikleri, elinizdeki verinin derinliklerine inerek değerli içgörüler elde etmenizi sağlar. Bu algoritmaları anlayarak ve uygulayarak, karmaşık veri kümelerinden anlam çıkarabilir ve bilinmeyeni keşfedebilirsiniz.
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
            "datePublished": "2023-10-28", 
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