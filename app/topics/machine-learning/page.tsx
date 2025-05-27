import Link from "next/link";
import Image from "next/image";
import { ArrowLeft, ArrowRight, Brain, CheckCircle2, Database, Code, ChartBar, Network, BookOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

export default function MachineLearningPage() {
  return (
    <div>
      {/* Hero section */}
      <section className="relative">
        <div className="relative h-[300px] md:h-[400px]">
          <Image 
            src="https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
            alt="Makine Öğrenmesi"
            fill
            className="object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-background via-background/80 to-transparent" />
        </div>
        <div className="container max-w-6xl mx-auto relative -mt-32 pb-12">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 mb-4">
              <Button asChild variant="ghost" size="sm" className="gap-1">
                <Link href="/topics">
                  <ArrowLeft className="h-4 w-4" />
                  Tüm Konular
                </Link>
              </Button>
            </div>
            <div className="flex items-center gap-4 mb-6">
              <div className="p-3 rounded-full bg-primary/10 backdrop-blur-sm">
                <Database className="h-8 w-8 text-primary" />
              </div>
              <h1 className="text-4xl font-bold">Makine Öğrenmesi</h1>
            </div>
            <p className="text-xl text-muted-foreground">
              Algoritmaların veri kullanarak nasıl öğrendiğini ve tahminlerde bulunduğunu keşfedin.
            </p>
          </div>
        </div>
      </section>
      
      {/* Main content */}
      <section className="container max-w-6xl mx-auto py-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
          <div className="lg:col-span-2">
            <div className="prose prose-lg dark:prose-invert max-w-none">
              <h2>Genel Bakış</h2>
              <p>
                Makine öğrenmesi, bilgisayarların açık programlamaya gerek kalmadan öğrenmesini sağlayan yapay zeka uygulamalarıdır. 
                Verilerden öğrenerek tahminlerde bulunma, sınıflandırma yapma ve karmaşık örüntüleri tespit etme yeteneği kazandırır.
              </p>
              
              <h2>Alt Konular</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 not-prose">
                <Card className="overflow-hidden">
                  <div className="relative h-40">
                    <Image 
                      src="https://images.pexels.com/photos/577585/pexels-photo-577585.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
                      alt="Denetimli Öğrenme"
                      fill
                      className="object-cover"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent" />
                  </div>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">Denetimli Öğrenme</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Etiketli verilerle modellerin nasıl eğitildiğini ve tahminlerde bulunduğunu öğrenin.
                    </p>
                  </CardContent>
                </Card>

                <Card className="overflow-hidden">
                  <div className="relative h-40">
                    <Image 
                      src="https://images.pexels.com/photos/373543/pexels-photo-373543.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
                      alt="Denetimsiz Öğrenme"
                      fill
                      className="object-cover"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent" />
                  </div>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">Denetimsiz Öğrenme</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Etiketlenmemiş verilerden kalıpları ve yapıları nasıl keşfedeceğinizi anlayın.
                    </p>
                  </CardContent>
                </Card>

                <Card className="overflow-hidden">
                  <div className="relative h-40">
                    <Image 
                      src="https://images.pexels.com/photos/6153354/pexels-photo-6153354.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
                      alt="Pekiştirmeli Öğrenme"
                      fill
                      className="object-cover"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent" />
                  </div>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">Pekiştirmeli Öğrenme</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Deneme yanılma yoluyla ajanların çevreleriyle nasıl etkileşime girdiğini ve öğrendiğini keşfedin.
                    </p>
                  </CardContent>
                </Card>

                <Card className="overflow-hidden">
                  <div className="relative h-40">
                    <Image 
                      src="https://images.pexels.com/photos/2599244/pexels-photo-2599244.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
                      alt="Derin Öğrenme Temelleri"
                      fill
                      className="object-cover"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent" />
                  </div>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">Derin Öğrenme Temelleri</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Derin öğrenmenin temel kavramlarını ve yapay sinir ağlarının çalışma prensiplerini öğrenin.
                    </p>
                  </CardContent>
                </Card>
              </div>

              <h2>Öğrenme Yolculuğunuz</h2>
              <div className="space-y-6 not-prose">
                <div className="flex gap-4">
                  <div className="p-3 rounded-full bg-primary/10 h-fit">
                    <Code className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="text-xl font-medium mb-2">Python Programlama</h3>
                    <p className="text-muted-foreground">
                      Makine öğrenmesi için gerekli Python programlama temellerini öğrenin.
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="p-3 rounded-full bg-primary/10 h-fit">
                    <ChartBar className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="text-xl font-medium mb-2">Veri Analizi</h3>
                    <p className="text-muted-foreground">
                      Veri manipülasyonu, görselleştirme ve istatistiksel analiz tekniklerini keşfedin.
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="p-3 rounded-full bg-primary/10 h-fit">
                    <Network className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="text-xl font-medium mb-2">ML Algoritmaları</h3>
                    <p className="text-muted-foreground">
                      Temel makine öğrenmesi algoritmalarını ve uygulama alanlarını öğrenin.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div>
            <div className="bg-muted rounded-lg p-6 sticky top-24">
              <h3 className="text-xl font-medium mb-4">Bu Konuda Kazanacağınız Beceriler</h3>
              <ul className="space-y-3 mb-6">
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                  <span>Veri Analizi ve Ön İşleme</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                  <span>Python ile ML Uygulamaları</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                  <span>Model Seçimi ve Değerlendirme</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                  <span>Hiperparametre Optimizasyonu</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                  <span>ML Pipeline Oluşturma</span>
                </li>
              </ul>
              
              <Separator className="my-6" />
              
              <h3 className="text-xl font-medium mb-4">Önerilen Kaynaklar</h3>
              <ul className="space-y-4">
                <li>
                  <Link 
                    href="#" 
                    className="flex items-center justify-between p-3 bg-background rounded-md hover:bg-secondary transition-colors"
                  >
                    <span className="font-medium">ML Temelleri</span>
                    <span className="text-sm text-muted-foreground">Kurs</span>
                  </Link>
                </li>
                <li>
                  <Link 
                    href="#" 
                    className="flex items-center justify-between p-3 bg-background rounded-md hover:bg-secondary transition-colors"
                  >
                    <span className="font-medium">Python ile ML</span>
                    <span className="text-sm text-muted-foreground">Pratik</span>
                  </Link>
                </li>
                <li>
                  <Link 
                    href="#" 
                    className="flex items-center justify-between p-3 bg-background rounded-md hover:bg-secondary transition-colors"
                  >
                    <span className="font-medium">ML Algoritmaları</span>
                    <span className="text-sm text-muted-foreground">E-Kitap</span>
                  </Link>
                </li>
              </ul>
              
              <Button className="w-full mt-6 rounded-full">Derse Kaydol</Button>
            </div>
          </div>
        </div>
      </section>
      
      {/* Projects section */}
      <section className="bg-muted py-16">
        <div className="container">
          <h2 className="text-2xl font-bold mb-8">Örnek Projeler</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Müşteri Segmentasyonu</CardTitle>
                <CardDescription>
                  Denetimsiz öğrenme ile müşteri gruplarının analizi
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  K-means clustering kullanarak müşteri davranışlarını analiz eden bir proje.
                </p>
              </CardContent>
              <CardFooter>
                <Button asChild variant="ghost" className="gap-1 ml-auto">
                  <Link href="#">
                    Projeyi İncele
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                </Button>
              </CardFooter>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Görüntü Sınıflandırma</CardTitle>
                <CardDescription>
                  CNN ile görüntü sınıflandırma uygulaması
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Derin öğrenme kullanarak görüntüleri kategorilere ayıran model.
                </p>
              </CardContent>
              <CardFooter>
                <Button asChild variant="ghost" className="gap-1 ml-auto">
                  <Link href="#">
                    Projeyi İncele
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                </Button>
              </CardFooter>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Öneri Sistemi</CardTitle>
                <CardDescription>
                  Collaborative filtering tabanlı öneri sistemi
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Kullanıcı davranışlarına göre ürün önerileri yapan sistem.
                </p>
              </CardContent>
              <CardFooter>
                <Button asChild variant="ghost" className="gap-1 ml-auto">
                  <Link href="#">
                    Projeyi İncele
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                </Button>
              </CardFooter>
            </Card>
          </div>
        </div>
      </section>
    </div>
  );
}