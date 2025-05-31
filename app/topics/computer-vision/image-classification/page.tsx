import { Metadata } from 'next';
import Link from 'next/link';
import Image from 'next/image';
import { ArrowLeft, Brain, Lightbulb, BarChart3, AlertTriangle, Zap, Microscope, Car, ImageIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';

export const metadata: Metadata = {
  title: 'Görüntü Sınıflandırma Nedir ve Nasıl Çalışır? | Kodleon CV Dersleri',
  description: 'Görüntü sınıflandırmanın temellerini, popüler algoritmaları (CNN, ResNet, VGG) ve Python ile pratik uygulama alanlarını Kodleon CV Platformunda öğrenin.',
  keywords: 'görüntü sınıflandırma, cnn, evrişimli sinir ağları, resnet, vgg, imagenet, bilgisayarlı görü, yapay zeka, makine öğrenmesi, görüntü tanıma, python görüntü işleme',
  alternates: {
    canonical: 'https://kodleon.com/topics/computer-vision/image-classification',
  },
  openGraph: {
    title: 'Görüntü Sınıflandırma: Derinlemesine Bakış | Kodleon',
    description: 'Bir görüntünün hangi sınıfa ait olduğunu belirleme sanatı olan görüntü sınıflandırmayı, temel adımlarını, popüler modellerini ve uygulama alanlarını keşfedin.',
    url: 'https://kodleon.com/topics/computer-vision/image-classification',
    images: [
      {
        url: '/images/og/topics/computer-vision/image-classification-og.png', // Bu görselin oluşturulması/var olması gerekli
        width: 1200,
        height: 630,
        alt: 'Kodleon Görüntü Sınıflandırma Eğitimi'
      }
    ]
  }
};

const algorithms = [
  { name: "LeNet-5", description: "İlk başarılı CNN yapılarından biri, özellikle el yazısı rakam tanıma için geliştirildi.", year: "1998" },
  { name: "AlexNet", description: "ImageNet yarışmasında büyük başarı göstererek derin öğrenmeye olan ilgiyi patlattı.", year: "2012" },
  { name: "VGGNet", description: "Daha derin ve daha küçük filtreler kullanarak performansı artırdı.", year: "2014" },
  { name: "GoogLeNet (Inception)", description: "Hesaplama verimliliğini artıran \"Inception\" modüllerini tanıttı.", year: "2014" },
  { name: "ResNet", description: "Çok derin ağların eğitilmesini sağlayan \"residual learning\" (artık öğrenme) kavramını getirdi.", year: "2015" },
  { name: "DenseNet", description: "Her katmanı diğer tüm katmanlara bağlayarak özellik yayılımını güçlendirdi.", year: "2017" },
  { name: "EfficientNet", description: "Model ölçeklendirme (derinlik, genişlik, çözünürlük) için dengeli bir yaklaşım sundu.", year: "2019" },
  { name: "Vision Transformer (ViT)", description: "NLP'deki Transformer mimarisini görüntü sınıflandırmaya uyarladı.", year: "2020" }
];

const applications = [
  { title: "Tıbbi Görüntü Analizi", description: "Röntgen, MR gibi tıbbi görüntülerden hastalıkların (örn: kanserli hücre tespiti) sınıflandırılması.", icon: <Microscope className="h-8 w-8 text-blue-500" /> },
  { title: "Otonom Araçlar", description: "Çevredeki nesnelerin (yaya, araç, trafik işareti) tanınması ve sınıflandırılması.", icon: <Car className="h-8 w-8 text-green-500" /> },
  { title: "İçerik Tabanlı Görüntü Alma", description: "Bir sorgu görüntüsüne benzer görüntülerin büyük veri tabanlarından bulunması.", icon: <ImageIcon className="h-8 w-8 text-purple-500" /> },
  { title: "Kalite Kontrol", description: "Üretim hatlarında hatalı ürünlerin otomatik olarak tespit edilmesi ve sınıflandırılması.", icon: <Zap className="h-8 w-8 text-yellow-500" /> },
  { title: "Tarım", description: "Bitki hastalıklarının tespiti, ürün verimliliğinin sınıflandırılması.", icon: <Lightbulb className="h-8 w-8 text-teal-500" /> },
  { title: "Güvenlik ve Gözetim", description: "Belirli nesnelerin veya olayların (örn: terk edilmiş paket) tespiti ve sınıflandırılması.", icon: <AlertTriangle className="h-8 w-8 text-red-500" /> }
];

export default function ImageClassificationPage() {
  return (
    <div className="bg-background text-foreground">
      {/* Hero Section */}
      <section className="relative py-16 md:py-24 bg-gradient-to-br from-primary/10 via-transparent to-secondary/10">
        <div className="container max-w-5xl mx-auto px-4 text-center">
          <div className="inline-flex items-center gap-2 mb-4">
            <Button asChild variant="ghost" size="sm" className="gap-1 text-primary hover:text-primary/80">
              <Link href="/topics/computer-vision">
                <ArrowLeft className="h-4 w-4" />
                Bilgisayarlı Görü Ana Sayfası
              </Link>
            </Button>
          </div>
          <Brain className="h-16 w-16 text-primary mx-auto mb-6" />
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
            Görüntü Sınıflandırma
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto">
            Bir görüntüye baktığımızda ne olduğunu anında anlayabiliriz. Peki makineler bunu nasıl başarıyor? Görüntü sınıflandırma, bir görüntüye uygun bir etiket (veya sınıf) atama görevidir. Bu, bilgisayarlı görünün en temel ve önemli görevlerinden biridir.
          </p>
        </div>
      </section>

      {/* Main Content Section */}
      <section className="py-12 md:py-20">
        <div className="container max-w-4xl mx-auto px-4">
          <article className="prose prose-lg dark:prose-invert max-w-none">
            
            <h2>Görüntü Sınıflandırma Nedir?</h2>
            <p>
              Görüntü sınıflandırma, bir girdiyi (görüntü) alır ve bu görüntüde hangi nesnenin veya sahnenin bulunduğunu belirleyerek bir çıktı (sınıf etiketi) üretir. Örneğin, bir kedi resmi verildiğinde, modelin "kedi" etiketini doğru bir şekilde tahmin etmesi beklenir. Bu görev, genellikle önceden tanımlanmış bir dizi kategori arasından seçim yapmayı içerir.
            </p>
            <p>
              Temelde, modelin bir görüntünün piksellerindeki desenleri öğrenmesi ve bu desenleri farklı sınıflarla ilişkilendirmesi gerekir. Bu, özellikle Evrişimli Sinir Ağları (CNN'ler) gibi derin öğrenme modelleri sayesinde son yıllarda büyük bir başarıyla gerçekleştirilmektedir.
            </p>

            <Separator className="my-8" />

            <h2>Temel Adımlar</h2>
            <p>Bir görüntü sınıflandırma sistemi geliştirmek genellikle aşağıdaki adımları içerir:</p>
            <ol>
              <li><strong>Veri Seti Toplama ve Hazırlama:</strong> Her sınıfa ait çok sayıda etiketli görüntüden oluşan bir veri seti toplanır. Bu veri seti eğitim, doğrulama ve test kümelerine ayrılır.</li>
              <li><strong>Veri Ön İşleme:</strong> Görüntüler, modelin daha iyi öğrenebilmesi için normalleştirme, yeniden boyutlandırma, veri artırma (data augmentation) gibi çeşitli ön işleme adımlarından geçirilir.</li>
              <li><strong>Model Seçimi ve Mimarisi:</strong> Göreve uygun bir model mimarisi seçilir. Genellikle önceden eğitilmiş (pre-trained) bir CNN modeli kullanılır ve bu model, hedef veri setine göre ince ayar (fine-tuning) yapılır.</li>
              <li><strong>Model Eğitimi:</strong> Seçilen model, eğitim veri seti kullanılarak eğitilir. Bu süreçte modelin ağırlıkları, kayıp fonksiyonunu (loss function) minimize edecek şekilde ayarlanır.</li>
              <li><strong>Model Değerlendirme:</strong> Eğitilen modelin performansı, daha önce görmediği test veri seti üzerinde çeşitli metrikler (doğruluk, kesinlik, duyarlılık, F1 skoru vb.) kullanılarak değerlendirilir.</li>
              <li><strong>Model Dağıtımı ve Kullanımı:</strong> Başarılı bulunan model, gerçek dünya uygulamalarında kullanılmak üzere dağıtılır.</li>
            </ol>

            <Separator className="my-8" />

            <h2>Popüler Algoritmalar ve Mimariler</h2>
            <p>Görüntü sınıflandırma alanında devrim yaratan birçok derin öğrenme mimarisi bulunmaktadır. İşte en bilinenlerden bazıları:</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-6">
              {algorithms.map(algo => (
                <Card key={algo.name} className="bg-secondary/30">
                  <CardHeader>
                    <CardTitle className="text-xl flex justify-between items-center">
                      {algo.name} 
                      <span className="text-sm font-normal text-muted-foreground bg-primary/10 px-2 py-0.5 rounded">
                        {algo.year}
                      </span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">{algo.description}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
            <p>
              Bu mimarilerin çoğu, ImageNet gibi büyük ölçekli veri setleri üzerinde eğitilmiş ve olağanüstü sonuçlar elde etmiştir. Transfer öğrenme (transfer learning) yaklaşımı sayesinde, bu önceden eğitilmiş modeller, daha küçük veri setleriyle bile farklı görevlere kolayca uyarlanabilir.
            </p>
            
            <Separator className="my-8" />

            <h2>Kullanım Alanları</h2>
            <p>Görüntü sınıflandırma, sayısız endüstri ve uygulamada kritik bir rol oynamaktadır:</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 my-6">
              {applications.map(app => (
                <Card key={app.title} className="flex flex-col items-center text-center p-6 hover:shadow-lg transition-shadow">
                   <div className="p-3 bg-primary/10 rounded-full mb-3">
                    {app.icon}
                  </div>
                  <CardTitle className="text-lg mb-2">{app.title}</CardTitle>
                  <CardContent className="text-sm text-muted-foreground p-0">
                    {app.description}
                  </CardContent>
                </Card>
              ))}
            </div>

            <Separator className="my-8" />

            <h2>Zorluklar ve Dikkat Edilmesi Gerekenler</h2>
            <p>Görüntü sınıflandırma güçlü bir teknoloji olmasına rağmen, bazı zorlukları da beraberinde getirir:</p>
            <ul>
              <li><strong>Veri İhtiyacı:</strong> Yüksek performanslı modeller genellikle büyük miktarda etiketli veri gerektirir.</li>
              <li><strong>Sınıflar Arası Dengesizlik:</strong> Bazı sınıfların diğerlerinden çok daha fazla örneğe sahip olması modelin performansını olumsuz etkileyebilir.</li>
              <li><strong>Görsel Çeşitlilik:</strong> Aynı nesnenin farklı açılardan, ışık koşullarında veya arka planlarda görünmesi modelin öğrenmesini zorlaştırabilir.</li>
              <li><strong>Hesaplama Kaynakları:</strong> Derin öğrenme modellerinin eğitimi ve dağıtımı önemli miktarda hesaplama kaynağı gerektirebilir.</li>
              <li><strong>Açıklanabilirlik:</strong> Özellikle kritik uygulamalarda, modelin neden belirli bir karar verdiğini anlamak (Explainable AI - XAI) önemlidir.</li>
            </ul>

            <Separator className="my-8" />

            <h2>Geleceği</h2>
            <p>
              Görüntü sınıflandırma alanı sürekli olarak gelişmektedir. Daha az veriyle daha iyi öğrenen modeller (few-shot learning, zero-shot learning), daha verimli ve daha küçük mimariler, ve daha iyi açıklanabilirlik yöntemleri üzerine aktif araştırmalar devam etmektedir. Vision Transformer (ViT) gibi yeni yaklaşımlar, CNN'lerin ötesine geçerek alana taze bir soluk getirmiştir. Gelecekte, görüntü sınıflandırmanın daha da karmaşık görevleri daha yüksek doğrulukla yerine getirebildiğini göreceğiz.
            </p>

          </article>

          <div className="mt-12 text-center">
            <Button asChild>
              <Link href="/topics/computer-vision">
                <ArrowLeft className="mr-2 h-4 w-4" /> Bilgisayarlı Görü Ana Konusuna Dön
              </Link>
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
} 