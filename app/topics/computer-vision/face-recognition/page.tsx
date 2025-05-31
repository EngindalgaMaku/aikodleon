import { Metadata } from 'next';
import Link from 'next/link';
import Image from 'next/image';
import { ArrowLeft, Users, UserCheck, UserX, LockKeyhole, ShieldQuestion, Camera, Building } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';

export const metadata: Metadata = {
  title: 'Yüz Tanıma Teknolojisi: Nasıl Çalışır, Uygulamaları ve Etik Boyutları | Kodleon',
  description: 'Yüz tanıma sistemlerinin temellerini, algoritmalarını (Eigenfaces, Fisherfaces, LBPH, Derin Öğrenme) ve güvenlikten pazarlamaya kadar geniş uygulama alanlarını keşfedin.',
  keywords: 'yüz tanıma, face recognition, eigenfaces, fisherfaces, lbph, derin öğrenme yüz tanıma, biyometri, bilgisayarlı görü, yapay zeka, güvenlik sistemleri, etik yapay zeka',
  alternates: {
    canonical: 'https://kodleon.com/topics/computer-vision/face-recognition',
  },
  openGraph: {
    title: 'Yüz Tanıma: Kimlik Doğrulamanın Ötesinde | Kodleon',
    description: 'Kişileri yüzlerinden tanıyabilen bu güçlü teknolojinin çalışma prensiplerini, yaygın kullanım senaryolarını ve beraberinde getirdiği etik tartışmaları inceleyin.',
    url: 'https://kodleon.com/topics/computer-vision/face-recognition',
    images: [
      {
        url: '/images/og/topics/computer-vision/face-recognition-og.png', // Bu görselin oluşturulması/var olması gerekli
        width: 1200,
        height: 630,
        alt: 'Kodleon Yüz Tanıma Teknolojisi Eğitimi'
      }
    ]
  }
};

const methodologies = [
  {
    name: "Geleneksel Yöntemler",
    approaches: [
      { name: "Eigenfaces (PCA)", description: "Yüz görüntülerini düşük boyutlu bir özellik uzayına indirgeyerek temel bileşenleri kullanır." },
      { name: "Fisherfaces (LDA)", description: "Sınıflar arası varyansı maksimize, sınıf içi varyansı minimize ederek daha iyi ayrımcılık sağlar." },
      { name: "Local Binary Patterns Histograms (LBPH)", description: "Yerel doku bilgisini kullanarak yüzleri temsil eder, aydınlatma değişimlerine karşı daha dirençlidir." }
    ]
  },
  {
    name: "Derin Öğrenme Tabanlı Yöntemler",
    approaches: [
      { name: "Evrişimli Sinir Ağları (CNN)", description: "Yüz özelliklerini otomatik olarak öğrenir ve yüksek doğrulukla tanıma yapar. DeepFace, FaceNet gibi mimariler bu yaklaşıma örnektir." },
      { name: "Siamese Networkler ve Triplet Loss", description: "Benzer yüzleri birbirine yakınlaştırıp farklı yüzleri uzaklaştırarak etkili bir özellik uzayı öğrenir." }
    ]
  }
];

const applications = [
  { title: "Güvenlik ve Erişim Kontrolü", description: "Binalara, cihazlara veya sistemlere yetkili kişilerin erişimini sağlamak.", icon: <LockKeyhole className="h-8 w-8 text-red-500" /> },
  { title: "Kolluk Kuvvetleri ve Gözetim", description: "Kalabalıklarda aranan kişilerin tespiti veya suç mahallerindeki kimlik belirleme.", icon: <Camera className="h-8 w-8 text-blue-700" /> },
  { title: "Mobil Cihaz Kilidi Açma", description: "Akıllı telefon ve tabletlerde kullanıcı kimlik doğrulaması.", icon: <UserCheck className="h-8 w-8 text-green-500" /> },
  { title: "Pazarlama ve Müşteri Deneyimi", description: "Müşteri demografisi analizi, kişiselleştirilmiş reklamcılık.", icon: <Building className="h-8 w-8 text-purple-500" /> },
  { title: "Sağlık Hizmetleri", description: "Hasta kimlik doğrulaması, genetik sendromların teşhisi.", icon: <UserX className="h-8 w-8 text-teal-500" /> }, // Icon changed to reflect a broader health context beyond just absence
  { title: "Sosyal Medya", description: "Fotoğraflardaki kişilerin otomatik olarak etiketlenmesi.", icon: <Users className="h-8 w-8 text-indigo-500" /> }
];

export default function FaceRecognitionPage() {
  return (
    <div className="bg-background text-foreground">
      <section className="relative py-16 md:py-24 bg-gradient-to-br from-indigo-500/5 via-transparent to-purple-500/5">
        <div className="container max-w-5xl mx-auto px-4 text-center">
          <div className="inline-flex items-center gap-2 mb-4">
            <Button asChild variant="ghost" size="sm" className="gap-1 text-primary hover:text-primary/80">
              <Link href="/topics/computer-vision">
                <ArrowLeft className="h-4 w-4" />
                Bilgisayarlı Görü Ana Sayfası
              </Link>
            </Button>
          </div>
          <Users className="h-16 w-16 text-primary mx-auto mb-6" />
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
            Yüz Tanıma Teknolojisi
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto">
            İnsan yüzü, kimliğimizin en belirgin özelliklerinden biridir. Yüz tanıma teknolojisi, bu benzersiz özellikleri kullanarak kişileri dijital ortamda tanımlama veya doğrulama yeteneği sunar. Güvenlikten sosyal medyaya kadar geniş bir yelpazede hayatımıza dokunmaktadır.
          </p>
        </div>
      </section>

      <section className="py-12 md:py-20">
        <div className="container max-w-4xl mx-auto px-4">
          <article className="prose prose-lg dark:prose-invert max-w-none">
            
            <h2>Yüz Tanıma Nedir?</h2>
            <p>
              Yüz tanıma, bir bireyi yüz özelliklerine dayanarak otomatik olarak tanımlayan veya doğrulayan bir biyometrik teknolojidir. Genellikle aşağıdaki adımları içerir:
            </p>
            <ol>
              <li><strong>Yüz Tespiti (Face Detection):</strong> Bir görüntü veya video karesi içinde insan yüzlerinin yerini bulma.</li>
              <li><strong>Yüz Hizalama (Face Alignment):</strong> Tespit edilen yüzü, poz ve ölçek farklılıklarını gidermek için standart bir konuma getirme (örneğin, gözler ve burun belirli noktalara gelecek şekilde döndürme ve ölçekleme).</li>
              <li><strong>Özellik Çıkarımı (Feature Extraction):</strong> Hizalanmış yüzden ayırt edici özellikleri (örneğin, gözler arasındaki mesafe, burun şekli, çene hattı) matematiksel bir vektör olarak çıkarma.</li>
              <li><strong>Eşleştirme (Matching):</strong> Çıkarılan özellik vektörünü, veri tabanındaki bilinen kişilere ait özellik vektörleriyle karşılaştırarak bir kimlik atama (tanımlama) veya iddia edilen kimliği doğrulama.
              </li>
            </ol>
            <p>
              Yüz tanıma sistemleri, 1:1 (doğrulama - verification) veya 1:N (tanımlama - identification) modunda çalışabilir. Doğrulamada, bir kişinin iddia ettiği kişi olup olmadığı kontrol edilirken (örneğin, telefon kilidini açma), tanımlamada ise bir yüzün veri tabanındaki hangi kişiye ait olduğu belirlenmeye çalışılır (örneğin, kalabalıkta bir şüphelinin aranması).
            </p>

            <Separator className="my-8" />

            <h2>Temel Metodolojiler</h2>
            <p>Yüz tanıma için geliştirilmiş başlıca metodolojiler şunlardır:</p>
            {methodologies.map(methodology => (
              <div key={methodology.name} className="mb-6">
                <h3 className="text-xl font-semibold mb-3">{methodology.name}</h3>
                <div className="space-y-4">
                  {methodology.approaches.map(approach => (
                    <Card key={approach.name} className="bg-secondary/30">
                      <CardHeader>
                        <CardTitle className="text-lg">{approach.name}</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-sm text-muted-foreground">{approach.description}</p>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            ))}
            
            <Separator className="my-8" />

            <h2>Kullanım Alanları</h2>
            <p>Yüz tanıma teknolojisi, çeşitli sektörlerde yaygın olarak kullanılmaktadır:</p>
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

            <h2>Etik Hususlar ve Zorluklar</h2>
            <p>Yüz tanıma teknolojisinin yaygınlaşması, önemli etik soruları ve zorlukları da beraberinde getirmektedir:</p>
            <ul className="list-disc pl-6 space-y-2">
              <li><strong>Mahremiyet:</strong> Bireylerin bilgisi veya rızası olmadan yüzlerinin taranması ve verilerinin saklanması ciddi mahremiyet endişeleri doğurur.</li>
              <li><strong>Önyargı ve Adalet:</strong> Yüz tanıma algoritmaları, eğitildikleri veri setlerindeki demografik dengesizlikler nedeniyle belirli etnik kökenlere, yaş gruplarına veya cinsiyetlere karşı daha düşük doğrulukla çalışabilir ve bu da haksız sonuçlara yol açabilir.</li>
              <li><strong>Yanlış Pozitif ve Yanlış Negatifler:</strong> Sistemlerin yanlış kişiyi teşhis etmesi (yanlış pozitif) veya doğru kişiyi tanıyamaması (yanlış negatif) ciddi sonuçlar doğurabilir.</li>
              <li><strong>Gözetim ve Kötüye Kullanım:</strong> Teknolojinin kitlesel gözetim veya baskıcı rejimler tarafından kötüye kullanılması potansiyeli bulunmaktadır.</li>
              <li><strong>Şeffaflık ve Hesap Verebilirlik:</strong> Bu sistemlerin nasıl karar verdiği ve hatalı durumlarda kimin sorumlu olacağı konularında şeffaflık eksikliği olabilir.</li>
              <li><strong>Veri Güvenliği:</strong> Toplanan yüz verilerinin güvenli bir şekilde saklanmaması, kimlik hırsızlığı gibi risklere yol açabilir.</li>
            </ul>
            <div className="mt-6 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg flex items-start">
              <ShieldQuestion className="h-8 w-8 text-yellow-600 mr-3 flex-shrink-0" />
              <p className="text-yellow-700 dark:text-yellow-300 text-sm">
                Yüz tanıma teknolojisinin kullanımı, potansiyel faydaları ile bireysel haklar ve toplumsal değerler arasında dikkatli bir denge kurulmasını gerektirir. Bu nedenle, etik kurallar, yasal düzenlemeler ve toplumsal tartışmalar büyük önem taşımaktadır.
              </p>
            </div>

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