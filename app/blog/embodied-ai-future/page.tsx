import { Metadata } from 'next';
import Link from 'next/link';
import Image from 'next/image';
import { ArrowLeft, Bot, Brain, Cpu, Eye, Globe, Lightbulb, AlertTriangle, CheckCircle, ExternalLink, ListChecks, Construction, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

export const metadata: Metadata = {
  title: 'Ekranın Ötesinde: Cisimleşmiş Yapay Zeka Dünyamızı Anlamayı ve Şekillendirmeyi Nasıl Öğreniyor? | Kodleon Blog',
  description: 'Cisimleşmiş Yapay Zeka, dünya modelleri ve yapay zekanın fiziksel dünyayla etkileşim kurmak ve anlamak için dijital alemlerin ötesine nasıl geçtiğini keşfedin. Uygulamaları, zorlukları ve akıllı robotiğin geleceğini keşfedin.',
  keywords: 'Cisimleşmiş Yapay Zeka, Dünya Modelleri, Robotik, Yapay Zeka, AI Trendleri, Fiziksel AI, İnsan-AI İşbirliği, AI Geleceği, Fiziksel Dünyada AI, Kodleon',
  openGraph: {
    title: 'Ekranın Ötesinde: Cisimleşmiş Yapay Zeka Dünyamızı Anlamayı ve Şekillendirmeyi Nasıl Öğreniyor?',
    description: 'Cisimleşmiş Yapay Zekanın büyüleyici dünyasına dalın ve makinelerin fiziksel gerçekliğimizle nasıl etkileşim kurduğunu öğrenin.',
    url: 'https://kodleon.com/blog/embodied-ai-future',
    type: 'article',
    images: [
      {
        url: 'https://images.pexels.com/photos/7661169/pexels-photo-7661169.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2', // İlgili bir görselle değiştirin
        width: 1260,
        height: 750,
        alt: 'Fiziksel dünyayla etkileşen robot, Cisimleşmiş Yapay Zekayı temsil ediyor',
      },
    ],
  },
};

const SectionTitle = ({ icon, title }: { icon: React.ReactNode, title: string }) => (
  <div className="flex items-center space-x-3 mb-6">
    {icon}
    <h2 className="text-3xl font-bold tracking-tight text-foreground">{title}</h2>
  </div>
);

const BulletPoint = ({ icon, children }: { icon: React.ReactNode, children: React.ReactNode }) => (
  <li className="flex items-start space-x-3 mb-3">
    <div className="flex-shrink-0 mt-1 text-primary">{icon}</div>
    <span className="text-muted-foreground">{children}</span>
  </li>
);

export default function CisimlesmisYapayZekaBlogPostPage() {
  return (
    <div className="bg-background text-foreground">
      {/* Hero Section */}
      <section className="relative py-20 md:py-32 overflow-hidden bg-gradient-to-b from-primary/5 via-transparent to-background">
        <div 
          className="absolute inset-0 opacity-10"
          style={{
            backgroundImage: 'url("data:image/svg+xml,%3Csvg width="80" height="80" viewBox="0 0 80 80" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="%239C92AC" fill-opacity="0.4"%3E%3Cpath fill-rule="evenodd" d="M0 0h40v40H0V0zm40 40h40v40H40V40zm0-40h20v20H40V0zm20 20h20v20H60V20zM0 60h20v20H0V60zm20-20h20v20H20V40z"/%3E%3C/g%3E%3C/svg%3E")',
          }}
          aria-hidden="true"
        />
        <div className="container max-w-4xl mx-auto relative z-10 px-4 text-center">
          <div className="inline-block p-4 mb-6 bg-primary/10 rounded-full border border-primary/20">
            <Bot className="h-12 w-12 text-primary" />
          </div>
          <h1 
            className="text-4xl md:text-5xl font-bold tracking-tight mb-6 
                       bg-clip-text text-transparent bg-gradient-to-r from-primary via-pink-500 to-orange-500"
          >
            Ekranın Ötesinde: Cisimleşmiş Yapay Zeka Dünyamızı Anlamayı ve Şekillendirmeyi Nasıl Öğreniyor?
          </h1>
          <p className="text-xl md:text-2xl text-muted-foreground mb-8">
            Yapay Zeka, dijital alemden çıkıp fiziksel gerçekliğimize adım atıyor. Cisimleşmiş Yapay Zekanın yükselişini, dünya modellerinin gücünü ve vaat ettikleri geleceği keşfedin.
          </p>
          <p className="text-sm text-muted-foreground">
            Yayınlanma Tarihi: {new Date().toLocaleDateString('tr-TR', { year: 'numeric', month: 'long', day: 'numeric' })}
          </p>
        </div>
      </section>

      {/* Main Content */}
      <article className="container max-w-4xl mx-auto py-12 md:py-16 px-4 md:px-6 prose prose-lg dark:prose-invert">
        <div className="mb-8">
          <Button variant="outline" asChild>
            <Link href="/blog" className="flex items-center space-x-2">
              <ArrowLeft className="h-4 w-4" />
              <span>Tüm Blog Yazılarına Dön</span>
            </Link>
          </Button>
        </div>

        <SectionTitle icon={<Globe className="h-8 w-8 text-primary" />} title="Giriş: Yapay Zeka İçin Bir Sonraki Sınır" />
        <p>
          On yıllardır Yapay Zeka, büyük ölçüde ekranlarımızın arkasında var oldu; karmaşık hesaplamalarda ustalaştı, anlayışlı metinler üretti ve çarpıcı görseller yarattı. Ancak yeni bir sınır ortaya çıkıyor: etrafımızdaki fiziksel dünyayı algılayabilen, anlayabilen ve onunla etkileşime girebilen Yapay Zeka. Bu, robotiğe ve akıllı makinelerle nasıl işbirliği yapacağımıza kadar her şeyi kökten değiştirmeye hazır dönüştürücü bir alan olan <strong>Cisimleşmiş Yapay Zeka</strong> alemidir.
        </p>
        <p>
          Bu evrimin kalbinde, Yapay Zekanın karmaşık gerçekliğimizin dinamiklerini kavramasını sağlayan sofistike iç temsiller olan "dünya modellerinin" geliştirilmesi yatmaktadır. Bu sadece verileri işlemekle ilgili değil; akıllı eylemi mümkün kılan temel bir anlayış oluşturmakla ilgilidir. Bu sıçrama neden bu kadar önemli? İnsan benzeri bir zarafetle öngörülemeyen ortamlarda gezinebilen robotları veya sadece kelimelerimizi değil fiziksel bağlamımızı da anlayan, görevlerde fiziksel olarak yardımcı olabilen Yapay Zeka asistanlarını hayal edin.
        </p>
        
        <figure className="my-8">
          <Image 
            src="https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2" // Daha spesifik Cisimleşmiş AI görseliyle değiştirin
            alt="Fiziksel nesnelerle etkileşen fütüristik robot kolu"
            width={1260}
            height={750}
            className="rounded-lg shadow-xl aspect-video object-cover"
          />
          <figcaption className="text-center text-sm text-muted-foreground mt-2">Cisimleşmiş Yapay Zeka, dijital zeka ile fiziksel etkileşim arasındaki boşluğu kapatmayı amaçlar.</figcaption>
        </figure>

        <SectionTitle icon={<Bot className="h-8 w-8 text-primary" />} title="Cisimleşmiş Yapay Zeka Nedir?" />
        <p>
          <strong>Cisimleşmiş Yapay Zeka</strong>, çevreleriyle doğrudan etkileşim yoluyla öğrenen ve kararlar alan akıllı ajanları (genellikle robotlar, aynı zamanda simüle edilmiş fiziksel dünyalardaki sanal ajanlar) ifade eder. Pasif olarak beslenen verileri işleyen geleneksel Yapay Zeka sistemlerinin aksine, Cisimleşmiş Yapay Zeka ajanları şunlara sahiptir:
        </p>
        <ul>
          <BulletPoint icon={<Eye className="h-5 w-5" />}>
            <strong>Algılama:</strong> Çevreleri hakkında bilgi toplamak için sensörler (kameralar, LiDAR, dokunsal sensörler vb.) kullanırlar.
          </BulletPoint>
          <BulletPoint icon={<Sparkles className="h-5 w-5" />}>
            <strong>Eylem:</strong> Fiziksel eylemler gerçekleştirmelerine ve nesneleri manipüle etmelerine olanak tanıyan aktüatörlere veya efektörlere (motorlar, tutucular, tekerlekler) sahiptirler.
          </BulletPoint>
          <BulletPoint icon={<Brain className="h-5 w-5" />}>
            <strong>Etkileşim Yoluyla Öğrenme:</strong> Çevredeki eylemlerinin sonuçlarını gözlemleyerek anlayışlarını ve becerilerini geliştirirler.
          </BulletPoint>
        </ul>
        <p>
          Bunu, yüzme hakkında bir kitap okumakla suya girerek yüzmeyi öğrenmek arasındaki fark gibi düşünün. Cisimleşmiş Yapay Zeka "suya girer", fiziği, belirsizlikleri ve gerçek dünyanın zenginliğini deneyimler.
        </p>

        <SectionTitle icon={<Cpu className="h-8 w-8 text-primary" />} title="Dünya Modellerinin Yükselişi" />
        <p>
          Cisimleşmiş Yapay Zeka alanındaki ilerlemenin temel katalizörlerinden biri <strong>dünya modelleri</strong> kavramıdır. Bunlar, dünyanın nasıl davrandığına dair öğrenilmiş, içsel temsillerdir. Bir dünya modeliyle donatılmış bir Yapay Zeka, yalnızca anlık duyusal girdilere tepki vermekle kalmaz, şunları yapabilir:
        </p>
        <ul>
          <BulletPoint icon={<Lightbulb className="h-5 w-5" />}>
            <strong>Gelecekteki Durumları Tahmin Etme:</strong> Mevcut koşullara ve potansiyel eylemlere dayanarak bundan sonra ne olabileceğini tahmin edin (örneğin, "Bu bloğu itersem, muhtemelen kayacaktır").
          </BulletPoint>
          <BulletPoint icon={<ListChecks className="h-5 w-5" />}>
            <strong>Karmaşık Görevleri Planlama:</strong> Potansiyel sonuçları göz önünde bulundurarak bir hedefe ulaşmak için eylem dizileri formüle edin.
          </BulletPoint>
          <BulletPoint icon={<CheckCircle className="h-5 w-5" />}>
            <strong>Neden ve Sonuç İlişkisini Anlama:</strong> Çevresindeki etkileşimleri yöneten temel ilkeleri öğrenin.
          </BulletPoint>
        </ul>
        <p>
          Meta'nın Baş Yapay Zeka Bilimcisi Yann LeCun gibi öncü araştırmacılar, dünya modellerinin daha insan benzeri zekaya veya Yapay Genel Zekaya (AGI) ulaşmak için çok önemli olduğunu uzun zamandır savunmaktadır. Google DeepMind'ın tek bir görüntüden etkileşimli 2D platform oyunu dünyaları üretebilen Genie gibi projeleri, Yapay Zekanın dinamik ortamları simüle etmeyi ve anlamayı öğrenmedeki gelişen yeteneklerini sergiliyor. Cisimleşmiş Yapay Zeka için sağlam dünya modelleri, daha az kırılgan, daha uyarlanabilir ve yeni durumlarda etkili bir şekilde çalışabilen robotlar anlamına gelir.
        </p>

        <Alert className="my-8 bg-primary/5 border-primary/20">
          <Brain className="h-5 w-5 text-primary" />
          <AlertTitle className="font-semibold text-primary">Moravec Paradoksu ve Cisimleşmiş Öğrenme</AlertTitle>
          <AlertDescription className="text-muted-foreground">
            İlginç bir şekilde, insanların kolay bulduğu görevler (algı ve motor kontrol gibi) Yapay Zeka için inanılmaz derecede zordur, karmaşık akıl yürütme (satranç gibi) ise nispeten daha kolaydır - bu Moravec Paradoksu olarak bilinir. Cisimleşmiş Yapay Zeka, Yapay Zekayı fiziksel etkileşim yoluyla öğrenmeye zorlayarak, algı ve motor becerilerinin "zor" sorunlarını doğrudan ele alır ve Yapay Zekanın neler başarabileceğinin sınırlarını zorlar.
          </AlertDescription>
        </Alert>

        <SectionTitle icon={<Construction className="h-8 w-8 text-primary" />} title="Cisimleşmiş Yapay Zeka Nasıl Öğrenir: Sanal ve Gerçek Arasındaki Köprü" />
        <p>
          Bir Yapay Zekayı fiziksel dünyanın karmaşıklıklarında gezinmek üzere eğitmek önemli bir zorluktur. Araştırmacılar birkaç strateji kullanır:
        </p>
        <ul>
          <BulletPoint icon={<Cpu className="h-5 w-5" />}>
            <strong>Simülasyon (Sim):</strong> Yapay Zeka ajanlarının güvenli ve hızlı bir şekilde öğrenebileceği ayrıntılı sanal ortamlar oluşturma. Milyonlarca etkileşim, gerçek dünyadan çok daha hızlı ve ucuza simüle edilebilir.
          </BulletPoint>
          <BulletPoint icon={<Globe className="h-5 w-5" />}>
            <strong>Gerçek Dünya Etkileşimi (Gerçek):</strong> Doğrudan deneyimden öğrenmek için Yapay Zeka ajanlarını fiziksel ortamlara dağıtma. Bu, simülasyonlarda mükemmel bir şekilde kopyalanamayan nüansları ve öngörülemeyen unsurları yakalamak için çok önemlidir.
          </BulletPoint>
          <BulletPoint icon={<ExternalLink className="h-5 w-5" />}>
            <strong>Simülasyondan Gerçeğe Aktarım (Sim-to-Real Transfer):</strong> Simülasyonda öğrenilen bilgi ve becerilerin gerçek dünya senaryolarına etkili bir şekilde aktarılmasını ve uygulanmasını sağlamak için teknikler geliştirme. Bu, "gerçeklik boşluğunu" kapatmayı amaçlayan önemli bir araştırma alanıdır.
          </BulletPoint>
        </ul>
        <p>
          Amaç erdemli bir döngü oluşturmaktır: gerçek dünya deneylerinden elde edilen bilgiler simülasyonları iyileştirir ve daha sağlam simülasyonlar gerçek dünya öğrenimini hızlandırır.
        </p>

        <SectionTitle icon={<Sparkles className="h-8 w-8 text-primary" />} title="Uygulamalar ve Potansiyel Etki" />
        <p>
          Olgunlaşmış Cisimleşmiş Yapay Zekanın potansiyel uygulamaları geniş ve dönüştürücüdür:
        </p>
        <div className="grid md:grid-cols-2 gap-6 my-6">
          {[
            { title: "Gelişmiş Robotik", description: "Üretim (örneğin karmaşık montaj), lojistik (örneğin otonom depolar), sağlık hizmetleri (cerrahi yardım, hasta bakımı) ve hatta ev işleri için son derece yetenekli robotlar.", icon: <Bot className="h-6 w-6 text-green-500" /> },
            { title: "İnsan-AI İşbirliği", description: "Fiziksel bağlamı anlayabilen, fiziksel görevlerde yardımcı olabilen ve insan gösterilerinden öğrenebilen, işyerlerini daha güvenli ve verimli hale getiren AI 'iş arkadaşları'.", icon: <Users className="h-6 w-6 text-blue-500" /> },
            { title: "Bilimsel Keşif", description: "Malzeme bilimi ve ilaç keşfi gibi alanlarda araştırmayı hızlandıran, otonom olarak fiziksel deneyler tasarlayabilen ve yürütebilen AI sistemleri.", icon: <Lightbulb className="h-6 w-6 text-yellow-500" /> },
            { title: "Gelişmiş Oyun ve Sanal Gerçeklik", description: "AI karakterlerinin gerçek fiziksel anlayışla davrandığı inanılmaz derecede gerçekçi ve etkileşimli sanal dünyalar yaratma.", icon: <Gamepad2 className="h-6 w-6 text-purple-500" /> },
            { title: "Erişilebilirlik Çözümleri", description: "Engelli bireyler için daha fazla bağımsızlık ve yaşam kalitesi sunan sofistike yardımcı teknolojiler.", icon: <Accessibility className="h-6 w-6 text-red-500" /> },
            { title: "Afet Müdahalesi", description: "Hayatta kalanları aramak veya kritik görevleri yerine getirmek için tehlikeli ve yapılandırılmamış ortamlarda gezinebilen robotlar.", icon: <FlameKindling className="h-6 w-6 text-orange-500" /> }
          ].map(item => (
            <Card key={item.title} className="bg-card border-border hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-center space-x-3 mb-2">
                  {item.icon}
                  <CardTitle className="text-lg">{item.title}</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">{item.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        <SectionTitle icon={<AlertTriangle className="h-8 w-8 text-destructive" />} title="Zorluklar ve Önümüzdeki Yol" />
        <p>
          Heyecan verici ilerlemeye rağmen, yaygın ve sofistike Cisimleşmiş Yapay Zekaya giden yol önemli zorluklarla doludur:
        </p>
        <ul>
          <BulletPoint icon={<Cpu className="h-5 w-5" />}>
            <strong>Donanım Sınırlamaları:</strong> Yeterince gelişmiş, sağlam ve uygun fiyatlı sensörler, aktüatörler ve yerleşik hesaplama gücü geliştirmek bir engel olmaya devam ediyor.
          </BulletPoint>
          <BulletPoint icon={<CheckCircle className="h-5 w-5" />}>
            <strong>Güvenlik ve Güvenilirlik:</strong> Bu sistemlerin karmaşık, dinamik insan ortamlarında güvenli ve öngörülebilir bir şekilde çalışmasını sağlamak her şeyden önemlidir.
          </BulletPoint>
          <BulletPoint icon={<Globe className="h-5 w-5" />}>
            <strong>Etik Hususlar:</strong> Gelişmiş robotik nedeniyle potansiyel işten çıkarmalar, fiziksel dünyadaki Yapay Zeka eylemlerinden hesap verebilirlik, veri gizliliği (bu sistemler çevremizi algıladığından) ve kötüye kullanımı önleme gibi konuların ele alınması.
          </BulletPoint>
          <BulletPoint icon={<ListChecks className="h-5 w-5" />}>
            <strong>Veri Kıtlığı ve Çeşitliliği:</strong> Sağlam eğitim için gereken büyük miktarda çeşitli, yüksek kaliteli gerçek dünya etkileşim verisini toplamak zor ve pahalıdır.
          </BulletPoint>
           <BulletPoint icon={<Construction className="h-5 w-5" />}>
            <strong>Gerçek Dünyanın Karmaşıklığı:</strong> Gerçek dünya senaryolarının salt öngörülemezliği ve zenginliği, Yapay Zekanın etkili bir şekilde genelleme yapmasını ve uyum sağlamasını inanılmaz derecede zorlaştırır.
          </BulletPoint>
        </ul>

        <Alert variant="destructive" className="my-8">
          <AlertTriangle className="h-5 w-5" />
          <AlertTitle className="font-semibold">Sorumlu Geliştirmeye Dair Bir Not</AlertTitle>
          <AlertDescription>
            Cisimleşmiş Yapay Zeka yetenekleri arttıkça, araştırma topluluğunun, endüstrinin ve politika yapıcıların bu güçlü teknolojinin insanlığa sorumlu bir şekilde fayda sağlamasını sağlamak için güçlü etik yönergeler ve güvenlik standartları oluşturmak üzere birlikte çalışması çok önemlidir.
          </AlertDescription>
        </Alert>

        <SectionTitle icon={<Sparkles className="h-8 w-8 text-primary" />} title="Sonuç: Cisimleşmiş Bir Gelecek" />
        <p>
          Cisimleşmiş Yapay Zeka, yapay zekanın yörüngesinde anıtsal bir değişimi temsil ediyor - dijital soyutlamadan fiziksel anlayışa ve etkileşime bir geçiş. Yolculuk karmaşık ve uzun vadeli olsa da, dünya modelleri, simülasyon ve robotikteki gelişmeler ilerlemeyi benzeri görülmemiş bir hızda hızlandırıyor.
        </p>
        <p>
          Fiziksel dünyamızda sadece düşünmekle kalmayıp aynı zamanda *yapabilen* Yapay Zekanın vaadi çok büyük. Endüstrileri dönüştürmekten günlük hayatımızı ancak hayal etmeye başladığımız şekillerde geliştirmeye kadar, Cisimleşmiş Yapay Zeka teknolojiyle ve belki de gerçekliğin kendisiyle ilişkimizi yeniden tanımlamaya hazırlanıyor. Gelecek sadece akıllı değil; cisimleşmiş.
        </p>

        <hr className="my-12" />

        <div className="text-center">
          <p className="text-lg text-muted-foreground mb-4">Cisimleşmiş Yapay Zeka ve Yapay Zekanın geleceği hakkında daha fazla bilgi edinmek ister misiniz?</p>
          <Button asChild size="lg">
            <Link href="/topics">AI Konularını Keşfedin</Link>
          </Button>
        </div>
      </article>

      {/* Footer navigation for blog - can be enhanced */}
      <section className="py-12 bg-muted/30">
        <div className="container max-w-4xl mx-auto px-4 text-center">
          <h3 className="text-2xl font-semibold mb-6">Diğer Yazılarımıza Göz Atın</h3>
          <div className="flex justify-center space-x-4">
            {/* Example: Link to the previous blog post if its path is known */}
            <Button variant="outline" asChild>
              <Link href="/blog/ai-kod-asistanlari-karsilastirmasi">AI Kod Asistanları Karşılaştırması</Link>
            </Button>
            <Button variant="outline" asChild>
              <Link href="/blog">Tüm Blog Yazıları</Link>
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
}

// Helper components for icons not directly available in lucide-react or for specific styling.
// (If you use any custom icons, define them here or import them. For now, using common ones from lucide-react)

const Users = (props: React.SVGProps<SVGSVGElement>) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
    <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>
    <circle cx="9" cy="7" r="4"/>
    <path d="M22 21v-2a4 4 0 0 0-3-3.87"/>
    <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
  </svg>
);

const Gamepad2 = (props: React.SVGProps<SVGSVGElement>) => (
 <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
    <line x1="6" y1="11" x2="10" y2="11"/>
    <line x1="8" y1="9" x2="8" y2="13"/>
    <line x1="15" y1="12" x2="15.01" y2="12"/>
    <line x1="18" y1="10" x2="18.01" y2="10"/>
    <path d="M17.32 5H6.68a4 4 0 0 0-3.978 3.59c-.006.052-.01.101-.01.152v0a4.831 4.831 0 0 0 4.491 4.818A4.5 4.5 0 0 0 11 16h2a4.5 4.5 0 0 0 4.002-2.44A4.831 4.831 0 0 0 21.31 8.743c0-.05-.004-.1-.01-.152A4 4 0 0 0 17.32 5Z"/>
 </svg>
);

const Accessibility = (props: React.SVGProps<SVGSVGElement>) => (
 <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
    <circle cx="16" cy="4" r="1"/>
    <path d="m18 19 1-7-6 1"/>
    <path d="m5 8 3-3 5.5 3-2.36 3.5"/>
    <path d="M4.24 14.5a5 5 0 0 0 6.88 6"/>
    <path d="M13.76 17.5a5 5 0 0 0-6.88-6"/>
 </svg>
);

const FlameKindling = (props: React.SVGProps<SVGSVGElement>) => (
 <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
    <path d="M12 22V18"/>
    <path d="M12 18H7.96a2 2 0 0 1-1.928-2.514l1.39-4.352A2 2 0 0 1 9.352 10H12"/>
    <path d="M12 18h4.04a2 2 0 0 0 1.928-2.514l-1.39-4.352A2 2 0 0 0 14.648 10H12"/>
    <path d="M4 15h16"/>
    <path d="M2 11h20"/>
    <path d="M17.5 11C17.5 9 16 6 16 6L12 2 8 6C8 6 6.5 9 6.5 11"/>
    <path d="M10.5 6h3"/>
 </svg>
); 