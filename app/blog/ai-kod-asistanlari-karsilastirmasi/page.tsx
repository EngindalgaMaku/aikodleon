import { Metadata } from 'next';
import Link from 'next/link';
import Image from 'next/image';
import { ArrowLeft, ExternalLink, Lightbulb, CheckCircle, XCircle, TrendingUp, Zap, Users, Target, ShieldCheck, Cpu } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import ClientImage from '@/components/ui/client-image';

// Placeholder for createPageMetadata if not available or you define metadata directly
const createPageMetadata = (data: any) => data; 

export const metadata: Metadata = createPageMetadata({
  title: 'AI Kod Asistanları Arenası: Cursor vs. Windsurf (Codeium) ve Diğerleri',
  description: 'Cursor, Windsurf (eski adıyla Codeium) ve diğer popüler AI kod asistanlarını karşılaştırıyoruz. Hangi aracın sizin için en uygun olduğunu keşfedin, artılarını ve eksilerini öğrenin.',
  path: '/blog/ai-kod-asistanlari-karsilastirmasi',
  keywords: ['ai kod asistanı', 'cursor', 'windsurf', 'codeium', 'github copilot', 'kodlama araçları', 'yapay zeka yazılım geliştirme', 'geliştirici verimliliği', 'ai programlama'],
  openGraph: {
    title: 'AI Kod Asistanları Arenası: Cursor vs. Windsurf (Codeium) ve Diğerleri',
    description: 'Cursor, Windsurf (Codeium) ve diğer AI kod asistanlarının derinlemesine karşılaştırması.',
    url: 'https://kodleon.com/blog/ai-kod-asistanlari-karsilastirmasi',
    type: 'article',
    images: [
      {
        url: '/blog-images/code-assistants.jpg',
        width: 1260,
        height: 750,
        alt: 'AI Kod Asistanları Karşılaştırması Blog Görseli',
      },
    ],
  },
});

export default function AiCodingAssistantsBlogPostPage() {
  const pageTitle = "AI Kod Asistanları Arenası: Cursor vs. Windsurf (Codeium) ve Diğerleri";
  const ogImages = metadata.openGraph?.images;
  let heroImageUrl = 'https://images.pexels.com/photos/546819/pexels-photo-546819.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2'; // Default fallback

  if (ogImages) {
    const imageEntry = Array.isArray(ogImages) ? ogImages[0] : ogImages;
    if (imageEntry) {
      if (typeof imageEntry === 'string') {
        heroImageUrl = imageEntry;
      } else if (imageEntry instanceof URL) {
        heroImageUrl = imageEntry.href;
      } else { // Should be OgImageDescriptor
        if (typeof imageEntry.url === 'string') {
          heroImageUrl = imageEntry.url;
        } else if (imageEntry.url instanceof URL) {
          heroImageUrl = imageEntry.url.href;
        }
      }
    }
  }

  // Helper component for Pros and Cons
  const ProsCons = ({ title, items, type }: { title: string; items: string[]; type: 'pros' | 'cons' }) => (
    <div className="mb-4">
      <h4 className={`text-lg font-semibold mb-2 flex items-center ${type === 'pros' ? 'text-green-600' : 'text-red-600'}`}>
        {type === 'pros' ? <CheckCircle className="h-5 w-5 mr-2" /> : <XCircle className="h-5 w-5 mr-2" />}
        {title}
      </h4>
      <div className={`p-3 rounded-md ${type === 'pros' ? 'bg-green-500/10 border border-green-500/20' : 'bg-red-500/10 border border-red-500/20'}`}>
        <ul className="list-disc list-inside space-y-1 text-muted-foreground">
          {items.map((item, index) => <li key={index}>{item}</li>)}
        </ul>
      </div>
    </div>
  );

  return (
    <div className="bg-background text-foreground">
      {/* Hero Section */}
      <section className="relative bg-gradient-to-b from-muted/30 to-background">
        <div className="absolute inset-0 opacity-10">
          <Image
            src={heroImageUrl}
            alt="AI Kod Asistanları Blog Arka Planı"
            fill
            className="object-cover"
            priority
          />
        </div>
        <div className="container max-w-4xl mx-auto py-16 md:py-24 px-4 md:px-6 relative z-10">
          <div className="mb-8">
            <Button asChild variant="ghost" size="sm" className="gap-1.5 text-muted-foreground hover:text-primary">
              <Link href="/blog" aria-label="Tüm blog yazılarına geri dön">
                <ArrowLeft className="h-4 w-4" aria-hidden="true" />
                Tüm Yazılar
              </Link>
            </Button>
          </div>
          <div className="text-center">
            <div className="inline-block p-3 mb-6 bg-primary/10 rounded-full border border-primary/20">
                <Zap className="h-12 w-12 text-primary" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-foreground mb-6">
              {pageTitle}
            </h1>
            <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
              Yazılım geliştirme dünyası yapay zeka ile hızla dönüşüyor. Bu dönüşümün en önemli oyuncularından biri de AI kod asistanları. Peki, hangi araç size en uygun? Gelin, popüler seçenekleri mercek altına alalım.
            </p>
            <p className="text-sm text-muted-foreground mt-4">Yayınlanma Tarihi: 10 Mart 2025</p>
          </div>
        </div>
      </section>

      {/* Main Content Area */}
      <article className="container max-w-4xl mx-auto py-12 md:py-16 px-4 md:px-6 prose prose-lg dark:prose-invert">
        <h2 id="giris" className="text-3xl font-bold mb-4 text-primary">Giriş: Yazılım Geliştirmede Yeni Bir Çağ</h2>
        <p>
          Yapay zeka (AI), yazılım geliştirme süreçlerini kökten değiştiren bir güç olarak karşımıza çıkıyor. Artık AI sadece bir konsept değil, geliştiricilerin günlük iş akışlarının bir parçası haline geliyor. Bu devrimin ön saflarında ise AI kod asistanları yer alıyor. Bu akıllı araçlar, kod yazmaktan hata ayıklamaya, yeni teknolojileri öğrenmekten verimliliği artırmaya kadar birçok alanda geliştiricilere destek oluyor.
        </p>
        <p>
          Bu yazıda, piyasadaki popüler AI kod asistanlarından bazılarını, özellikle de <a href="https://cursor.sh" target="_blank" rel="noopener noreferrer">Cursor</a> ve <a href="https://windsurf.com" target="_blank" rel="noopener noreferrer">Windsurf</a> (eski adıyla Codeium) gibi dikkat çeken araçları derinlemesine inceleyeceğiz. Özelliklerini, artılarını, eksilerini ve hangi geliştirici profilleri için daha uygun olabileceklerini tartışacağız.
        </p>

        <Separator className="my-8" />

        <h2 id="ai-kod-asistani-nedir" className="text-3xl font-bold mb-4 text-primary">AI Kod Asistanı Nedir? Temel Özellikler</h2>
        <p>
          AI kod asistanları, geliştiricilere kodlama süreçlerinde yardımcı olmak için tasarlanmış yapay zeka destekli yazılımlardır. Genellikle büyük dil modelleri (LLM&apos;ler) üzerine kurulu olan bu araçlar, kod bağlamını anlayarak çeşitli görevleri otomatikleştirebilir ve hızlandırabilirler. İşte bu asistanların sunduğu bazı temel yetenekler:
        </p>
        <ul>
          <li><strong>Akıllı Kod Tamamlama:</strong> Sadece birkaç karakter yazdığınızda bile, bağlama uygun kod blokları veya fonksiyonlar önerirler.</li>
          <li><strong>Kod Üretme ve Düzenleme (Refactoring):</strong> Doğal dil komutlarıyla (örneğin, &quot;bu fonksiyona test yaz&quot; veya &quot;bu kodu optimize et&quot;) sıfırdan kod üretebilir veya mevcut kodu yeniden düzenleyebilirler.</li>
          <li><strong>Kod Tabanıyla Sohbet:</strong> Projenizdeki kodlar hakkında sorular sorabilir, belirli bir fonksiyonun ne işe yaradığını öğrenebilir veya karmaşık kod bloklarını açıklamasını isteyebilirsiniz.</li>
          <li><strong>Hata Ayıklama Desteği:</strong> Hatalı kodları tespit etmede ve olası çözümler sunmada yardımcı olabilirler.</li>
          <li><strong>Çoklu Dosya İşlemleri:</strong> Özellikle daha karmaşık görevlerde, birden fazla dosyayı etkileyen değişiklikler yapabilirler.</li>
          <li><strong>Dökümantasyon ve Öğrenme:</strong> Kod hakkında hızlıca dökümantasyon oluşturabilir veya yeni bir kütüphane/framework hakkında bilgi sağlayabilirler.</li>
        </ul>
        <Alert className="mb-6">
          <Lightbulb className="h-4 w-4" />
          <AlertTitle>Unutmayın!</AlertTitle>
          <AlertDescription>
            AI kod asistanları güçlü araçlar olsa da, ürettikleri kodu her zaman dikkatlice gözden geçirmek ve test etmek kritik öneme sahiptir. Onları bir &quot;yardımcı pilot&quot; olarak düşünmek en doğrusudur.
          </AlertDescription>
        </Alert>
        
        <Separator className="my-8" />

        <h2 id="cursor-incelemesi" className="text-3xl font-bold mb-4 text-primary">Derinlemesine Bakış: Cursor</h2>
        <div className="flex items-center mb-4">
          <ClientImage 
            src="/images/cursor.jpg" 
            fallbackSrc="https://via.placeholder.com/40?text=C"
            alt="Cursor Logo" 
            width={40} 
            height={40} 
            className="mr-3 rounded" 
          />
          <h3 className="text-2xl font-semibold m-0">Cursor: Güçlü ve Esnek Bir AI IDE</h3>
        </div>
        <p>
          Cursor, özellikle VS Code&apos;a aşina olan geliştiriciler için tasarlanmış, yapay zeka özellikleriyle donatılmış bir kod editörüdür. VS Code&apos;un bir &quot;fork&quot;u olması sayesinde, mevcut eklenti ekosisteminden ve kullanıcı arayüzünden faydalanırken, üzerine güçlü AI yetenekleri ekler.
        </p>
        <h4 className="font-semibold mt-6 mb-2">Öne Çıkan Özellikleri:</h4>
        <ul className="list-disc list-inside space-y-1 mb-4">
          <li><strong>Gelişmiş Bağlam Yönetimi:</strong> &quot;@dosyaadı&quot; veya &quot;@semboladı&quot; gibi komutlarla AI&apos;a projenizdeki belirli dosyalar veya kod parçacıkları hakkında kolayca bağlam sağlayabilirsiniz.</li>
          <li><strong>&quot;Agentic&quot; Mod:</strong> AI&apos;ın sadece kod önermekle kalmayıp, terminal komutları çalıştırabilmesi, dosya sistemi üzerinde değişiklikler yapabilmesi gibi daha otonom görevler üstlenebilmesini sağlar.</li>
          <li><strong>Kod Tabanında Akıllı Arama:</strong> Sadece metin tabanlı arama yerine, kodun anlamını anlayarak arama yapabilir ve ilgili bölümleri bulabilir.</li>
          <li><strong>VS Code Uyumluluğu:</strong> Mevcut VS Code temanızı, ayarlarınızı ve eklentilerinizi büyük ölçüde kullanmaya devam edebilirsiniz.</li>
          <li><strong>İleri Düzey Özellikler:</strong> Otomatik commit mesajı oluşturma, potansiyel bug&apos;ları tespit etme gibi &quot;power user&quot;lara yönelik özellikler sunar.</li>
        </ul>
        
        <ProsCons 
          title="Artıları"
          type="pros"
          items={[
            "Çok kapsamlı ve güçlü özellik seti.",
            "Genişletilmiş bağlam yönetimi sayesinde AI'ın daha isabetli öneriler sunması.",
            "VS Code kullanıcıları için neredeyse sıfır öğrenme eğrisi (arayüz ve temel işlevler açısından).",
            "Aktif geliştirme ve sık güncellemeler."
          ]}
        />
        <ProsCons 
          title="Eksileri"
          type="cons"
          items={[
            "Özellik yoğunluğu nedeniyle yeni başlayanlar için biraz karmaşık gelebilir.",
            "Bazı gelişmiş özellikleri (örneğin, 'Auto-debug') ek krediler veya daha üst abonelik gerektirebilir.",
            "Tamamen agentic modda bazen beklenmedik sonuçlar verebilir, dikkatli kullanım gerektirir."
          ]}
        />
        <Button asChild variant="outline" className="mt-4">
          <Link href="https://cursor.sh" target="_blank" rel="noopener noreferrer">
            Cursor Web Sitesi <ExternalLink className="h-4 w-4 ml-2" />
          </Link>
        </Button>

        <Separator className="my-8" />

        <h2 id="windsurf-incelemesi" className="text-3xl font-bold mb-4 text-primary">Derinlemesine Bakış: Windsurf (eski adıyla Codeium)</h2>
         <div className="flex items-center mb-4">
          <ClientImage 
            src="/images/windsurf.png" 
            fallbackSrc="https://via.placeholder.com/40?text=W" 
            alt="Windsurf (Codeium) Logo" 
            width={40} 
            height={40} 
            className="mr-3 rounded" 
          />
          <h3 className="text-2xl font-semibold m-0">Windsurf: Hızlı ve Kullanıcı Dostu</h3>
        </div>
        <p>
          Windsurf, eski adıyla Codeium, hızlı ve kullanıcı dostu bir AI kod asistanıdır. Özellikle hızlı kod tamamlama ve kod üretme konusunda güçlü özellikler sunar.
        </p>
        <h4 className="font-semibold mt-6 mb-2">Öne Çıkan Özellikleri:</h4>
        <ul className="list-disc list-inside space-y-1 mb-4">
          <li><strong>Hızlı Kod Tamamlama:</strong> Kullanıcı dostu arayüzü ile hızlı ve doğru kod tamamlama önerileri sunar.</li>
          <li><strong>Kod Üretme:</strong> Doğal dil komutlarıyla kod üretme yeteneği ile geliştiricilere yardımcı olur.</li>
          <li><strong>Kullanıcı Dostu Arayüz:</strong> Basit ve anlaşılır arayüzü ile yeni başlayanlar için idealdir.</li>
        </ul>
        <ProsCons 
          title="Artıları"
          type="pros"
          items={[
            "Hızlı ve kullanıcı dostu arayüz.",
            "Güçlü kod tamamlama ve üretme özellikleri.",
            "Yeni başlayanlar için ideal."
          ]}
        />
        <ProsCons 
          title="Eksileri"
          type="cons"
          items={[
            "Bazı gelişmiş özellikler eksik olabilir.",
            "Daha az özelleştirme seçeneği."
          ]}
        />
        <Button asChild variant="outline" className="mt-4">
          <Link href="https://windsurf.com" target="_blank" rel="noopener noreferrer">
            Windsurf Web Sitesi <ExternalLink className="h-4 w-4 ml-2" />
          </Link>
        </Button>
        
        <Separator className="my-8" />

        <h2 id="sonuc" className="text-3xl font-bold mb-4 text-primary">Sonuç</h2>
        <p>
          AI kod asistanları, yazılım geliştirme süreçlerini hızlandırmak ve verimliliği artırmak için güçlü araçlardır. Cursor ve Windsurf gibi araçlar, farklı geliştirici profillerine hitap eden özellikler sunar. Hangi aracın size en uygun olduğunu belirlemek için, ihtiyaçlarınızı ve kullanım alışkanlıklarınızı göz önünde bulundurmanız önemlidir.
        </p>
      </article>
      {/* Missing Images List:
          - Cursor Logo: /images/cursor.jpg
          - Windsurf Logo: /images/windsurf.png
      */}
       {/* Footer/Navigation back to blog list or homepage could be added here */}
       <section className="container max-w-3xl mx-auto py-8 px-4 md:px-6">
        <Separator />
        <div className="flex justify-between items-center mt-8">
            <Button asChild variant="outline">
                <Link href="/blog">
                    <ArrowLeft className="h-4 w-4 mr-2" /> Tüm Blog Yazılarına Dön
                </Link>
            </Button>
            <p className="text-sm text-muted-foreground">Okuduğunuz için teşekkürler!</p>
        </div>
      </section>
    </div>
  );
} 