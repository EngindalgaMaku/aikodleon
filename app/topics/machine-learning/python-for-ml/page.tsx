import { Metadata } from 'next';
import { createPageMetadata } from '@/lib/seo';
import Link from 'next/link';
import Image from 'next/image'; // Added Image
import {
  ArrowLeft, CheckCircle, Lightbulb, Code2, Library, Layers, BarChart3, BrainCircuit, PlayCircle, Settings, PackageCheck, TerminalSquare, Wand2, FolderGit2 // Added icons
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';

export const metadata: Metadata = createPageMetadata({
  title: 'Makine Öğrenmesi için Python Programlama',
  description: "Makine öğrenmesi projelerinizde Python'un gücünü keşfedin. Temel kütüphaneler (NumPy, Pandas, Scikit-learn) ve en iyi uygulamalarla donanın.",
  path: '/topics/machine-learning/python-for-ml',
  keywords: ['python makine öğrenmesi', 'python ml', 'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn', 'tensorflow', 'pytorch', 'makine öğrenmesi kütüphaneleri', 'kodleon', 'türkçe ai eğitimi', 'python veri bilimi'],
  imageUrl: 'https://images.pexels.com/photos/546819/pexels-photo-546819.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2' // Example image
});

const libraries = [
  {
    name: "NumPy",
    description: "Sayısal hesaplamalar için temel paket. Çok boyutlu diziler ve matrisler üzerinde hızlı işlemler sunar.",
    icon: <Layers className="w-8 h-8 text-blue-500" />,
    features: ["ndarray nesnesi", "Vektörize operasyonlar", "Lineer cebir, Fourier dönüşümü"]
  },
  {
    name: "Pandas",
    description: "Veri manipülasyonu ve analizi için güçlü araçlar. DataFrame ve Series yapılarıyla yapılandırılmış verilerle çalışmayı kolaylaştırır.",
    icon: <Library className="w-8 h-8 text-purple-500" />,
    features: ["Veri temizleme ve hazırlama", "CSV, Excel okuma/yazma", "Zaman serisi analizi"]
  },
  {
    name: "Matplotlib & Seaborn",
    description: "Kapsamlı veri görselleştirme kütüphaneleri. Matplotlib esneklik, Seaborn ise daha estetik ve istatistiksel grafikler sunar.",
    icon: <BarChart3 className="w-8 h-8 text-green-500" />,
    features: ["Çizgi, bar, histogram, dağılım grafikleri", "Grafik özelleştirme", "İstatistiksel görselleştirmeler"]
  },
  {
    name: "Scikit-learn",
    description: "Klasik makine öğrenmesi algoritmaları için endüstri standardı. Sınıflandırma, regresyon, kümeleme ve model değerlendirme araçları içerir.",
    icon: <BrainCircuit className="w-8 h-8 text-orange-500" />,
    features: ["Tutarlı API", "Ön işleme araçları", "Model seçimi ve değerlendirme"]
  },
  {
    name: "TensorFlow & PyTorch",
    description: "Derin öğrenme modelleri (sinir ağları) geliştirmek ve eğitmek için lider kütüphaneler. GPU desteği ile yüksek performans sunarlar.",
    icon: <Code2 className="w-8 h-8 text-red-500" />,
    features: ["Dinamik/Statik grafikler", "Otomatik türev alma", "Geniş topluluk ve kaynaklar"]
  }
];

const projectIdeas = [
  {
    title: "Iris Çiçeği Sınıflandırma",
    description: "Scikit-learn'ün ünlü Iris veri setini kullanarak farklı çiçek türlerini taç ve çanak yaprak ölçülerine göre sınıflandırın.",
    difficulty: "Başlangıç",
    tags: ["Sınıflandırma", "Scikit-learn", "Veri Analizi"]
  },
  {
    title: "Titanic Hayatta Kalma Tahmini",
    description: "Kaggle'ın Titanic veri setini kullanarak yolcuların özelliklerine (yaş, cinsiyet, sınıf vb.) göre hayatta kalma olasılıklarını tahmin edin.",
    difficulty: "Orta",
    tags: ["Sınıflandırma", "Pandas", "Özellik Mühendisliği"]
  },
  {
    title: "Basit Doğrusal Regresyon",
    description: "Kendi oluşturacağınız veya bulacağınız basit bir veri seti ile (örneğin, ev fiyatları ve metrekare) doğrusal regresyon modeli kurun.",
    difficulty: "Başlangıç",
    tags: ["Regresyon", "NumPy", "Matplotlib"]
  }
];

const pythonCodeExample = `import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Örnek veri (genellikle bir CSV dosyasından okunur)
data = {'Metrekare': [50, 75, 100, 120, 150, 180, 200],
        'OdaSayisi': [1, 1, 2, 2, 3, 3, 4],
        'Fiyat': [150000, 220000, 300000, 350000, 450000, 520000, 580000]}
df = pd.DataFrame(data)

# Bağımsız değişkenler (X) ve bağımlı değişken (y)
X = df[['Metrekare', 'OdaSayisi']]
y = df['Fiyat']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Model performansını değerlendirme (Ortalama Karesel Hata)
mse = mean_squared_error(y_test, y_pred)
print(f"Modelin Ortalama Karesel Hatası (MSE): {mse:.2f}")

# Yeni bir veri için tahmin yapma
yeni_ev = [[130, 3]]  # 130 metrekare, 3 oda
tahmini_fiyat = model.predict(yeni_ev)
print(f"130m², 3 odalı evin tahmini fiyatı: {tahmini_fiyat[0]:.2f} TL")`;

export default function PythonForMlPage() {
  const pageTitle = "Makine Öğrenmesi için Python Programlama";
  const heroImageUrl = "https://images.pexels.com/photos/546819/pexels-photo-546819.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"; // Coding image

  return (
    <div className="bg-background text-foreground">
      {/* Hero Section */}
      <section className="relative bg-gradient-to-b from-muted/30 to-background">
        <div className="absolute inset-0 opacity-10">
          <Image
            src={heroImageUrl}
            alt="Python Programlama Arka Planı"
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
                <Code2 className="h-12 w-12 text-primary" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-foreground mb-6">
              {pageTitle}
            </h1>
            <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
              Makine öğrenmesi projelerinizde Python'un gücünü keşfedin. Bu kapsamlı rehberle temel kütüphaneler, pratik örnekler ve en iyi uygulamalarla donanın.
            </p>
          </div>
        </div>
      </section>

      {/* Main Content Area */}
      <div className="container max-w-6xl mx-auto py-12 md:py-16 px-4 md:px-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 lg:gap-12">
          {/* Left Content Column */}
          <div className="lg:col-span-2 space-y-12">
            <section aria-labelledby="why-python-heading">
              <Card className="shadow-lg border-border hover:border-primary/30 transition-all">
                <CardHeader className="bg-muted/30">
                  <div className="flex items-center gap-3">
                    <Lightbulb className="h-7 w-7 text-yellow-500" />
                    <CardTitle id="why-python-heading" className="text-2xl font-semibold">Neden Makine Öğrenmesi için Python?</CardTitle>
                  </div>
                </CardHeader>
                <CardContent className="pt-6 text-base space-y-4">
                  <p>
                    Python, okunabilir sözdizimi, devasa kütüphane ekosistemi ve geniş topluluk desteği sayesinde makine öğrenmesi (ML) ve veri bilimi alanlarında fiili standart haline gelmiştir. Hızlı prototipleme ve kolay öğrenme eğrisi, hem yeni başlayanlar hem de deneyimli geliştiriciler için caziptir.
                  </p>
                  <ul className="list-disc list-inside space-y-2">
                    <li><strong>Basit ve Anlaşılır Sözdizimi:</strong> İngilizceye yakın yapısı sayesinde kod yazmak ve okumak kolaydır.</li>
                    <li><strong>Geniş Kütüphane Desteği:</strong> NumPy, Pandas, Scikit-learn gibi güçlü kütüphaneler ML süreçlerini hızlandırır.</li>
                    <li><strong>Büyük ve Aktif Topluluk:</strong> Karşılaşılan sorunlara çözüm bulmak ve yeni şeyler öğrenmek kolaydır.</li>
                    <li><strong>Esneklik ve Ölçeklenebilirlik:</strong> Küçük denemelerden büyük ölçekli üretim sistemlerine kadar kullanılabilir.</li>
                    <li><strong>Çoklu Platform Desteği:</strong> Windows, macOS ve Linux gibi farklı işletim sistemlerinde sorunsuz çalışır.</li>
                  </ul>
                </CardContent>
              </Card>
            </section>

            <section aria-labelledby="key-libraries-heading">
              <h2 id="key-libraries-heading" className="text-3xl font-bold text-foreground mb-8 border-b pb-3">Temel Kütüphaneler</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {libraries.map((lib) => (
                  <Card key={lib.name} className="shadow-md hover:shadow-xl transition-shadow border-border flex flex-col">
                    <CardHeader className="bg-muted/20">
                      <div className="flex items-center gap-3 mb-2">
                        {lib.icon}
                        <CardTitle className="text-xl font-semibold">{lib.name}</CardTitle>
                      </div>
                      <CardDescription>{lib.description}</CardDescription>
                    </CardHeader>
                    <CardContent className="pt-4 flex-grow">
                      <h4 className="font-semibold mb-2 text-sm text-muted-foreground">Öne Çıkan Özellikler:</h4>
                      <ul className="list-disc list-inside space-y-1 text-sm">
                        {lib.features.map((feature, i) => (
                          <li key={i}>{feature}</li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </section>

            <section aria-labelledby="code-example-heading">
              <h2 id="code-example-heading" className="text-3xl font-bold text-foreground mb-6 border-b pb-3">Pratik Bir Örnek: Basit Doğrusal Regresyon</h2>
              <p className="mb-4 text-muted-foreground">
                Aşağıda, Pandas ve Scikit-learn kütüphanelerini kullanarak basit bir doğrusal regresyon modelinin nasıl oluşturulup eğitileceğine dair bir Python kodu örneği bulunmaktadır. Bu örnek, temel veri hazırlama, model eğitimi ve tahmin adımlarını göstermektedir.
              </p>
              <div className="bg-muted/50 p-4 rounded-lg border border-border shadow-sm overflow-x-auto">
                <pre className="language-python !bg-transparent !p-0">
                  <code
                    className="text-sm"
                    dangerouslySetInnerHTML={{ __html: pythonCodeExample.replace(/</g, "&lt;").replace(/>/g, "&gt;") }}
                  />
                </pre>
              </div>
              <p className="mt-4 text-sm text-muted-foreground">
                <strong>Not:</strong> Bu kodu çalıştırmadan önce <code>pandas</code> ve <code>scikit-learn</code> kütüphanelerinin kurulu olduğundan emin olun (<code>pip install pandas scikit-learn</code>).
              </p>
            </section>

             <section aria-labelledby="project-ideas-heading">
              <h2 id="project-ideas-heading" className="text-3xl font-bold text-foreground mb-8 border-b pb-3">Pratik Proje Fikirleri</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {projectIdeas.map((project) => (
                  <Card key={project.title} className="shadow-md hover:shadow-lg transition-shadow border-border flex flex-col">
                    <CardHeader className="bg-muted/20">
                      <div className="flex items-center gap-3 mb-1">
                         <PlayCircle className="h-6 w-6 text-green-600" />
                        <CardTitle className="text-lg font-semibold">{project.title}</CardTitle>
                      </div>
                    </CardHeader>
                    <CardContent className="pt-3 flex-grow">
                      <p className="text-sm text-muted-foreground mb-3">{project.description}</p>
                      <p className="text-xs font-medium text-primary mb-2">Zorluk: {project.difficulty}</p>
                      <div className="flex flex-wrap gap-1.5">
                        {project.tags.map(tag => (
                          <span key={tag} className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded-full">{tag}</span>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </section>

          </div>

          {/* Right Sticky Sidebar */}
          <aside className="lg:col-span-1 space-y-8">
            <div className="bg-muted/50 rounded-lg p-6 sticky top-24 border border-border shadow-sm">
              <h3 className="text-xl font-semibold mb-6 text-foreground border-b pb-3 flex items-center">
                <Settings className="h-6 w-6 text-sky-500 mr-2.5" /> Kurulum ve Ortam Yönetimi
              </h3>
              <div className="space-y-4 text-sm">
                <p className="text-muted-foreground">
                  Python ve kütüphanelerini kurmanın en iyi yolu Anaconda veya Miniconda kullanmaktır. Sanal ortamlar (<code>conda create -n myenv python=3.9</code>) proje bağımlılıklarınızı izole eder.
                </p>
                <div>
                  <h4 className="font-semibold text-foreground mb-1.5 flex items-center"><PackageCheck className="h-4 w-4 text-green-500 mr-2" /> Kütüphane Kurulumu:</h4>
                  <p className="text-muted-foreground">
                    Genellikle <code>pip install kütüphane_adı</code> komutu kullanılır. Proje bağımlılıkları için <code>requirements.txt</code> dosyası oluşturup <code>pip install -r requirements.txt</code> ile toplu kurulum yapabilirsiniz.
                  </p>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground mb-1.5 flex items-center"><TerminalSquare className="h-4 w-4 text-gray-500 mr-2" /> Geliştirme Araçları:</h4>
                  <p className="text-muted-foreground">
                    Jupyter Notebook/Lab, VS Code (Python eklentisi ile) veya PyCharm gibi IDE'ler popüler seçeneklerdir.
                  </p>
                </div>
              </div>

              <Separator className="my-6" />

              <h3 className="text-xl font-semibold mb-6 text-foreground border-b pb-3 flex items-center">
                <Wand2 className="h-6 w-6 text-indigo-500 mr-2.5" /> İpuçları ve En İyi Uygulamalar
              </h3>
              <ul className="space-y-2.5 text-sm list-disc list-outside pl-5 text-muted-foreground">
                <li><strong>Okunabilir Kod Yazın:</strong> PEP 8 stil rehberini takip edin, açıklayıcı değişken ve fonksiyon isimleri kullanın.</li>
                <li><strong>Yorum Satırları Ekleyin:</strong> Karmaşık mantıkları ve önemli kararları açıklayan yorumlar ekleyin.</li>
                <li><strong>Fonksiyonlar ve Modüller Kullanın:</strong> Kodunuzu yeniden kullanılabilir fonksiyonlara ve modüllere ayırın.</li>
                <li><strong>Verimli Veri Yapıları Seçin:</strong> Probleminize uygun NumPy dizileri, Pandas DataFrame'leri veya Python list/dict'lerini kullanın.</li>
                <li><strong>Versiyon Kontrolü Kullanın:</strong> Projeleriniz için Git ve GitHub/GitLab gibi platformları kullanmayı alışkanlık haline getirin. <FolderGit2 className="inline h-4 w-4 ml-1" /></li>
                <li><strong>Sürekli Öğrenin:</strong> Yeni kütüphaneleri, teknikleri ve topluluk tartışmalarını takip edin.</li>
              </ul>
            </div>
          </aside>
        </div>
      </div>

      <section className="border-t border-border bg-muted/20 py-12 md:py-16">
        <div className="container max-w-4xl mx-auto px-4 md:px-6 text-center">
          <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-4">Python ile Makine Öğrenmesine Hazır mısınız?</h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Bu temel bilgilerle, artık makine öğrenmesi dünyasında Python'ı etkili bir şekilde kullanmaya başlayabilirsiniz. Daha fazla pratik yaparak ve projeler geliştirerek becerilerinizi bir sonraki seviyeye taşıyın.
          </p>
          <div className="flex flex-wrap justify-center items-center gap-4">
            <Button asChild size="lg" className="bg-primary hover:bg-primary/90 text-primary-foreground">
              <Link href="/topics/machine-learning/supervised-learning">
                <BrainCircuit className="mr-2 h-5 w-5" /> Denetimli Öğrenmeye Başla
              </Link>
            </Button>
            <Button asChild variant="outline" size="lg">
              <Link href="/topics">
                <Lightbulb className="mr-2 h-5 w-5" /> Tüm Yapay Zeka Konuları
              </Link>
            </Button>
          </div>
        </div>
      </section>

    </div>
  );
}