import { Metadata } from 'next';
import Link from 'next/link';
import Image from 'next/image';
import { ArrowRight, BookOpen, Code, Sigma, FileText, HeartPulse, BookMarked, Router, Cpu, Factory, Building, Zap, Microscope } from 'lucide-react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export const metadata: Metadata = {
  title: 'Metasezgisel Optimizasyon Algoritmaları | Kodleon',
  description: 'Karmaşık optimizasyon problemlerini çözmek için doğadan esinlenen ve sezgisel yöntemler kullanan metasezgisel optimizasyon algoritmaları hakkında bilgi edinin.',
  keywords: 'metasezgisel optimizasyon, genetik algoritma, parçacık sürü optimizasyonu, karınca kolonisi optimizasyonu, yapay arı kolonisi, tavlama benzetimi',
};

const optimizationAlgorithms = [
  {
    title: "Genetik Algoritmalar",
    description: "Doğal seçilim ve genetik mekanizmaları taklit eden popülasyon tabanlı metasezgisel optimizasyon algoritması.",
    icon: <Sigma className="h-6 w-6 text-purple-500" />,
    href: "/topics/metasezgisel-optimizasyon/genetik-algoritmalar",
    category: "evolutionary",
    fields: ["Bilgisayar", "Mühendislik", "Finans"]
  },
  {
    title: "Parçacık Sürü Optimizasyonu",
    description: "Kuş ve balık sürülerinin sosyal davranışlarından esinlenen, sürü zekasına dayalı optimizasyon tekniği.",
    icon: <Sigma className="h-6 w-6 text-blue-500" />,
    href: "/topics/metasezgisel-optimizasyon/parcacik-suru-optimizasyonu",
    category: "swarm",
    fields: ["Elektrik", "Finans", "Mühendislik"]
  },
  {
    title: "Tavlama Benzetimi",
    description: "Metallerin tavlama işleminden esinlenen, yerel optimumlardan kaçmak için olasılıksal bir yaklaşım sunan algoritma.",
    icon: <Sigma className="h-6 w-6 text-red-500" />,
    href: "/topics/metasezgisel-optimizasyon/tavlama-benzetimi",
    category: "physics",
    fields: ["Üretim", "Lojistik", "Bilgisayar"]
  },
  {
    title: "Karınca Kolonisi Optimizasyonu",
    description: "Karıncaların yiyecek arama davranışlarından esinlenen, feromon izlerine dayalı optimizasyon algoritması.",
    icon: <Sigma className="h-6 w-6 text-amber-500" />,
    href: "/topics/metasezgisel-optimizasyon/karinca-kolonisi-optimizasyonu",
    category: "swarm",
    fields: ["Lojistik", "İletişim", "Bilgisayar"]
  },
  {
    title: "Yapay Arı Kolonisi Optimizasyonu",
    description: "Bal arılarının yiyecek arama davranışlarından esinlenen popülasyon tabanlı optimizasyon algoritması.",
    icon: <Sigma className="h-6 w-6 text-yellow-500" />,
    href: "/topics/metasezgisel-optimizasyon/yapay-ari-kolonisi-optimizasyonu",
    category: "swarm",
    fields: ["Sağlık", "Elektrik", "Bilgisayar"]
  },
  {
    title: "Yasaklı Arama",
    description: "Yerel arama algoritmalarını geliştiren, önceki çözümleri yasaklayarak döngüleri önleyen optimizasyon tekniği.",
    icon: <Sigma className="h-6 w-6 text-emerald-500" />,
    href: "/topics/metasezgisel-optimizasyon/yasakli-arama",
    category: "local-search",
    fields: ["Lojistik", "Finans", "Üretim"]
  },
  {
    title: "Diferansiyel Gelişim",
    description: "Popülasyon tabanlı, vektör farkına dayalı mutasyon operatörü kullanan evrimsel optimizasyon algoritması.",
    icon: <Sigma className="h-6 w-6 text-cyan-500" />,
    href: "/topics/metasezgisel-optimizasyon/diferansiyel-gelisim",
    category: "evolutionary",
    fields: ["Mühendislik", "Finans", "Enerji"]
  },
  {
    title: "Uyum Araması",
    description: "Müzisyenlerin doğaçlama yaparken estetik standartları karşılamak için birlikte çalıştığı süreçten esinlenen optimizasyon algoritması.",
    icon: <Sigma className="h-6 w-6 text-indigo-500" />,
    href: "/topics/metasezgisel-optimizasyon/uyum-aramasi",
    category: "other",
    fields: ["Bilgisayar", "Tasarım", "Mühendislik"]
  },
  {
    title: "Yarasa Algoritması",
    description: "Yarasaların ekolokasyon davranışlarından esinlenen, frekans ayarlı ekolokasyon kullanan optimizasyon tekniği.",
    icon: <Sigma className="h-6 w-6 text-violet-500" />,
    href: "/topics/metasezgisel-optimizasyon/yarasa-algoritmasi",
    category: "swarm",
    fields: ["Mühendislik", "İletişim", "Görüntü İşleme"]
  },
  {
    title: "Ateşböceği Algoritması",
    description: "Ateşböceklerinin parıldama davranışlarını taklit eden, çekiciliğe dayalı optimizasyon algoritması.",
    icon: <Sigma className="h-6 w-6 text-amber-400" />,
    href: "/topics/metasezgisel-optimizasyon/atesbocegi-algoritmasi",
    category: "swarm",
    fields: ["Görüntü İşleme", "Sinyal İşleme", "Elektrik"]
  },
  {
    title: "Guguk Kuşu Araması",
    description: "Guguk kuşlarının yumurtlama stratejisini ve yumurta tanıma mekanizmasını taklit eden optimizasyon algoritması.",
    icon: <Sigma className="h-6 w-6 text-teal-500" />,
    href: "/topics/metasezgisel-optimizasyon/guguk-kusu-aramasi",
    category: "other",
    fields: ["Bilgisayar", "Finans", "Mühendislik"]
  },
  {
    title: "Gri Kurt Optimizasyonu",
    description: "Gri kurtların hiyerarşik liderlik yapısını ve avlanma davranışlarını taklit eden optimizasyon tekniği.",
    icon: <Sigma className="h-6 w-6 text-gray-500" />,
    href: "/topics/metasezgisel-optimizasyon/gri-kurt-optimizasyonu",
    category: "swarm",
    fields: ["Elektrik", "Robotik", "Görüntü İşleme"]
  },
  {
    title: "Balina Optimizasyon Algoritması",
    description: "Kambur balinaların avlanma stratejisini, özellikle baloncuk ağı besleme davranışını taklit eden optimizasyon algoritması.",
    icon: <Sigma className="h-6 w-6 text-blue-400" />,
    href: "/topics/metasezgisel-optimizasyon/balina-optimizasyon-algoritmasi",
    category: "swarm",
    fields: ["Mühendislik", "Elektrik", "Veri Madenciliği"]
  }
];

const applicationCategories = [
  { id: "all", name: "Tümü" },
  { id: "swarm", name: "Sürü Zekası" },
  { id: "evolutionary", name: "Evrimsel Algoritmalar" },
  { id: "physics", name: "Fizik Tabanlı" },
  { id: "local-search", name: "Yerel Arama" },
  { id: "other", name: "Diğer" }
];

const applicationFields = [
  { 
    title: "Bilgisayar/Teknoloji", 
    icon: <Code className="h-8 w-8" />,
    items: [
      "Bilgisayar paket sıralamalarına bağlantıların atanması (ACO)",
      "Ağ ve bilgisayar güvenliği ve saldırı tespiti (AIS)",
      "Kablosuz ağ tasarımı optimizasyonu (ABC)",
      "Robot tasarımı (YSA)",
      "Görüntü tanıma ve sıkıştırma (TS)",
      "Anti-virüs programı (AIS)"
    ]
  },
  { 
    title: "Sağlık/Tıp", 
    icon: <HeartPulse className="h-8 w-8" />,
    items: [
      "Biyoenformatik (ACO)",
      "Kanser teşhisi (YSA)",
      "DNA kod dizilimi (GA, ACO)",
      "Protein 3d yapısının belirlenmesi (ACO)",
      "DNA tox dizilimi (GA, ACO)"
    ]
  },
  { 
    title: "Eğitim/Bilim", 
    icon: <BookMarked className="h-8 w-8" />,
    items: [
      "Tam öğretme problemi (GA)",
      "Çift çenekli bitki damar deseni modelleme (GA)",
      "Grafik renklendirme (ACO)",
      "Karakter teşhisi (TS)",
      "Astronomi ve astrofizik (GA)",
      "Karşılaştırmalı tasarım (AIS)"
    ]
  },
  { 
    title: "Lojistik/Finans", 
    icon: <Router className="h-8 w-8" />,
    items: [
      "Lojistik araç rotalama ve yönlendirme (SA, GA, TS)",
      "Okul otobüsü yönlendirme (GA)",
      "Koli yükleme (ACO, GA)",
      "Hisse senedi tahmini (GA)",
      "Ekonometrik tahmin (GA)",
      "Havayolu filosu çizelgeleme (SA)",
      "Kaynak tahsisi (SA)",
      "Karayak tahsisi (SA)",
      "Havayolu personel planlaması (GA)"
    ]
  },
  { 
    title: "Elektrik/Elektronik", 
    icon: <Cpu className="h-8 w-8" />,
    items: [
      "Elektrik sistemlerinde yük dengelenmesi (ABC)",
      "Dijital filtre optimizasyonu (ABC)",
      "Elektrik yalıtışçı dağılımı (TS, GA, PSO)",
      "Sinyal işleme (PSO)",
      "Elektrik sistemlerinde yük dengelemesi (ABC)",
      "Güç sistemlerinde arıza tespiti (AIS)",
      "Bijudik devreler için bütçe yönlendirme (TS)",
      "Elektrik yüklenişçı dağılımı (TS, GA, PSO)"
    ]
  },
  { 
    title: "Üretim", 
    icon: <Factory className="h-8 w-8" />,
    items: [
      "Parti büyüklüğü belirleme (SA)",
      "Seri üretim (akış tipi) çizelgelemesi (ACO, SA, TS)",
      "Kutu paketleme (GA, ACO)",
      "Darbüğaz belirleme (SA)"
    ]
  },
  { 
    title: "Makine/Enerji/Sistemler", 
    icon: <Zap className="h-8 w-8" />,
    items: [
      "Rulman makinelerinin otomatik testi (AIS)",
      "Akış çizelgeleme problemi (SA)",
      "Rotor parçalarında hata tespiti (GA)",
      "Bakim planlaması (AIS)",
      "Güç sistemlerinde arıza tespiti (AIS)"
    ]
  },
  { 
    title: "İnşaat/Mühendislik", 
    icon: <Building className="h-8 w-8" />,
    items: [
      "Basınç Yükü Altındaki Boru Şekilli Kolonun Maliyet Optimizasyonu (BA)",
      "Konsel Kirişin Ağırlık Optimizasyonu (FA)",
      "Üç Elemanlı Kafes Sisteminin Ağırlık ve Maliyet Minimizasyonu (DE)",
      "Beş Elemanlı Kafes Sistemin Toplam Boy Minimizasyonu (PSO)",
      "Tek Eksenli Eğilme Etkisi Altındaki Dikdörtgen Kesitli Betonarme Kolonların Maliyet Minimizasyonu (HS)"
    ]
  }
];

// Bilim insanlarından alıntı yapılan optimizasyon tanımları
const optimizationDefinitions = [
  {
    author: "Edgar ve diğ. (2001)",
    definition: "Optimizasyon, amacı performans ölçütünün en iyi değerini veren süreçteki değişkenlerin değerlerini elde etmek olan süreçtir."
  },
  {
    author: "Onwubolu ve Babu (2004)",
    definition: "Optimizasyon, istenen sonuçlara ulaşmak için farklı olası çözümler arasından en iyi çözümün nasıl elde edileceğine yönelik olan iyileştirme arayışıdır."
  },
  {
    author: "Mikki ve Kishk (2008)",
    definition: "Optimizasyon, önceden belirlenmiş bir hedefe ulaşmayı amaçlayan sürecin sistematik değişimi, modifikasyonu ve uyarlanması olarak tanımlanabilir."
  },
  {
    author: "Vrahatis (2010)",
    definition: "Optimizasyon, bir problem için alternatifler arasındaki en uygun çözümlerin tespiti ile ilgilenen bilimsel bir disiplindir."
  },
  {
    author: "Parkinson ve diğ. (2013)",
    definition: "En iyi tasarımı belirleme süreci optimizasyon olarak adlandırılmaktadır."
  },
  {
    author: "Kırmızıgül ve diğ. (2014)",
    definition: "Optimizasyon, sayıların, fonksiyonların veya sistemlerin aşırı yanı minimum ve maksimum değerlerinin bulunmasıyla ilgilenen matematiksel bir disiplin olduğu belirtmiştir."
  },
  {
    author: "Rao (2014)",
    definition: "Optimizasyon terimi, verilen koşullar altında veya girdi değerleri ile en iyi sonucu/çıktıyı elde etmeyi ifade ettiğini söylemiştir."
  }
];

export default function MetaheuristicOptimizationPage() {
  return (
    <div className="bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800 min-h-screen pb-16">
      <div className="container mx-auto py-10 px-4 sm:px-6 lg:px-8 max-w-6xl">
        <div className="mb-12 text-center">
          <h1 className="text-4xl font-extrabold tracking-tight mb-4 text-gradient bg-clip-text text-transparent bg-gradient-to-r from-purple-600 to-indigo-600">
            Metasezgisel Optimizasyon Algoritmaları
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Karmaşık optimizasyon problemlerini çözmek için doğadan esinlenen ve sezgisel yöntemler kullanan
            algoritmalar hakkında detaylı bilgiler ve uygulama alanları.
          </p>
        </div>

        <div className="mb-16 bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
          <div className="p-6 sm:p-10">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
              <div>
                <h2 className="text-2xl font-bold mb-4">Optimizasyon Nedir?</h2>
                <p className="mb-4 text-muted-foreground">
                  Optimizasyon, bir problem, olay veya duruma ait sonuca ulaşırken en uygun çözüm yollarının belirlenmesi veya bu gibi öğelere yönelik en iyi performansın sergilenmesi amacıyla gerçekleştirilen süreçtir.
                </p>
                <p className="mb-4 text-muted-foreground">
                  Başka bir açıdan bakılacak olursa optimizasyon, pratik olarak gerekli görülen başarı veya beklenen faydadan ötürü, önceden belirlenmiş karar değişkenlerinin bir fonksiyonu olarak ifade edilebileceğinden, bir fonksiyonun maksimum veya minimum değerini veren koşulların bulunması süreci olarak tanımlanabilir.
                </p>
                <p className="mb-6 text-muted-foreground">
                  Klasik optimizasyon problemleri matematiksel işlemlerle çözümlenebilmektedir. Ancak gerçek hayat problemlerinin bu gibi matematiksel fonksiyonlarla ifade edilmesi pek mümkün olmamaktadır. Bu nedenle metasezgisel algoritmalar geliştirilmiştir.
                </p>
                <div className="flex flex-wrap gap-2 mb-6">
                  <Badge variant="outline" className="text-sm bg-purple-50 dark:bg-purple-900/30">Matematiksel Optimizasyon</Badge>
                  <Badge variant="outline" className="text-sm bg-blue-50 dark:bg-blue-900/30">Doğadan İlham Alan Algoritmalar</Badge>
                  <Badge variant="outline" className="text-sm bg-green-50 dark:bg-green-900/30">Sezgisel Yöntemler</Badge>
                  <Badge variant="outline" className="text-sm bg-amber-50 dark:bg-amber-900/30">Kompleks Problemler</Badge>
                </div>
                <Button asChild className="mt-4">
                  <Link href="#algorithms">
                    Algoritmaları Keşfet <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </div>
              <div className="relative h-64 lg:h-80 overflow-hidden rounded-xl">
                <Image 
                  src="/images/metasezgisel_algoritm.jpg"
                  alt="Metasezgisel Optimizasyon"
                  fill
                  className="object-cover"
                />
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-4">
                  <p className="text-white text-sm font-medium">Şekil: f(x) fonksiyonunun minimum değeri ile -f(x) fonksiyonunun maksimum değerinin eşitliği</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Bilim İnsanlarının Tanımları */}
        <div className="mb-16">
          <h2 className="text-3xl font-bold mb-8 text-center">Literatürde Optimizasyon Tanımları</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {optimizationDefinitions.map((def, idx) => (
              <Card key={idx} className="bg-white dark:bg-gray-800">
                <CardContent className="pt-6">
                  <blockquote className="border-l-4 border-purple-500 pl-4 italic text-muted-foreground">
                    "{def.definition}"
                  </blockquote>
                  <p className="text-right mt-4 text-sm font-semibold">- {def.author}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        <div className="mb-16">
          <h2 className="text-3xl font-bold mb-6 text-center">Metasezgisel Optimizasyonun Özellikleri</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card className="bg-white dark:bg-gray-800">
              <CardHeader>
                <CardTitle>Doğadan İlham Alma</CardTitle>
              </CardHeader>
              <CardContent>
                <p>Metasezgisel algoritmalar genellikle doğadaki süreçlerden, hayvan davranışlarından veya fiziksel olaylardan ilham alır. Bu, algoritmaların sezgisel yaklaşımlar geliştirmesine olanak tanır.</p>
              </CardContent>
            </Card>
            <Card className="bg-white dark:bg-gray-800">
              <CardHeader>
                <CardTitle>Stokastik Yapı</CardTitle>
              </CardHeader>
              <CardContent>
                <p>Bu algoritmaların rastgeleliğe dayalı bir yapısı vardır. Bu özellik, algoritmanın yerel optimumlara takılmadan global optimuma yaklaşmasına yardımcı olur.</p>
              </CardContent>
            </Card>
            <Card className="bg-white dark:bg-gray-800">
              <CardHeader>
                <CardTitle>Parametre Ayarı</CardTitle>
              </CardHeader>
              <CardContent>
                <p>Algoritmanın performansını etkileyen çeşitli parametreler vardır. Bu parametrelerin doğru ayarlanması, algoritmanın etkinliğini büyük ölçüde artırabilir.</p>
              </CardContent>
            </Card>
            <Card className="bg-white dark:bg-gray-800">
              <CardHeader>
                <CardTitle>Keşif ve Sömürü Dengesi</CardTitle>
              </CardHeader>
              <CardContent>
                <p>Tüm metasezgisel algoritmaların ortak özelliği, çözüm uzayının keşfi (exploration) ile bulunan iyi çözümlerin sömürüsü (exploitation) arasında denge kurmasıdır.</p>
              </CardContent>
            </Card>
            <Card className="bg-white dark:bg-gray-800">
              <CardHeader>
                <CardTitle>Problem Bağımsızlık</CardTitle>
              </CardHeader>
              <CardContent>
                <p>Metasezgisel algoritmalar, spesifik problem türlerinden bağımsız olarak tasarlanmıştır. Bu sayede çok çeşitli optimizasyon problemlerine uygulanabilirler.</p>
              </CardContent>
            </Card>
            <Card className="bg-white dark:bg-gray-800">
              <CardHeader>
                <CardTitle>Karmaşık Alanlarda Etkinlik</CardTitle>
              </CardHeader>
              <CardContent>
                <p>Klasik yöntemlerin zorlandığı, yüksek boyutlu, çok modlu veya türevi alınamayan fonksiyonların optimizasyonunda başarılı sonuçlar verirler.</p>
              </CardContent>
            </Card>
          </div>
        </div>

        <div id="algorithms" className="mb-16">
          <h2 className="text-3xl font-bold mb-6 text-center">Metasezgisel Optimizasyon Algoritmaları</h2>
          
          <Tabs defaultValue="all" className="mb-8">
            <div className="flex justify-center mb-6">
              <TabsList>
                {applicationCategories.map((category) => (
                  <TabsTrigger value={category.id} key={category.id}>
                    {category.name}
                  </TabsTrigger>
                ))}
              </TabsList>
            </div>
            
            {applicationCategories.map((category) => (
              <TabsContent value={category.id} key={category.id} className="mt-0">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {optimizationAlgorithms
                    .filter((algo) => category.id === "all" || algo.category === category.id)
                    .map((algorithm, idx) => (
                      <Card key={idx} className="bg-white dark:bg-gray-800 hover:shadow-lg transition-shadow">
                        <CardHeader>
                          <div className="flex items-center gap-3">
                            {algorithm.icon}
                            <CardTitle className="text-xl">{algorithm.title}</CardTitle>
                          </div>
                        </CardHeader>
                        <CardContent>
                          <CardDescription className="text-base">
                            {algorithm.description}
                          </CardDescription>
                          <div className="flex flex-wrap gap-2 mt-4">
                            {algorithm.fields.map((field, fidx) => (
                              <Badge key={fidx} variant="secondary" className="text-xs">
                                {field}
                              </Badge>
                            ))}
                          </div>
                        </CardContent>
                        <CardFooter>
                          <Button asChild variant="ghost" className="w-full">
                            <Link href={algorithm.href}>
                              Detaylı Bilgi <ArrowRight className="ml-2 h-4 w-4" />
                            </Link>
                          </Button>
                        </CardFooter>
                      </Card>
                    ))}
                </div>
              </TabsContent>
            ))}
          </Tabs>
        </div>

        <div className="mb-16">
          <h2 className="text-3xl font-bold mb-8 text-center">Uygulama Alanları</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {applicationFields.map((field, idx) => (
              <Card key={idx} className="bg-white dark:bg-gray-800 h-full">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    {field.icon}
                    <CardTitle>{field.title}</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2 list-disc list-inside text-sm text-muted-foreground">
                    {field.items.map((item, iidx) => (
                      <li key={iidx}>{item}</li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Optimizasyon teorisinden açıklama */}
        <div className="mb-16 bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
          <div className="p-6 sm:p-10">
            <h2 className="text-2xl font-bold mb-6 text-center">Optimizasyon Teorisi</h2>
            <p className="mb-4">
              Bu açıdan bir bilim alanı olarak optimizasyon genellikle matematiksel programlama olarak adlandırılmakta ve bu matematiksel programlama modeline ait problemlerin genel formülizasyonu Denklem (1.1)'deki gibi olmak üzere, problemin amacı fonksiyonunu belirtmekte olup M=1 olduğu koşulda optimizasyon problemi tek hedefli veya çok kriterli optimizasyon olarak adlandırılmaktadır.
            </p>
            <p className="mb-6">
              Ancak birçok gerçek hayat sorununun, çeşitli matematiksel ifadelerle somutlaştırılarak modellenmesi ve klasik yöntemlerle çözümlenebilmesi, problemlerin bazı özelliklerinden dolayı optimizasyon sürecinde çeşitli zorluklar oluşmasına neden olmakta ve bu durum nedeniyle de teknolojinin ilerlemesi ile klasik yöntemler yerini daha üstün/akıllı olarak kabul edilen sezgisel ve hatta metasezgisel algoritmalara bırakmış olup problemlerin daha hızlı, daha ekonomik ve en önemlisi en uygun şekilde çözümlenebilmesi sağlanmış olmaktadır.
            </p>
          </div>
        </div>

        <div className="mt-12 text-center">
          <p className="text-sm text-muted-foreground mb-4">
            Metasezgisel optimizasyon algoritmalarıyla ilgili daha fazla bilgi edinmek ve uygulama örneklerini incelemek için algoritma sayfalarını ziyaret edebilirsiniz.
          </p>
          <Button asChild variant="outline">
            <Link href="/topics">
              <ArrowRight className="mr-2 h-4 w-4 rotate-180" /> Tüm Konular
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 