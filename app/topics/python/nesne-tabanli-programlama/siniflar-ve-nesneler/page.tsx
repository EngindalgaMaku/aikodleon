import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import Image from "next/image";
import { ArrowRight, Code2, Settings, Terminal, BookOpen, GraduationCap } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python Sınıflar ve Nesneler | AIKOD',
  description: 'Python\'da sınıf ve nesne kavramları, constructor, instance metodları, self parametresi ve daha fazlası.',
};

const content = `
# Python'da Sınıflar ve Nesneler

Nesne tabanlı programlamanın temel yapı taşları olan sınıflar ve nesneler, kodunuzu daha organize ve yeniden kullanılabilir hale getirmenizi sağlar.

## Sınıf Nedir?

Sınıf (class), nesnelerin özelliklerini ve davranışlarını tanımlayan bir şablondur. Örneğin, bir "Araba" sınıfı şu özelliklere sahip olabilir:

\`\`\`python
class Araba:
    def __init__(self, marka, model, yil):
        self.marka = marka
        self.model = model
        self.yil = yil
        self.hiz = 0
    
    def hizlan(self, artis):
        self.hiz += artis
    
    def yavasla(self, azalis):
        self.hiz = max(0, self.hiz - azalis)
    
    def bilgi_goster(self):
        return f"{self.yil} {self.marka} {self.model}, Hız: {self.hiz} km/s"
\`\`\`

## Nesne (Object) Nedir?

Nesne, bir sınıfın örneğidir. Yani sınıf şablonundan oluşturulan somut bir varlıktır:

\`\`\`python
# Nesne oluşturma
araba1 = Araba("Toyota", "Corolla", 2020)
araba2 = Araba("BMW", "X5", 2021)

# Nesne metodlarını kullanma
araba1.hizlan(50)
print(araba1.bilgi_goster())  # 2020 Toyota Corolla, Hız: 50 km/s
\`\`\`

## Constructor (__init__)

Constructor, nesne oluşturulduğunda otomatik olarak çağrılan özel bir metoddur:

\`\`\`python
class Ogrenci:
    def __init__(self, ad, numara):
        self.ad = ad
        self.numara = numara
        self.dersler = []  # Boş liste ile başlat
        print(f"{ad} isimli öğrenci oluşturuldu!")

# Constructor otomatik çağrılır
ogrenci1 = Ogrenci("Ahmet", "123")
\`\`\`

## Self Parametresi

\`self\` parametresi, nesnenin kendisini temsil eder ve instance metodlarında ilk parametre olarak kullanılır:

\`\`\`python
class Dikdortgen:
    def __init__(self, en, boy):
        self.en = en    # self ile nesne özelliği
        self.boy = boy  # self ile nesne özelliği
    
    def alan_hesapla(self):  # self parametresi
        return self.en * self.boy
    
    def cevre_hesapla(self):  # self parametresi
        return 2 * (self.en + self.boy)

# self otomatik gönderilir
d1 = Dikdortgen(5, 3)
print(d1.alan_hesapla())  # 15
\`\`\`

## Instance Metodları

Instance metodları, nesnenin davranışlarını tanımlayan ve \`self\` parametresi alan metodlardır:

\`\`\`python
class BankaHesabi:
    def __init__(self, hesap_no, bakiye=0):
        self.hesap_no = hesap_no
        self.bakiye = bakiye
    
    def para_yatir(self, miktar):
        self.bakiye += miktar
        return f"{miktar} TL yatırıldı. Yeni bakiye: {self.bakiye} TL"
    
    def para_cek(self, miktar):
        if miktar <= self.bakiye:
            self.bakiye -= miktar
            return f"{miktar} TL çekildi. Yeni bakiye: {self.bakiye} TL"
        return "Yetersiz bakiye!"

# Instance metodlarını kullanma
hesap = BankaHesabi("123456", 1000)
print(hesap.para_yatir(500))  # 500 TL yatırıldı. Yeni bakiye: 1500 TL
print(hesap.para_cek(2000))   # Yetersiz bakiye!
\`\`\`

## Instance Değişkenleri

Instance değişkenleri her nesne için ayrı ayrı oluşturulur ve saklanır:

\`\`\`python
class Calisan:
    def __init__(self, ad, maas):
        self.ad = ad        # Instance değişkeni
        self.maas = maas    # Instance değişkeni
        self.projeler = []  # Instance değişkeni
    
    def proje_ekle(self, proje):
        self.projeler.append(proje)

# Her nesnenin kendi değişkenleri vardır
c1 = Calisan("Ali", 5000)
c2 = Calisan("Veli", 6000)

c1.proje_ekle("Web Sitesi")
c2.proje_ekle("Mobil Uygulama")

print(c1.projeler)  # ['Web Sitesi']
print(c2.projeler)  # ['Mobil Uygulama']
\`\`\`

## İyi Pratikler

1. Sınıf isimleri PascalCase ile yazılır (Her kelimenin ilk harfi büyük)
2. Metod ve değişken isimleri snake_case ile yazılır
3. Her sınıf tek bir sorumluluğa sahip olmalıdır
4. Constructor'da gerekli tüm başlangıç değerleri atanmalıdır
5. Metodlar anlamlı isimler ile adlandırılmalıdır

## Nesne Dizileriyle Çalışma

Nesnelerle çalışırken, birden fazla nesneyi bir dizide (liste veya tuple) saklayabilir ve üzerlerinde döngülerle işlem yapabilirsiniz:

\`\`\`python
class Oyuncu:
    def __init__(self, x, y):
        self.x = x  # x koordinatı
        self.y = y  # y koordinatı

# Birden fazla nesne oluşturma
oyuncu1 = Oyuncu(5, 6)
oyuncu2 = Oyuncu(2, 4)
oyuncu3 = Oyuncu(3, 6)

# Nesneleri bir listede saklama
oyuncular = [oyuncu1, oyuncu2, oyuncu3]

# Tüm oyuncuların koordinatlarını yazdırma
for oyuncu in oyuncular:
    print(f"X: {oyuncu.x} Y: {oyuncu.y}")
\`\`\`

Bu örnekte:
1. \`Oyuncu\` sınıfı, 2D dünyada bir oyuncunun konumunu temsil eder
2. Üç farklı oyuncu nesnesi oluşturulur
3. Nesneler bir liste içinde saklanır
4. \`for\` döngüsü ile her oyuncunun koordinatları yazdırılır

Döngü her iterasyonda:
- İlk iterasyonda \`oyuncu1\` nesnesi \`oyuncu\` değişkenine atanır
- İkinci iterasyonda \`oyuncu2\` nesnesi \`oyuncu\` değişkenine atanır
- Üçüncü iterasyonda \`oyuncu3\` nesnesi \`oyuncu\` değişkenine atanır

💡 İpucu: Döngü değişkenine, içinde saklayacağı nesne tipini temsil eden bir isim vermeniz önerilir. Bu örnekte \`oyuncu\` kullanılmıştır çünkü değişken \`Oyuncu\` sınıfı nesnelerini tutar.

### Nesne Dizileriyle Çalışmanın Avantajları

1. **Kod Tekrarını Azaltma**: Aynı işlemi birden fazla nesne için tek bir döngüde yapabilirsiniz
2. **Bakım Kolaylığı**: Tüm nesneler için yapılacak değişiklikler tek bir yerde yapılır
3. **Dinamik İşlemler**: Çalışma zamanında nesne sayısı değişse bile kod çalışmaya devam eder
4. **Toplu İşlemler**: Filtreleme, sıralama, dönüştürme gibi işlemleri kolayca yapabilirsiniz

## Alıştırmalar

Pratik yaparak öğrenin:

1. Başlangıç seviyesi örnekler
2. Orta seviye projeler
3. İleri seviye uygulamalar
4. Test senaryoları

`;

const sections = [
  {
    title: "Sınıf Kavramı",
    description: "Sınıfların tanımı ve temel özellikleri",
    icon: <Code2 className="h-6 w-6" />,
    link: "/topics/python/nesne-tabanli-programlama/siniflar-ve-nesneler/sinif-kavrami",
    topics: [
      "Sınıf tanımlama",
      "Özellikler (attributes)",
      "Metodlar (methods)",
      "Sınıf yapısı"
    ]
  },
  {
    title: "Nesne Oluşturma",
    description: "Sınıflardan nesne örnekleri oluşturma",
    icon: <Settings className="h-6 w-6" />,
    link: "/topics/python/nesne-tabanli-programlama/siniflar-ve-nesneler/nesne-olusturma",
    topics: [
      "Constructor kullanımı",
      "Nesne örnekleme",
      "Parametre geçme",
      "Nesne yaşam döngüsü"
    ]
  },
  {
    title: "Instance Metodları",
    description: "Nesne davranışlarını tanımlayan metodlar",
    icon: <Terminal className="h-6 w-6" />,
    link: "/topics/python/nesne-tabanli-programlama/siniflar-ve-nesneler/instance-metodlari",
    topics: [
      "Self parametresi",
      "Metod tanımlama",
      "Metod çağırma",
      "Return değerleri"
    ]
  },
  {
    title: "İyi Pratikler",
    description: "Sınıf ve nesne kullanım önerileri",
    icon: <BookOpen className="h-6 w-6" />,
    link: "/topics/python/nesne-tabanli-programlama/siniflar-ve-nesneler/iyi-pratikler",
    topics: [
      "İsimlendirme kuralları",
      "Kod organizasyonu",
      "Hata yönetimi",
      "Dokümantasyon"
    ]
  },
  {
    title: "Nesne Dizileri",
    description: "Birden fazla nesneyle çalışma teknikleri",
    icon: <Code2 className="h-6 w-6" />,
    link: "/topics/python/nesne-tabanli-programlama/siniflar-ve-nesneler/nesne-dizileri",
    topics: [
      "Liste ve tuple kullanımı",
      "Döngülerle işlem yapma",
      "Toplu nesne işlemleri",
      "Nesne koleksiyonları"
    ]
  },
  {
    title: "Alıştırmalar",
    description: "Pratik yaparak öğrenin",
    icon: <GraduationCap className="h-6 w-6" />,
    link: "/topics/python/nesne-tabanli-programlama/siniflar-ve-nesneler/alistirmalar",
    topics: [
      "Başlangıç seviyesi örnekler",
      "Orta seviye projeler",
      "İleri seviye uygulamalar",
      "Test senaryoları"
    ]
  }
];

export default function SiniflarVeNesnelerPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Concept Cards */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Önemli Kavramlar</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {sections.map((section, index) => (
              <Card key={index} className="bg-blue-50 hover:bg-blue-100 dark:bg-blue-950/50 dark:hover:bg-blue-950/70 transition-all duration-300">
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg text-blue-600 dark:text-blue-400">
                      {section.icon}
                    </div>
                    <CardTitle>{section.title}</CardTitle>
                  </div>
                  <CardDescription className="dark:text-gray-300">{section.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground dark:text-gray-400">
                    {section.topics.map((topic, i) => (
                      <li key={i}>{topic}</li>
                    ))}
                  </ul>
                </CardContent>
                {section.link && (
                  <CardFooter>
                    <Button asChild variant="outline" className="w-full group">
                      <Link href={section.link}>
                        Detaylı İncele
                        <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
                      </Link>
                    </Button>
                  </CardFooter>
                )}
              </Card>
            ))}
          </div>
        </div>

        {/* Next Topic Button */}
        <div className="mt-12 flex justify-end">
          <Button asChild className="group">
            <Link href="/topics/python/nesne-tabanli-programlama/kalitim">
              Sonraki Konu: Kalıtım
              <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 