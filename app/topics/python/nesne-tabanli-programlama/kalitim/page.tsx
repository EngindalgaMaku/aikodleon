import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import Image from "next/image";
import { ArrowRight, GitFork, GitMerge, GitBranch, GitPullRequest } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python Kalıtım (Inheritance) | AIKOD',
  description: 'Python\'da kalıtım kavramı, tek ve çoklu kalıtım, method overriding, super() kullanımı ve daha fazlası.',
};

const content = `
# Python'da Kalıtım (Inheritance)

Kalıtım, bir sınıfın başka bir sınıfın özelliklerini ve metodlarını miras almasını sağlayan OOP özelliğidir. Bu sayede kod tekrarını önler ve hiyerarşik bir yapı oluşturmanıza olanak tanır.

## Temel Kalıtım

Bir sınıfın başka bir sınıftan kalıtım alması için, sınıf tanımında parantez içinde üst sınıfı belirtmemiz yeterlidir:

\`\`\`python
class Hayvan:
    def __init__(self, isim, yas):
        self.isim = isim
        self.yas = yas
    
    def ses_cikar(self):
        print("Genel hayvan sesi")

class Kopek(Hayvan):  # Hayvan sınıfından kalıtım alır
    def __init__(self, isim, yas, cins):
        super().__init__(isim, yas)  # Üst sınıfın constructor'ını çağır
        self.cins = cins
    
    def ses_cikar(self):  # Method overriding
        print("Hav hav!")

# Kullanımı
kopek = Kopek("Karabaş", 3, "Golden")
print(kopek.isim)  # Üst sınıftan miras alınan özellik
kopek.ses_cikar()  # Override edilmiş method
\`\`\`

## Çoklu Kalıtım

Python'da bir sınıf birden fazla sınıftan kalıtım alabilir:

\`\`\`python
class Ucabilen:
    def uc(self):
        print("Uçuyor...")

class Yuzebilen:
    def yuz(self):
        print("Yüzüyor...")

class Penguen(Yuzebilen, Ucabilen):  # Çoklu kalıtım
    def __init__(self, isim):
        self.isim = isim
    
    def uc(self):  # Override
        print(f"{self.isim} uçamaz!")

# Kullanımı
penguen = Penguen("Happy Feet")
penguen.yuz()   # Yüzebilen sınıfından
penguen.uc()    # Override edilmiş method
\`\`\`

## Method Resolution Order (MRO)

Python'da çoklu kalıtımda metodların hangi sırayla aranacağını belirleyen mekanizmadır:

\`\`\`python
class A:
    def metod(self):
        print("A'dan")

class B(A):
    def metod(self):
        print("B'den")

class C(A):
    def metod(self):
        print("C'den")

class D(B, C):
    pass

# MRO'yu görüntüle
print(D.__mro__)  # D -> B -> C -> A -> object
\`\`\`

## super() Kullanımı

\`super()\` fonksiyonu, üst sınıfın metodlarına erişmemizi sağlar:

\`\`\`python
class Calisan:
    def __init__(self, isim, maas):
        self.isim = isim
        self.maas = maas
    
    def bilgi_goster(self):
        return f"{self.isim} - {self.maas} TL"

class Yonetici(Calisan):
    def __init__(self, isim, maas, departman):
        super().__init__(isim, maas)  # Üst sınıfın constructor'ı
        self.departman = departman
    
    def bilgi_goster(self):
        temel_bilgi = super().bilgi_goster()  # Üst sınıfın methodu
        return f"{temel_bilgi} - {self.departman}"

# Kullanımı
yonetici = Yonetici("Ahmet", 10000, "IT")
print(yonetici.bilgi_goster())  # Ahmet - 10000 TL - IT
\`\`\`

## Mixin Sınıfları

Mixin'ler, sınıflara ek özellikler eklemek için kullanılan özel sınıflardır:

\`\`\`python
class LogMixin:
    def log(self, message):
        print(f"[LOG] {message}")

class JSONMixin:
    def to_json(self):
        import json
        return json.dumps(self.__dict__)

class Urun(LogMixin, JSONMixin):
    def __init__(self, ad, fiyat):
        self.ad = ad
        self.fiyat = fiyat
    
    def fiyat_guncelle(self, yeni_fiyat):
        self.log(f"Fiyat güncellendi: {self.fiyat} -> {yeni_fiyat}")
        self.fiyat = yeni_fiyat

# Kullanımı
urun = Urun("Laptop", 15000)
urun.fiyat_guncelle(16000)  # LogMixin'den
print(urun.to_json())       # JSONMixin'den
\`\`\`

## İyi Kalıtım Pratikleri

1. Kalıtımı "is-a" ilişkisi varsa kullanın (örn: Köpek bir Hayvandır)
2. Çoklu kalıtımı dikkatli kullanın, karmaşıklığı artırabilir
3. Mixin'leri özellik eklemek için tercih edin
4. Derin kalıtım hiyerarşilerinden kaçının (en fazla 2-3 seviye)
5. Her zaman \`super()\` ile üst sınıf metodlarını çağırın
`;

const sections = [
  {
    title: "Temel Kalıtım",
    description: "Sınıflar arası temel kalıtım işlemleri",
    icon: <GitFork className="h-6 w-6" />,
    topics: [
      "Kalıtım tanımlama",
      "Üst sınıf özellikleri",
      "Method overriding",
      "Constructor kalıtımı"
    ]
  },
  {
    title: "Çoklu Kalıtım",
    description: "Birden fazla sınıftan kalıtım alma",
    icon: <GitMerge className="h-6 w-6" />,
    topics: [
      "Çoklu kalıtım tanımlama",
      "MRO (Method Resolution Order)",
      "Diamond problemi",
      "Mixin sınıfları"
    ]
  },
  {
    title: "super() Kullanımı",
    description: "Üst sınıf metodlarına erişim",
    icon: <GitBranch className="h-6 w-6" />,
    topics: [
      "super() fonksiyonu",
      "Constructor zinciri",
      "Method zincirleme",
      "Özellik erişimi"
    ]
  },
  {
    title: "İyi Pratikler",
    description: "Kalıtım için önerilen yaklaşımlar",
    icon: <GitPullRequest className="h-6 w-6" />,
    topics: [
      "Kalıtım prensipleri",
      "Kod organizasyonu",
      "Hata yönetimi",
      "Dokümantasyon"
    ]
  }
];

export default function KalitimPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Concept Cards */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Önemli Kavramlar</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {sections.map((section, index) => (
              <Card key={index} className="bg-green-50 hover:bg-green-100 dark:bg-green-950/50 dark:hover:bg-green-950/70 transition-all duration-300">
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg text-green-600 dark:text-green-400">
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
              </Card>
            ))}
          </div>
        </div>

        {/* Next Topic Button */}
        <div className="mt-12 flex justify-end">
          <Button asChild className="group">
            <Link href="/topics/python/nesne-tabanli-programlama/kapsulleme">
              Sonraki Konu: Kapsülleme
              <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 