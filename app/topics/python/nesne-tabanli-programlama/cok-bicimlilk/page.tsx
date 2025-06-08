import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import Image from "next/image";
import { ArrowRight, Layers, Code2, Workflow, Settings } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python Çok Biçimlilik (Polymorphism) | Kodleon',
  description: 'Python\'da çok biçimlilik kavramı, method overriding, duck typing, abstract base classes ve daha fazlası.',
};

const content = `
# Python'da Çok Biçimlilik (Polymorphism)

Çok biçimlilik, farklı sınıfların aynı arayüzü (metod isimlerini) kullanarak farklı davranışlar sergileyebilmesidir. Bu özellik, kodunuzu daha esnek ve genişletilebilir hale getirir.

## Method Overriding

Alt sınıfların üst sınıftan gelen metodları kendi ihtiyaçlarına göre yeniden tanımlamasıdır:

\`\`\`python
class Sekil:
    def alan_hesapla(self):
        pass
    
    def cevre_hesapla(self):
        pass

class Dikdortgen(Sekil):
    def __init__(self, en, boy):
        self.en = en
        self.boy = boy
    
    def alan_hesapla(self):  # Override
        return self.en * self.boy
    
    def cevre_hesapla(self):  # Override
        return 2 * (self.en + self.boy)

class Daire(Sekil):
    def __init__(self, yaricap):
        self.yaricap = yaricap
    
    def alan_hesapla(self):  # Override
        import math
        return math.pi * self.yaricap ** 2
    
    def cevre_hesapla(self):  # Override
        import math
        return 2 * math.pi * self.yaricap

# Polimorfik kullanım
sekiller = [
    Dikdortgen(5, 3),
    Daire(4)
]

for sekil in sekiller:
    print(f"Alan: {sekil.alan_hesapla():.2f}")
    print(f"Çevre: {sekil.cevre_hesapla():.2f}")
\`\`\`

## Duck Typing

Python'da bir nesnenin tipinden çok, desteklediği metodlar önemlidir. "Eğer bir kuş gibi yürüyor ve bir kuş gibi ötüyorsa, o bir kuştur" prensibi:

\`\`\`python
class Ordek:
    def ses_cikar(self):
        return "Vak vak!"
    
    def yuz(self):
        return "Ordek yüzüyor"

class RobotOrdek:
    def ses_cikar(self):
        return "Elektronik vak vak!"
    
    def yuz(self):
        return "Robot ordek yüzüyor"

def ordek_testi(ordek):
    # Tip kontrolü yapmıyoruz!
    print(ordek.ses_cikar())
    print(ordek.yuz())

# Her iki sınıf da aynı arayüzü destekliyor
ordek_testi(Ordek())
ordek_testi(RobotOrdek())
\`\`\`

## Abstract Base Classes (ABC)

Soyut sınıflar ve metodlar tanımlamak için ABC modülünü kullanabiliriz:

\`\`\`python
from abc import ABC, abstractmethod

class Hayvan(ABC):
    @abstractmethod
    def ses_cikar(self):
        pass
    
    @abstractmethod
    def hareket_et(self):
        pass

class Kedi(Hayvan):
    def ses_cikar(self):
        return "Miyav!"
    
    def hareket_et(self):
        return "Kedi yürüyor"

class Kus(Hayvan):
    def ses_cikar(self):
        return "Cik cik!"
    
    def hareket_et(self):
        return "Kuş uçuyor"

# Soyut sınıftan nesne oluşturulamaz
# hayvan = Hayvan()  # TypeError

# Alt sınıflardan nesne oluşturulabilir
kedi = Kedi()
kus = Kus()
\`\`\`

## Interface Tanımlama

Python'da formal interface yapısı yoktur, ancak ABC ile benzer bir yapı oluşturabiliriz:

\`\`\`python
from abc import ABC, abstractmethod

class IVeriTabani(ABC):
    @abstractmethod
    def baglan(self):
        pass
    
    @abstractmethod
    def kaydet(self, veri):
        pass
    
    @abstractmethod
    def sorgula(self, sorgu):
        pass

class PostgreSQL(IVeriTabani):
    def baglan(self):
        return "PostgreSQL'e bağlanıldı"
    
    def kaydet(self, veri):
        return f"PostgreSQL'e kaydedildi: {veri}"
    
    def sorgula(self, sorgu):
        return f"PostgreSQL sorgusu çalıştırıldı: {sorgu}"

class MongoDB(IVeriTabani):
    def baglan(self):
        return "MongoDB'ye bağlanıldı"
    
    def kaydet(self, veri):
        return f"MongoDB'ye kaydedildi: {veri}"
    
    def sorgula(self, sorgu):
        return f"MongoDB sorgusu çalıştırıldı: {sorgu}"

# Kullanım
def veri_isle(db: IVeriTabani, veri: str):
    db.baglan()
    db.kaydet(veri)
    return db.sorgula("SELECT *")

# Her iki veritabanı da aynı arayüzü kullanıyor
postgres = PostgreSQL()
mongo = MongoDB()

veri_isle(postgres, "test")
veri_isle(mongo, "test")
\`\`\`

## Polymorphic Functions

Fonksiyonlar da çok biçimli olabilir:

\`\`\`python
def topla(x, y):
    return x + y

# Farklı veri tipleriyle çalışır
print(topla(5, 3))          # 8
print(topla("Merhaba ", "Dünya"))  # Merhaba Dünya
print(topla([1, 2], [3, 4]))       # [1, 2, 3, 4]
\`\`\`

## İyi Çok Biçimlilik Pratikleri

1. Arayüzleri açık ve tutarlı tasarlayın
2. Duck typing'i akıllıca kullanın
3. Gerektiğinde ABC kullanarak zorunlu metodları belirtin
4. Method isimlerini anlamlı ve tutarlı seçin
5. Tip kontrolü yerine EAFP (Easier to Ask for Forgiveness than Permission) yaklaşımını benimseyin
`;

const sections = [
  {
    title: "Method Overriding",
    description: "Alt sınıfların üst sınıf metodlarını yeniden tanımlaması",
    icon: <Layers className="h-6 w-6" />,
    topics: [
      "Override kavramı",
      "super() kullanımı",
      "Metod ezme",
      "Polimorfik davranış"
    ]
  },
  {
    title: "Duck Typing",
    description: "Tip kontrolü yerine davranış kontrolü",
    icon: <Code2 className="h-6 w-6" />,
    topics: [
      "Duck typing prensibi",
      "Dinamik tipli yapı",
      "Arayüz uyumu",
      "Esnek tasarım"
    ]
  },
  {
    title: "Abstract Classes",
    description: "Soyut sınıflar ve zorunlu metodlar",
    icon: <Workflow className="h-6 w-6" />,
    topics: [
      "ABC modülü",
      "Abstract metodlar",
      "Soyut sınıflar",
      "Interface benzeri yapılar"
    ]
  },
  {
    title: "İyi Pratikler",
    description: "Çok biçimlilik için önerilen yaklaşımlar",
    icon: <Settings className="h-6 w-6" />,
    topics: [
      "Tasarım prensipleri",
      "Kod organizasyonu",
      "Hata yönetimi",
      "EAFP yaklaşımı"
    ]
  }
];

export default function CokBicimlilkPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Concept Cards */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Önemli Kavramlar</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {sections.map((section, index) => (
              <Card key={index} className="bg-orange-50 hover:bg-orange-100 dark:bg-orange-950/50 dark:hover:bg-orange-950/70 transition-all duration-300">
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg text-orange-600 dark:text-orange-400">
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
            <Link href="/topics/python/nesne-tabanli-programlama/soyut-siniflar-ve-arayuzler">
              Sonraki Konu: Soyut Sınıflar ve Arayüzler
              <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 