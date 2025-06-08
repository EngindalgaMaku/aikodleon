import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowRight, Code2, BookOpen, GraduationCap, Lightbulb } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python OOP Terimler Sözlüğü | Kodleon',
  description: 'Python nesne tabanlı programlamada kullanılan temel kavramlar ve terimler sözlüğü.',
};

const content = `
# Python OOP Terimler Sözlüğü

Nesne tabanlı programlamada kullanılan temel kavramlar ve terimlerin açıklamaları.

## Temel Kavramlar

### Sınıf (Class)
Nesnelerin özelliklerini ve davranışlarını tanımlayan şablondur. Bir veri yapısı ve bu veri üzerinde çalışacak metodların birleşimidir.

\`\`\`python
class Araba:
    def __init__(self, marka, model):
        self.marka = marka
        self.model = model
\`\`\`

### Nesne (Object)
Bir sınıfın örneğidir. Sınıf şablonundan oluşturulan somut bir varlıktır.

\`\`\`python
araba1 = Araba("Toyota", "Corolla")  # Nesne oluşturma
\`\`\`

### Özellik (Attribute)
Bir nesnenin sahip olduğu veri değişkenleridir.

\`\`\`python
print(araba1.marka)  # Özelliğe erişim
araba1.model = "Camry"  # Özellik değiştirme
\`\`\`

### Metod (Method)
Bir sınıfın davranışlarını tanımlayan fonksiyonlardır.

\`\`\`python
class Hesap:
    def para_yatir(self, miktar):  # Metod
        self.bakiye += miktar
\`\`\`

## Özel Terimler

### Constructor (__init__)
Nesne oluşturulduğunda otomatik çağrılan özel metoddur.

\`\`\`python
class Oyuncu:
    def __init__(self, isim):  # Constructor
        self.isim = isim
        self.skor = 0
\`\`\`

### Self
Instance metodlarında nesnenin kendisini temsil eden parametredir.

\`\`\`python
class Kare:
    def alan_hesapla(self):  # self parametresi
        return self.kenar * self.kenar
\`\`\`

### Instance
Bir sınıftan oluşturulan tekil nesnedir.

\`\`\`python
k1 = Kare()  # k1 bir instance'dır
k2 = Kare()  # k2 farklı bir instance'dır
\`\`\`

## Erişim Belirteçleri

### Public
Herhangi bir alt çizgi ile başlamayan, dışarıdan erişilebilen özellik ve metodlar.

\`\`\`python
class Urun:
    def fiyat_goster(self):  # Public metod
        return self.fiyat
\`\`\`

### Protected (_)
Tek alt çizgi ile başlayan, dolaylı olarak korunan özellik ve metodlar.

\`\`\`python
class Veritabani:
    def __init__(self):
        self._baglanti = None  # Protected özellik
\`\`\`

### Private (__)
Çift alt çizgi ile başlayan, dışarıdan erişilemeyen özellik ve metodlar.

\`\`\`python
class Sifreleme:
    def __init__(self):
        self.__anahtar = "gizli"  # Private özellik
\`\`\`

## İleri Düzey Kavramlar

### Inheritance (Kalıtım)
Bir sınıfın başka bir sınıftan özellik ve metodları miras almasıdır.

\`\`\`python
class Hayvan:
    def ses_cikar(self):
        pass

class Kopek(Hayvan):  # Kalıtım
    def ses_cikar(self):
        return "Hav hav!"
\`\`\`

### Polymorphism (Çok Biçimlilik)
Aynı isimli metodların farklı sınıflarda farklı davranışlar sergilemesidir.

\`\`\`python
def hayvan_sesi(hayvan):
    print(hayvan.ses_cikar())  # Polymorphism

kopek = Kopek()
kedi = Kedi()
hayvan_sesi(kopek)  # "Hav hav!"
hayvan_sesi(kedi)   # "Miyav!"
\`\`\`

### Encapsulation (Kapsülleme)
Veri ve metodların bir arada tutulması ve dış erişimin kontrol edilmesidir.

\`\`\`python
class BankaHesabi:
    def __init__(self):
        self.__bakiye = 0  # Private
    
    @property
    def bakiye(self):  # Getter
        return self.__bakiye
    
    @bakiye.setter
    def bakiye(self, deger):  # Setter
        if deger >= 0:
            self.__bakiye = deger
\`\`\`

### Abstraction (Soyutlama)
Karmaşık sistemlerin basitleştirilmesi ve temel özelliklerinin öne çıkarılmasıdır.

\`\`\`python
from abc import ABC, abstractmethod

class Sekil(ABC):
    @abstractmethod
    def alan_hesapla(self):
        pass
\`\`\`

## Özel Metodlar

### Magic Methods
Çift alt çizgi ile başlayıp biten özel metodlardır.

\`\`\`python
class Nokta:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"({self.x}, {self.y})"
    
    def __add__(self, other):
        return Nokta(self.x + other.x, self.y + other.y)
\`\`\`

### Property Decorator
Metodları özellik gibi kullanmamızı sağlayan dekoratördür.

\`\`\`python
class Sicaklik:
    def __init__(self, celsius):
        self._celsius = celsius
    
    @property
    def fahrenheit(self):
        return (self._celsius * 9/5) + 32
\`\`\`

## Tasarım Kalıpları

### Singleton
Bir sınıfın yalnızca bir örneğinin olmasını sağlayan tasarım kalıbıdır.

\`\`\`python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
\`\`\`

### Factory
Nesne oluşturma işlemini merkezileştiren tasarım kalıbıdır.

\`\`\`python
class HayvanFactory:
    @staticmethod
    def hayvan_olustur(tur):
        if tur == "kopek":
            return Kopek()
        elif tur == "kedi":
            return Kedi()
\`\`\`
`;

const sections = [
  {
    title: "Temel Kavramlar",
    description: "OOP'nin temel yapı taşları",
    icon: <Code2 className="h-6 w-6" />,
    topics: [
      "Sınıf (Class)",
      "Nesne (Object)",
      "Özellik (Attribute)",
      "Metod (Method)"
    ]
  },
  {
    title: "Özel Terimler",
    description: "Python'a özgü OOP terimleri",
    icon: <BookOpen className="h-6 w-6" />,
    topics: [
      "Constructor",
      "Self",
      "Instance",
      "Magic Methods"
    ]
  },
  {
    title: "OOP Prensipleri",
    description: "Temel OOP prensipleri",
    icon: <GraduationCap className="h-6 w-6" />,
    topics: [
      "Inheritance",
      "Polymorphism",
      "Encapsulation",
      "Abstraction"
    ]
  },
  {
    title: "Tasarım Kalıpları",
    description: "Yaygın tasarım kalıpları",
    icon: <Lightbulb className="h-6 w-6" />,
    topics: [
      "Singleton",
      "Factory",
      "Observer",
      "Strategy"
    ]
  }
];

export default function TerimlerSozluguPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Concept Cards */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Terim Kategorileri</h2>
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
              </Card>
            ))}
          </div>
        </div>

        {/* Back Button */}
        <div className="mt-12 flex justify-end">
          <Button asChild variant="outline" className="group">
            <Link href="/topics/python/nesne-tabanli-programlama">
              Nesne Tabanlı Programlama Sayfasına Dön
              <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 