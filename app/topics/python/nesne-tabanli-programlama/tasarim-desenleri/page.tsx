import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import Image from "next/image";
import { ArrowRight, Factory, Copy, Command, Shield } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python Tasarım Desenleri | Kodleon',
  description: 'Python\'da yaygın kullanılan tasarım desenleri: Creational, Structural ve Behavioral patterns.',
};

const content = `
# Python'da Tasarım Desenleri

Tasarım desenleri, yazılım geliştirmede sık karşılaşılan problemlere yönelik test edilmiş, kanıtlanmış çözüm şablonlarıdır. Python'da en sık kullanılan tasarım desenlerini inceleyelim.

## Yaratımsal (Creational) Desenler

### Singleton Deseni

Bir sınıfın yalnızca bir örneğinin olmasını sağlar:

\`\`\`python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.value = 0

# Kullanım
s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True - Aynı nesne
\`\`\`

### Factory Method Deseni

Nesne oluşturma işlemini alt sınıflara devreder:

\`\`\`python
from abc import ABC, abstractmethod

class Hayvan(ABC):
    @abstractmethod
    def ses_cikar(self):
        pass

class Kopek(Hayvan):
    def ses_cikar(self):
        return "Hav hav!"

class Kedi(Hayvan):
    def ses_cikar(self):
        return "Miyav!"

class HayvanFactory:
    def hayvan_olustur(self, tur):
        if tur == "kopek":
            return Kopek()
        elif tur == "kedi":
            return Kedi()
        raise ValueError(f"Bilinmeyen hayvan türü: {tur}")

# Kullanım
factory = HayvanFactory()
hayvan = factory.hayvan_olustur("kopek")
print(hayvan.ses_cikar())  # Hav hav!
\`\`\`

### Builder Deseni

Karmaşık nesnelerin adım adım oluşturulmasını sağlar:

\`\`\`python
class Bilgisayar:
    def __init__(self):
        self.parcalar = []
    
    def parca_ekle(self, parca):
        self.parcalar.append(parca)
    
    def ozellikleri_goster(self):
        return f"Bilgisayar özellikleri: {', '.join(self.parcalar)}"

class BilgisayarBuilder:
    def __init__(self):
        self.bilgisayar = Bilgisayar()
    
    def islemci_ekle(self, islemci):
        self.bilgisayar.parca_ekle(f"İşlemci: {islemci}")
        return self
    
    def ram_ekle(self, ram):
        self.bilgisayar.parca_ekle(f"RAM: {ram}GB")
        return self
    
    def depolama_ekle(self, depolama):
        self.bilgisayar.parca_ekle(f"Depolama: {depolama}GB")
        return self
    
    def olustur(self):
        return self.bilgisayar

# Kullanım
builder = BilgisayarBuilder()
bilgisayar = builder.islemci_ekle("Intel i7") \\
                    .ram_ekle(16) \\
                    .depolama_ekle(512) \\
                    .olustur()
print(bilgisayar.ozellikleri_goster())
\`\`\`

## Yapısal (Structural) Desenler

### Adapter Deseni

Uyumsuz arayüzleri birlikte çalışabilir hale getirir:

\`\`\`python
class EskiSistem:
    def eski_metod(self):
        return "Eski sistem çalışıyor"

class YeniArayuz:
    def yeni_metod(self):
        pass

class Adapter(YeniArayuz):
    def __init__(self, eski_sistem):
        self.eski_sistem = eski_sistem
    
    def yeni_metod(self):
        return self.eski_sistem.eski_metod()

# Kullanım
eski = EskiSistem()
adapter = Adapter(eski)
print(adapter.yeni_metod())  # Eski sistem çalışıyor
\`\`\`

### Decorator Deseni

Nesnelere dinamik olarak yeni davranışlar ekler:

\`\`\`python
from functools import wraps

def log_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"{func.__name__} fonksiyonu çağrıldı")
        result = func(*args, **kwargs)
        print(f"{func.__name__} fonksiyonu tamamlandı")
        return result
    return wrapper

@log_decorator
def toplama(a, b):
    return a + b

# Kullanım
sonuc = toplama(3, 5)  # Loglama otomatik yapılır
\`\`\`

## Davranışsal (Behavioral) Desenler

### Observer Deseni

Nesneler arasında one-to-many ilişkisi kurar:

\`\`\`python
class Subject:
    def __init__(self):
        self._observers = []
        self._state = None
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self._state)
    
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, value):
        self._state = value
        self.notify()

class Observer:
    def __init__(self, name):
        self.name = name
    
    def update(self, state):
        print(f"{self.name} güncellendi: {state}")

# Kullanım
subject = Subject()
observer1 = Observer("Gözlemci 1")
observer2 = Observer("Gözlemci 2")

subject.attach(observer1)
subject.attach(observer2)
subject.state = "Yeni durum"
\`\`\`

### Strategy Deseni

Algoritmaları dinamik olarak değiştirmeyi sağlar:

\`\`\`python
from abc import ABC, abstractmethod

class OdemeStratejisi(ABC):
    @abstractmethod
    def odeme_yap(self, miktar):
        pass

class KrediKartiOdeme(OdemeStratejisi):
    def odeme_yap(self, miktar):
        return f"Kredi kartı ile {miktar}TL ödendi"

class HavaleOdeme(OdemeStratejisi):
    def odeme_yap(self, miktar):
        return f"Havale ile {miktar}TL ödendi"

class OdemeIslemcisi:
    def __init__(self, strateji: OdemeStratejisi):
        self.strateji = strateji
    
    def odeme_yap(self, miktar):
        return self.strateji.odeme_yap(miktar)

# Kullanım
kredi_karti = KrediKartiOdeme()
havale = HavaleOdeme()

islemci = OdemeIslemcisi(kredi_karti)
print(islemci.odeme_yap(100))  # Kredi kartı ile ödeme

islemci.strateji = havale
print(islemci.odeme_yap(100))  # Havale ile ödeme
\`\`\`

## İyi Pratikler

1. Tasarım desenlerini ihtiyaç olduğunda kullanın, her yerde değil
2. Desenin amacını ve kullanım senaryolarını iyi anlayın
3. SOLID prensiplerini göz önünde bulundurun
4. Kodun okunabilirliğini ve bakım yapılabilirliğini önceliklendirin
5. Desenleri projenin gereksinimlerine göre uyarlayın
`;

const sections = [
  {
    title: "Yaratımsal Desenler",
    description: "Nesne oluşturma mekanizmaları",
    icon: <Factory className="h-6 w-6" />,
    topics: [
      "Singleton Pattern",
      "Factory Method",
      "Abstract Factory",
      "Builder Pattern"
    ]
  },
  {
    title: "Yapısal Desenler",
    description: "Nesnelerin birleştirilmesi",
    icon: <Copy className="h-6 w-6" />,
    topics: [
      "Adapter Pattern",
      "Bridge Pattern",
      "Composite Pattern",
      "Decorator Pattern"
    ]
  },
  {
    title: "Davranışsal Desenler",
    description: "Nesneler arası iletişim",
    icon: <Command className="h-6 w-6" />,
    topics: [
      "Observer Pattern",
      "Strategy Pattern",
      "Command Pattern",
      "State Pattern"
    ]
  },
  {
    title: "İyi Pratikler",
    description: "Tasarım desenleri kullanım önerileri",
    icon: <Shield className="h-6 w-6" />,
    topics: [
      "Doğru desen seçimi",
      "SOLID prensipler",
      "Kod organizasyonu",
      "Performans dengesi"
    ]
  }
];

export default function TasarimDesenleriPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Concept Cards */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Önemli Kavramlar</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {sections.map((section, index) => (
              <Card key={index} className="bg-purple-50 hover:bg-purple-100 dark:bg-purple-950/50 dark:hover:bg-purple-950/70 transition-all duration-300">
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg text-purple-600 dark:text-purple-400">
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
            <Link href="/topics/python/nesne-tabanli-programlama/pratik-ornekler">
              Sonraki Konu: Pratik Örnekler
              <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 