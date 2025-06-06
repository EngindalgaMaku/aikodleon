import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import Image from "next/image";
import { ArrowRight, Component, Blocks, Workflow, Code2 } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python Soyut Sınıflar ve Arayüzler | AIKOD',
  description: 'Python\'da soyut sınıflar, abstract metodlar, arayüzler, protokoller ve daha fazlası.',
};

const content = `
# Python'da Soyut Sınıflar ve Arayüzler

Soyut sınıflar ve arayüzler, nesne yönelimli programlamada kod organizasyonu ve standartlaştırma için kullanılan önemli yapılardır. Python'da bu yapıları \`abc\` (Abstract Base Classes) modülü ile oluşturabiliriz.

## Soyut Sınıf Nedir?

Soyut sınıf, doğrudan örneklenemeyen ve alt sınıflar için şablon görevi gören bir sınıf türüdür:

\`\`\`python
from abc import ABC, abstractmethod

class SoyutHayvan(ABC):
    def __init__(self, isim):
        self.isim = isim
    
    @abstractmethod
    def ses_cikar(self):
        pass
    
    @abstractmethod
    def hareket_et(self):
        pass
    
    def bilgi_goster(self):  # Normal metod
        return f"{self.isim} isimli hayvan"

# Soyut sınıftan nesne oluşturulamaz
# hayvan = SoyutHayvan("Test")  # TypeError
\`\`\`

## Abstract Metodlar

Abstract metodlar, alt sınıfların mutlaka uygulaması gereken metodlardır:

\`\`\`python
class Kedi(SoyutHayvan):
    def ses_cikar(self):  # Abstract metod implementasyonu
        return "Miyav!"
    
    def hareket_et(self):  # Abstract metod implementasyonu
        return "Kedi yürüyor"

class Kus(SoyutHayvan):
    def ses_cikar(self):
        return "Cik cik!"
    
    def hareket_et(self):
        return "Kuş uçuyor"

# Şimdi nesne oluşturabiliriz
kedi = Kedi("Pamuk")
kus = Kus("Maviş")

print(kedi.ses_cikar())  # Miyav!
print(kus.hareket_et())  # Kuş uçuyor
\`\`\`

## Arayüz (Interface) Tanımlama

Python'da formal interface yapısı yoktur, ancak soyut sınıflar ile benzer bir yapı oluşturabiliriz:

\`\`\`python
from abc import ABC, abstractmethod

class IVeriDepolama(ABC):
    @abstractmethod
    def kaydet(self, veri: str) -> bool:
        pass
    
    @abstractmethod
    def oku(self, id: int) -> str:
        pass
    
    @abstractmethod
    def sil(self, id: int) -> bool:
        pass

class DosyaDepolama(IVeriDepolama):
    def kaydet(self, veri: str) -> bool:
        print(f"Dosyaya kaydediliyor: {veri}")
        return True
    
    def oku(self, id: int) -> str:
        return f"Dosyadan okunan veri: {id}"
    
    def sil(self, id: int) -> bool:
        print(f"Dosyadan siliniyor: {id}")
        return True

class VeritabaniDepolama(IVeriDepolama):
    def kaydet(self, veri: str) -> bool:
        print(f"Veritabanına kaydediliyor: {veri}")
        return True
    
    def oku(self, id: int) -> str:
        return f"Veritabanından okunan veri: {id}"
    
    def sil(self, id: int) -> bool:
        print(f"Veritabanından siliniyor: {id}")
        return True

# Kullanım
def veri_isle(depolama: IVeriDepolama, veri: str):
    if depolama.kaydet(veri):
        print("Veri başarıyla kaydedildi")
        okunan = depolama.oku(1)
        print(okunan)
\`\`\`

## Protokoller (Python 3.8+)

Modern Python'da protokoller, yapısal alt tipleme için kullanılır:

\`\`\`python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Yazilabilir(Protocol):
    def yaz(self, data: str) -> None:
        ...

class DosyaYazici:
    def yaz(self, data: str) -> None:
        print(f"Dosyaya yazılıyor: {data}")

class KonsolYazici:
    def yaz(self, data: str) -> None:
        print(f"Konsola yazılıyor: {data}")

def veri_yaz(hedef: Yazilabilir, mesaj: str):
    hedef.yaz(mesaj)

# Her iki sınıf da Yazilabilir protokolüne uyuyor
dosya = DosyaYazici()
konsol = KonsolYazici()

veri_yaz(dosya, "Merhaba")
veri_yaz(konsol, "Dünya")
\`\`\`

## Çoklu Arayüz Implementasyonu

Bir sınıf birden fazla arayüzü implement edebilir:

\`\`\`python
class IYazilabilir(ABC):
    @abstractmethod
    def yaz(self, data: str):
        pass

class IOkunabilir(ABC):
    @abstractmethod
    def oku(self) -> str:
        pass

class Dosya(IYazilabilir, IOkunabilir):
    def __init__(self, icerik=""):
        self.icerik = icerik
    
    def yaz(self, data: str):
        self.icerik += data
    
    def oku(self) -> str:
        return self.icerik

# Kullanım
dosya = Dosya()
dosya.yaz("Merhaba ")
dosya.yaz("Dünya!")
print(dosya.oku())  # Merhaba Dünya!
\`\`\`

## İyi Pratikler

1. Soyut sınıfları ortak davranışları tanımlamak için kullanın
2. Her abstract metodun amacını dokümante edin
3. Interface'leri mümkün olduğunca küçük ve odaklı tutun
4. Type hinting kullanarak kod okunabilirliğini artırın
5. Protokolleri duck typing ile birlikte kullanın
`;

const sections = [
  {
    title: "Soyut Sınıflar",
    description: "Soyut sınıf kavramı ve kullanımı",
    icon: <Component className="h-6 w-6" />,
    topics: [
      "ABC modülü",
      "Abstract metodlar",
      "Soyut sınıf tanımlama",
      "Alt sınıf implementasyonu"
    ]
  },
  {
    title: "Arayüzler",
    description: "Python'da arayüz tanımlama ve kullanma",
    icon: <Blocks className="h-6 w-6" />,
    topics: [
      "Interface tanımlama",
      "Çoklu interface",
      "Interface implementasyonu",
      "Type hinting"
    ]
  },
  {
    title: "Protokoller",
    description: "Modern Python'da protokol kullanımı",
    icon: <Workflow className="h-6 w-6" />,
    topics: [
      "Protocol sınıfı",
      "Runtime checking",
      "Yapısal alt tipleme",
      "Duck typing"
    ]
  },
  {
    title: "İyi Pratikler",
    description: "Soyut sınıf ve arayüz kullanım önerileri",
    icon: <Code2 className="h-6 w-6" />,
    topics: [
      "Tasarım prensipleri",
      "Dokümantasyon",
      "Kod organizasyonu",
      "Test edilebilirlik"
    ]
  }
];

export default function SoyutSiniflarVeArayuzlerPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Concept Cards */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Önemli Kavramlar</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {sections.map((section, index) => (
              <Card key={index} className="bg-pink-50 hover:bg-pink-100 dark:bg-pink-950/50 dark:hover:bg-pink-950/70 transition-all duration-300">
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg text-pink-600 dark:text-pink-400">
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
            <Link href="/topics/python/nesne-tabanli-programlama/tasarim-desenleri">
              Sonraki Konu: Tasarım Desenleri
              <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 