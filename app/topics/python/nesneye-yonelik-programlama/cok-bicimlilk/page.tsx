import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Info, Lightbulb, AlertTriangle } from "lucide-react";
import Image from "next/image";

export const metadata: Metadata = {
  title: 'Python OOP: Çok Biçimlilik (Polymorphism) | Kodleon',
  description: 'Python\'da çok biçimlilik kavramını, türlerini ve kullanım örneklerini öğrenin.',
};

const content = `
# Çok Biçimlilik (Polymorphism)

Çok biçimlilik, bir nesnenin farklı şekillerde davranabilme yeteneğidir. Python'da çok biçimlilik, kodun daha esnek ve yeniden kullanılabilir olmasını sağlar.

## Method Overriding (Metod Ezme)

Alt sınıfların üst sınıftan gelen metodları yeniden tanımlaması:

\`\`\`python
class Hayvan:
    def ses_cikar(self):
        return "Ses yok"
    
    def hareket_et(self):
        return "Hareket ediyor"

class Kopek(Hayvan):
    def ses_cikar(self):  # Metod ezme
        return "Hav hav!"

class Kedi(Hayvan):
    def ses_cikar(self):  # Metod ezme
        return "Miyav!"

# Polimorfik kullanım
hayvanlar = [Kopek(), Kedi(), Hayvan()]
for hayvan in hayvanlar:
    print(hayvan.ses_cikar())  # Her hayvan kendi sesini çıkarır
\`\`\`

## Duck Typing

Python'da bir nesnenin tipinden çok, davranışı önemlidir:

\`\`\`python
class Ordek:
    def yuz(self):
        return "Ördek yüzüyor"
    
    def uc(self):
        return "Ördek uçuyor"

class BalikAdam:
    def yuz(self):
        return "Balık adam yüzüyor"

def havuzda_yuz(yuzen):
    # Tip kontrolü yok, sadece yuz() metodunun varlığı önemli
    print(yuzen.yuz())

# Her iki nesne de yuz() metoduna sahip olduğu için çalışır
havuzda_yuz(Ordek())      # Ördek yüzüyor
havuzda_yuz(BalikAdam())  # Balık adam yüzüyor
\`\`\`

## Operator Overloading

Python'da operatörlerin davranışını özelleştirebiliriz:

\`\`\`python
class Vektor:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"Vektor({self.x}, {self.y})"
    
    def __add__(self, other):
        return Vektor(self.x + other.x, self.y + other.y)
    
    def __mul__(self, skaler):
        return Vektor(self.x * skaler, self.y * skaler)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

# Kullanım
v1 = Vektor(2, 3)
v2 = Vektor(3, 4)
print(v1 + v2)        # Vektor(5, 7)
print(v1 * 2)         # Vektor(4, 6)
print(v1 == v2)       # False
\`\`\`

## Abstract Base Classes (ABC)

Soyut sınıflar ve metodlar ile arayüz tanımlama:

\`\`\`python
from abc import ABC, abstractmethod

class Sekil(ABC):
    @abstractmethod
    def alan(self):
        pass
    
    @abstractmethod
    def cevre(self):
        pass
    
    def bilgi(self):  # Normal metod
        return f"Alan: {self.alan()}, Çevre: {self.cevre()}"

class Dikdortgen(Sekil):
    def __init__(self, genislik, yukseklik):
        self.genislik = genislik
        self.yukseklik = yukseklik
    
    def alan(self):
        return self.genislik * self.yukseklik
    
    def cevre(self):
        return 2 * (self.genislik + self.yukseklik)

class Daire(Sekil):
    def __init__(self, yaricap):
        self.yaricap = yaricap
    
    def alan(self):
        return 3.14 * self.yaricap ** 2
    
    def cevre(self):
        return 2 * 3.14 * self.yaricap

# Kullanım
# sekil = Sekil()  # TypeError: Can't instantiate abstract class
d1 = Dikdortgen(5, 3)
d2 = Daire(4)
print(d1.bilgi())  # Alan: 15, Çevre: 16
print(d2.bilgi())  # Alan: 50.24, Çevre: 25.12
\`\`\`

## Method Overloading

Python'da geleneksel method overloading yoktur, ancak varsayılan parametreler ve \`*args\`, \`**kwargs\` ile benzer işlevsellik sağlanabilir:

\`\`\`python
class Hesaplama:
    def topla(self, *args):
        return sum(args)
    
    def carp(self, x, y=1, z=1):
        return x * y * z

# Kullanım
h = Hesaplama()
print(h.topla(1, 2))        # 3
print(h.topla(1, 2, 3, 4))  # 10
print(h.carp(2))            # 2
print(h.carp(2, 3))         # 6
print(h.carp(2, 3, 4))      # 24
\`\`\`

## İyi Uygulama Örnekleri

1. **Strateji Deseni**
\`\`\`python
class OdemeStratejisi:
    @abstractmethod
    def ode(self, miktar):
        pass

class KrediKarti(OdemeStratejisi):
    def ode(self, miktar):
        return f"{miktar}TL kredi kartı ile ödendi"

class Havale(OdemeStratejisi):
    def ode(self, miktar):
        return f"{miktar}TL havale ile ödendi"

class Odeme:
    def __init__(self, strateji: OdemeStratejisi):
        self.strateji = strateji
    
    def ode(self, miktar):
        return self.strateji.ode(miktar)

# Kullanım
odeme = Odeme(KrediKarti())
print(odeme.ode(100))  # 100TL kredi kartı ile ödendi
odeme.strateji = Havale()
print(odeme.ode(100))  # 100TL havale ile ödendi
\`\`\`

2. **Template Method**
\`\`\`python
class Rapor(ABC):
    def olustur(self):
        self.baslik()
        self.icerik()
        self.ozet()
    
    @abstractmethod
    def baslik(self):
        pass
    
    @abstractmethod
    def icerik(self):
        pass
    
    def ozet(self):  # Hook method
        pass

class PDFRapor(Rapor):
    def baslik(self):
        print("PDF Başlık")
    
    def icerik(self):
        print("PDF İçerik")
\`\`\`

## Alıştırmalar

1. **Şekil Hesaplayıcı**
Farklı geometrik şekillerin alan ve çevre hesaplamalarını yapan bir sistem oluşturun.

2. **Dosya İşleyici**
Farklı dosya formatlarını (txt, csv, json) okuyabilen ve yazabilen bir sistem tasarlayın.

3. **Oyun Karakterleri**
Farklı karakterlerin (savaşçı, büyücü, okçu) saldırı ve savunma davranışlarını modelleyin.

## Kaynaklar

- [Python ABC Documentation](https://docs.python.org/3/library/abc.html)
- [Real Python - Object-Oriented Programming (OOP) in Python 3](https://realpython.com/python3-object-oriented-programming/)
- [Python Special Methods](https://docs.python.org/3/reference/datamodel.html#special-method-names)
`;

export default function PolymorphismPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Interactive Examples Section */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">İnteraktif Örnekler</h2>
          <Tabs defaultValue="example1">
            <TabsList>
              <TabsTrigger value="example1">Method Override</TabsTrigger>
              <TabsTrigger value="example2">Duck Typing</TabsTrigger>
              <TabsTrigger value="example3">Operator Overloading</TabsTrigger>
            </TabsList>
            <TabsContent value="example1">
              <Card>
                <CardHeader>
                  <CardTitle>Şekil Sınıfları</CardTitle>
                  <CardDescription>
                    Method override örneği
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                    <code>{`class Sekil:
    def alan_hesapla(self):
        return 0

class Kare(Sekil):
    def __init__(self, kenar):
        self.kenar = kenar
    
    def alan_hesapla(self):
        return self.kenar ** 2

# Kullanım
sekil = Sekil()
kare = Kare(5)
print(sekil.alan_hesapla())  # 0
print(kare.alan_hesapla())   # 25`}</code>
                  </pre>
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="example2">
              <Card>
                <CardHeader>
                  <CardTitle>Dosya İşlemleri</CardTitle>
                  <CardDescription>
                    Duck typing örneği
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                    <code>{`class TextDosya:
    def oku(self):
        return "Text dosyası okundu"

class PDFDosya:
    def oku(self):
        return "PDF dosyası okundu"

def dosya_oku(dosya):
    print(dosya.oku())

# Kullanım
text = TextDosya()
pdf = PDFDosya()
dosya_oku(text)  # Text dosyası okundu
dosya_oku(pdf)   # PDF dosyası okundu`}</code>
                  </pre>
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="example3">
              <Card>
                <CardHeader>
                  <CardTitle>Para Sınıfı</CardTitle>
                  <CardDescription>
                    Operator overloading örneği
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                    <code>{`class Para:
    def __init__(self, miktar):
        self.miktar = miktar
    
    def __add__(self, other):
        return Para(self.miktar + other.miktar)
    
    def __str__(self):
        return f"{self.miktar}TL"

# Kullanım
p1 = Para(100)
p2 = Para(200)
print(p1 + p2)  # 300TL`}</code>
                  </pre>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        {/* Tips and Best Practices */}
        <div className="my-12 space-y-4">
          <h2 className="text-3xl font-bold mb-8">İpuçları ve En İyi Pratikler</h2>
          
          <Alert>
            <Info className="h-4 w-4" />
            <AlertTitle>Duck Typing</AlertTitle>
            <AlertDescription>
              Python'da tip kontrolü yerine davranış kontrolü yapın. "Eğer bir şey ördek gibi yüzüyor ve ördek gibi ses çıkarıyorsa, o bir ördektir."
            </AlertDescription>
          </Alert>

          <Alert>
            <Lightbulb className="h-4 w-4" />
            <AlertTitle>Abstract Base Classes</AlertTitle>
            <AlertDescription>
              Ortak arayüzleri tanımlamak için ABC kullanın. Bu, kodunuzun daha tutarlı olmasını sağlar.
            </AlertDescription>
          </Alert>

          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Method Overloading</AlertTitle>
            <AlertDescription>
              Python'da geleneksel method overloading yerine varsayılan parametreler ve *args/**kwargs kullanın.
            </AlertDescription>
          </Alert>
        </div>
      </div>
    </div>
  );
} 