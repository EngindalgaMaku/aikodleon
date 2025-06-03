import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Info, Lightbulb, AlertTriangle } from "lucide-react";
import Image from "next/image";

export const metadata: Metadata = {
  title: 'Python OOP: Çok Biçimlilik (Polymorphism) | Kodleon',
  description: 'Python\'da çok biçimlilik kavramını, metod overriding, duck typing ve abstract sınıfları öğrenin.',
};

const content = `
# Python'da Çok Biçimlilik (Polymorphism)

Çok biçimlilik, aynı arayüzün farklı sınıflarda farklı şekillerde uygulanmasıdır. Python'da bu özellik sayesinde kodumuz daha esnek ve yeniden kullanılabilir hale gelir.

## Metod Overriding

Alt sınıflar, üst sınıfın metodlarını kendi ihtiyaçlarına göre yeniden tanımlayabilir:

\`\`\`python
class Hayvan:
    def ses_cikar(self):
        return "..."
    
    def hareket_et(self):
        return "Hareket ediyor"

class Kopek(Hayvan):
    def ses_cikar(self):  # Metod override edildi
        return "Hav hav!"

class Kus(Hayvan):
    def ses_cikar(self):  # Metod override edildi
        return "Cik cik!"
    
    def hareket_et(self):  # Metod override edildi
        return "Uçuyor"

# Kullanım
hayvanlar = [Kopek(), Kus()]
for hayvan in hayvanlar:
    print(hayvan.ses_cikar())    # Her hayvan kendi sesini çıkarır
    print(hayvan.hareket_et())   # Her hayvan kendi hareketini yapar
\`\`\`

## Duck Typing

Python'da bir nesnenin tipi, sahip olduğu metodlar ve özelliklerle belirlenir:

\`\`\`python
class Ordek:
    def yuz(self):
        return "Ördek yüzüyor"
    
    def ses_cikar(self):
        return "Vak vak!"

class Robot:
    def yuz(self):
        return "Robot yüzüyor"
    
    def ses_cikar(self):
        return "Bip bip!"

def havuzda_yuz(nesne):
    print(nesne.yuz())
    print(nesne.ses_cikar())

# Her iki nesne de yüzebilir ve ses çıkarabilir
havuzda_yuz(Ordek())  # Tip kontrolü yok
havuzda_yuz(Robot())  # Duck typing sayesinde çalışır
\`\`\`

## Abstract Base Classes (ABC)

Soyut sınıflar, alt sınıfların uygulaması gereken metodları tanımlar:

\`\`\`python
from abc import ABC, abstractmethod

class Sekil(ABC):
    @abstractmethod
    def alan(self):
        pass
    
    @abstractmethod
    def cevre(self):
        pass

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
        import math
        return math.pi * self.yaricap ** 2
    
    def cevre(self):
        import math
        return 2 * math.pi * self.yaricap

# Kullanım
sekiller = [Dikdortgen(5, 3), Daire(4)]
for sekil in sekiller:
    print(f"Alan: {sekil.alan():.2f}")
    print(f"Çevre: {sekil.cevre():.2f}")
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
    
    def __sub__(self, other):
        return Vektor(self.x - other.x, self.y - other.y)
    
    def __mul__(self, skaler):
        return Vektor(self.x * skaler, self.y * skaler)

# Kullanım
v1 = Vektor(2, 3)
v2 = Vektor(3, 4)
print(v1 + v2)        # Vektor(5, 7)
print(v1 - v2)        # Vektor(-1, -1)
print(v1 * 2)         # Vektor(4, 6)
\`\`\`

## Pratik Örnek: Medya Oynatıcı

\`\`\`python
class MedyaDosyasi:
    def __init__(self, dosya_adi):
        self.dosya_adi = dosya_adi
    
    def oynat(self):
        raise NotImplementedError("Bu metod alt sınıflarda uygulanmalıdır")
    
    def durdur(self):
        return "Medya durduruldu"

class Ses(MedyaDosyasi):
    def oynat(self):
        return f"{self.dosya_adi} ses dosyası çalınıyor"
    
    def ses_seviyesi_ayarla(self, seviye):
        return f"Ses seviyesi {seviye} olarak ayarlandı"

class Video(MedyaDosyasi):
    def oynat(self):
        return f"{self.dosya_adi} video dosyası oynatılıyor"
    
    def tam_ekran(self):
        return "Video tam ekran moduna geçti"

class Resim(MedyaDosyasi):
    def oynat(self):
        return f"{self.dosya_adi} resim dosyası görüntüleniyor"
    
    def yakınlaştır(self, oran):
        return f"Resim {oran}x yakınlaştırıldı"

# Kullanım
medya_listesi = [
    Ses("muzik.mp3"),
    Video("film.mp4"),
    Resim("foto.jpg")
]

for medya in medya_listesi:
    print(medya.oynat())    # Her medya kendi tipine göre oynatılır
    print(medya.durdur())   # Ortak metod
\`\`\`

## Alıştırmalar

1. Bir \`Hesaplayici\` soyut sınıfı oluşturun:
   - \`topla\`, \`cikar\`, \`carp\`, \`bol\` metodları olsun
   - \`BasitHesaplayici\` ve \`BilimselHesaplayici\` alt sınıfları yazın

2. Bir \`Oyun\` sistemi tasarlayın:
   - \`Karakter\` soyut sınıfı oluşturun
   - \`Savasci\`, \`Buyucu\`, \`Okcu\` alt sınıfları yazın
   - Her sınıf için farklı saldırı ve savunma metodları ekleyin

3. Bir \`Sekil\` hiyerarşisi oluşturun:
   - \`+\` operatörü ile şekillerin alanlarını toplayın
   - \`*\` operatörü ile şekli ölçeklendirin
   - \`str\` ve \`repr\` metodlarını uygulayın

## Sonraki Adımlar

Çok biçimlilik konusunu öğrendiniz. Şimdi soyut sınıflar ve arayüzler konusuna geçerek, sınıflar arası sözleşmeleri nasıl tanımlayacağımızı öğrenebilirsiniz.
`;

export default function PolymorphismPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <Button asChild variant="ghost" size="sm" className="gap-1">
          <Link href="/topics/python/nesneye-yonelik-programlama">
            <ArrowLeft className="h-4 w-4" />
            OOP Konularına Dön
          </Link>
        </Button>
      </div>
      
      <div className="prose prose-lg dark:prose-invert max-w-none">
        <MarkdownContent content={content} />
      </div>
      
      {/* Navigasyon Linkleri */}
      <div className="mt-12 flex flex-col sm:flex-row justify-between gap-4">
        <Button asChild variant="outline" className="gap-2">
          <Link href="/topics/python/nesneye-yonelik-programlama/kapsulleme">
            <ArrowLeft className="h-4 w-4" />
            Önceki Konu: Kapsülleme
          </Link>
        </Button>
        
        <Button asChild variant="default" className="gap-2">
          <Link href="/topics/python/nesneye-yonelik-programlama/soyut-siniflar-ve-arayuzler">
            Sonraki Konu: Soyut Sınıflar ve Arayüzler
            <ArrowRight className="h-4 w-4" />
          </Link>
        </Button>
      </div>
      
      <div className="mt-16 text-center text-sm text-muted-foreground">
        <p>© {new Date().getFullYear()} Kodleon | Yapay Zeka Eğitim Platformu</p>
      </div>
    </div>
  );
} 