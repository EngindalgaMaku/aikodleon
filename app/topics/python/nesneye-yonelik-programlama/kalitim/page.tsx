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
  title: 'Python OOP: Kalıtım (Inheritance) | Kodleon',
  description: 'Python\'da kalıtım kavramını, türetilmiş sınıfları ve çoklu kalıtımı öğrenin.',
};

const content = `
# Python'da Kalıtım (Inheritance)

Kalıtım, bir sınıfın başka bir sınıfın özelliklerini ve metodlarını miras almasıdır. Bu sayede kod tekrarını önler ve hiyerarşik bir yapı oluşturabiliriz.

## Temel Kalıtım

Bir sınıftan türetme yapmak için, yeni sınıf tanımında parantez içinde temel sınıfı belirtiriz:

\`\`\`python
class Hayvan:
    def __init__(self, isim, yas):
        self.isim = isim
        self.yas = yas
    
    def ses_cikar(self):
        pass

class Kopek(Hayvan):
    def ses_cikar(self):
        return "Hav hav!"

class Kedi(Hayvan):
    def ses_cikar(self):
        return "Miyav!"
\`\`\`

## super() Fonksiyonu

\`super()\` fonksiyonu, üst sınıfın metodlarını çağırmak için kullanılır:

\`\`\`python
class Ogrenci:
    def __init__(self, ad, soyad):
        self.ad = ad
        self.soyad = soyad

class LiseOgrencisi(Ogrenci):
    def __init__(self, ad, soyad, sinif):
        super().__init__(ad, soyad)  # Üst sınıfın __init__ metodunu çağır
        self.sinif = sinif
\`\`\`

## Çoklu Kalıtım

Python'da bir sınıf birden fazla sınıftan türetilebilir:

\`\`\`python
class A:
    def metod_a(self):
        return "A sınıfından"

class B:
    def metod_b(self):
        return "B sınıfından"

class C(A, B):  # C sınıfı hem A hem B'den türetildi
    pass

c = C()
print(c.metod_a())  # "A sınıfından"
print(c.metod_b())  # "B sınıfından"
\`\`\`

## Method Resolution Order (MRO)

Python'da çoklu kalıtımda metodların aranma sırası MRO ile belirlenir:

\`\`\`python
class A:
    def kim(self):
        return "A"

class B(A):
    def kim(self):
        return "B"

class C(A):
    def kim(self):
        return "C"

class D(B, C):
    pass

d = D()
print(D.mro())  # MRO sırasını gösterir
print(d.kim())  # "B" (soldan sağa arama yapılır)
\`\`\`

## isinstance() ve issubclass()

Nesne ve sınıf ilişkilerini kontrol etmek için kullanılan fonksiyonlar:

\`\`\`python
kopek = Kopek("Karabaş", 3)
print(isinstance(kopek, Kopek))      # True
print(isinstance(kopek, Hayvan))     # True
print(issubclass(Kopek, Hayvan))     # True
\`\`\`

## Pratik Örnek: Şekiller

\`\`\`python
import math

class Sekil:
    def alan(self):
        pass
    
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
        return math.pi * self.yaricap ** 2
    
    def cevre(self):
        return 2 * math.pi * self.yaricap

# Kullanım
d1 = Dikdortgen(5, 3)
print(f"Dikdörtgen Alanı: {d1.alan()}")
print(f"Dikdörtgen Çevresi: {d1.cevre()}")

d2 = Daire(4)
print(f"Daire Alanı: {d2.alan():.2f}")
print(f"Daire Çevresi: {d2.cevre():.2f}")
\`\`\`

## Alıştırmalar

1. Bir \`Calisan\` temel sınıfı oluşturun ve bu sınıftan \`Muhendis\` ve \`Yonetici\` sınıflarını türetin.
2. Bir \`Arac\` temel sınıfı oluşturun ve farklı araç türleri için alt sınıflar oluşturun.
3. Çoklu kalıtım kullanarak bir \`SuperKahraman\` sınıfı oluşturun.

## Sonraki Adımlar

Kalıtım konusunu öğrendiniz. Şimdi kapsülleme (encapsulation) konusuna geçerek, sınıf içi verileri nasıl koruyacağımızı ve erişimi nasıl kontrol edeceğimizi öğrenebilirsiniz.
`;

export default function InheritancePage() {
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
          <Link href="/topics/python/nesneye-yonelik-programlama/siniflar-ve-nesneler">
            <ArrowLeft className="h-4 w-4" />
            Önceki Konu: Sınıflar ve Nesneler
          </Link>
        </Button>
        
        <Button asChild variant="default" className="gap-2">
          <Link href="/topics/python/nesneye-yonelik-programlama/kapsulleme">
            Sonraki Konu: Kapsülleme
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