'use client';

import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Info, Lightbulb, AlertTriangle } from "lucide-react";
import Image from "next/image";
import CodeRunner from '../siniflar-ve-nesneler/components/CodeRunner';

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

1. **Çalışan Yönetim Sistemi** [Detaylı çözüm için tıklayın](/topics/python/nesneye-yonelik-programlama/kalitim/calisan-yonetim-sistemi)
   - Bir şirketin çalışan yönetim sistemini modelleyin:
     - `Calisan` temel sınıfı
     - `Muhendis`, `Yonetici`, `Pazarlamaci` gibi alt sınıflar
     - Maaş hesaplama, izin takibi, proje atama gibi özellikler
     - Departman bazlı raporlama sistemi

2. **Oyun Karakter Sistemi** [Detaylı çözüm için tıklayın](/topics/python/nesneye-yonelik-programlama/kalitim/oyun-karakter-sistemi)
   - Bir RPG oyunu için karakter sistemi geliştirin:
     - `Karakter` temel sınıfı
     - `Savasci`, `Buyucu`, `Okcu` gibi alt sınıflar
     - Yetenek sistemi ve seviye atlama
     - Envanter yönetimi ve ekipman sistemi

3. **Medya Oynatıcı Sistemi** [Detaylı çözüm için tıklayın](/topics/python/nesneye-yonelik-programlama/kalitim/medya-oynatici-sistemi)
   - Farklı medya türlerini destekleyen bir oynatıcı sistemi oluşturun:
     - `MedyaOynatici` temel sınıfı
     - `MuzikOynatici`, `VideoOynatici`, `PodcastOynatici` alt sınıfları
     - Çalma listesi yönetimi
     - Format dönüştürme ve kalite ayarları

## Sonraki Adımlar

Kalıtım konusunu detaylı örneklerle öğrendiniz. Şimdi kapsülleme (encapsulation) konusuna geçerek, sınıf içi verileri nasıl koruyacağımızı ve erişimi nasıl kontrol edeceğimizi öğrenebilirsiniz.
`;

const temelKalitimCode = `# Temel bir Sekil sınıfı tanımlayalım
class Sekil:
    def __init__(self, x, y):
        self.x = x  # x koordinatı
        self.y = y  # y koordinatı
        
    def konum_goster(self):
        return f"X: {self.x}, Y: {self.y}"
    
    def alan_hesapla(self):
        return 0  # Temel sınıfta alan hesabı yok
    
    def bilgi_goster(self):
        return f"Bu bir şekildir. Konumu: {self.konum_goster()}"

# Sekil sınıfından türetilen Dikdortgen sınıfı
class Dikdortgen(Sekil):
    def __init__(self, x, y, genislik, yukseklik):
        # Üst sınıfın constructor'ını çağır
        super().__init__(x, y)
        self.genislik = genislik
        self.yukseklik = yukseklik
    
    def alan_hesapla(self):
        return self.genislik * self.yukseklik
    
    def bilgi_goster(self):
        return f"Bu bir dikdörtgendir. Konumu: {self.konum_goster()}, Alanı: {self.alan_hesapla()}"

# Sekil sınıfından türetilen Daire sınıfı
class Daire(Sekil):
    def __init__(self, x, y, yaricap):
        super().__init__(x, y)
        self.yaricap = yaricap
    
    def alan_hesapla(self):
        import math
        return math.pi * self.yaricap ** 2
    
    def bilgi_goster(self):
        return f"Bu bir dairedir. Konumu: {self.konum_goster()}, Alanı: {self.alan_hesapla():.2f}"

# Test edelim
sekil = Sekil(0, 0)
print(sekil.bilgi_goster())

dikdortgen = Dikdortgen(2, 3, 4, 5)
print(dikdortgen.bilgi_goster())

daire = Daire(1, 1, 3)
print(daire.bilgi_goster())`;

const cokluKalitimCode = `# Yetenek sınıfları
class Yuzebilir:
    def yuz(self):
        return "Yüzüyor..."
    
    def dalis_yap(self):
        return "Dalış yapıyor..."

class Ucabilir:
    def uc(self):
        return "Uçuyor..."
    
    def kanat_cap(self):
        return "Kanatlarını çırpıyor..."

class Yuruyebilir:
    def yuru(self):
        return "Yürüyor..."
    
    def kos(self):
        return "Koşuyor..."

# Hayvan sınıfı - temel sınıf
class Hayvan:
    def __init__(self, isim, yas):
        self.isim = isim
        self.yas = yas
    
    def bilgi_goster(self):
        return f"{self.isim} ({self.yas} yaşında)"

# Penguen - Hem yüzebilir hem yürüyebilir
class Penguen(Hayvan, Yuzebilir, Yuruyebilir):
    def __init__(self, isim, yas):
        super().__init__(isim, yas)
    
    def ozel_yetenek(self):
        return "Buzda kayabilir"

# Ördek - Yüzebilir, uçabilir ve yürüyebilir
class Ordek(Hayvan, Yuzebilir, Ucabilir, Yuruyebilir):
    def __init__(self, isim, yas):
        super().__init__(isim, yas)
    
    def ozel_yetenek(self):
        return "Gagasıyla yem toplayabilir"

# Test edelim
penguen = Penguen("Happy Feet", 3)
print(f"\\nPenguen: {penguen.bilgi_goster()}")
print(f"Yüzme: {penguen.yuz()}")
print(f"Yürüme: {penguen.yuru()}")
print(f"Özel yetenek: {penguen.ozel_yetenek()}")

ordek = Ordek("Donald", 2)
print(f"\\nÖrdek: {ordek.bilgi_goster()}")
print(f"Yüzme: {ordek.yuz()}")
print(f"Uçma: {ordek.uc()}")
print(f"Yürüme: {ordek.yuru()}")
print(f"Özel yetenek: {ordek.ozel_yetenek()}")

# MRO (Method Resolution Order) gösterimi
print("\\nÖrdek sınıfının metod arama sırası:")
print(Ordek.mro())`;

const abstractClassCode = `from abc import ABC, abstractmethod

# Soyut temel sınıf
class CihazArayuzu(ABC):
    def __init__(self, marka, model):
        self.marka = marka
        self.model = model
        self.acik_mi = False
    
    @abstractmethod
    def ac(self):
        pass
    
    @abstractmethod
    def kapat(self):
        pass
    
    @abstractmethod
    def ses_ayarla(self, seviye):
        pass

# Televizyon sınıfı
class Televizyon(CihazArayuzu):
    def __init__(self, marka, model):
        super().__init__(marka, model)
        self.kanal = 1
        self.ses_seviyesi = 50
    
    def ac(self):
        self.acik_mi = True
        return f"{self.marka} {self.model} TV açıldı."
    
    def kapat(self):
        self.acik_mi = False
        return f"{self.marka} {self.model} TV kapandı."
    
    def ses_ayarla(self, seviye):
        if 0 <= seviye <= 100:
            self.ses_seviyesi = seviye
            return f"Ses seviyesi {seviye} olarak ayarlandı."
        return "Geçersiz ses seviyesi!"
    
    def kanal_degistir(self, yeni_kanal):
        self.kanal = yeni_kanal
        return f"Kanal {yeni_kanal} olarak değiştirildi."

# Akıllı Telefon sınıfı
class AkilliTelefon(CihazArayuzu):
    def __init__(self, marka, model):
        super().__init__(marka, model)
        self.ses_seviyesi = 70
        self.uygulama_acik = None
    
    def ac(self):
        self.acik_mi = True
        return f"{self.marka} {self.model} telefon açıldı."
    
    def kapat(self):
        self.acik_mi = False
        return f"{self.marka} {self.model} telefon kapandı."
    
    def ses_ayarla(self, seviye):
        if 0 <= seviye <= 100:
            self.ses_seviyesi = seviye
            return f"Ses seviyesi {seviye} olarak ayarlandı."
        return "Geçersiz ses seviyesi!"
    
    def uygulama_calistir(self, uygulama):
        self.uygulama_acik = uygulama
        return f"{uygulama} uygulaması çalıştırıldı."

# Test edelim
tv = Televizyon("Samsung", "Smart TV")
print(tv.ac())
print(tv.ses_ayarla(75))
print(tv.kanal_degistir(8))
print(tv.kapat())

print("\\n" + "="*50 + "\\n")

telefon = AkilliTelefon("iPhone", "14 Pro")
print(telefon.ac())
print(telefon.ses_ayarla(60))
print(telefon.uygulama_calistir("YouTube"))
print(telefon.kapat())`;

export default function InheritancePage() {
  return (
    <div className="container mx-auto py-8">
      <nav className="flex flex-col sm:flex-row justify-between items-center gap-4 mb-8 bg-muted/30 p-4 rounded-lg">
        <Link href="/topics/python/nesneye-yonelik-programlama/siniflar-ve-nesneler" className="w-full sm:w-auto">
          <Button variant="outline" className="w-full">
            <ArrowLeft className="mr-2 h-4 w-4" />
            <div className="flex flex-col items-start">
              <span className="text-xs text-muted-foreground">Önceki Konu</span>
              <span>Sınıflar ve Nesneler</span>
            </div>
          </Button>
        </Link>
        <Link href="/topics/python/nesneye-yonelik-programlama/kapsulleme" className="w-full sm:w-auto">
          <Button variant="outline" className="w-full">
            <div className="flex flex-col items-end">
              <span className="text-xs text-muted-foreground">Sonraki Konu</span>
              <span>Kapsülleme</span>
            </div>
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </Link>
      </nav>

      <h1 className="text-4xl font-bold mb-6">Kalıtım (Inheritance)</h1>

      <section className="prose prose-lg max-w-none mb-12">
        <h2 className="text-3xl font-semibold mb-4">Kalıtım Nedir?</h2>
        <p>
          Kalıtım, nesne yönelimli programlamanın temel prensiplerinden biridir. Bir sınıfın başka bir sınıfın özelliklerini ve davranışlarını miras almasını sağlar.
          Bu sayede kod tekrarını önler ve sınıflar arasında hiyerarşik bir ilişki kurulmasını sağlar.
        </p>

        <div className="bg-blue-50 dark:bg-blue-900/10 p-6 rounded-lg mb-8">
          <h3 className="text-2xl font-semibold mb-4">🎯 Kalıtımın Avantajları</h3>
          <ul className="list-disc pl-6">
            <li><strong>Kod Tekrarını Önleme:</strong> Ortak özellikleri temel sınıfta tanımlayarak kod tekrarını önler.</li>
            <li><strong>Hiyerarşik Yapı:</strong> Sınıflar arasında mantıksal bir hiyerarşi oluşturur.</li>
            <li><strong>Kodun Yeniden Kullanılabilirliği:</strong> Var olan kodun yeni sınıflarda kullanılmasını sağlar.</li>
            <li><strong>Genişletilebilirlik:</strong> Mevcut sınıfları değiştirmeden yeni özellikler eklenebilir.</li>
          </ul>
        </div>

        <h3 className="text-2xl font-semibold mb-4">Temel Kalıtım Örneği</h3>
        <p>
          Aşağıdaki örnekte, geometrik şekilleri modelleyen bir sınıf hiyerarşisi oluşturuyoruz. <code>Sekil</code> sınıfı temel sınıf olarak kullanılıyor
          ve <code>Dikdortgen</code> ve <code>Daire</code> sınıfları bu temel sınıftan türetiliyor.
        </p>
        <CodeRunner initialCode={temelKalitimCode} />
      </section>

      <section className="prose prose-lg max-w-none mb-12">
        <h2 className="text-3xl font-semibold mb-4">Çoklu Kalıtım</h2>
        <p>
          Python, bir sınıfın birden fazla sınıftan kalıtım almasına izin verir. Bu özellik, çoklu kalıtım olarak adlandırılır.
          Çoklu kalıtım güçlü bir özellik olmakla birlikte, dikkatli kullanılması gerekir.
        </p>

        <div className="bg-yellow-50 dark:bg-yellow-900/10 p-6 rounded-lg mb-8">
          <h3 className="text-2xl font-semibold mb-4">⚠️ Çoklu Kalıtımda Dikkat Edilecek Noktalar</h3>
          <ul className="list-disc pl-6">
            <li><strong>Elmas Problemi:</strong> Aynı metodun farklı üst sınıflarda farklı şekillerde tanımlanması durumu.</li>
            <li><strong>Karmaşıklık:</strong> Çok sayıda üst sınıf kullanımı kodun anlaşılmasını zorlaştırabilir.</li>
            <li><strong>MRO (Method Resolution Order):</strong> Python'ın metod arama sırasını anlamak önemlidir.</li>
          </ul>
        </div>

        <h3 className="text-2xl font-semibold mb-4">Çoklu Kalıtım Örneği</h3>
        <p>
          Bu örnekte, farklı yetenekleri (yüzme, uçma, yürüme) temsil eden sınıfları kullanarak hayvanları modelleyeceğiz.
          Bu yaklaşım, davranışların kompozisyonunu göstermek için ideal bir örnektir.
        </p>
        <CodeRunner initialCode={cokluKalitimCode} />
      </section>

      <section className="prose prose-lg max-w-none mb-12">
        <h2 className="text-3xl font-semibold mb-4">Soyut Sınıflar ve Arayüzler</h2>
        <p>
          Soyut sınıflar, doğrudan örneklenemeyen ve türetilmiş sınıflar için bir şablon görevi gören sınıflardır.
          Python'da <code>abc</code> modülü kullanılarak soyut sınıflar oluşturulabilir.
        </p>

        <div className="bg-green-50 dark:bg-green-900/10 p-6 rounded-lg mb-8">
          <h3 className="text-2xl font-semibold mb-4">🔑 Soyut Sınıfların Özellikleri</h3>
          <ul className="list-disc pl-6">
            <li><strong>Şablon Oluşturma:</strong> Alt sınıflar için zorunlu metodları tanımlar.</li>
            <li><strong>Kod Standardizasyonu:</strong> Alt sınıfların belirli bir arayüzü uygulamasını sağlar.</li>
            <li><strong>Polimorfizm:</strong> Farklı sınıfların aynı arayüzü kullanmasını sağlar.</li>
          </ul>
        </div>

        <h3 className="text-2xl font-semibold mb-4">Soyut Sınıf Örneği</h3>
        <p>
          Aşağıdaki örnekte, elektronik cihazlar için bir arayüz tanımlayacağız. Bu arayüz, tüm cihazların uygulaması
          gereken temel metodları belirler.
        </p>
        <CodeRunner initialCode={abstractClassCode} />
      </section>

      <section className="prose prose-lg max-w-none mb-12">
        <h2 className="text-3xl font-semibold mb-4">İyi Pratikler ve Öneriler</h2>
        
        <div className="bg-purple-50 dark:bg-purple-900/10 p-6 rounded-lg mb-8">
          <h3 className="text-2xl font-semibold mb-4">💡 Kalıtım Kullanırken Dikkat Edilecek Noktalar</h3>
          <ul className="list-disc pl-6">
            <li><strong>IS-A İlişkisi:</strong> Kalıtım kullanırken "is-a" ilişkisinin varlığından emin olun.</li>
            <li><strong>Kompozisyon vs Kalıtım:</strong> Bazen kalıtım yerine kompozisyon kullanmak daha uygun olabilir.</li>
            <li><strong>Liskov Substitution Prensibi:</strong> Alt sınıflar, üst sınıfların yerine kullanılabilmelidir.</li>
            <li><strong>DRY Prensibi:</strong> Kendini tekrar eden kodları ortak bir üst sınıfa taşıyın.</li>
            <li><strong>SOLID Prensipleri:</strong> Kalıtım hiyerarşisini tasarlarken SOLID prensiplerine uyun.</li>
          </ul>
        </div>
      </section>
    </div>
  );
} 