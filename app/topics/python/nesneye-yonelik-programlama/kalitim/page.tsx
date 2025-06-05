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
import MarkdownContent from '@/components/MarkdownContent';
import { content } from './content';

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

export default function Page() {
  return (
    <div className="max-w-5xl mx-auto py-8">
      <MarkdownContent content={content} />
      
      <section className="mt-8">
        <h2 className="text-2xl font-semibold mb-4">Kod Örnekleri</h2>
        
        <div className="space-y-8">
          <div>
            <h3 className="text-xl font-semibold mb-2">Temel Kalıtım Örneği</h3>
            <CodeRunner initialCode={temelKalitimCode} />
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Çoklu Kalıtım Örneği</h3>
            <CodeRunner initialCode={cokluKalitimCode} />
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-2">Soyut Sınıf Örneği</h3>
            <CodeRunner initialCode={abstractClassCode} />
          </div>
        </div>
      </section>
    </div>
  );
} 