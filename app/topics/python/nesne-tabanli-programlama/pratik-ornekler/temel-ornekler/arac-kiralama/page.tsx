import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowRight, Car, Calendar, CreditCard, FileText } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python OOP Araç Kiralama Sistemi Örneği | AIKOD',
  description: 'Python nesne tabanlı programlama ile araç kiralama sistemi uygulaması örneği ve detaylı açıklamalar.',
};

const content = `
# Araç Kiralama Sistemi Örneği

Bu örnek, nesne tabanlı programlama kavramlarını kullanarak bir araç kiralama sistemi oluşturur. Sistem, araç yönetimi, müşteri kayıtları, kiralama işlemleri ve faturalandırma gibi temel özellikleri içerir.

## Sistem Bileşenleri

### 1. Araç Sınıfı
- Araç bilgilerini tutar
- Kiralama durumu takibi
- Bakım ve servis kayıtları

### 2. Müşteri Sınıfı
- Müşteri bilgilerini tutar
- Kiralama geçmişi
- Ödeme bilgileri

### 3. Kiralama Sınıfı
- Kiralama detayları
- Fiyat hesaplama
- Tarih kontrolü

### 4. Fatura Sınıfı
- Fatura oluşturma
- Ödeme takibi
- Raporlama

## Kod Örneği

\`\`\`python
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

class AracTipi(Enum):
    EKONOMIK = "Ekonomik"
    ORTA = "Orta"
    LUKS = "Lüks"
    SUV = "SUV"
    TICARI = "Ticari"

class AracDurumu(Enum):
    MUSAIT = "Müsait"
    KIRADA = "Kirada"
    SERVISTE = "Serviste"
    REZERVE = "Rezerve"

@dataclass
class Adres:
    sokak: str
    sehir: str
    ulke: str
    posta_kodu: str

class OdemeYontemi(ABC):
    @abstractmethod
    def odeme_yap(self, tutar: float) -> bool:
        pass

class KrediKarti(OdemeYontemi):
    def __init__(self, kart_no: str, son_kullanma: str, cvv: str):
        self.kart_no = kart_no
        self.son_kullanma = son_kullanma
        self.cvv = cvv
    
    def odeme_yap(self, tutar: float) -> bool:
        # Gerçek uygulamada payment gateway entegrasyonu
        print(f"Kredi kartı ile {tutar}₺ ödeme yapıldı")
        return True

class Arac:
    def __init__(self, plaka: str, marka: str, model: str, yil: int, tip: AracTipi):
        self.plaka = plaka
        self.marka = marka
        self.model = model
        self.yil = yil
        self.tip = tip
        self.durum = AracDurumu.MUSAIT
        self._gunluk_ucret = self._hesapla_gunluk_ucret()
        self._bakim_gecmisi: List[str] = []
    
    def _hesapla_gunluk_ucret(self) -> float:
        baz_ucret = 500.0
        carpanlar = {
            AracTipi.EKONOMIK: 1.0,
            AracTipi.ORTA: 1.5,
            AracTipi.LUKS: 2.5,
            AracTipi.SUV: 2.0,
            AracTipi.TICARI: 1.8
        }
        return baz_ucret * carpanlar[self.tip]
    
    def bakim_ekle(self, aciklama: str) -> None:
        tarih = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._bakim_gecmisi.append(f"{tarih}: {aciklama}")
        self.durum = AracDurumu.SERVISTE
    
    def bakimdan_cik(self) -> None:
        self.durum = AracDurumu.MUSAIT
    
    def __str__(self) -> str:
        return f"{self.yil} {self.marka} {self.model} ({self.tip.value})"

class Musteri:
    def __init__(self, tc_no: str, ad: str, soyad: str, telefon: str, adres: Adres):
        self.tc_no = tc_no
        self.ad = ad
        self.soyad = soyad
        self.telefon = telefon
        self.adres = adres
        self._kiralama_gecmisi: List["Kiralama"] = []
        self._odeme_yontemleri: List[OdemeYontemi] = []
    
    def odeme_yontemi_ekle(self, yontem: OdemeYontemi) -> None:
        self._odeme_yontemleri.append(yontem)
    
    def kiralama_ekle(self, kiralama: "Kiralama") -> None:
        self._kiralama_gecmisi.append(kiralama)
    
    def __str__(self) -> str:
        return f"{self.ad} {self.soyad} ({self.tc_no})"

class Kiralama:
    def __init__(self, arac: Arac, musteri: Musteri, baslangic: datetime, bitis: datetime):
        self.arac = arac
        self.musteri = musteri
        self.baslangic = baslangic
        self.bitis = bitis
        self.toplam_ucret = self._hesapla_ucret()
        self.odendi = False
        self._fatura: Optional[Fatura] = None
        
        if arac.durum != AracDurumu.MUSAIT:
            raise ValueError("Araç müsait değil")
        
        arac.durum = AracDurumu.KIRADA
        musteri.kiralama_ekle(self)
    
    def _hesapla_ucret(self) -> float:
        gun_sayisi = (self.bitis - self.baslangic).days
        return self.arac._gunluk_ucret * gun_sayisi
    
    def fatura_olustur(self) -> "Fatura":
        if not self._fatura:
            self._fatura = Fatura(self)
        return self._fatura
    
    def __str__(self) -> str:
        return f"{self.arac} - {self.baslangic.date()} -> {self.bitis.date()}"

class Fatura:
    def __init__(self, kiralama: Kiralama):
        self.kiralama = kiralama
        self.olusturma_tarihi = datetime.now()
        self.son_odeme_tarihi = self.olusturma_tarihi + timedelta(days=7)
        self.fatura_no = self._fatura_no_uret()
    
    def _fatura_no_uret(self) -> str:
        tarih_str = self.olusturma_tarihi.strftime("%Y%m%d")
        return f"FTR{tarih_str}{hash(self.kiralama) % 1000:03d}"
    
    def odeme_al(self, odeme_yontemi: OdemeYontemi) -> bool:
        if odeme_yontemi.odeme_yap(self.kiralama.toplam_ucret):
            self.kiralama.odendi = True
            self.kiralama.arac.durum = AracDurumu.MUSAIT
            return True
        return False
    
    def fatura_detayi(self) -> str:
        return f"""
FATURA
======
Fatura No: {self.fatura_no}
Tarih: {self.olusturma_tarihi.strftime("%d/%m/%Y")}
Son Ödeme: {self.son_odeme_tarihi.strftime("%d/%m/%Y")}

MÜŞTERİ BİLGİLERİ
----------------
{self.kiralama.musteri}
{self.kiralama.musteri.adres.sokak}
{self.kiralama.musteri.adres.sehir}, {self.kiralama.musteri.adres.posta_kodu}
{self.kiralama.musteri.adres.ulke}

KİRALAMA DETAYLARI
----------------
Araç: {self.kiralama.arac}
Başlangıç: {self.kiralama.baslangic.strftime("%d/%m/%Y")}
Bitiş: {self.kiralama.bitis.strftime("%d/%m/%Y")}
Gün Sayısı: {(self.kiralama.bitis - self.kiralama.baslangic).days}
Günlük Ücret: {self.kiralama.arac._gunluk_ucret:.2f}₺

TOPLAM: {self.kiralama.toplam_ucret:.2f}₺
"""

class AracKiralamaServisi:
    def __init__(self):
        self._araclar: List[Arac] = []
        self._musteriler: Dict[str, Musteri] = {}  # {tc_no: musteri}
        self._kiralamalar: List[Kiralama] = []
    
    def arac_ekle(self, arac: Arac) -> None:
        self._araclar.append(arac)
    
    def musteri_ekle(self, musteri: Musteri) -> None:
        self._musteriler[musteri.tc_no] = musteri
    
    def musait_araclar(self, tip: Optional[AracTipi] = None) -> List[Arac]:
        araclar = [a for a in self._araclar if a.durum == AracDurumu.MUSAIT]
        if tip:
            araclar = [a for a in araclar if a.tip == tip]
        return araclar
    
    def arac_kirala(self, tc_no: str, plaka: str, 
                    baslangic: datetime, bitis: datetime) -> Kiralama:
        musteri = self._musteriler.get(tc_no)
        if not musteri:
            raise ValueError("Müşteri bulunamadı")
        
        arac = next((a for a in self._araclar if a.plaka == plaka), None)
        if not arac:
            raise ValueError("Araç bulunamadı")
        
        kiralama = Kiralama(arac, musteri, baslangic, bitis)
        self._kiralamalar.append(kiralama)
        return kiralama
    
    def rapor_olustur(self) -> str:
        aktif_kiralamalar = [k for k in self._kiralamalar if not k.odendi]
        tamamlanan_kiralamalar = [k for k in self._kiralamalar if k.odendi]
        
        return f"""
ARAÇ KİRALAMA SİSTEMİ RAPORU
============================
Toplam Araç Sayısı: {len(self._araclar)}
Müsait Araç Sayısı: {len(self.musait_araclar())}
Toplam Müşteri Sayısı: {len(self._musteriler)}

Aktif Kiralamalar: {len(aktif_kiralamalar)}
Tamamlanan Kiralamalar: {len(tamamlanan_kiralamalar)}

Toplam Ciro: {sum(k.toplam_ucret for k in tamamlanan_kiralamalar):.2f}₺
"""

# Kullanım örneği
def main():
    # Servis oluştur
    servis = AracKiralamaServisi()
    
    # Araçlar ekle
    arac1 = Arac("34ABC123", "Toyota", "Corolla", 2022, AracTipi.EKONOMIK)
    arac2 = Arac("34XYZ789", "BMW", "X5", 2023, AracTipi.LUKS)
    servis.arac_ekle(arac1)
    servis.arac_ekle(arac2)
    
    # Müşteri oluştur
    adres = Adres("Atatürk Cad. No:123", "İstanbul", "Türkiye", "34100")
    musteri = Musteri("12345678901", "Mehmet", "Yılmaz", "5551234567", adres)
    kart = KrediKarti("1234-5678-9012-3456", "12/25", "123")
    musteri.odeme_yontemi_ekle(kart)
    servis.musteri_ekle(musteri)
    
    # Kiralama yap
    baslangic = datetime.now()
    bitis = baslangic + timedelta(days=3)
    kiralama = servis.arac_kirala(musteri.tc_no, arac1.plaka, baslangic, bitis)
    
    # Fatura oluştur ve öde
    fatura = kiralama.fatura_olustur()
    print(fatura.fatura_detayi())
    fatura.odeme_al(kart)
    
    # Sistem raporu
    print(servis.rapor_olustur())

if __name__ == "__main__":
    main()
\`\`\`

## Kullanılan OOP Kavramları

1. **Sınıf Hiyerarşisi**
   - Temel sınıflar: Arac, Musteri, Kiralama, Fatura
   - Yardımcı sınıflar: Adres, OdemeYontemi
   - Enum kullanımı: AracTipi, AracDurumu

2. **Soyut Sınıflar ve Arayüzler**
   - OdemeYontemi abstract sınıfı
   - KrediKarti concrete implementasyonu
   - Genişletilebilir ödeme sistemi

3. **Kapsülleme**
   - Protected özellikler (\`_gunluk_ucret\`, \`_bakim_gecmisi\`, vb.)
   - Private metotlar (\`_hesapla_ucret\`, \`_fatura_no_uret\`)
   - Veri doğrulama ve kontrol

4. **Kompozisyon**
   - AracKiralamaServisi -> Arac, Musteri, Kiralama
   - Kiralama -> Arac, Musteri
   - Musteri -> Adres, OdemeYontemi

5. **Type Hints ve Dataclass**
   - Type hints kullanımı
   - Optional ve List tipleri
   - Adres için dataclass

## Geliştirme Önerileri

1. Veritabanı entegrasyonu
2. Web arayüzü
3. Rezervasyon sistemi
4. Araç takip sistemi (GPS)
5. Mobil uygulama
`;

const sections = [
  {
    title: "Araç Yönetimi",
    description: "Araç kayıt ve takip sistemi",
    icon: <Car className="h-6 w-6" />,
    topics: [
      "Araç kaydı",
      "Durum takibi",
      "Bakım planı",
      "Filo yönetimi"
    ]
  },
  {
    title: "Kiralama İşlemleri",
    description: "Kiralama ve rezervasyon yönetimi",
    icon: <Calendar className="h-6 w-6" />,
    topics: [
      "Müsaitlik kontrolü",
      "Rezervasyon",
      "Teslim alma",
      "Teslim etme"
    ]
  },
  {
    title: "Ödeme Sistemi",
    description: "Ödeme ve faturalandırma",
    icon: <CreditCard className="h-6 w-6" />,
    topics: [
      "Fiyatlandırma",
      "Ödeme alma",
      "Fatura oluşturma",
      "Ödeme geçmişi"
    ]
  },
  {
    title: "Raporlama",
    description: "Sistem raporları ve analizler",
    icon: <FileText className="h-6 w-6" />,
    topics: [
      "Gelir raporu",
      "Doluluk oranı",
      "Müşteri analizi",
      "Araç performansı"
    ]
  }
];

export default function AracKiralamaPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Feature Cards */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Sistem Özellikleri</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {sections.map((section, index) => (
              <Card key={index} className="bg-yellow-50 hover:bg-yellow-100 dark:bg-yellow-950/50 dark:hover:bg-yellow-950/70 transition-all duration-300">
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg text-yellow-600 dark:text-yellow-400">
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

        {/* Back to Basic Examples Button */}
        <div className="mt-12 flex justify-end">
          <Button asChild variant="outline" className="group">
            <Link href="/topics/python/nesne-tabanli-programlama/pratik-ornekler/temel-ornekler">
              Temel Örneklere Dön
              <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 