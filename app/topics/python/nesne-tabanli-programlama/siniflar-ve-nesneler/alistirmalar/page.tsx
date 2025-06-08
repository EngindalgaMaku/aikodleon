import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowRight, Code2, BookOpen, GraduationCap, Lightbulb } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python OOP Alıştırmalar | Kodleon',
  description: 'Python nesne tabanlı programlama alıştırmaları ve çözümleri.',
};

const content = `
# Python OOP Alıştırmalar

Bu bölümde, nesne tabanlı programlama kavramlarını pekiştirmeniz için çeşitli zorluk seviyelerinde alıştırmalar bulacaksınız.

## Başlangıç Seviyesi

### Alıştırma 1: Öğrenci Sınıfı
Öğrenci bilgilerini ve notlarını yöneten bir sınıf oluşturun.

**İstenenler:**
- İsim, numara ve not listesi özellikleri
- Not ekleme metodu
- Ortalama hesaplama metodu
- Durum kontrolü (geçti/kaldı) metodu

**Çözüm:**
\`\`\`python
class Ogrenci:
    def __init__(self, isim, numara):
        self.isim = isim
        self.numara = numara
        self.notlar = []
    
    def not_ekle(self, not_degeri):
        if 0 <= not_degeri <= 100:
            self.notlar.append(not_degeri)
            return True
        return False
    
    def ortalama_hesapla(self):
        if not self.notlar:
            return 0
        return sum(self.notlar) / len(self.notlar)
    
    def durum_kontrol(self):
        ort = self.ortalama_hesapla()
        return "Geçti" if ort >= 60 else "Kaldı"

# Test
ogrenci = Ogrenci("Ali", "101")
ogrenci.not_ekle(70)
ogrenci.not_ekle(85)
print(f"Ortalama: {ogrenci.ortalama_hesapla()}")
print(f"Durum: {ogrenci.durum_kontrol()}")
\`\`\`

### Alıştırma 2: Dikdörtgen Sınıfı
Dikdörtgenin alanını ve çevresini hesaplayan bir sınıf oluşturun.

**İstenenler:**
- En ve boy özellikleri
- Alan hesaplama metodu
- Çevre hesaplama metodu
- Kare kontrolü metodu

**Çözüm:**
\`\`\`python
class Dikdortgen:
    def __init__(self, en, boy):
        self.en = en
        self.boy = boy
    
    def alan_hesapla(self):
        return self.en * self.boy
    
    def cevre_hesapla(self):
        return 2 * (self.en + self.boy)
    
    def kare_mi(self):
        return self.en == self.boy
    
    def __str__(self):
        return f"Dikdörtgen(en={self.en}, boy={self.boy})"

# Test
d1 = Dikdortgen(5, 3)
print(f"Alan: {d1.alan_hesapla()}")
print(f"Çevre: {d1.cevre_hesapla()}")
print(f"Kare mi? {d1.kare_mi()}")
\`\`\`

## Orta Seviye

### Alıştırma 3: Banka Hesabı
Banka hesap işlemlerini yöneten bir sınıf oluşturun.

**İstenenler:**
- Hesap numarası ve bakiye özellikleri
- Para yatırma ve çekme metodları
- İşlem geçmişi tutma
- Hesap özeti görüntüleme

**Çözüm:**
\`\`\`python
from datetime import datetime

class BankaHesabi:
    def __init__(self, hesap_no, baslangic_bakiye=0):
        self.hesap_no = hesap_no
        self.__bakiye = baslangic_bakiye
        self.__islemler = []
    
    def para_yatir(self, miktar):
        if miktar > 0:
            self.__bakiye += miktar
            self.__islem_kaydet("Para Yatırma", miktar)
            return True
        return False
    
    def para_cek(self, miktar):
        if miktar > 0 and self.__bakiye >= miktar:
            self.__bakiye -= miktar
            self.__islem_kaydet("Para Çekme", miktar)
            return True
        return False
    
    def __islem_kaydet(self, islem_tipi, miktar):
        tarih = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.__islemler.append({
            "tarih": tarih,
            "tip": islem_tipi,
            "miktar": miktar,
            "bakiye": self.__bakiye
        })
    
    def hesap_ozeti(self):
        ozet = f"Hesap No: {self.hesap_no}\\n"
        ozet += f"Güncel Bakiye: {self.__bakiye} TL\\n\\n"
        ozet += "Son İşlemler:\\n"
        for islem in self.__islemler[-5:]:  # Son 5 işlem
            ozet += f"{islem['tarih']} - {islem['tip']}: {islem['miktar']} TL "
            ozet += f"(Bakiye: {islem['bakiye']} TL)\\n"
        return ozet

# Test
hesap = BankaHesabi("12345")
hesap.para_yatir(1000)
hesap.para_cek(500)
hesap.para_yatir(2000)
print(hesap.hesap_ozeti())
\`\`\`

### Alıştırma 4: Kütüphane Sistemi
Kitap ve üye yönetimi yapan bir kütüphane sistemi oluşturun.

**İstenenler:**
- Kitap ve Üye sınıfları
- Ödünç alma ve iade işlemleri
- Gecikme ücreti hesaplama
- Kitap ve üye durumu takibi

**Çözüm:**
\`\`\`python
from datetime import datetime, timedelta

class Kitap:
    def __init__(self, isbn, baslik, yazar):
        self.isbn = isbn
        self.baslik = baslik
        self.yazar = yazar
        self.durum = "Rafta"  # Rafta veya Ödünç
        self.odunc_alan = None
        self.iade_tarihi = None
    
    def __str__(self):
        return f"{self.baslik} ({self.yazar})"

class Uye:
    def __init__(self, id, ad, soyad):
        self.id = id
        self.ad = ad
        self.soyad = soyad
        self.odunc_kitaplar = []
        self.ceza = 0
    
    def __str__(self):
        return f"{self.ad} {self.soyad}"

class Kutuphane:
    def __init__(self):
        self.kitaplar = {}
        self.uyeler = {}
        self.gunluk_ceza = 1  # TL
    
    def kitap_ekle(self, kitap):
        self.kitaplar[kitap.isbn] = kitap
    
    def uye_ekle(self, uye):
        self.uyeler[uye.id] = uye
    
    def odunc_ver(self, isbn, uye_id, gun=14):
        kitap = self.kitaplar.get(isbn)
        uye = self.uyeler.get(uye_id)
        
        if not (kitap and uye):
            return False
        
        if kitap.durum != "Rafta":
            return False
        
        kitap.durum = "Ödünç"
        kitap.odunc_alan = uye
        kitap.iade_tarihi = datetime.now() + timedelta(days=gun)
        uye.odunc_kitaplar.append(kitap)
        return True
    
    def iade_al(self, isbn):
        kitap = self.kitaplar.get(isbn)
        if not kitap or kitap.durum == "Rafta":
            return False
        
        # Gecikme kontrolü
        if datetime.now() > kitap.iade_tarihi:
            geciken_gun = (datetime.now() - kitap.iade_tarihi).days
            ceza = geciken_gun * self.gunluk_ceza
            kitap.odunc_alan.ceza += ceza
        
        kitap.odunc_alan.odunc_kitaplar.remove(kitap)
        kitap.durum = "Rafta"
        kitap.odunc_alan = None
        kitap.iade_tarihi = None
        return True
    
    def durum_raporu(self):
        rapor = "Kütüphane Durum Raporu\\n"
        rapor += "======================\\n\\n"
        
        rapor += "Kitaplar:\\n"
        for kitap in self.kitaplar.values():
            rapor += f"- {kitap}: {kitap.durum}\\n"
        
        rapor += "\\nÜyeler:\\n"
        for uye in self.uyeler.values():
            rapor += f"- {uye} (Ceza: {uye.ceza} TL)\\n"
            if uye.odunc_kitaplar:
                rapor += "  Ödünç Kitaplar:\\n"
                for kitap in uye.odunc_kitaplar:
                    rapor += f"  * {kitap} (İade: {kitap.iade_tarihi})\\n"
        
        return rapor

# Test
kutuphane = Kutuphane()

# Kitap ve üye ekleme
k1 = Kitap("123", "Python Programlama", "Ahmet Yılmaz")
k2 = Kitap("456", "Veri Bilimi", "Ayşe Demir")
kutuphane.kitap_ekle(k1)
kutuphane.kitap_ekle(k2)

u1 = Uye("1", "Ali", "Kaya")
u2 = Uye("2", "Zeynep", "Yıldız")
kutuphane.uye_ekle(u1)
kutuphane.uye_ekle(u2)

# İşlemler
kutuphane.odunc_ver("123", "1")
kutuphane.odunc_ver("456", "2")
print(kutuphane.durum_raporu())
\`\`\`

## İleri Seviye

### Alıştırma 5: Oyun Karakterleri
Bir rol yapma oyunu için karakter sınıfları oluşturun.

**İstenenler:**
- Temel Karakter sınıfı
- Farklı karakter türleri (Savaşçı, Büyücü, Okçu)
- Özellik ve yetenek sistemi
- Savaş mekanizması
- Deneyim ve seviye sistemi

**Çözüm:**
\`\`\`python
from abc import ABC, abstractmethod
import random

class Karakter(ABC):
    def __init__(self, isim, seviye=1):
        self.isim = isim
        self.seviye = seviye
        self.deneyim = 0
        self.deneyim_limiti = 100
        
        # Temel özellikler
        self.can = self._hesapla_can()
        self.guc = self._hesapla_guc()
        self.savunma = self._hesapla_savunma()
        
        self.yetenekler = []
        self.maksimum_can = self.can
    
    @abstractmethod
    def _hesapla_can(self):
        pass
    
    @abstractmethod
    def _hesapla_guc(self):
        pass
    
    @abstractmethod
    def _hesapla_savunma(self):
        pass
    
    def seviye_atla(self):
        if self.deneyim >= self.deneyim_limiti:
            self.seviye += 1
            self.deneyim = 0
            self.deneyim_limiti *= 1.5
            
            # Özellikleri güncelle
            self.maksimum_can = self._hesapla_can()
            self.can = self.maksimum_can
            self.guc = self._hesapla_guc()
            self.savunma = self._hesapla_savunma()
            
            return True
        return False
    
    def saldir(self, hedef):
        hasar = max(1, self.guc - hedef.savunma)
        hedef.hasar_al(hasar)
        return hasar
    
    def hasar_al(self, hasar):
        self.can = max(0, self.can - hasar)
        return self.can > 0
    
    def iyiles(self, miktar):
        self.can = min(self.maksimum_can, self.can + miktar)
    
    def deneyim_kazan(self, miktar):
        self.deneyim += miktar
        if self.deneyim >= self.deneyim_limiti:
            self.seviye_atla()
    
    def __str__(self):
        return (
            f"{self.__class__.__name__} {self.isim} "
            f"(Seviye {self.seviye})\\n"
            f"Can: {self.can}/{self.maksimum_can}\\n"
            f"Güç: {self.guc}\\n"
            f"Savunma: {self.savunma}\\n"
            f"Deneyim: {self.deneyim}/{self.deneyim_limiti}"
        )

class Savasci(Karakter):
    def _hesapla_can(self):
        return 100 + (self.seviye * 10)
    
    def _hesapla_guc(self):
        return 10 + (self.seviye * 2)
    
    def _hesapla_savunma(self):
        return 5 + (self.seviye * 1.5)
    
    def guc_saldirisi(self, hedef):
        """Özel yetenek: Güçlü saldırı"""
        hasar = int(self.guc * 1.5)
        hedef.hasar_al(hasar)
        return hasar

class Buyucu(Karakter):
    def __init__(self, isim, seviye=1):
        super().__init__(isim, seviye)
        self.mana = self._hesapla_mana()
        self.maksimum_mana = self.mana
    
    def _hesapla_can(self):
        return 70 + (self.seviye * 7)
    
    def _hesapla_guc(self):
        return 15 + (self.seviye * 2.5)
    
    def _hesapla_savunma(self):
        return 3 + (self.seviye * 1)
    
    def _hesapla_mana(self):
        return 100 + (self.seviye * 10)
    
    def ates_topu(self, hedef):
        """Özel yetenek: Ateş topu"""
        if self.mana >= 30:
            hasar = int(self.guc * 2)
            hedef.hasar_al(hasar)
            self.mana -= 30
            return hasar
        return 0

class Okcu(Karakter):
    def __init__(self, isim, seviye=1):
        super().__init__(isim, seviye)
        self.ok = 20
    
    def _hesapla_can(self):
        return 80 + (self.seviye * 8)
    
    def _hesapla_guc(self):
        return 12 + (self.seviye * 2.2)
    
    def _hesapla_savunma(self):
        return 4 + (self.seviye * 1.2)
    
    def ok_yagmuru(self, hedefler):
        """Özel yetenek: Ok yağmuru - birden fazla hedefe saldırı"""
        if self.ok >= 5:
            hasar = int(self.guc * 0.8)
            for hedef in hedefler:
                hedef.hasar_al(hasar)
            self.ok -= 5
            return hasar * len(hedefler)
        return 0

# Test
def savas_simulasyonu():
    savasci = Savasci("Aragorn", 1)
    buyucu = Buyucu("Gandalf", 1)
    okcu = Okcu("Legolas", 1)
    
    print("Savaş Başlıyor!\\n")
    print(savasci)
    print("\\n" + "="*30 + "\\n")
    print(buyucu)
    print("\\n" + "="*30 + "\\n")
    print(okcu)
    print("\\n" + "="*30 + "\\n")
    
    # Birkaç tur savaş
    for tur in range(1, 4):
        print(f"\\nTur {tur}:")
        
        # Savaşçı saldırısı
        hasar = savasci.guc_saldirisi(buyucu)
        print(f"{savasci.isim} -> {buyucu.isim}: {hasar} hasar!")
        
        # Büyücü saldırısı
        if buyucu.can > 0:
            hasar = buyucu.ates_topu(savasci)
            print(f"{buyucu.isim} -> {savasci.isim}: {hasar} hasar!")
        
        # Okçu saldırısı
        hasar = okcu.ok_yagmuru([savasci, buyucu])
        print(f"{okcu.isim} ok yağmuru: Toplam {hasar} hasar!")
        
        # Durum raporu
        print("\\nDurum:")
        print(f"{savasci.isim}: {savasci.can}/{savasci.maksimum_can} HP")
        print(f"{buyucu.isim}: {buyucu.can}/{buyucu.maksimum_can} HP")
        print(f"{okcu.isim}: {okcu.can}/{okcu.maksimum_can} HP")

# Simulasyonu çalıştır
savas_simulasyonu()
\`\`\`

## Bonus: Test Senaryoları

Her alıştırma için kapsamlı test senaryoları yazmanız önerilir. Örnek bir test senaryosu:

\`\`\`python
import unittest

class OgrenciTests(unittest.TestCase):
    def setUp(self):
        self.ogrenci = Ogrenci("Test Öğrenci", "123")
    
    def test_not_ekleme(self):
        self.assertTrue(self.ogrenci.not_ekle(85))
        self.assertFalse(self.ogrenci.not_ekle(-10))
        self.assertFalse(self.ogrenci.not_ekle(110))
    
    def test_ortalama_hesaplama(self):
        self.ogrenci.not_ekle(80)
        self.ogrenci.not_ekle(90)
        self.assertEqual(self.ogrenci.ortalama_hesapla(), 85)
    
    def test_durum_kontrol(self):
        self.ogrenci.not_ekle(55)
        self.assertEqual(self.ogrenci.durum_kontrol(), "Kaldı")
        self.ogrenci.not_ekle(65)
        self.assertEqual(self.ogrenci.durum_kontrol(), "Geçti")

if __name__ == '__main__':
    unittest.main()
\`\`\`
`;

const sections = [
  {
    title: "Başlangıç Seviyesi",
    description: "Temel OOP kavramlarını pekiştiren alıştırmalar",
    icon: <Code2 className="h-6 w-6" />,
    topics: [
      "Öğrenci Sınıfı",
      "Dikdörtgen Sınıfı",
      "Basit Hesap Makinesi",
      "Araba Sınıfı"
    ]
  },
  {
    title: "Orta Seviye",
    description: "Daha karmaşık OOP uygulamaları",
    icon: <BookOpen className="h-6 w-6" />,
    topics: [
      "Banka Hesabı",
      "Kütüphane Sistemi",
      "E-Ticaret Ürünleri",
      "Personel Yönetimi"
    ]
  },
  {
    title: "İleri Seviye",
    description: "Gelişmiş OOP konseptleri",
    icon: <GraduationCap className="h-6 w-6" />,
    topics: [
      "Oyun Karakterleri",
      "Veritabanı Sistemi",
      "GUI Uygulaması",
      "API İstemcisi"
    ]
  },
  {
    title: "Test Senaryoları",
    description: "Kod test etme örnekleri",
    icon: <Lightbulb className="h-6 w-6" />,
    topics: [
      "Unit Testler",
      "Test Senaryoları",
      "Edge Cases",
      "Test Coverage"
    ]
  }
];

export default function AlistirmalarPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Concept Cards */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Alıştırma Kategorileri</h2>
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
            <Link href="/topics/python/nesne-tabanli-programlama/siniflar-ve-nesneler">
              Sınıflar ve Nesneler Sayfasına Dön
              <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 