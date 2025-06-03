import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Info, Lightbulb, AlertTriangle } from "lucide-react";
import Image from "next/image";

export const metadata: Metadata = {
  title: 'Python OOP: Pratik Örnekler ve Projeler | Kodleon',
  description: 'Python\'da nesne yönelimli programlama konularını pekiştirmek için pratik örnekler ve gerçek dünya projeleri.',
};

const content = `
# Python OOP: Pratik Örnekler ve Projeler

Bu bölümde, öğrendiğimiz OOP kavramlarını gerçek dünya problemlerine nasıl uygulayacağımızı göreceğiz.

## 1. Banka Hesap Yönetimi

Basit bir banka hesap yönetim sistemi:

\`\`\`python
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict

class Hesap(ABC):
    def __init__(self, hesap_no: str, sahip: str, bakiye: float = 0):
        self._hesap_no = hesap_no
        self._sahip = sahip
        self._bakiye = bakiye
        self._islemler: List[Dict] = []
    
    @property
    def bakiye(self) -> float:
        return self._bakiye
    
    def islem_ekle(self, islem_tipi: str, miktar: float):
        self._islemler.append({
            'tarih': datetime.now(),
            'tip': islem_tipi,
            'miktar': miktar,
            'bakiye': self._bakiye
        })
    
    @abstractmethod
    def para_yatir(self, miktar: float) -> bool:
        pass
    
    @abstractmethod
    def para_cek(self, miktar: float) -> bool:
        pass
    
    def ozet_goruntule(self) -> str:
        return f"""
        Hesap No: {self._hesap_no}
        Sahip: {self._sahip}
        Bakiye: {self._bakiye:.2f}TL
        """
    
    def hesap_ozeti(self) -> List[Dict]:
        return self._islemler

class VadesizHesap(Hesap):
    def para_yatir(self, miktar: float) -> bool:
        if miktar > 0:
            self._bakiye += miktar
            self.islem_ekle("Para Yatırma", miktar)
            return True
        return False
    
    def para_cek(self, miktar: float) -> bool:
        if 0 < miktar <= self._bakiye:
            self._bakiye -= miktar
            self.islem_ekle("Para Çekme", -miktar)
            return True
        return False

class VadeliHesap(Hesap):
    def __init__(self, hesap_no: str, sahip: str, bakiye: float = 0, 
                 vade_suresi: int = 30):
        super().__init__(hesap_no, sahip, bakiye)
        self.vade_suresi = vade_suresi
        self.vade_baslangic = datetime.now()
    
    def para_yatir(self, miktar: float) -> bool:
        if miktar >= 1000:  # Minimum vade miktarı
            self._bakiye += miktar
            self.islem_ekle("Vadeli Para Yatırma", miktar)
            return True
        return False
    
    def para_cek(self, miktar: float) -> bool:
        vade_farki = (datetime.now() - self.vade_baslangic).days
        
        if vade_farki < self.vade_suresi:
            print(f"Vade süresinin dolmasına {self.vade_suresi - vade_farki} gün var")
            return False
        
        if 0 < miktar <= self._bakiye:
            self._bakiye -= miktar
            self.islem_ekle("Vadeli Para Çekme", -miktar)
            return True
        return False

class KrediHesabi(Hesap):
    def __init__(self, hesap_no: str, sahip: str, kredi_limiti: float):
        super().__init__(hesap_no, sahip)
        self.kredi_limiti = kredi_limiti
    
    def para_yatir(self, miktar: float) -> bool:
        if miktar > 0:
            self._bakiye += miktar
            self.islem_ekle("Kredi Ödemesi", miktar)
            return True
        return False
    
    def para_cek(self, miktar: float) -> bool:
        if miktar > 0 and self._bakiye - miktar >= -self.kredi_limiti:
            self._bakiye -= miktar
            self.islem_ekle("Kredi Kullanımı", -miktar)
            return True
        return False

# Kullanım Örneği
def banka_islemleri():
    # Hesaplar oluştur
    vadesiz = VadesizHesap("123", "Ahmet Yılmaz", 1000)
    vadeli = VadeliHesap("456", "Mehmet Demir", 5000)
    kredi = KrediHesabi("789", "Ayşe Kaya", 10000)
    
    # İşlemler
    print("Vadesiz Hesap İşlemleri:")
    vadesiz.para_yatir(500)
    vadesiz.para_cek(200)
    print(vadesiz.ozet_goruntule())
    
    print("\nVadeli Hesap İşlemleri:")
    vadeli.para_yatir(2000)
    vadeli.para_cek(1000)  # Vade dolmadığı için başarısız olacak
    print(vadeli.ozet_goruntule())
    
    print("\nKredi Hesabı İşlemleri:")
    kredi.para_cek(5000)  # Kredi kullan
    kredi.para_yatir(1000)  # Ödeme yap
    print(kredi.ozet_goruntule())
    
    # Hesap özetlerini görüntüle
    print("\nVadesiz Hesap Hareketleri:")
    for islem in vadesiz.hesap_ozeti():
        print(f"{islem['tarih']}: {islem['tip']} - {islem['miktar']}TL")

if __name__ == "__main__":
    banka_islemleri()
\`\`\`

## 2. E-Ticaret Sistemi

Basit bir e-ticaret sistemi tasarımı:

\`\`\`python
from abc import ABC, abstractmethod
from typing import List, Dict
from datetime import datetime
from enum import Enum

class UrunKategorisi(Enum):
    ELEKTRONIK = "Elektronik"
    GIYIM = "Giyim"
    KITAP = "Kitap"
    GIDA = "Gıda"

class Urun:
    def __init__(self, id: int, ad: str, fiyat: float, kategori: UrunKategorisi):
        self.id = id
        self.ad = ad
        self._fiyat = fiyat
        self.kategori = kategori
        self._stok = 0
    
    @property
    def fiyat(self) -> float:
        return self._fiyat
    
    @fiyat.setter
    def fiyat(self, yeni_fiyat: float):
        if yeni_fiyat >= 0:
            self._fiyat = yeni_fiyat
    
    @property
    def stok(self) -> int:
        return self._stok
    
    def stok_ekle(self, miktar: int):
        if miktar > 0:
            self._stok += miktar
            return True
        return False
    
    def stok_dus(self, miktar: int):
        if 0 < miktar <= self._stok:
            self._stok -= miktar
            return True
        return False

class Sepet:
    def __init__(self):
        self.urunler: Dict[Urun, int] = {}  # Ürün: Miktar
    
    def urun_ekle(self, urun: Urun, miktar: int = 1):
        if urun.stok >= miktar:
            if urun in self.urunler:
                self.urunler[urun] += miktar
            else:
                self.urunler[urun] = miktar
            return True
        return False
    
    def urun_cikar(self, urun: Urun, miktar: int = 1):
        if urun in self.urunler:
            if self.urunler[urun] <= miktar:
                del self.urunler[urun]
            else:
                self.urunler[urun] -= miktar
            return True
        return False
    
    def toplam_fiyat(self) -> float:
        return sum(urun.fiyat * miktar for urun, miktar in self.urunler.items())
    
    def sepeti_temizle(self):
        self.urunler.clear()

class OdemeStratejisi(ABC):
    @abstractmethod
    def ode(self, miktar: float) -> bool:
        pass

class KrediKartiOdeme(OdemeStratejisi):
    def __init__(self, kart_no: str, son_kullanma: str, cvv: str):
        self.kart_no = kart_no
        self.son_kullanma = son_kullanma
        self.cvv = cvv
    
    def ode(self, miktar: float) -> bool:
        # Gerçek uygulamada burada ödeme işlemi yapılır
        print(f"Kredi kartı ile {miktar}TL ödeme yapıldı")
        return True

class HavaleOdeme(OdemeStratejisi):
    def __init__(self, iban: str):
        self.iban = iban
    
    def ode(self, miktar: float) -> bool:
        # Gerçek uygulamada burada havale işlemi yapılır
        print(f"Havale ile {miktar}TL ödeme yapıldı")
        return True

class Siparis:
    def __init__(self, musteri_id: int, sepet: Sepet):
        self.siparis_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.musteri_id = musteri_id
        self.urunler = dict(sepet.urunler)
        self.toplam_tutar = sepet.toplam_fiyat()
        self.tarih = datetime.now()
        self.durum = "Beklemede"
    
    def odeme_yap(self, odeme: OdemeStratejisi) -> bool:
        if odeme.ode(self.toplam_tutar):
            self.durum = "Ödendi"
            # Stok güncelleme
            for urun, miktar in self.urunler.items():
                urun.stok_dus(miktar)
            return True
        return False
    
    def siparis_ozeti(self) -> str:
        ozet = f"""
        Sipariş ID: {self.siparis_id}
        Müşteri ID: {self.musteri_id}
        Tarih: {self.tarih}
        Durum: {self.durum}
        
        Ürünler:
        """
        for urun, miktar in self.urunler.items():
            ozet += f"- {urun.ad} x {miktar}: {urun.fiyat * miktar}TL\n"
        ozet += f"\nToplam Tutar: {self.toplam_tutar}TL"
        return ozet

# Kullanım Örneği
def e_ticaret_islemleri():
    # Ürünler oluştur
    laptop = Urun(1, "Laptop", 15000, UrunKategorisi.ELEKTRONIK)
    telefon = Urun(2, "Telefon", 8000, UrunKategorisi.ELEKTRONIK)
    kitap = Urun(3, "Python Kitabı", 100, UrunKategorisi.KITAP)
    
    # Stok ekle
    laptop.stok_ekle(5)
    telefon.stok_ekle(10)
    kitap.stok_ekle(20)
    
    # Sepet oluştur ve ürün ekle
    sepet = Sepet()
    sepet.urun_ekle(laptop)
    sepet.urun_ekle(telefon, 2)
    sepet.urun_ekle(kitap, 3)
    
    # Sipariş oluştur
    siparis = Siparis(1, sepet)
    print(siparis.siparis_ozeti())
    
    # Ödeme yap
    kredi_karti = KrediKartiOdeme("1234-5678", "12/24", "123")
    if siparis.odeme_yap(kredi_karti):
        print("\nÖdeme başarılı!")
        print("\nGüncel stok durumu:")
        print(f"Laptop: {laptop.stok}")
        print(f"Telefon: {telefon.stok}")
        print(f"Kitap: {kitap.stok}")

if __name__ == "__main__":
    e_ticaret_islemleri()
\`\`\`

## 3. Blog Sistemi

Basit bir blog yönetim sistemi:

\`\`\`python
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional
from enum import Enum

class IcerikDurumu(Enum):
    TASLAK = "Taslak"
    INCELENIYOR = "İnceleniyor"
    YAYINDA = "Yayında"
    ARSIVLENDI = "Arşivlendi"

class Kullanici:
    def __init__(self, id: int, ad: str, email: str):
        self.id = id
        self.ad = ad
        self.email = email
        self._sifre = None
    
    @property
    def sifre(self):
        raise AttributeError("Şifre görüntülenemez")
    
    @sifre.setter
    def sifre(self, yeni_sifre: str):
        if len(yeni_sifre) >= 8:
            # Gerçek uygulamada şifre hash'lenir
            self._sifre = yeni_sifre
        else:
            raise ValueError("Şifre en az 8 karakter olmalıdır")

class Yorum:
    def __init__(self, yazar: Kullanici, icerik: str):
        self.yazar = yazar
        self.icerik = icerik
        self.tarih = datetime.now()
        self.duzenlenme_tarihi: Optional[datetime] = None
    
    def duzenle(self, yeni_icerik: str):
        self.icerik = yeni_icerik
        self.duzenlenme_tarihi = datetime.now()

class Icerik(ABC):
    def __init__(self, baslik: str, yazar: Kullanici):
        self.baslik = baslik
        self.yazar = yazar
        self.olusturma_tarihi = datetime.now()
        self.duzenlenme_tarihi: Optional[datetime] = None
        self.durum = IcerikDurumu.TASLAK
        self.yorumlar: List[Yorum] = []
    
    def yorum_ekle(self, yorum: Yorum):
        self.yorumlar.append(yorum)
    
    def yorum_sil(self, yorum: Yorum):
        if yorum in self.yorumlar:
            self.yorumlar.remove(yorum)
    
    @abstractmethod
    def onizleme(self) -> str:
        pass

class BlogYazisi(Icerik):
    def __init__(self, baslik: str, yazar: Kullanici, icerik: str):
        super().__init__(baslik, yazar)
        self.icerik = icerik
        self.etiketler: List[str] = []
    
    def etiket_ekle(self, etiket: str):
        if etiket not in self.etiketler:
            self.etiketler.append(etiket)
    
    def etiket_sil(self, etiket: str):
        if etiket in self.etiketler:
            self.etiketler.remove(etiket)
    
    def onizleme(self) -> str:
        # İlk 100 karakter
        return self.icerik[:100] + "..."

class Sayfa(Icerik):
    def __init__(self, baslik: str, yazar: Kullanici, icerik: str):
        super().__init__(baslik, yazar)
        self.icerik = icerik
        self.menu_sirasi: Optional[int] = None
    
    def onizleme(self) -> str:
        return f"Sayfa: {self.baslik}"

class Blog:
    def __init__(self, ad: str, aciklama: str):
        self.ad = ad
        self.aciklama = aciklama
        self.yazilar: List[BlogYazisi] = []
        self.sayfalar: List[Sayfa] = []
        self.yazarlar: List[Kullanici] = []
    
    def yazi_ekle(self, yazi: BlogYazisi):
        self.yazilar.append(yazi)
    
    def sayfa_ekle(self, sayfa: Sayfa):
        self.sayfalar.append(sayfa)
    
    def yazar_ekle(self, yazar: Kullanici):
        if yazar not in self.yazarlar:
            self.yazarlar.append(yazar)
    
    def etiket_ara(self, etiket: str) -> List[BlogYazisi]:
        return [yazi for yazi in self.yazilar if etiket in yazi.etiketler]
    
    def yazar_yazilari(self, yazar: Kullanici) -> List[BlogYazisi]:
        return [yazi for yazi in self.yazilar if yazi.yazar == yazar]

# Kullanım Örneği
def blog_islemleri():
    # Blog oluştur
    blog = Blog("Python Blog", "Python programlama hakkında her şey")
    
    # Kullanıcılar oluştur
    yazar1 = Kullanici(1, "Ahmet", "ahmet@example.com")
    yazar2 = Kullanici(2, "Ayşe", "ayse@example.com")
    
    # Yazarları ekle
    blog.yazar_ekle(yazar1)
    blog.yazar_ekle(yazar2)
    
    # Blog yazıları oluştur
    yazi1 = BlogYazisi(
        "Python'da OOP",
        yazar1,
        "Python'da nesne yönelimli programlama temel kavramları..."
    )
    yazi1.etiket_ekle("python")
    yazi1.etiket_ekle("oop")
    yazi1.durum = IcerikDurumu.YAYINDA
    
    yazi2 = BlogYazisi(
        "Python ile Web Geliştirme",
        yazar2,
        "Django ve Flask kullanarak web uygulamaları geliştirme..."
    )
    yazi2.etiket_ekle("python")
    yazi2.etiket_ekle("web")
    yazi2.durum = IcerikDurumu.YAYINDA
    
    # Yazıları bloga ekle
    blog.yazi_ekle(yazi1)
    blog.yazi_ekle(yazi2)
    
    # Yorum ekle
    yorum = Yorum(yazar2, "Harika bir yazı olmuş!")
    yazi1.yorum_ekle(yorum)
    
    # Blog içeriğini görüntüle
    print(f"Blog: {blog.ad}")
    print(f"Açıklama: {blog.aciklama}")
    print("\nYazılar:")
    for yazi in blog.yazilar:
        print(f"\nBaşlık: {yazi.baslik}")
        print(f"Yazar: {yazi.yazar.ad}")
        print(f"Durum: {yazi.durum.value}")
        print(f"Etiketler: {', '.join(yazi.etiketler)}")
        print("Önizleme:", yazi.onizleme())
        if yazi.yorumlar:
            print("\nYorumlar:")
            for yorum in yazi.yorumlar:
                print(f"- {yorum.yazar.ad}: {yorum.icerik}")
    
    # Etiket arama
    print("\nPython etiketli yazılar:")
    python_yazilar = blog.etiket_ara("python")
    for yazi in python_yazilar:
        print(f"- {yazi.baslik}")

if __name__ == "__main__":
    blog_islemleri()
\`\`\`

Bu örnekler, OOP kavramlarının gerçek dünya problemlerine nasıl uygulanabileceğini göstermektedir. Her örnek, farklı OOP prensiplerini ve tasarım kalıplarını kullanmaktadır:

1. **Banka Hesap Yönetimi**
   - Soyut sınıflar ve kalıtım
   - Kapsülleme ve property kullanımı
   - Polimorfizm

2. **E-Ticaret Sistemi**
   - Strateji tasarım kalıbı (Ödeme stratejileri)
   - Kompozisyon (Sepet ve Ürün ilişkisi)
   - Enum kullanımı

3. **Blog Sistemi**
   - Soyut sınıflar ve arayüzler
   - Kalıtım ve polimorfizm
   - İlişkisel yapılar

Bu projeleri kendi ihtiyaçlarınıza göre genişletebilir ve özelleştirebilirsiniz.
`;

export default function PracticalExamplesPage() {
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
      
      <div className="mt-16 text-center text-sm text-muted-foreground">
        <p>© {new Date().getFullYear()} Kodleon | Yapay Zeka Eğitim Platformu</p>
      </div>
    </div>
  );
} 