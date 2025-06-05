import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python OOP: Kütüphane Üye Sistemi | Kodleon',
  description: 'Python\'da kapsülleme prensiplerini kullanarak kapsamlı bir kütüphane üye yönetim sistemi oluşturmayı öğrenin.',
};

const content = `
# Kütüphane Üye Yönetim Sistemi

Bu örnekte, kapsülleme prensiplerini kullanarak bir kütüphane üye yönetim sistemi geliştireceğiz.

## Özellikler

1. **Üye Bilgileri Yönetimi**
   - Kişisel bilgilerin güvenli saklanması
   - KVKK uyumlu veri yönetimi
   - Üyelik durumu takibi

2. **Kitap Ödünç Sistemi**
   - Kitap ödünç alma/iade
   - Kitap rezervasyon
   - Ödünç geçmişi

3. **Gecikme Takibi**
   - Otomatik ceza hesaplama
   - Ödeme takibi
   - Üyelik kısıtlamaları

## Kod Örneği ve Açıklamalar

\`\`\`python
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass

# Üyelik durumları için enum
class UyelikDurumu(Enum):
    AKTIF = "Aktif"
    ASKIDA = "Askıda"
    CEZALI = "Cezalı"
    IPTAL = "İptal"

# Kitap durumu için enum
class KitapDurumu(Enum):
    RAFTA = "Rafta"
    ODUNC = "Ödünç Verilmiş"
    REZERVE = "Rezerve Edilmiş"
    KAYIP = "Kayıp"

@dataclass
class Kitap:
    """Kitap bilgilerini tutan veri sınıfı"""
    isbn: str
    baslik: str
    yazar: str
    durum: KitapDurumu = KitapDurumu.RAFTA
    odunc_alan: Optional[str] = None
    son_iade_tarihi: Optional[datetime] = None

class KutuphaneUyesi:
    # Sınıf sabitleri
    __MAKSIMUM_KITAP_SAYISI = 5
    __GUNLUK_GECIKME_CEZASI = 1.0  # TL
    __ODUNC_SURESI = 14  # gün
    
    def __init__(self, tc_no: str, ad: str, soyad: str, email: str, telefon: str):
        # Private üye bilgileri
        self.__tc_no = tc_no
        self.__ad = ad
        self.__soyad = soyad
        self.__email = email
        self.__telefon = telefon
        
        # Private sistem bilgileri
        self.__uyelik_durumu = UyelikDurumu.AKTIF
        self.__odunc_kitaplar: List[Kitap] = []
        self.__gecmis_oduncler: List[Dict] = []
        self.__toplam_ceza = 0.0
        self.__son_ceza_kontrolu = datetime.now()
        
        # Protected bilgiler
        self._uyelik_baslangic = datetime.now()
        self._son_islem_tarihi = datetime.now()

    @property
    def tam_ad(self) -> str:
        """Üyenin tam adını döndürür"""
        return f"{self.__ad} {self.__soyad}"

    @property
    def uyelik_durumu(self) -> UyelikDurumu:
        """Üyelik durumunu döndürür"""
        return self.__uyelik_durumu

    @property
    def toplam_ceza(self) -> float:
        """Toplam ceza miktarını döndürür"""
        self.__ceza_kontrolu()
        return self.__toplam_ceza

    def __ceza_kontrolu(self) -> None:
        """
        Gecikmiş kitaplar için ceza hesaplar
        """
        simdi = datetime.now()
        
        for kitap in self.__odunc_kitaplar:
            if kitap.son_iade_tarihi and simdi > kitap.son_iade_tarihi:
                gecikme_gun = (simdi - kitap.son_iade_tarihi).days
                ceza = gecikme_gun * self.__GUNLUK_GECIKME_CEZASI
                self.__toplam_ceza += ceza

        self.__son_ceza_kontrolu = simdi
        
        # Ceza durumuna göre üyelik durumu güncelleme
        if self.__toplam_ceza > 50:
            self.__uyelik_durumu = UyelikDurumu.CEZALI

    def kitap_odunc_al(self, kitap: Kitap) -> bool:
        """
        Kitap ödünç alma işlemi
        
        Args:
            kitap: Ödünç alınacak kitap
            
        Returns:
            bool: İşlem başarılıysa True
            
        Raises:
            ValueError: Uygun olmayan durumlar için
        """
        if self.__uyelik_durumu != UyelikDurumu.AKTIF:
            raise ValueError(f"Üyelik durumu uygun değil: {self.__uyelik_durumu.value}")
            
        if len(self.__odunc_kitaplar) >= self.__MAKSIMUM_KITAP_SAYISI:
            raise ValueError(f"Maksimum kitap sayısına ulaşıldı: {self.__MAKSIMUM_KITAP_SAYISI}")
            
        if kitap.durum != KitapDurumu.RAFTA:
            raise ValueError(f"Kitap ödünç alınamaz: {kitap.durum.value}")
            
        # Kitabı ödünç ver
        kitap.durum = KitapDurumu.ODUNC
        kitap.odunc_alan = self.__tc_no
        kitap.son_iade_tarihi = datetime.now() + timedelta(days=self.__ODUNC_SURESI)
        
        self.__odunc_kitaplar.append(kitap)
        self._son_islem_tarihi = datetime.now()
        
        # İşlem kaydı
        self.__gecmis_oduncler.append({
            "isbn": kitap.isbn,
            "islem": "Ödünç Alma",
            "tarih": datetime.now(),
            "iade_tarihi": kitap.son_iade_tarihi
        })
        
        return True

    def kitap_iade_et(self, kitap: Kitap) -> bool:
        """
        Kitap iade işlemi
        
        Args:
            kitap: İade edilecek kitap
            
        Returns:
            bool: İşlem başarılıysa True
            
        Raises:
            ValueError: Uygun olmayan durumlar için
        """
        if kitap not in self.__odunc_kitaplar:
            raise ValueError("Bu kitap size ait değil")
            
        # Ceza kontrolü
        self.__ceza_kontrolu()
        
        # Kitabı iade et
        kitap.durum = KitapDurumu.RAFTA
        kitap.odunc_alan = None
        kitap.son_iade_tarihi = None
        
        self.__odunc_kitaplar.remove(kitap)
        self._son_islem_tarihi = datetime.now()
        
        # İşlem kaydı
        self.__gecmis_oduncler.append({
            "isbn": kitap.isbn,
            "islem": "İade",
            "tarih": datetime.now()
        })
        
        return True

    def ceza_ode(self, miktar: float) -> bool:
        """
        Ceza ödeme işlemi
        
        Args:
            miktar: Ödenecek miktar
            
        Returns:
            bool: İşlem başarılıysa True
        """
        if miktar <= 0:
            raise ValueError("Geçersiz ödeme miktarı")
            
        if miktar > self.__toplam_ceza:
            raise ValueError("Ödeme miktarı cezadan büyük olamaz")
            
        self.__toplam_ceza -= miktar
        
        # Ceza tamamen ödendiyse üyelik durumunu güncelle
        if self.__toplam_ceza == 0 and self.__uyelik_durumu == UyelikDurumu.CEZALI:
            self.__uyelik_durumu = UyelikDurumu.AKTIF
            
        return True

    def odunc_gecmisi(self, baslangic: Optional[datetime] = None, 
                     bitis: Optional[datetime] = None) -> List[Dict]:
        """
        Ödünç geçmişini döndürür
        
        Args:
            baslangic: Başlangıç tarihi
            bitis: Bitiş tarihi
            
        Returns:
            List[Dict]: İşlem kayıtları
        """
        if not baslangic:
            baslangic = self._uyelik_baslangic
        if not bitis:
            bitis = datetime.now()
            
        return [
            islem for islem in self.__gecmis_oduncler
            if baslangic <= islem["tarih"] <= bitis
        ]

    def uye_ozeti(self) -> str:
        """Üye bilgilerinin özetini döndürür"""
        return f"""
        Üye Bilgileri
        -------------
        Ad Soyad: {self.tam_ad}
        E-posta: {self.__email}
        Üyelik Durumu: {self.__uyelik_durumu.value}
        Ödünç Kitap Sayısı: {len(self.__odunc_kitaplar)}
        Toplam Ceza: {self.toplam_ceza} TL
        Son İşlem: {self._son_islem_tarihi}
        """

# Kullanım örneği
def ornek_kullanim():
    # Üye oluşturma
    uye = KutuphaneUyesi(
        "12345678901",
        "Ahmet",
        "Yılmaz",
        "ahmet@email.com",
        "5551234567"
    )
    
    # Kitap oluşturma
    kitap1 = Kitap(
        "978-0-7475-3269-9",
        "Harry Potter",
        "J.K. Rowling"
    )
    
    kitap2 = Kitap(
        "978-0-7475-3269-8",
        "Yüzüklerin Efendisi",
        "J.R.R. Tolkien"
    )
    
    try:
        # Kitap ödünç alma
        uye.kitap_odunc_al(kitap1)
        print("Kitap başarıyla ödünç alındı")
        
        # Gecikme simülasyonu
        kitap1.son_iade_tarihi = datetime.now() - timedelta(days=5)
        
        # Ceza kontrolü
        print(f"Toplam ceza: {uye.toplam_ceza} TL")
        
        # Kitap iade
        uye.kitap_iade_et(kitap1)
        print("Kitap başarıyla iade edildi")
        
        # Ceza ödeme
        if uye.toplam_ceza > 0:
            uye.ceza_ode(uye.toplam_ceza)
            print("Ceza ödendi")
        
    except ValueError as e:
        print(f"Hata: {e}")
    
    # Üye özeti
    print(uye.uye_ozeti())
    
    # Ödünç geçmişi
    gecmis = uye.odunc_gecmisi()
    print("\\nÖdünç Geçmişi:")
    for islem in gecmis:
        print(f"{islem['islem']}: {islem['isbn']} - {islem['tarih']}")

if __name__ == "__main__":
    ornek_kullanim()
\`\`\`

## Kod Açıklamaları

### 1. Veri Yapıları

- **Enum Sınıfları**: Üyelik ve kitap durumları için sabit değerler
- **Dataclass**: Kitap bilgileri için veri sınıfı
- **Private Attributes**: Hassas üye bilgileri için
- **Protected Attributes**: Sistem bilgileri için

### 2. Üye Bilgileri Yönetimi

- Kişisel bilgiler private olarak saklanır
- Property'ler ile kontrollü erişim sağlanır
- Üyelik durumu otomatik güncellenir

### 3. Kitap Ödünç Sistemi

- Maksimum kitap sayısı kontrolü
- Ödünç alma/iade işlemleri
- Detaylı işlem kaydı

### 4. Gecikme ve Ceza Sistemi

- Otomatik ceza hesaplama
- Günlük gecikme ücreti
- Ceza ödeme ve durum güncelleme

## Önerilen Geliştirmeler

1. **Kitap Rezervasyon Sistemi**
   - Bekleme listesi
   - Otomatik bildirimler
   - Öncelik sırası

2. **Gelişmiş Raporlama**
   - İstatistikler
   - Popüler kitaplar
   - Üye davranış analizi

3. **Otomatik Bildirimler**
   - İade hatırlatmaları
   - Ceza bildirimleri
   - Rezervasyon bildirimleri

4. **Kategori ve Etiket Sistemi**
   - Kitap kategorileri
   - Öneriler
   - Benzer kitaplar

## Sonraki Adımlar

Bu sistemi daha da geliştirebilirsiniz:

1. Veritabanı entegrasyonu
2. Web arayüzü
3. QR kod sistemi
4. Mobil uygulama

Bu geliştirmeler, sistemin gerçek bir kütüphanede kullanılabilir hale gelmesini sağlayacaktır.
`;

export default function KutuphaneUyeSistemiPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <Button asChild variant="ghost" size="sm" className="gap-1">
          <Link href="/topics/python/nesneye-yonelik-programlama/kapsulleme">
            <ArrowLeft className="h-4 w-4" />
            Kapsülleme Konusuna Dön
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