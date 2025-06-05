import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python OOP: Araç Kiralama Sistemi | Kodleon',
  description: 'Python\'da kapsülleme prensiplerini kullanarak kapsamlı bir araç kiralama sistemi oluşturmayı öğrenin.',
};

const content = `
# Araç Kiralama Sistemi

Bu örnekte, kapsülleme prensiplerini kullanarak profesyonel bir araç kiralama sistemi geliştireceğiz.

## Özellikler

1. **Araç Yönetimi**
   - Araç durumu takibi
   - Bakım planlaması
   - Kilometre takibi

2. **Kiralama İşlemleri**
   - Rezervasyon sistemi
   - Fiyat hesaplama
   - Sözleşme yönetimi

3. **Bakım Takibi**
   - Periyodik bakım planı
   - Arıza kayıtları
   - Servis geçmişi

## Kod Örneği ve Açıklamalar

\`\`\`python
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass

# Araç durumu için enum
class AracDurumu(Enum):
    MUSAIT = "Müsait"
    KIRADA = "Kirada"
    BAKIMDA = "Bakımda"
    SERVISTE = "Serviste"
    PASIF = "Pasif"

# Araç sınıfı için enum
class AracSinifi(Enum):
    EKONOMIK = "Ekonomik"
    ORTA = "Orta"
    LUKS = "Lüks"
    VIP = "VIP"

# Bakım türü için enum
class BakimTuru(Enum):
    PERIYODIK = "Periyodik Bakım"
    ARIZA = "Arıza Bakımı"
    KAZA = "Kaza Tamiri"
    LASTIK = "Lastik Değişimi"

@dataclass
class BakimKaydi:
    """Bakım kaydı için veri sınıfı"""
    tarih: datetime
    tur: BakimTuru
    km: int
    aciklama: str
    maliyet: float
    sonraki_bakim_km: Optional[int] = None

class Arac:
    # Sınıf sabitleri
    __GUNLUK_KM_LIMITI = 500
    __PERIYODIK_BAKIM_KM = 10000
    __LASTIK_DEGISIM_KM = 40000
    
    def __init__(self, plaka: str, marka: str, model: str, yil: int, 
                 sinif: AracSinifi, gunluk_ucret: float):
        # Private attributes
        self.__plaka = plaka
        self.__marka = marka
        self.__model = model
        self.__yil = yil
        self.__sinif = sinif
        self.__gunluk_ucret = gunluk_ucret
        self.__km = 0
        self.__durum = AracDurumu.MUSAIT
        self.__son_bakim_km = 0
        self.__son_lastik_km = 0
        self.__bakim_gecmisi: List[BakimKaydi] = []
        self.__kiralama_gecmisi: List[Dict] = []
        
        # Protected attributes
        self._eklenme_tarihi = datetime.now()
        self._son_guncelleme = datetime.now()

    @property
    def plaka(self) -> str:
        """Plaka bilgisini döndürür"""
        return self.__plaka

    @property
    def tam_ad(self) -> str:
        """Araç tam adını döndürür"""
        return f"{self.__marka} {self.__model} ({self.__yil})"

    @property
    def durum(self) -> AracDurumu:
        """Araç durumunu döndürür"""
        return self.__durum

    @property
    def gunluk_ucret(self) -> float:
        """Günlük kiralama ücretini döndürür"""
        return self.__gunluk_ucret

    @gunluk_ucret.setter
    def gunluk_ucret(self, yeni_ucret: float) -> None:
        """Günlük kiralama ücretini günceller"""
        if yeni_ucret <= 0:
            raise ValueError("Ücret pozitif olmalıdır")
        self.__gunluk_ucret = yeni_ucret
        self._son_guncelleme = datetime.now()

    def __bakim_kontrolu(self) -> Optional[str]:
        """
        Bakım ihtiyacını kontrol eder
        
        Returns:
            Optional[str]: Bakım gerekiyorsa nedeni, gerekmiyorsa None
        """
        if self.__km - self.__son_bakim_km >= self.__PERIYODIK_BAKIM_KM:
            return "Periyodik bakım zamanı geldi"
            
        if self.__km - self.__son_lastik_km >= self.__LASTIK_DEGISIM_KM:
            return "Lastik değişim zamanı geldi"
            
        return None

    def km_guncelle(self, yeni_km: int) -> None:
        """
        Kilometre bilgisini günceller ve bakım kontrolü yapar
        
        Args:
            yeni_km: Yeni kilometre
            
        Raises:
            ValueError: Geçersiz kilometre değeri için
        """
        if yeni_km < self.__km:
            raise ValueError("Yeni kilometre mevcut kilometreden küçük olamaz")
            
        self.__km = yeni_km
        self._son_guncelleme = datetime.now()
        
        # Bakım kontrolü
        bakim_durumu = self.__bakim_kontrolu()
        if bakim_durumu:
            print(f"Uyarı: {self.plaka} - {bakim_durumu}")

    def bakim_ekle(self, bakim: BakimKaydi) -> None:
        """
        Yeni bakım kaydı ekler
        
        Args:
            bakim: Bakım kaydı
        """
        self.__bakim_gecmisi.append(bakim)
        
        # Bakım türüne göre son bakım bilgilerini güncelle
        if bakim.tur == BakimTuru.PERIYODIK:
            self.__son_bakim_km = bakim.km
        elif bakim.tur == BakimTuru.LASTIK:
            self.__son_lastik_km = bakim.km
            
        # Araç durumunu güncelle
        self.__durum = AracDurumu.MUSAIT
        self._son_guncelleme = datetime.now()

    def kirala(self, musteri_id: str, baslangic: datetime, 
               bitis: datetime, km_limiti: Optional[int] = None) -> Dict:
        """
        Araç kiralama işlemi
        
        Args:
            musteri_id: Müşteri ID
            baslangic: Kiralama başlangıç tarihi
            bitis: Kiralama bitiş tarihi
            km_limiti: Özel kilometre limiti
            
        Returns:
            Dict: Kiralama bilgileri
            
        Raises:
            ValueError: Uygun olmayan durumlar için
        """
        if self.__durum != AracDurumu.MUSAIT:
            raise ValueError(f"Araç müsait değil: {self.__durum.value}")
            
        # Bakım kontrolü
        bakim_durumu = self.__bakim_kontrolu()
        if bakim_durumu:
            raise ValueError(f"Araç bakım gerektiriyor: {bakim_durumu}")
            
        # Kiralama süresi hesaplama
        gun_sayisi = (bitis - baslangic).days
        if gun_sayisi < 1:
            raise ValueError("Minimum kiralama süresi 1 gündür")
            
        # Kilometre limiti belirleme
        km_limiti = km_limiti or (gun_sayisi * self.__GUNLUK_KM_LIMITI)
        
        # Kiralama kaydı oluştur
        kiralama = {
            "musteri_id": musteri_id,
            "baslangic": baslangic,
            "bitis": bitis,
            "km_limiti": km_limiti,
            "baslangic_km": self.__km,
            "gunluk_ucret": self.__gunluk_ucret,
            "toplam_ucret": gun_sayisi * self.__gunluk_ucret
        }
        
        self.__kiralama_gecmisi.append(kiralama)
        self.__durum = AracDurumu.KIRADA
        self._son_guncelleme = datetime.now()
        
        return kiralama

    def teslim_al(self, son_km: int, hasar_var: bool = False) -> Dict:
        """
        Kiralık aracı teslim alma işlemi
        
        Args:
            son_km: Teslim kilometresi
            hasar_var: Hasar durumu
            
        Returns:
            Dict: Teslim alma bilgileri
            
        Raises:
            ValueError: Uygun olmayan durumlar için
        """
        if self.__durum != AracDurumu.KIRADA:
            raise ValueError("Araç kirada değil")
            
        # Son kiralamayı bul
        son_kiralama = self.__kiralama_gecmisi[-1]
        
        # Kilometre kontrolü
        km_farki = son_km - son_kiralama["baslangic_km"]
        km_asimi = max(0, km_farki - son_kiralama["km_limiti"])
        
        # Ek ücret hesaplama
        km_asim_ucreti = km_asimi * 0.5  # km başına 0.5 TL
        
        # Teslim bilgilerini güncelle
        teslim_bilgisi = {
            "teslim_tarihi": datetime.now(),
            "son_km": son_km,
            "km_farki": km_farki,
            "km_asimi": km_asimi,
            "km_asim_ucreti": km_asim_ucreti,
            "hasar_var": hasar_var,
            "toplam_ucret": son_kiralama["toplam_ucret"] + km_asim_ucreti
        }
        
        # Araç bilgilerini güncelle
        self.km_guncelle(son_km)
        self.__durum = AracDurumu.MUSAIT if not hasar_var else AracDurumu.SERVISTE
        
        return teslim_bilgisi

    def bakim_gecmisi_getir(self, baslangic: Optional[datetime] = None,
                           bitis: Optional[datetime] = None) -> List[BakimKaydi]:
        """
        Bakım geçmişini döndürür
        
        Args:
            baslangic: Başlangıç tarihi
            bitis: Bitiş tarihi
            
        Returns:
            List[BakimKaydi]: Bakım kayıtları
        """
        if not baslangic:
            baslangic = self._eklenme_tarihi
        if not bitis:
            bitis = datetime.now()
            
        return [
            kayit for kayit in self.__bakim_gecmisi
            if baslangic <= kayit.tarih <= bitis
        ]

    def kiralama_gecmisi_getir(self, baslangic: Optional[datetime] = None,
                              bitis: Optional[datetime] = None) -> List[Dict]:
        """
        Kiralama geçmişini döndürür
        
        Args:
            baslangic: Başlangıç tarihi
            bitis: Bitiş tarihi
            
        Returns:
            List[Dict]: Kiralama kayıtları
        """
        if not baslangic:
            baslangic = self._eklenme_tarihi
        if not bitis:
            bitis = datetime.now()
            
        return [
            kiralama for kiralama in self.__kiralama_gecmisi
            if baslangic <= kiralama["baslangic"] <= bitis
        ]

    def arac_ozeti(self) -> str:
        """Araç bilgilerinin özetini döndürür"""
        return f"""
        Araç Bilgileri
        --------------
        Plaka: {self.__plaka}
        Araç: {self.tam_ad}
        Sınıf: {self.__sinif.value}
        Durum: {self.__durum.value}
        Kilometre: {self.__km}
        Günlük Ücret: {self.__gunluk_ucret} TL
        Son Bakım: {self.__son_bakim_km} km
        Son Lastik: {self.__son_lastik_km} km
        """

# Kullanım örneği
def ornek_kullanim():
    # Araç oluşturma
    arac = Arac(
        "34ABC123",
        "Toyota",
        "Corolla",
        2022,
        AracSinifi.ORTA,
        500.0  # Günlük ücret
    )
    
    try:
        # Kilometre güncelleme
        arac.km_guncelle(5000)
        
        # Periyodik bakım ekleme
        bakim = BakimKaydi(
            datetime.now(),
            BakimTuru.PERIYODIK,
            5000,
            "Periyodik bakım yapıldı",
            1500.0,
            15000
        )
        arac.bakim_ekle(bakim)
        
        # Kiralama işlemi
        kiralama = arac.kirala(
            "MUS123",
            datetime.now(),
            datetime.now() + timedelta(days=3)
        )
        print("Kiralama başarılı:", kiralama)
        
        # Teslim alma simülasyonu
        teslim = arac.teslim_al(5800)
        print("Teslim bilgileri:", teslim)
        
    except ValueError as e:
        print(f"Hata: {e}")
    
    # Araç özeti
    print(arac.arac_ozeti())
    
    # Bakım geçmişi
    bakimlar = arac.bakim_gecmisi_getir()
    print("\\nBakım Geçmişi:")
    for bakim in bakimlar:
        print(f"{bakim.tarih}: {bakim.tur.value} - {bakim.aciklama}")

if __name__ == "__main__":
    ornek_kullanim()
\`\`\`

## Kod Açıklamaları

### 1. Veri Yapıları

- **Enum Sınıfları**: Araç durumu, sınıfı ve bakım türleri için sabit değerler
- **Dataclass**: Bakım kayıtları için veri sınıfı
- **Private Attributes**: Araç bilgileri ve işlem geçmişi için
- **Protected Attributes**: Sistem bilgileri için

### 2. Araç Yönetimi

- Kilometre takibi
- Durum kontrolü
- Bakım planlaması
- Lastik değişim takibi

### 3. Kiralama Sistemi

- Müsaitlik kontrolü
- Fiyat hesaplama
- Kilometre limiti kontrolü
- Teslim alma işlemleri

### 4. Bakım Takibi

- Periyodik bakım planı
- Lastik değişim planı
- Servis geçmişi
- Maliyet takibi

## Önerilen Geliştirmeler

1. **Rezervasyon Sistemi**
   - Online rezervasyon
   - Ön ödeme
   - İptal politikaları

2. **Filo Yönetimi**
   - Araç gruplaması
   - Şube yönetimi
   - Transfer planlaması

3. **Müşteri Yönetimi**
   - Üyelik sistemi
   - Puan sistemi
   - Özel teklifler

4. **Raporlama Sistemi**
   - Doluluk oranları
   - Gelir analizi
   - Bakım maliyetleri

## Sonraki Adımlar

Bu sistemi daha da geliştirebilirsiniz:

1. Veritabanı entegrasyonu
2. Web arayüzü
3. Mobil uygulama
4. GPS entegrasyonu

Bu geliştirmeler, sistemin profesyonel bir araç kiralama işletmesinde kullanılabilir hale gelmesini sağlayacaktır.
`;

export default function AracKiralamaSistemiPage() {
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