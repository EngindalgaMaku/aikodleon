import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python OOP: Çalışan Yönetim Sistemi | Kodleon',
  description: 'Python\'da kalıtım prensiplerini kullanarak kapsamlı bir çalışan yönetim sistemi oluşturmayı öğrenin.',
};

const content = `
# Çalışan Yönetim Sistemi

Bu örnekte, kalıtım prensiplerini kullanarak bir şirketin çalışan yönetim sistemini modelleyeceğiz.

## Özellikler

1. **Çalışan Yönetimi**
   - Temel çalışan bilgileri
   - Departman bazlı organizasyon
   - Performans takibi

2. **Maaş ve İzin Sistemi**
   - Pozisyona göre maaş hesaplama
   - İzin takibi ve onay süreci
   - Fazla mesai hesaplaması

3. **Proje Yönetimi**
   - Proje atama ve takip
   - İş yükü analizi
   - Raporlama sistemi

## Kod Örneği ve Açıklamalar

\`\`\`python
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass

# Departman türleri için enum
class Departman(Enum):
    YAZILIM = "Yazılım"
    PAZARLAMA = "Pazarlama"
    INSAN_KAYNAKLARI = "İnsan Kaynakları"
    YONETIM = "Yönetim"
    FINANS = "Finans"

# İzin türleri için enum
class IzinTuru(Enum):
    YILLIK = "Yıllık İzin"
    HASTALIK = "Hastalık İzni"
    MAZERET = "Mazeret İzni"
    UCRETSIZ = "Ücretsiz İzin"

@dataclass
class Izin:
    """İzin kaydı için veri sınıfı"""
    baslangic: datetime
    bitis: datetime
    tur: IzinTuru
    onaylandi: bool = False
    onaylayan: Optional[str] = None

@dataclass
class Proje:
    """Proje bilgileri için veri sınıfı"""
    kod: str
    ad: str
    baslangic: datetime
    bitis: Optional[datetime] = None
    durum: str = "Devam Ediyor"

class Calisan:
    """Temel çalışan sınıfı"""
    def __init__(self, tc_no: str, ad: str, soyad: str, departman: Departman,
                 ise_giris: datetime, maas: float):
        self.tc_no = tc_no
        self.ad = ad
        self.soyad = soyad
        self.departman = departman
        self.ise_giris = ise_giris
        self._maas = maas  # protected
        self._izinler: List[Izin] = []
        self._projeler: List[Proje] = []
        self._performans_puani = 100
    
    @property
    def tam_ad(self) -> str:
        return f"{self.ad} {self.soyad}"
    
    @property
    def kidem_yili(self) -> int:
        return (datetime.now() - self.ise_giris).days // 365
    
    @property
    def maas(self) -> float:
        """Kıdeme göre artırılmış maaşı döndürür"""
        kidem_katsayisi = 1 + (self.kidem_yili * 0.1)  # Her yıl %10 artış
        return round(self._maas * kidem_katsayisi, 2)
    
    def izin_ekle(self, izin: Izin) -> bool:
        """Yeni izin talebi ekler"""
        # Tarih kontrolü
        if izin.baslangic < datetime.now():
            raise ValueError("Geçmiş tarihli izin eklenemez")
        
        # İzin çakışması kontrolü
        for mevcut_izin in self._izinler:
            if (izin.baslangic <= mevcut_izin.bitis and 
                izin.bitis >= mevcut_izin.baslangic):
                raise ValueError("İzin tarihleri çakışıyor")
        
        self._izinler.append(izin)
        return True
    
    def proje_ekle(self, proje: Proje) -> bool:
        """Çalışana yeni proje atar"""
        # Aktif proje sayısı kontrolü
        aktif_proje_sayisi = sum(
            1 for p in self._projeler 
            if p.durum == "Devam Ediyor"
        )
        if aktif_proje_sayisi >= 3:
            raise ValueError("Maksimum aktif proje sayısına ulaşıldı")
        
        self._projeler.append(proje)
        return True
    
    def performans_guncelle(self, yeni_puan: int) -> None:
        """Performans puanını günceller"""
        if not 0 <= yeni_puan <= 100:
            raise ValueError("Performans puanı 0-100 arasında olmalıdır")
        self._performans_puani = yeni_puan
    
    def ozet_bilgi(self) -> str:
        """Çalışan özet bilgilerini döndürür"""
        return f"""
        Çalışan Bilgileri
        -----------------
        Ad Soyad: {self.tam_ad}
        Departman: {self.departman.value}
        Kıdem: {self.kidem_yili} yıl
        Maaş: {self.maas} TL
        Performans: {self._performans_puani}
        Aktif Proje Sayısı: {sum(1 for p in self._projeler if p.durum == "Devam Ediyor")}
        """

class Muhendis(Calisan):
    """Mühendis sınıfı"""
    def __init__(self, tc_no: str, ad: str, soyad: str, 
                 ise_giris: datetime, maas: float,
                 uzmanlik_alani: str):
        super().__init__(tc_no, ad, soyad, Departman.YAZILIM, 
                        ise_giris, maas)
        self.uzmanlik_alani = uzmanlik_alani
        self._teknik_seviye = 1
    
    def proje_ekle(self, proje: Proje) -> bool:
        """Mühendis için özelleştirilmiş proje atama"""
        # Teknik seviyeye göre maksimum proje sayısı
        max_proje = self._teknik_seviye + 1
        
        aktif_proje_sayisi = sum(
            1 for p in self._projeler 
            if p.durum == "Devam Ediyor"
        )
        if aktif_proje_sayisi >= max_proje:
            raise ValueError(f"Teknik seviye {self._teknik_seviye} için "
                           f"maksimum {max_proje} aktif proje atanabilir")
        
        return super().proje_ekle(proje)
    
    def teknik_seviye_artir(self) -> None:
        """Teknik seviyeyi bir kademe artırır"""
        if self._teknik_seviye < 5:  # Maksimum 5. seviye
            self._teknik_seviye += 1
            # Seviye artışı ile maaş artışı
            self._maas *= 1.15  # %15 artış

class Yonetici(Calisan):
    """Yönetici sınıfı"""
    def __init__(self, tc_no: str, ad: str, soyad: str,
                 ise_giris: datetime, maas: float):
        super().__init__(tc_no, ad, soyad, Departman.YONETIM,
                        ise_giris, maas)
        self._ekip: List[Calisan] = []
    
    def ekip_uyesi_ekle(self, calisan: Calisan) -> bool:
        """Ekibe yeni üye ekler"""
        if len(self._ekip) >= 10:  # Maksimum ekip büyüklüğü
            raise ValueError("Maksimum ekip büyüklüğüne ulaşıldı")
        
        self._ekip.append(calisan)
        return True
    
    def ekip_performansi(self) -> float:
        """Ekibin ortalama performansını hesaplar"""
        if not self._ekip:
            return 0
        
        return sum(uye._performans_puani for uye in self._ekip) / len(self._ekip)
    
    @property
    def maas(self) -> float:
        """Ekip büyüklüğüne göre artırılmış maaşı döndürür"""
        ekip_katsayisi = 1 + (len(self._ekip) * 0.1)  # Her ekip üyesi için %10
        return super().maas * ekip_katsayisi

class Pazarlamaci(Calisan):
    """Pazarlama çalışanı sınıfı"""
    def __init__(self, tc_no: str, ad: str, soyad: str,
                 ise_giris: datetime, maas: float):
        super().__init__(tc_no, ad, soyad, Departman.PAZARLAMA,
                        ise_giris, maas)
        self._satislar: List[float] = []
    
    def satis_ekle(self, miktar: float) -> None:
        """Yeni satış kaydı ekler"""
        if miktar <= 0:
            raise ValueError("Satış miktarı pozitif olmalıdır")
        
        self._satislar.append(miktar)
    
    @property
    def maas(self) -> float:
        """Satış performansına göre artırılmış maaşı döndürür"""
        if not self._satislar:
            return super().maas
        
        # Son 3 ayın satış ortalaması
        son_satislar = self._satislar[-3:]
        satis_ortalamasi = sum(son_satislar) / len(son_satislar)
        
        # Her 10000 TL satış için %1 bonus
        bonus_katsayisi = 1 + ((satis_ortalamasi // 10000) * 0.01)
        return super().maas * bonus_katsayisi

# Test fonksiyonu
def ornek_kullanim():
    # Çalışanlar oluşturalım
    muhendis = Muhendis(
        "12345678901",
        "Ali",
        "Yılmaz",
        datetime(2020, 1, 1),
        15000,
        "Python Geliştirici"
    )
    
    yonetici = Yonetici(
        "23456789012",
        "Ayşe",
        "Demir",
        datetime(2018, 1, 1),
        25000
    )
    
    pazarlamaci = Pazarlamaci(
        "34567890123",
        "Mehmet",
        "Kaya",
        datetime(2021, 6, 1),
        12000
    )
    
    try:
        # Mühendis için işlemler
        proje1 = Proje("PRJ001", "Web Sitesi", datetime.now())
        proje2 = Proje("PRJ002", "Mobile App", datetime.now())
        muhendis.proje_ekle(proje1)
        muhendis.proje_ekle(proje2)
        muhendis.teknik_seviye_artir()
        
        # Yönetici için işlemler
        yonetici.ekip_uyesi_ekle(muhendis)
        yonetici.ekip_uyesi_ekle(pazarlamaci)
        
        # Pazarlamacı için işlemler
        pazarlamaci.satis_ekle(15000)
        pazarlamaci.satis_ekle(25000)
        pazarlamaci.satis_ekle(20000)
        
        # İzin işlemleri
        izin = Izin(
            datetime.now() + timedelta(days=5),
            datetime.now() + timedelta(days=10),
            IzinTuru.YILLIK
        )
        muhendis.izin_ekle(izin)
        
        # Bilgileri göster
        print(muhendis.ozet_bilgi())
        print(yonetici.ozet_bilgi())
        print(f"Ekip Performansı: {yonetici.ekip_performansi():.2f}")
        print(pazarlamaci.ozet_bilgi())
        
    except ValueError as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    ornek_kullanim()
\`\`\`

## Kod Açıklamaları

### 1. Veri Yapıları
- **Enum Sınıfları**: Departman ve izin türleri için sabit değerler
- **Dataclass'lar**: İzin ve proje bilgileri için veri sınıfları
- **Temel Sınıf**: Tüm çalışanlar için ortak özellikleri içeren \`Calisan\` sınıfı

### 2. Kalıtım Hiyerarşisi
- \`Calisan\`: Temel sınıf
- \`Muhendis\`: Teknik seviye ve proje yönetimi özellikleri
- \`Yonetici\`: Ekip yönetimi ve performans takibi
- \`Pazarlamaci\`: Satış takibi ve bonus sistemi

### 3. Özelleştirilmiş Davranışlar
- Her alt sınıf kendi ihtiyaçlarına göre metodları override eder
- Maaş hesaplama her pozisyon için farklı
- Proje ve izin yönetimi pozisyona göre özelleştirilmiş

## Önerilen Geliştirmeler

1. **Raporlama Sistemi**
   - Departman bazlı raporlar
   - Performans analizleri
   - Maliyet raporları

2. **İK Modülü**
   - İşe alım süreci
   - Eğitim takibi
   - Kariyer planlama

3. **Proje Yönetimi**
   - Gantt şeması
   - Kaynak planlaması
   - Bütçe takibi

4. **Entegrasyonlar**
   - Muhasebe sistemi
   - Documan yönetimi
   - Mesajlaşma sistemi

## Sonraki Adımlar

Bu sistemi daha da geliştirebilirsiniz:

1. Veritabanı entegrasyonu
2. Web arayüzü
3. API servisleri
4. Mobil uygulama

Bu geliştirmeler, sistemin gerçek bir şirkette kullanılabilir hale gelmesini sağlayacaktır.
`;

export default function CalisanYonetimSistemiPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <Button asChild variant="ghost" size="sm" className="gap-1">
          <Link href="/topics/python/nesneye-yonelik-programlama/kalitim">
            <ArrowLeft className="h-4 w-4" />
            Kalıtım Konusuna Dön
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