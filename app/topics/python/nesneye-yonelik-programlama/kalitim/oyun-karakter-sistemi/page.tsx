import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python OOP: Oyun Karakter Sistemi | Kodleon',
  description: 'Python\'da kalıtım prensiplerini kullanarak kapsamlı bir RPG oyun karakter sistemi oluşturmayı öğrenin.',
};

const content = `
# Oyun Karakter Sistemi

Bu örnekte, kalıtım prensiplerini kullanarak bir RPG oyunu için karakter sistemi geliştireceğiz.

## Özellikler

1. **Karakter Yönetimi**
   - Temel karakter özellikleri
   - Sınıf bazlı yetenekler
   - Seviye sistemi

2. **Envanter Sistemi**
   - Ekipman yönetimi
   - Eşya kullanımı
   - Ağırlık limiti

3. **Savaş Sistemi**
   - Hasar hesaplama
   - Savunma mekanizması
   - Özel yetenekler

## Kod Örneği ve Açıklamalar

\`\`\`python
from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass
import random

# Karakter sınıfları için enum
class KarakterSinifi(Enum):
    SAVASCI = "Savaşçı"
    BUYUCU = "Büyücü"
    OKCU = "Okçu"
    HIRSIZ = "Hırsız"

# Ekipman türleri için enum
class EkipmanTuru(Enum):
    SILAH = "Silah"
    ZIRH = "Zırh"
    KALKAN = "Kalkan"
    AKSESUAR = "Aksesuar"

# Yetenek türleri için enum
class YetenekTuru(Enum):
    SALDIRI = "Saldırı"
    SAVUNMA = "Savunma"
    DESTEK = "Destek"
    OZEL = "Özel"

@dataclass
class Ekipman:
    """Ekipman için veri sınıfı"""
    id: str
    ad: str
    tur: EkipmanTuru
    seviye_gereksinimi: int
    bonus: Dict[str, float]
    agirlik: float
    deger: int

@dataclass
class Yetenek:
    """Yetenek için veri sınıfı"""
    id: str
    ad: str
    tur: YetenekTuru
    mana_maliyeti: int
    bekleme_suresi: int
    hasar_katsayisi: float
    etki_suresi: Optional[int] = None
    etki_alani: Optional[float] = None

class Karakter:
    """Temel karakter sınıfı"""
    def __init__(self, isim: str, sinif: KarakterSinifi):
        self.isim = isim
        self.sinif = sinif
        self.seviye = 1
        self.deneyim = 0
        self.can = 100
        self.mana = 100
        self.guc = 10
        self.ceviklik = 10
        self.zeka = 10
        self.dayaniklilik = 10
        
        self._max_can = 100
        self._max_mana = 100
        self._envanter: List[Ekipman] = []
        self._ekipmanlar: Dict[EkipmanTuru, Optional[Ekipman]] = {
            tur: None for tur in EkipmanTuru
        }
        self._yetenekler: List[Yetenek] = []
        self._yetenek_bekleme: Dict[str, int] = {}
    
    @property
    def max_tasima_kapasitesi(self) -> float:
        """Maksimum taşıma kapasitesini hesaplar"""
        return self.guc * 5 + self.dayaniklilik * 2
    
    @property
    def mevcut_agirlik(self) -> float:
        """Envanterdeki toplam ağırlığı hesaplar"""
        return sum(esya.agirlik for esya in self._envanter)
    
    def seviye_atlat(self) -> None:
        """Karakter seviyesini artırır ve özellikleri günceller"""
        self.seviye += 1
        self._max_can += 20
        self._max_mana += 10
        self.can = self._max_can
        self.mana = self._max_mana
    
    def deneyim_kazan(self, miktar: int) -> None:
        """Deneyim puanı ekler ve gerekirse seviye atlatır"""
        self.deneyim += miktar
        gereken_deneyim = self.seviye * 100
        
        while self.deneyim >= gereken_deneyim:
            self.deneyim -= gereken_deneyim
            self.seviye_atlat()
            gereken_deneyim = self.seviye * 100
    
    def ekipman_giy(self, ekipman: Ekipman) -> bool:
        """Ekipman giyme işlemi"""
        if ekipman.seviye_gereksinimi > self.seviye:
            raise ValueError(f"Bu ekipman için seviye yetersiz: {ekipman.seviye_gereksinimi}")
        
        # Mevcut ekipmanı çıkar
        if self._ekipmanlar[ekipman.tur]:
            self.ekipman_cikar(ekipman.tur)
        
        # Yeni ekipmanı giy
        self._ekipmanlar[ekipman.tur] = ekipman
        
        # Bonusları uygula
        for ozellik, deger in ekipman.bonus.items():
            if hasattr(self, ozellik):
                setattr(self, ozellik, getattr(self, ozellik) + deger)
        
        return True
    
    def ekipman_cikar(self, tur: EkipmanTuru) -> Optional[Ekipman]:
        """Ekipman çıkarma işlemi"""
        ekipman = self._ekipmanlar[tur]
        if not ekipman:
            return None
        
        # Bonusları kaldır
        for ozellik, deger in ekipman.bonus.items():
            if hasattr(self, ozellik):
                setattr(self, ozellik, getattr(self, ozellik) - deger)
        
        self._ekipmanlar[tur] = None
        return ekipman
    
    def yetenek_ekle(self, yetenek: Yetenek) -> bool:
        """Yeni yetenek ekler"""
        if len(self._yetenekler) >= 8:  # Maksimum 8 yetenek
            raise ValueError("Maksimum yetenek sayısına ulaşıldı")
        
        self._yetenekler.append(yetenek)
        self._yetenek_bekleme[yetenek.id] = 0
        return True
    
    def yetenek_kullan(self, yetenek_id: str, hedef: 'Karakter') -> float:
        """Yetenek kullanma işlemi"""
        yetenek = next((y for y in self._yetenekler if y.id == yetenek_id), None)
        if not yetenek:
            raise ValueError("Yetenek bulunamadı")
        
        if self._yetenek_bekleme[yetenek.id] > 0:
            raise ValueError("Yetenek bekleme süresinde")
        
        if self.mana < yetenek.mana_maliyeti:
            raise ValueError("Yetersiz mana")
        
        # Yeteneği kullan
        self.mana -= yetenek.mana_maliyeti
        self._yetenek_bekleme[yetenek.id] = yetenek.bekleme_suresi
        
        # Hasarı hesapla
        temel_hasar = self.guc * yetenek.hasar_katsayisi
        kritik_sans = min(self.ceviklik / 100, 0.5)  # Maksimum %50 kritik şans
        
        if random.random() < kritik_sans:
            temel_hasar *= 2  # Kritik vuruş
        
        return temel_hasar
    
    def tur_baslat(self) -> None:
        """Yeni tur başlangıcında çağrılır"""
        # Bekleme sürelerini güncelle
        for yetenek_id in self._yetenek_bekleme:
            if self._yetenek_bekleme[yetenek_id] > 0:
                self._yetenek_bekleme[yetenek_id] -= 1
        
        # Can ve mana rejenerasyonu
        self.can = min(self._max_can, self.can + self.dayaniklilik)
        self.mana = min(self._max_mana, self.mana + self.zeka)
    
    def durum_bilgisi(self) -> str:
        """Karakter durum bilgilerini döndürür"""
        return f"""
        Karakter Bilgileri
        -----------------
        İsim: {self.isim}
        Sınıf: {self.sinif.value}
        Seviye: {self.seviye} (Deneyim: {self.deneyim})
        Can: {self.can}/{self._max_can}
        Mana: {self.mana}/{self._max_mana}
        
        Özellikler
        ----------
        Güç: {self.guc}
        Çeviklik: {self.ceviklik}
        Zeka: {self.zeka}
        Dayanıklılık: {self.dayaniklilik}
        
        Envanter
        --------
        Ağırlık: {self.mevcut_agirlik}/{self.max_tasima_kapasitesi}
        """

class Savasci(Karakter):
    """Savaşçı sınıfı"""
    def __init__(self, isim: str):
        super().__init__(isim, KarakterSinifi.SAVASCI)
        self.guc += 5
        self.dayaniklilik += 5
        
        # Başlangıç yetenekleri
        self.yetenek_ekle(Yetenek(
            "giyotin",
            "Giyotin",
            YetenekTuru.SALDIRI,
            20,
            3,
            1.5
        ))
        
        self.yetenek_ekle(Yetenek(
            "savunma_durusu",
            "Savunma Duruşu",
            YetenekTuru.SAVUNMA,
            15,
            4,
            0.5,
            etki_suresi=2
        ))
    
    def seviye_atlat(self) -> None:
        """Savaşçıya özel seviye atlama"""
        super().seviye_atlat()
        self.guc += 3
        self.dayaniklilik += 2
        
        # 5. seviyede yeni yetenek
        if self.seviye == 5:
            self.yetenek_ekle(Yetenek(
                "carkifelek",
                "Çarkıfelek",
                YetenekTuru.SALDIRI,
                35,
                5,
                2.0,
                etki_alani=3
            ))

class Buyucu(Karakter):
    """Büyücü sınıfı"""
    def __init__(self, isim: str):
        super().__init__(isim, KarakterSinifi.BUYUCU)
        self.zeka += 7
        self.mana += 50
        self._max_mana += 50
        
        # Başlangıç yetenekleri
        self.yetenek_ekle(Yetenek(
            "alev_topu",
            "Alev Topu",
            YetenekTuru.SALDIRI,
            30,
            2,
            2.0,
            etki_alani=2
        ))
        
        self.yetenek_ekle(Yetenek(
            "buz_kalkan",
            "Buz Kalkanı",
            YetenekTuru.SAVUNMA,
            25,
            4,
            1.0,
            etki_suresi=3
        ))
    
    def seviye_atlat(self) -> None:
        """Büyücüye özel seviye atlama"""
        super().seviye_atlat()
        self.zeka += 4
        self._max_mana += 20
        self.mana = self._max_mana
        
        # 5. seviyede yeni yetenek
        if self.seviye == 5:
            self.yetenek_ekle(Yetenek(
                "meteor_yagmuru",
                "Meteor Yağmuru",
                YetenekTuru.SALDIRI,
                50,
                6,
                3.0,
                etki_alani=5
            ))

class Okcu(Karakter):
    """Okçu sınıfı"""
    def __init__(self, isim: str):
        super().__init__(isim, KarakterSinifi.OKCU)
        self.ceviklik += 7
        self.guc += 3
        
        # Başlangıç yetenekleri
        self.yetenek_ekle(Yetenek(
            "hizli_atis",
            "Hızlı Atış",
            YetenekTuru.SALDIRI,
            15,
            1,
            1.2
        ))
        
        self.yetenek_ekle(Yetenek(
            "zehirli_ok",
            "Zehirli Ok",
            YetenekTuru.SALDIRI,
            25,
            3,
            1.0,
            etki_suresi=3
        ))
    
    def seviye_atlat(self) -> None:
        """Okçuya özel seviye atlama"""
        super().seviye_atlat()
        self.ceviklik += 3
        self.guc += 2
        
        # 5. seviyede yeni yetenek
        if self.seviye == 5:
            self.yetenek_ekle(Yetenek(
                "ok_firtinasi",
                "Ok Fırtınası",
                YetenekTuru.SALDIRI,
                40,
                5,
                2.5,
                etki_alani=4
            ))

# Test fonksiyonu
def ornek_kullanim():
    # Karakterler oluştur
    savasci = Savasci("Aragorn")
    buyucu = Buyucu("Gandalf")
    okcu = Okcu("Legolas")
    
    # Ekipmanlar oluştur
    kilic = Ekipman(
        "sword1",
        "Çelik Kılıç",
        EkipmanTuru.SILAH,
        1,
        {"guc": 5},
        3.0,
        100
    )
    
    asa = Ekipman(
        "staff1",
        "Bilge Asası",
        EkipmanTuru.SILAH,
        1,
        {"zeka": 5, "mana": 20},
        2.0,
        120
    )
    
    yay = Ekipman(
        "bow1",
        "Uzun Yay",
        EkipmanTuru.SILAH,
        1,
        {"ceviklik": 3, "guc": 2},
        2.5,
        110
    )
    
    try:
        # Ekipmanları giy
        savasci.ekipman_giy(kilic)
        buyucu.ekipman_giy(asa)
        okcu.ekipman_giy(yay)
        
        # Deneyim kazan ve seviye atla
        savasci.deneyim_kazan(250)
        buyucu.deneyim_kazan(250)
        okcu.deneyim_kazan(250)
        
        # Yetenekleri kullan
        print("Savaşçı -> Büyücü:", savasci.yetenek_kullan("giyotin", buyucu))
        print("Büyücü -> Okçu:", buyucu.yetenek_kullan("alev_topu", okcu))
        print("Okçu -> Savaşçı:", okcu.yetenek_kullan("hizli_atis", savasci))
        
        # Durum bilgilerini göster
        print(savasci.durum_bilgisi())
        print(buyucu.durum_bilgisi())
        print(okcu.durum_bilgisi())
        
    except ValueError as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    ornek_kullanim()
\`\`\`

## Kod Açıklamaları

### 1. Veri Yapıları
- **Enum Sınıfları**: Karakter sınıfları, ekipman ve yetenek türleri için sabit değerler
- **Dataclass'lar**: Ekipman ve yetenek bilgileri için veri sınıfları
- **Temel Sınıf**: Tüm karakterler için ortak özellikleri içeren \`Karakter\` sınıfı

### 2. Kalıtım Hiyerarşisi
- \`Karakter\`: Temel sınıf
- \`Savasci\`: Yakın dövüş odaklı
- \`Buyucu\`: Büyü ve uzak mesafe hasarı odaklı
- \`Okcu\`: Çeviklik ve kritik vuruş odaklı

### 3. Özelleştirilmiş Davranışlar
- Her sınıf kendi başlangıç özelliklerine sahip
- Sınıfa özel yetenek setleri
- Farklı seviye atlama bonusları

## Önerilen Geliştirmeler

1. **Sınıf Sistemi**
   - Alt uzmanlıklar
   - Yetenek ağacı
   - Prestij sınıfları

2. **Envanter Sistemi**
   - Eşya birleştirme
   - Set bonusları
   - Eşya geliştirme

3. **Savaş Sistemi**
   - Zincir yetenekler
   - Takım sinerjileri
   - Durum etkileri

4. **Görev Sistemi**
   - Görev takibi
   - Ödül sistemi
   - Başarım sistemi

## Sonraki Adımlar

Bu sistemi daha da geliştirebilirsiniz:

1. Veritabanı entegrasyonu
2. Çoklu oyuncu desteği
3. Grafik arayüzü
4. Yapay zeka rakipler

Bu geliştirmeler, sistemin gerçek bir oyunda kullanılabilir hale gelmesini sağlayacaktır.
`;

export default function OyunKarakterSistemiPage() {
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