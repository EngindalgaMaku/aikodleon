import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python OOP: Medya Oynatıcı Sistemi | Kodleon',
  description: 'Python\'da kalıtım prensiplerini kullanarak kapsamlı bir medya oynatıcı sistemi oluşturmayı öğrenin.',
};

const content = `
# Medya Oynatıcı Sistemi

Bu örnekte, kalıtım prensiplerini kullanarak farklı medya türlerini destekleyen bir oynatıcı sistemi geliştireceğiz.

## Özellikler

1. **Medya Yönetimi**
   - Farklı medya türleri desteği
   - Format dönüştürme
   - Metadata yönetimi

2. **Çalma Listesi**
   - Liste oluşturma ve düzenleme
   - Sıralama ve filtreleme
   - Otomatik oynatma

3. **Kalite Ayarları**
   - Ses kalitesi kontrolü
   - Video çözünürlüğü
   - Akış hızı yönetimi

## Kod Örneği ve Açıklamalar

\`\`\`python
from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import random

# Medya türleri için enum
class MedyaTuru(Enum):
    MUZIK = "Müzik"
    VIDEO = "Video"
    PODCAST = "Podcast"

# Format türleri için enum
class FormatTuru(Enum):
    # Müzik formatları
    MP3 = "MP3"
    WAV = "WAV"
    FLAC = "FLAC"
    # Video formatları
    MP4 = "MP4"
    MKV = "MKV"
    AVI = "AVI"

# Kalite seviyeleri için enum
class KaliteSeviyesi(Enum):
    DUSUK = "Düşük"
    ORTA = "Orta"
    YUKSEK = "Yüksek"
    KAYIPSIZ = "Kayıpsız"

@dataclass
class Metadata:
    """Medya metadata bilgileri için veri sınıfı"""
    baslik: str
    sanatci: str
    album: Optional[str] = None
    yil: Optional[int] = None
    tur: Optional[str] = None
    sure: Optional[int] = None  # saniye cinsinden
    etiketler: List[str] = None
    
    def __post_init__(self):
        if self.etiketler is None:
            self.etiketler = []

@dataclass
class KaliteAyari:
    """Kalite ayarları için veri sınıfı"""
    seviye: KaliteSeviyesi
    bit_hizi: int  # kbps
    ornekleme_hizi: Optional[int] = None  # Hz
    kanal_sayisi: int = 2
    cozunurluk: Optional[tuple[int, int]] = None  # (genişlik, yükseklik)

class MedyaOynatici:
    """Temel medya oynatıcı sınıfı"""
    def __init__(self):
        self._medya_listesi: List[Dict] = []
        self._mevcut_medya: Optional[Dict] = None
        self._oynatma_listesi: List[Dict] = []
        self._oynatiliyor: bool = False
        self._duraklatildi: bool = False
        self._mevcut_konum: int = 0  # saniye cinsinden
    
    def medya_ekle(self, dosya_yolu: str, metadata: Metadata,
                   format: FormatTuru) -> bool:
        """Yeni medya ekler"""
        medya = {
            "dosya_yolu": dosya_yolu,
            "metadata": metadata,
            "format": format,
            "eklenme_tarihi": datetime.now()
        }
        self._medya_listesi.append(medya)
        return True
    
    def oynatma_listesi_olustur(self, ad: str) -> int:
        """Yeni oynatma listesi oluşturur"""
        liste = {
            "id": len(self._oynatma_listesi) + 1,
            "ad": ad,
            "medyalar": [],
            "olusturma_tarihi": datetime.now(),
            "toplam_sure": 0
        }
        self._oynatma_listesi.append(liste)
        return liste["id"]
    
    def listeye_ekle(self, liste_id: int, medya: Dict) -> bool:
        """Oynatma listesine medya ekler"""
        liste = next(
            (l for l in self._oynatma_listesi if l["id"] == liste_id),
            None
        )
        if not liste:
            raise ValueError("Liste bulunamadı")
        
        liste["medyalar"].append(medya)
        if medya["metadata"].sure:
            liste["toplam_sure"] += medya["metadata"].sure
        return True
    
    def oynat(self, medya: Dict) -> bool:
        """Medya oynatma işlemi"""
        self._mevcut_medya = medya
        self._oynatiliyor = True
        self._duraklatildi = False
        self._mevcut_konum = 0
        return True
    
    def duraklat(self) -> bool:
        """Oynatmayı duraklatır"""
        if self._oynatiliyor and not self._duraklatildi:
            self._duraklatildi = True
            return True
        return False
    
    def devam_et(self) -> bool:
        """Duraklatılmış medyayı devam ettirir"""
        if self._oynatiliyor and self._duraklatildi:
            self._duraklatildi = False
            return True
        return False
    
    def durdur(self) -> bool:
        """Oynatmayı durdurur"""
        if self._oynatiliyor:
            self._oynatiliyor = False
            self._duraklatildi = False
            self._mevcut_konum = 0
            return True
        return False
    
    def ileri_sar(self, saniye: int) -> bool:
        """İleri sarar"""
        if not self._mevcut_medya or not self._oynatiliyor:
            return False
        
        yeni_konum = self._mevcut_konum + saniye
        if self._mevcut_medya["metadata"].sure:
            yeni_konum = min(yeni_konum, self._mevcut_medya["metadata"].sure)
        
        self._mevcut_konum = yeni_konum
        return True
    
    def geri_sar(self, saniye: int) -> bool:
        """Geri sarar"""
        if not self._mevcut_medya or not self._oynatiliyor:
            return False
        
        self._mevcut_konum = max(0, self._mevcut_konum - saniye)
        return True
    
    def durum_bilgisi(self) -> str:
        """Oynatıcı durum bilgilerini döndürür"""
        if not self._mevcut_medya:
            return "Medya yok"
        
        metadata = self._mevcut_medya["metadata"]
        durum = "Oynatılıyor" if self._oynatiliyor and not self._duraklatildi else "Duraklatıldı"
        
        return f"""
        Şu an oynatılıyor:
        -----------------
        Başlık: {metadata.baslik}
        Sanatçı: {metadata.sanatci}
        Albüm: {metadata.album or 'N/A'}
        Süre: {timedelta(seconds=metadata.sure) if metadata.sure else 'N/A'}
        Konum: {timedelta(seconds=self._mevcut_konum)}
        Durum: {durum}
        """

class MuzikOynatici(MedyaOynatici):
    """Müzik oynatıcı sınıfı"""
    def __init__(self):
        super().__init__()
        self._equalizer_aktif = False
        self._equalizer_ayarlari = {
            "bass": 0,
            "treble": 0,
            "mid": 0
        }
        self._varsayilan_format = FormatTuru.MP3
    
    def equalizer_ayarla(self, bass: int, treble: int, mid: int) -> bool:
        """Equalizer ayarlarını günceller"""
        if not all(-12 <= x <= 12 for x in [bass, treble, mid]):
            raise ValueError("Equalizer değerleri -12 ile +12 arasında olmalıdır")
        
        self._equalizer_ayarlari.update({
            "bass": bass,
            "treble": treble,
            "mid": mid
        })
        return True
    
    def equalizer_sifirla(self) -> None:
        """Equalizer ayarlarını sıfırlar"""
        self._equalizer_ayarlari = {
            "bass": 0,
            "treble": 0,
            "mid": 0
        }
    
    def format_donustur(self, medya: Dict, hedef_format: FormatTuru) -> bool:
        """Müzik dosyası formatını dönüştürür"""
        if medya["format"] == hedef_format:
            return False
        
        if hedef_format not in [FormatTuru.MP3, FormatTuru.WAV, FormatTuru.FLAC]:
            raise ValueError("Desteklenmeyen format")
        
        # Format dönüştürme simülasyonu
        medya["format"] = hedef_format
        return True

class VideoOynatici(MedyaOynatici):
    """Video oynatıcı sınıfı"""
    def __init__(self):
        super().__init__()
        self._altyazi_aktif = False
        self._altyazi_dili = None
        self._varsayilan_format = FormatTuru.MP4
        self._ekran_modu = "normal"  # normal, tam ekran, sinema
    
    def altyazi_ac(self, dil: str) -> bool:
        """Altyazıyı açar"""
        self._altyazi_aktif = True
        self._altyazi_dili = dil
        return True
    
    def altyazi_kapat(self) -> bool:
        """Altyazıyı kapatır"""
        self._altyazi_aktif = False
        self._altyazi_dili = None
        return True
    
    def ekran_modu_degistir(self, mod: str) -> bool:
        """Ekran modunu değiştirir"""
        if mod not in ["normal", "tam ekran", "sinema"]:
            raise ValueError("Geçersiz ekran modu")
        
        self._ekran_modu = mod
        return True
    
    def cozunurluk_degistir(self, medya: Dict, 
                           cozunurluk: tuple[int, int]) -> bool:
        """Video çözünürlüğünü değiştirir"""
        if not medya["metadata"].sure:  # Video değilse
            return False
        
        # Çözünürlük değiştirme simülasyonu
        medya["kalite"].cozunurluk = cozunurluk
        return True

class PodcastOynatici(MedyaOynatici):
    """Podcast oynatıcı sınıfı"""
    def __init__(self):
        super().__init__()
        self._oynatma_hizi = 1.0  # normal hız
        self._sessiz_bolge_atla = False
        self._varsayilan_format = FormatTuru.MP3
        self._bolum_isaretleri: List[Dict] = []
    
    def hiz_ayarla(self, hiz: float) -> bool:
        """Oynatma hızını ayarlar"""
        if not 0.5 <= hiz <= 3.0:
            raise ValueError("Hız 0.5x ile 3.0x arasında olmalıdır")
        
        self._oynatma_hizi = hiz
        return True
    
    def sessiz_bolge_atlamayi_ac(self) -> None:
        """Sessiz bölge atlama özelliğini açar"""
        self._sessiz_bolge_atla = True
    
    def bolum_isareti_ekle(self, konum: int, aciklama: str) -> bool:
        """Bölüm işareti ekler"""
        if not self._mevcut_medya:
            return False
        
        isaret = {
            "konum": konum,
            "aciklama": aciklama,
            "ekleme_tarihi": datetime.now()
        }
        self._bolum_isaretleri.append(isaret)
        return True
    
    def bolum_isaretlerine_git(self, indeks: int) -> bool:
        """Belirtilen bölüm işaretine gider"""
        if not 0 <= indeks < len(self._bolum_isaretleri):
            return False
        
        self._mevcut_konum = self._bolum_isaretleri[indeks]["konum"]
        return True

# Test fonksiyonu
def ornek_kullanim():
    # Oynatıcıları oluştur
    muzik = MuzikOynatici()
    video = VideoOynatici()
    podcast = PodcastOynatici()
    
    try:
        # Müzik için örnek
        muzik_metadata = Metadata(
            "Bohemian Rhapsody",
            "Queen",
            "A Night at the Opera",
            1975,
            "Rock",
            354,
            ["rock", "classic rock"]
        )
        
        muzik.medya_ekle(
            "muzik/bohemian_rhapsody.mp3",
            muzik_metadata,
            FormatTuru.MP3
        )
        
        # Video için örnek
        video_metadata = Metadata(
            "Inception",
            "Christopher Nolan",
            yil=2010,
            tur="Bilim Kurgu",
            sure=8880
        )
        
        video.medya_ekle(
            "video/inception.mp4",
            video_metadata,
            FormatTuru.MP4
        )
        
        # Podcast için örnek
        podcast_metadata = Metadata(
            "Yazılım Geliştirme Süreçleri",
            "Kod Akademi",
            sure=3600,
            etiketler=["yazılım", "geliştirme", "agile"]
        )
        
        podcast.medya_ekle(
            "podcast/yazilim_gelistirme.mp3",
            podcast_metadata,
            FormatTuru.MP3
        )
        
        # Oynatma listeleri oluştur
        muzik_listesi = muzik.oynatma_listesi_olustur("Rock Klasikleri")
        video_listesi = video.oynatma_listesi_olustur("Bilim Kurgu")
        podcast_listesi = podcast.oynatma_listesi_olustur("Yazılım Dersleri")
        
        # Medyaları oynat ve özelleştir
        muzik.oynat(muzik._medya_listesi[0])
        muzik.equalizer_ayarla(5, 2, -1)
        
        video.oynat(video._medya_listesi[0])
        video.altyazi_ac("Türkçe")
        video.ekran_modu_degistir("sinema")
        
        podcast.oynat(podcast._medya_listesi[0])
        podcast.hiz_ayarla(1.5)
        podcast.sessiz_bolge_atlamayi_ac()
        
        # Durum bilgilerini göster
        print("Müzik Oynatıcı:")
        print(muzik.durum_bilgisi())
        
        print("\\nVideo Oynatıcı:")
        print(video.durum_bilgisi())
        
        print("\\nPodcast Oynatıcı:")
        print(podcast.durum_bilgisi())
        
    except ValueError as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    ornek_kullanim()
\`\`\`

## Kod Açıklamaları

### 1. Veri Yapıları
- **Enum Sınıfları**: Medya türleri, format türleri ve kalite seviyeleri için sabit değerler
- **Dataclass'lar**: Metadata ve kalite ayarları için veri sınıfları
- **Temel Sınıf**: Tüm oynatıcılar için ortak özellikleri içeren \`MedyaOynatici\` sınıfı

### 2. Kalıtım Hiyerarşisi
- \`MedyaOynatici\`: Temel sınıf
- \`MuzikOynatici\`: Equalizer ve format dönüştürme özellikleri
- \`VideoOynatici\`: Altyazı ve ekran modu yönetimi
- \`PodcastOynatici\`: Oynatma hızı ve bölüm işaretleri

### 3. Özelleştirilmiş Davranışlar
- Her oynatıcı türü kendi özel özelliklerine sahip
- Format ve kalite yönetimi her tür için özelleştirilmiş
- Oynatma kontrolleri medya türüne göre adapte edilmiş

## Önerilen Geliştirmeler

1. **Medya Yönetimi**
   - Otomatik metadata çıkarma
   - Akıllı sıralama
   - Çevrimdışı depolama

2. **Kullanıcı Arayüzü**
   - Görsel equalizer
   - Gelişmiş kontroller
   - Tema desteği

3. **Akış Özellikleri**
   - Çevrimiçi akış
   - Kalite otomatik ayarlama
   - Önbellek yönetimi

4. **Sosyal Özellikler**
   - Paylaşım seçenekleri
   - Yorum sistemi
   - Oynatma listeleri paylaşımı

## Sonraki Adımlar

Bu sistemi daha da geliştirebilirsiniz:

1. Veritabanı entegrasyonu
2. Web arayüzü
3. Mobil uygulama
4. Bulut senkronizasyonu

Bu geliştirmeler, sistemin profesyonel bir medya oynatıcı olarak kullanılabilir hale gelmesini sağlayacaktır.
`;

export default function MedyaOynaticiSistemiPage() {
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