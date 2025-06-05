import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python OOP: Banka Hesabı Geliştirme | Kodleon',
  description: 'Python\'da kapsülleme prensiplerini kullanarak gelişmiş bir banka hesabı sistemi oluşturmayı öğrenin.',
};

const content = `
# Gelişmiş Banka Hesabı Sistemi

Bu örnekte, kapsülleme prensiplerini kullanarak kapsamlı bir banka hesabı sistemi geliştireceğiz. Sistem, gerçek dünya senaryolarına uygun olarak tasarlanmıştır.

## Özellikler

1. **Hesap Limiti Kontrolü**
   - Hesap türüne göre farklı limitler
   - Eksiye düşme kontrolü
   - Limit aşım uyarıları

2. **Günlük Para Çekme Limiti**
   - 24 saatlik periyot takibi
   - Limit sıfırlama mekanizması
   - Limit aşım kontrolü

3. **Hesap Dondurma/Açma**
   - Güvenlik kontrolü
   - Dondurma nedeni kaydı
   - Otomatik/manuel dondurma

4. **Detaylı İşlem Raporu**
   - Tarih/saat kaydı
   - İşlem kategorileri
   - Bakiye değişim geçmişi

## Kod Örneği ve Açıklamalar

\`\`\`python
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional

# Hesap türleri için enum sınıfı
class HesapTuru(Enum):
    STANDART = "Standart"
    PREMIUM = "Premium"
    KURUMSAL = "Kurumsal"

# İşlem kategorileri için enum sınıfı
class IslemTuru(Enum):
    PARA_YATIRMA = "Para Yatırma"
    PARA_CEKME = "Para Çekme"
    HAVALE_GELEN = "Havale Gelen"
    HAVALE_GIDEN = "Havale Giden"
    HESAP_DONDURMA = "Hesap Dondurma"
    HESAP_AKTIVASYON = "Hesap Aktivasyon"

class BankaHesabi:
    # Hesap türlerine göre limitler (private class attribute)
    __HESAP_LIMITLERI = {
        HesapTuru.STANDART: {"gunluk_cekim": 5000, "eksiye_dusme": 0},
        HesapTuru.PREMIUM: {"gunluk_cekim": 10000, "eksiye_dusme": -1000},
        HesapTuru.KURUMSAL: {"gunluk_cekim": 50000, "eksiye_dusme": -5000},
    }

    def __init__(self, hesap_no: str, sahip_adi: str, hesap_turu: HesapTuru = HesapTuru.STANDART):
        """
        Banka hesabı başlatıcı metodu.
        
        Args:
            hesap_no (str): Benzersiz hesap numarası
            sahip_adi (str): Hesap sahibinin adı
            hesap_turu (HesapTuru): Hesap türü (varsayılan: STANDART)
        """
        # Private attributes
        self.__hesap_no = hesap_no
        self.__bakiye = 0.0
        self.__hesap_turu = hesap_turu
        self.__dondurulmus = False
        self.__dondurma_nedeni = None
        self.__islem_gecmisi: List[Dict] = []
        self.__gunluk_cekim = 0.0
        self.__son_cekim_tarihi = datetime.now()
        
        # Protected attributes
        self._sahip_adi = sahip_adi
        self._olusturma_tarihi = datetime.now()
        
        # İlk işlem kaydı: Hesap açılışı
        self.__islem_kaydet("Hesap Açılış", 0, "Hesap başarıyla oluşturuldu")

    @property
    def hesap_no(self) -> str:
        """Hesap numarasını döndürür."""
        return self.__hesap_no

    @property
    def bakiye(self) -> float:
        """Güncel bakiyeyi döndürür."""
        return self.__bakiye

    @property
    def hesap_turu(self) -> HesapTuru:
        """Hesap türünü döndürür."""
        return self.__hesap_turu

    @property
    def dondurulmus(self) -> bool:
        """Hesabın dondurulma durumunu döndürür."""
        return self.__dondurulmus

    def __gunluk_limit_kontrolu(self, miktar: float) -> bool:
        """
        Günlük para çekme limitini kontrol eder.
        
        Args:
            miktar (float): Çekilmek istenen miktar
            
        Returns:
            bool: Limit aşılmamışsa True, aşılmışsa False
        """
        simdi = datetime.now()
        
        # Yeni güne geçildiyse limiti sıfırla
        if (simdi - self.__son_cekim_tarihi).days >= 1:
            self.__gunluk_cekim = 0
            self.__son_cekim_tarihi = simdi
        
        # Limit kontrolü
        yeni_toplam = self.__gunluk_cekim + miktar
        gunluk_limit = self.__HESAP_LIMITLERI[self.__hesap_turu]["gunluk_cekim"]
        
        return yeni_toplam <= gunluk_limit

    def __eksiye_dusme_kontrolu(self, miktar: float) -> bool:
        """
        Eksiye düşme limitini kontrol eder.
        
        Args:
            miktar (float): Çekilmek istenen miktar
            
        Returns:
            bool: Limit aşılmamışsa True, aşılmışsa False
        """
        yeni_bakiye = self.__bakiye - miktar
        izin_verilen_eksi = self.__HESAP_LIMITLERI[self.__hesap_turu]["eksiye_dusme"]
        
        return yeni_bakiye >= izin_verilen_eksi

    def __islem_kaydet(self, islem_turu: str, miktar: float, aciklama: str) -> None:
        """
        İşlem geçmişine yeni bir kayıt ekler.
        
        Args:
            islem_turu (str): İşlemin türü
            miktar (float): İşlem miktarı
            aciklama (str): İşlem açıklaması
        """
        islem = {
            "tarih": datetime.now(),
            "islem_turu": islem_turu,
            "miktar": miktar,
            "bakiye": self.__bakiye,
            "aciklama": aciklama
        }
        self.__islem_gecmisi.append(islem)

    def para_yatir(self, miktar: float) -> bool:
        """
        Hesaba para yatırma işlemi.
        
        Args:
            miktar (float): Yatırılacak miktar
            
        Returns:
            bool: İşlem başarılıysa True, değilse False
        """
        if self.__dondurulmus:
            raise ValueError("Hesap dondurulmuş durumda!")
            
        if miktar <= 0:
            raise ValueError("Geçersiz miktar!")
            
        self.__bakiye += miktar
        self.__islem_kaydet(
            IslemTuru.PARA_YATIRMA.value,
            miktar,
            f"{miktar} TL yatırıldı"
        )
        return True

    def para_cek(self, miktar: float) -> bool:
        """
        Hesaptan para çekme işlemi.
        
        Args:
            miktar (float): Çekilecek miktar
            
        Returns:
            bool: İşlem başarılıysa True, değilse False
            
        Raises:
            ValueError: Hesap dondurulmuşsa, miktar geçersizse veya limitler aşılmışsa
        """
        if self.__dondurulmus:
            raise ValueError("Hesap dondurulmuş durumda!")
            
        if miktar <= 0:
            raise ValueError("Geçersiz miktar!")
            
        if not self.__gunluk_limit_kontrolu(miktar):
            raise ValueError("Günlük para çekme limiti aşıldı!")
            
        if not self.__eksiye_dusme_kontrolu(miktar):
            raise ValueError("Yetersiz bakiye veya eksiye düşme limiti aşıldı!")
            
        self.__bakiye -= miktar
        self.__gunluk_cekim += miktar
        self.__son_cekim_tarihi = datetime.now()
        
        self.__islem_kaydet(
            IslemTuru.PARA_CEKME.value,
            miktar,
            f"{miktar} TL çekildi"
        )
        return True

    def hesabi_dondur(self, nedeni: str) -> bool:
        """
        Hesabı dondurma işlemi.
        
        Args:
            nedeni (str): Dondurma nedeni
            
        Returns:
            bool: İşlem başarılıysa True, değilse False
        """
        if self.__dondurulmus:
            return False
            
        self.__dondurulmus = True
        self.__dondurma_nedeni = nedeni
        
        self.__islem_kaydet(
            IslemTuru.HESAP_DONDURMA.value,
            0,
            f"Hesap donduruldu. Nedeni: {nedeni}"
        )
        return True

    def hesabi_ac(self) -> bool:
        """
        Dondurulmuş hesabı açma işlemi.
        
        Returns:
            bool: İşlem başarılıysa True, değilse False
        """
        if not self.__dondurulmus:
            return False
            
        self.__dondurulmus = False
        self.__dondurma_nedeni = None
        
        self.__islem_kaydet(
            IslemTuru.HESAP_AKTIVASYON.value,
            0,
            "Hesap aktif edildi"
        )
        return True

    def islem_raporu_al(self, baslangic_tarih: Optional[datetime] = None, 
                       bitis_tarih: Optional[datetime] = None) -> List[Dict]:
        """
        Belirtilen tarih aralığındaki işlem geçmişini döndürür.
        
        Args:
            baslangic_tarih (datetime, optional): Başlangıç tarihi
            bitis_tarih (datetime, optional): Bitiş tarihi
            
        Returns:
            List[Dict]: İşlem kayıtları listesi
        """
        if not baslangic_tarih:
            baslangic_tarih = self._olusturma_tarihi
        if not bitis_tarih:
            bitis_tarih = datetime.now()
            
        return [
            islem for islem in self.__islem_gecmisi
            if baslangic_tarih <= islem["tarih"] <= bitis_tarih
        ]

    def ozet_bilgi(self) -> str:
        """
        Hesap özet bilgilerini döndürür.
        
        Returns:
            str: Hesap özeti
        """
        return f"""
        Hesap Özeti
        -----------
        Hesap No: {self.__hesap_no}
        Sahip: {self._sahip_adi}
        Tür: {self.__hesap_turu.value}
        Bakiye: {self.__bakiye} TL
        Durum: {"Dondurulmuş" if self.__dondurulmus else "Aktif"}
        {f"Dondurma Nedeni: {self.__dondurma_nedeni}" if self.__dondurulmus else ""}
        Günlük Çekim Limiti: {self.__HESAP_LIMITLERI[self.__hesap_turu]["gunluk_cekim"]} TL
        Bugünkü Çekim: {self.__gunluk_cekim} TL
        """

# Kullanım Örneği
def ornek_kullanim():
    # Hesap oluşturma
    hesap = BankaHesabi("12345", "Ahmet Yılmaz", HesapTuru.PREMIUM)
    
    try:
        # Para yatırma
        hesap.para_yatir(5000)
        
        # Para çekme
        hesap.para_cek(2000)
        
        # Hesap dondurma
        hesap.hesabi_dondur("Güvenlik şüphesi")
        
        # Dondurulmuş hesaptan para çekme denemesi
        hesap.para_cek(1000)  # ValueError: Hesap dondurulmuş durumda!
        
    except ValueError as e:
        print(f"Hata: {e}")
        
    # Hesabı tekrar açma
    hesap.hesabi_ac()
    
    # İşlem raporu alma
    bir_hafta_once = datetime.now() - timedelta(days=7)
    islemler = hesap.islem_raporu_al(bir_hafta_once)
    
    # Özet bilgi
    print(hesap.ozet_bilgi())

if __name__ == "__main__":
    ornek_kullanim()
\`\`\`

## Kod Açıklamaları

### 1. Sınıf Yapısı

- **Enum Sınıfları**: \`HesapTuru\` ve \`IslemTuru\` ile sabit değerler tanımlanmıştır.
- **Private Class Attribute**: \`__HESAP_LIMITLERI\` ile hesap türlerine göre limitler belirlenmiştir.
- **Instance Attributes**: Private ve protected üye değişkenler tanımlanmıştır.

### 2. Hesap Limiti Kontrolü

- Her hesap türü için farklı limitler tanımlanmıştır:
  - Standart: 5.000 TL günlük, eksiye düşemez
  - Premium: 10.000 TL günlük, -1.000 TL'ye kadar eksi
  - Kurumsal: 50.000 TL günlük, -5.000 TL'ye kadar eksi

### 3. Günlük Para Çekme Limiti

- \`__gunluk_limit_kontrolu\` metodu ile:
  - 24 saatlik periyot takibi yapılır
  - Gün değişiminde limit sıfırlanır
  - Toplam çekim miktarı kontrol edilir

### 4. Hesap Dondurma/Açma

- \`hesabi_dondur\` ve \`hesabi_ac\` metodları ile:
  - Dondurma nedeni kaydedilir
  - İşlem geçmişine kayıt eklenir
  - Dondurulmuş hesapta işlem yapılması engellenir

### 5. Detaylı İşlem Raporu

- \`__islem_kaydet\` metodu ile her işlem için:
  - Tarih ve saat
  - İşlem türü
  - Miktar
  - İşlem sonrası bakiye
  - Açıklama kaydedilir

### 6. Güvenlik Önlemleri

- Private metodlar ve değişkenler
- İşlem öncesi kontroller
- Hata yönetimi
- Durum kontrolleri

## Örnek Senaryolar

### Senaryo 1: Normal Kullanım

\`\`\`python
hesap = BankaHesabi("12345", "Ahmet Yılmaz", HesapTuru.PREMIUM)
hesap.para_yatir(5000)
hesap.para_cek(2000)
print(hesap.ozet_bilgi())
\`\`\`

### Senaryo 2: Limit Aşımı

\`\`\`python
hesap = BankaHesabi("12345", "Ahmet Yılmaz", HesapTuru.STANDART)
hesap.para_yatir(10000)
try:
    hesap.para_cek(6000)  # Günlük limit aşımı
except ValueError as e:
    print(f"Hata: {e}")
\`\`\`

### Senaryo 3: Dondurulmuş Hesap

\`\`\`python
hesap = BankaHesabi("12345", "Ahmet Yılmaz")
hesap.para_yatir(1000)
hesap.hesabi_dondur("Güvenlik şüphesi")
try:
    hesap.para_cek(500)  # Dondurulmuş hesaptan çekim
except ValueError as e:
    print(f"Hata: {e}")
\`\`\`

## Önerilen Geliştirmeler

1. **SMS/Email Bildirimleri**
   - İşlem onayları
   - Limit uyarıları
   - Güvenlik bildirimleri

2. **İşlem Kategorileri**
   - Otomatik ödemeler
   - Düzenli transferler
   - E-ticaret işlemleri

3. **Güvenlik Geliştirmeleri**
   - İşlem şifreleri
   - IP kontrolü
   - Şüpheli işlem tespiti

4. **Raporlama Geliştirmeleri**
   - Grafik raporlar
   - Kategori bazlı analiz
   - Aylık/yıllık özetler

## Sonraki Adımlar

Bu örnek projeyi daha da geliştirebilirsiniz:

1. Veritabanı entegrasyonu ekleyerek
2. Web API'ye dönüştürerek
3. Kullanıcı arayüzü ekleyerek
4. Birim testler yazarak

Bu geliştirmeler, projenin gerçek dünya uygulamalarına daha uygun hale gelmesini sağlayacaktır.
`;

export default function BankaHesabiGelistirmePage() {
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