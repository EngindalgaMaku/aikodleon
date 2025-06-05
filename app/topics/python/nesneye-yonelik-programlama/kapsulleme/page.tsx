import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python OOP: Kapsülleme (Encapsulation) | Kodleon',
  description: 'Python\'da kapsülleme kavramını, private ve protected üyeleri, property dekoratörlerini detaylı örneklerle öğrenin.',
};

const content = `
# Python'da Kapsülleme (Encapsulation)

Kapsülleme (Encapsulation), nesne yönelimli programlamanın temel prensiplerinden biridir. Bu kavram, bir sınıfın verilerini ve bu veriler üzerinde çalışan metodları bir arada tutarak, dış dünyadan gizlemeyi ve korumayı amaçlar.

## Kapsüllemenin Temel Prensipleri

1. **Veri Gizleme**: Sınıf içindeki verilere doğrudan erişimi kısıtlama
2. **Kontrollü Erişim**: Verilere sadece metodlar aracılığıyla erişim sağlama
3. **Veri Doğrulama**: Verilerin değiştirilmesi sırasında kontrol mekanizmaları oluşturma

## Private ve Protected Üyeler

Python'da üç tür erişim belirleyici vardır:

1. **Public** (Genel): Herhangi bir alt çizgi olmayan üyeler
2. **Protected** (Korumalı): Tek alt çizgi (_) ile başlayan üyeler
3. **Private** (Özel): Çift alt çizgi (__) ile başlayan üyeler

### Detaylı Örnek: Banka Hesabı

\`\`\`python
class BankaHesabi:
    def __init__(self, hesap_no, bakiye, sahip_adi):
        self.hesap_no = hesap_no      # Public üye
        self._sahip_adi = sahip_adi   # Protected üye
        self.__bakiye = bakiye        # Private üye
        self.__islemler = []          # Private işlem geçmişi
    
    def para_yatir(self, miktar):
        if miktar > 0:
            self.__bakiye += miktar
            self.__islem_kaydet(f"Para yatırma: +{miktar} TL")
            return True
        return False
    
    def para_cek(self, miktar):
        if miktar > 0 and self.__bakiye >= miktar:
            self.__bakiye -= miktar
            self.__islem_kaydet(f"Para çekme: -{miktar} TL")
            return True
        return False
    
    def bakiye_goruntule(self):
        return f"Güncel bakiye: {self.__bakiye} TL"
    
    def __islem_kaydet(self, islem):
        from datetime import datetime
        tarih = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.__islemler.append(f"{tarih} - {islem}")
    
    def islem_gecmisi(self):
        return "\\n".join(self.__islemler)

# Kullanım örneği
hesap = BankaHesabi("12345", 1000, "Ahmet Yılmaz")

# Public üyeye erişim
print(hesap.hesap_no)  # Direkt erişilebilir

# Protected üyeye erişim
print(hesap._sahip_adi)  # Erişilebilir ama önerilmez

# Private üyeye erişim denemesi
try:
    print(hesap.__bakiye)  # AttributeError hatası verir
except AttributeError:
    print("Private üyelere doğrudan erişilemez!")

# Doğru kullanım
print(hesap.bakiye_goruntule())
hesap.para_yatir(500)
hesap.para_cek(200)
print(hesap.islem_gecmisi())
\`\`\`

### Açıklama:
- \`hesap_no\`: Public üye, herkes tarafından erişilebilir ve değiştirilebilir
- \`_sahip_adi\`: Protected üye, erişilebilir ama değiştirilmemesi önerilir
- \`__bakiye\`: Private üye, sadece sınıf içinden erişilebilir
- \`__islemler\`: Private liste, işlem geçmişini güvenli şekilde tutar
- \`__islem_kaydet\`: Private metod, sadece sınıf içinden çağrılabilir

## Property Dekoratörleri

Property'ler, sınıf özelliklerine güvenli ve kontrollü erişim sağlar. Üç temel bileşeni vardır:
1. **getter**: Veriyi okuma
2. **setter**: Veriyi değiştirme
3. **deleter**: Veriyi silme

### Detaylı Örnek: Sıcaklık Dönüştürücü

\`\`\`python
class SicaklikDonusturucu:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        """Sıcaklığı Celsius cinsinden döndürür"""
        return self._celsius
    
    @celsius.setter
    def celsius(self, deger):
        """Celsius değerini kontrollü şekilde ayarlar"""
        if deger < -273.15:  # Mutlak sıfır kontrolü
            raise ValueError("Sıcaklık mutlak sıfırdan düşük olamaz!")
        self._celsius = deger
    
    @property
    def fahrenheit(self):
        """Sıcaklığı Fahrenheit cinsinden döndürür"""
        return (self._celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, deger):
        """Fahrenheit değerini Celsius'a çevirip ayarlar"""
        celsius = (deger - 32) * 5/9
        if celsius < -273.15:
            raise ValueError("Sıcaklık mutlak sıfırdan düşük olamaz!")
        self._celsius = celsius
    
    @property
    def kelvin(self):
        """Sıcaklığı Kelvin cinsinden döndürür"""
        return self._celsius + 273.15
    
    @kelvin.setter
    def kelvin(self, deger):
        """Kelvin değerini Celsius'a çevirip ayarlar"""
        celsius = deger - 273.15
        if celsius < -273.15:
            raise ValueError("Sıcaklık mutlak sıfırdan düşük olamaz!")
        self._celsius = celsius

# Kullanım örnekleri
sicaklik = SicaklikDonusturucu(25)  # 25°C ile başla

# Farklı birimlerde okuma
print(f"Celsius: {sicaklik.celsius}°C")
print(f"Fahrenheit: {sicaklik.fahrenheit}°F")
print(f"Kelvin: {sicaklik.kelvin}K")

# Farklı birimlerde değer atama
sicaklik.fahrenheit = 68  # 20°C
print(f"Yeni Celsius: {sicaklik.celsius}°C")

sicaklik.kelvin = 300  # 26.85°C
print(f"Yeni Celsius: {sicaklik.celsius}°C")

# Hata kontrolü
try:
    sicaklik.celsius = -300  # Mutlak sıfırdan düşük!
except ValueError as e:
    print(f"Hata: {e}")
\`\`\`

## Pratik Örnek: E-Ticaret Ürün Yönetimi

\`\`\`python
class Urun:
    def __init__(self, ad, fiyat, stok):
        self._ad = ad
        self._fiyat = fiyat
        self._stok = stok
        self._satis_sayisi = 0
        self._indirim_orani = 0
    
    @property
    def ad(self):
        return self._ad
    
    @property
    def fiyat(self):
        """İndirimli fiyatı hesaplar ve döndürür"""
        indirimli_fiyat = self._fiyat * (1 - self._indirim_orani)
        return round(indirimli_fiyat, 2)
    
    @fiyat.setter
    def fiyat(self, yeni_fiyat):
        """Fiyatı kontrollü şekilde günceller"""
        if yeni_fiyat < 0:
            raise ValueError("Fiyat negatif olamaz!")
        self._fiyat = yeni_fiyat
    
    @property
    def stok(self):
        return self._stok
    
    def stok_ekle(self, miktar):
        """Stok miktarını artırır"""
        if miktar > 0:
            self._stok += miktar
            return True
        return False
    
    def stok_cikar(self, miktar):
        """Stok miktarını azaltır ve satış sayısını günceller"""
        if 0 < miktar <= self._stok:
            self._stok -= miktar
            self._satis_sayisi += miktar
            return True
        return False
    
    @property
    def satis_sayisi(self):
        return self._satis_sayisi
    
    def indirim_uygula(self, oran):
        """İndirim oranını ayarlar (0-1 arası)"""
        if 0 <= oran <= 1:
            self._indirim_orani = oran
            return True
        return False
    
    def __str__(self):
        return f"""
Ürün: {self._ad}
Orijinal Fiyat: {self._fiyat} TL
İndirim Oranı: %{self._indirim_orani * 100}
Güncel Fiyat: {self.fiyat} TL
Stok: {self._stok}
Satış Sayısı: {self._satis_sayisi}
"""

# Kullanım örneği
laptop = Urun("Gaming Laptop", 25000, 10)
print(laptop)

# Fiyat güncelleme
laptop.fiyat = 27500
print(f"Yeni fiyat: {laptop.fiyat} TL")

# İndirim uygulama
laptop.indirim_uygula(0.15)  # %15 indirim
print(f"İndirimli fiyat: {laptop.fiyat} TL")

# Stok işlemleri
laptop.stok_ekle(5)
print(f"Güncel stok: {laptop.stok}")

laptop.stok_cikar(3)
print(f"Kalan stok: {laptop.stok}")
print(f"Toplam satış: {laptop.satis_sayisi}")

print(laptop)
\`\`\`

## Alıştırmalar

1. **Banka Hesabı Geliştirme** 
   <a href="/topics/python/nesneye-yonelik-programlama/kapsulleme/banka-hesabi-gelistirme" target="_blank" rel="noopener noreferrer">Detaylı çözüm için tıklayın</a>
   - Yukarıdaki BankaHesabi sınıfına şu özellikleri ekleyin:
     - Hesap limiti kontrolü
     - Günlük para çekme limiti
     - Hesap dondurma/açma özelliği
     - Detaylı işlem raporu

2. **Kütüphane Üye Sistemi**
   <a href="/topics/python/nesneye-yonelik-programlama/kapsulleme/kutuphane-uye-sistemi" target="_blank" rel="noopener noreferrer">Detaylı çözüm için tıklayın</a>
   - Üye bilgilerini kapsülleyen bir sistem oluşturun:
     - Üye kişisel bilgilerinin gizliliği
     - Ödünç alınan kitap takibi
     - Gecikme cezası hesaplama
     - Üyelik durumu kontrolü

3. **Araç Kiralama Sistemi**
   <a href="/topics/python/nesneye-yonelik-programlama/kapsulleme/arac-kiralama-sistemi" target="_blank" rel="noopener noreferrer">Detaylı çözüm için tıklayın</a>
   - Araç bilgilerini ve kiralama işlemlerini yöneten bir sistem yapın:
     - Araç durumu takibi
     - Kiralama ücreti hesaplama
     - Kilometre sınırı kontrolü
     - Bakım takibi

## Sonraki Adımlar

Kapsülleme konusunu detaylı örneklerle öğrendiniz. Şimdi sırada çok biçimlilik (polymorphism) konusu var. Bu konuda, aynı arayüzü farklı sınıflarda nasıl kullanacağımızı ve kodumuzun esnekliğini nasıl artıracağımızı öğreneceğiz.
`;

export default function EncapsulationPage() {
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
      
      <div className="mt-12 flex flex-col sm:flex-row justify-between gap-4">
        <Button asChild variant="outline" className="gap-2">
          <Link href="/topics/python/nesneye-yonelik-programlama/kalitim">
            <ArrowLeft className="h-4 w-4" />
            Önceki Konu: Kalıtım
          </Link>
        </Button>
        
        <Button asChild variant="default" className="gap-2">
          <Link href="/topics/python/nesneye-yonelik-programlama/cok-bicimlilk">
            Sonraki Konu: Çok Biçimlilik
            <ArrowRight className="h-4 w-4" />
          </Link>
        </Button>
      </div>
      
      <div className="mt-16 text-center text-sm text-muted-foreground">
        <p>© {new Date().getFullYear()} Kodleon | Yapay Zeka Eğitim Platformu</p>
      </div>
    </div>
  );
} 