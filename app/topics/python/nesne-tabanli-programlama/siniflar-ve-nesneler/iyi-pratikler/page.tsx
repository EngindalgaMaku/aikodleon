import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowRight, Code2, Settings, Terminal, BookOpen } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python OOP İyi Pratikler | Kodleon',
  description: 'Python nesne tabanlı programlamada iyi pratikler, tasarım prensipleri ve kod organizasyonu.',
};

const content = `
# Python OOP İyi Pratikler

Nesne tabanlı programlamada iyi pratikler, kodunuzun daha okunabilir, bakımı kolay ve yeniden kullanılabilir olmasını sağlar.

## İsimlendirme Kuralları

### Sınıf İsimleri
- PascalCase kullanın (her kelimenin ilk harfi büyük)
- Anlamlı ve açıklayıcı isimler seçin
- Tek bir varlığı temsil eden tekil isimler kullanın

\`\`\`python
# YANLIŞ
class ogrenci:
    pass

class ARABA:
    pass

# DOĞRU
class Ogrenci:
    pass

class OtomobilParcasi:
    pass
\`\`\`

### Metod ve Değişken İsimleri
- snake_case kullanın (kelimeler alt çizgi ile ayrılır)
- Metodlar için eylem bildiren fiiller kullanın
- Değişkenler için açıklayıcı isimler seçin

\`\`\`python
class BankaHesabi:
    def __init__(self):
        # YANLIŞ
        self.Bakiye = 0
        self.HesapNo = ""
        
        # DOĞRU
        self.bakiye = 0
        self.hesap_no = ""
    
    # YANLIŞ
    def ParaYatir(self, miktar):
        pass
    
    # DOĞRU
    def para_yatir(self, miktar):
        pass
\`\`\`

## Kod Organizasyonu

### Sınıf Yapısı
1. İlk olarak \`__init__\` metodu
2. Özel metodlar (\`__str__\`, \`__len__\` vb.)
3. Public metodlar
4. Protected metodlar (tek alt çizgi ile başlayan)
5. Private metodlar (çift alt çizgi ile başlayan)

\`\`\`python
class Urun:
    def __init__(self, ad, fiyat):
        self.ad = ad
        self.fiyat = fiyat
        self._stok = 0
        self.__son_guncelleme = None
    
    def __str__(self):
        return f"{self.ad} - {self.fiyat} TL"
    
    def fiyat_guncelle(self, yeni_fiyat):
        self.fiyat = yeni_fiyat
        self.__guncelleme_kaydet()
    
    def _stok_kontrol(self):
        return self._stok > 0
    
    def __guncelleme_kaydet(self):
        from datetime import datetime
        self.__son_guncelleme = datetime.now()
\`\`\`

### Modül Organizasyonu
- İlgili sınıfları aynı modülde gruplandırın
- Bağımlılıkları en aza indirin
- Döngüsel bağımlılıklardan kaçının

\`\`\`python
# urunler.py
class Urun:
    pass

class UrunKategorisi:
    pass

# stok.py
from .urunler import Urun

class StokHareketi:
    pass

class StokDurumu:
    pass
\`\`\`

## Hata Yönetimi

### Özel İstisnalar
- Sınıfa özel hatalar için özel istisna sınıfları oluşturun
- Anlamlı hata mesajları kullanın
- Hataları uygun seviyede yakalayın

\`\`\`python
class YetersizBakiyeError(Exception):
    pass

class BankaHesabi:
    def __init__(self, bakiye=0):
        if bakiye < 0:
            raise ValueError("Başlangıç bakiyesi negatif olamaz")
        self.bakiye = bakiye
    
    def para_cek(self, miktar):
        try:
            if miktar > self.bakiye:
                raise YetersizBakiyeError(
                    f"Çekilmek istenen {miktar} TL için yetersiz bakiye. "
                    f"Mevcut bakiye: {self.bakiye} TL"
                )
            self.bakiye -= miktar
        except YetersizBakiyeError as e:
            print(f"Hata: {e}")
            raise
\`\`\`

## Dokümantasyon

### Sınıf ve Metod Dokümantasyonu
- Docstring kullanın
- Parametreleri ve dönüş değerlerini belirtin
- Örnekler ekleyin
- Özel durumları ve istisnaları belirtin

\`\`\`python
class Hesaplayici:
    """
    Matematiksel hesaplamalar yapan bir sınıf.
    
    Bu sınıf temel matematiksel işlemleri gerçekleştirir
    ve sonuçları saklar.
    """
    
    def bol(self, bolunen: float, bolen: float) -> float:
        """
        İki sayının bölümünü hesaplar.
        
        Args:
            bolunen (float): Bölünecek sayı
            bolen (float): Bölen sayı
        
        Returns:
            float: Bölüm sonucu
        
        Raises:
            ValueError: Bölen sıfır olduğunda
            TypeError: Sayısal olmayan değerler için
        
        Example:
            >>> h = Hesaplayici()
            >>> h.bol(10, 2)
            5.0
        """
        if bolen == 0:
            raise ValueError("Sıfıra bölme hatası")
        return bolunen / bolen
\`\`\`

## SOLID Prensipleri

### Tek Sorumluluk Prensibi (Single Responsibility)
Her sınıf tek bir sorumluluğa sahip olmalıdır.

\`\`\`python
# YANLIŞ
class Kullanici:
    def __init__(self, ad, email):
        self.ad = ad
        self.email = email
    
    def email_dogrula(self):
        # Email doğrulama işlemleri
        pass
    
    def email_gonder(self, mesaj):
        # Email gönderme işlemleri
        pass

# DOĞRU
class Kullanici:
    def __init__(self, ad, email):
        self.ad = ad
        self.email = email

class EmailServisi:
    @staticmethod
    def dogrula(email):
        # Email doğrulama işlemleri
        pass
    
    @staticmethod
    def gonder(email, mesaj):
        # Email gönderme işlemleri
        pass
\`\`\`

### Açık/Kapalı Prensibi (Open/Closed)
Sınıflar genişlemeye açık, değişime kapalı olmalıdır.

\`\`\`python
from abc import ABC, abstractmethod

class Sekil(ABC):
    @abstractmethod
    def alan_hesapla(self):
        pass

class Dikdortgen(Sekil):
    def __init__(self, en, boy):
        self.en = en
        self.boy = boy
    
    def alan_hesapla(self):
        return self.en * self.boy

class Daire(Sekil):
    def __init__(self, yaricap):
        self.yaricap = yaricap
    
    def alan_hesapla(self):
        from math import pi
        return pi * self.yaricap ** 2

# Yeni şekil eklemek için mevcut kodu değiştirmemize gerek yok
class Ucgen(Sekil):
    def __init__(self, taban, yukseklik):
        self.taban = taban
        self.yukseklik = yukseklik
    
    def alan_hesapla(self):
        return (self.taban * self.yukseklik) / 2
\`\`\`

## Performans İyileştirmeleri

### Bellek Yönetimi
- Büyük nesneleri gerektiğinde temizleyin
- Döngüsel referanslardan kaçının
- \`__slots__\` kullanarak bellek kullanımını optimize edin

\`\`\`python
class Nokta:
    __slots__ = ['x', 'y']  # Sadece bu özelliklere izin ver
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
\`\`\`

### Lazy Loading
Büyük nesneleri sadece gerektiğinde yükleyin:

\`\`\`python
class Rapor:
    def __init__(self):
        self._veri = None
    
    @property
    def veri(self):
        if self._veri is None:
            self._veri = self._veri_yukle()
        return self._veri
    
    def _veri_yukle(self):
        # Büyük veriyi yükle
        return "Büyük Veri"
\`\`\`

## Yaygın Hatalar ve Çözümleri

### 1. Mutable Varsayılan Argümanlar

\`\`\`python
# YANLIŞ
class Ogrenci:
    def __init__(self, ad, notlar=[]):  # Mutable varsayılan!
        self.ad = ad
        self.notlar = notlar

# DOĞRU
class Ogrenci:
    def __init__(self, ad, notlar=None):
        self.ad = ad
        self.notlar = notlar if notlar is not None else []
\`\`\`

### 2. Yanlış Kapsülleme

\`\`\`python
# YANLIŞ
class Hesap:
    def __init__(self):
        self.bakiye = 0  # Direkt erişilebilir

# DOĞRU
class Hesap:
    def __init__(self):
        self._bakiye = 0  # Protected
    
    @property
    def bakiye(self):
        return self._bakiye
    
    @bakiye.setter
    def bakiye(self, deger):
        if deger < 0:
            raise ValueError("Bakiye negatif olamaz")
        self._bakiye = deger
\`\`\`

### 3. Gereksiz Inheritance

\`\`\`python
# YANLIŞ
class Arac:
    def __init__(self, marka, model):
        self.marka = marka
        self.model = model

class AracYoneticisi(Arac):  # Gereksiz kalıtım
    def arac_ekle(self, arac):
        pass

# DOĞRU
class AracYoneticisi:
    def __init__(self):
        self.araclar = []
    
    def arac_ekle(self, arac: Arac):
        self.araclar.append(arac)
\`\`\`
`;

const sections = [
  {
    title: "İsimlendirme",
    description: "Sınıf ve metod isimlendirme kuralları",
    icon: <Code2 className="h-6 w-6" />,
    topics: [
      "PascalCase sınıf isimleri",
      "snake_case metod isimleri",
      "Anlamlı isimler",
      "Tutarlı terminoloji"
    ]
  },
  {
    title: "Kod Yapısı",
    description: "Kod organizasyonu ve yapılandırma",
    icon: <Terminal className="h-6 w-6" />,
    topics: [
      "Sınıf yapısı",
      "Modül organizasyonu",
      "Bağımlılık yönetimi",
      "Kod tekrarını önleme"
    ]
  },
  {
    title: "Hata Yönetimi",
    description: "Exception handling ve hata kontrolü",
    icon: <Settings className="h-6 w-6" />,
    topics: [
      "Özel istisnalar",
      "Hata mesajları",
      "Kontrol akışı",
      "Güvenli kod"
    ]
  },
  {
    title: "Dokümantasyon",
    description: "Kod dokümantasyonu ve örnekler",
    icon: <BookOpen className="h-6 w-6" />,
    topics: [
      "Docstring kullanımı",
      "Parametre açıklamaları",
      "Örnek kullanımlar",
      "API dokümantasyonu"
    ]
  }
];

export default function IyiPratiklerPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Concept Cards */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Önemli Kavramlar</h2>
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