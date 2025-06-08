import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowRight, Code2, Settings, Terminal, BookOpen } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python Sınıf Kavramı | Kodleon',
  description: 'Python\'da sınıf kavramı, sınıf tanımlama, özellikler, metodlar ve sınıf yapısı hakkında detaylı bilgi.',
};

const content = `
# Python'da Sınıf Kavramı

Sınıf (class), nesne tabanlı programlamanın temel yapı taşıdır. Nesnelerin özelliklerini ve davranışlarını tanımlayan bir şablondur.

## Sınıf Tanımlama

Python'da sınıf tanımlamak için \`class\` anahtar kelimesi kullanılır:

\`\`\`python
class Ogrenci:
    # Sınıf özellikleri (class attributes)
    okul = "Python Akademi"
    ogrenci_sayisi = 0
    
    def __init__(self, ad, numara):
        # Instance özellikleri (instance attributes)
        self.ad = ad
        self.numara = numara
        self.dersler = []
        Ogrenci.ogrenci_sayisi += 1
    
    def ders_ekle(self, ders):
        self.dersler.append(ders)
    
    def bilgi_goster(self):
        return f"Ad: {self.ad}, Numara: {self.numara}"

# Sınıftan nesne oluşturma
ogrenci1 = Ogrenci("Ali", "101")
ogrenci2 = Ogrenci("Veli", "102")
\`\`\`

## Özellikler (Attributes)

Sınıflarda iki tür özellik bulunur:

### 1. Sınıf Özellikleri (Class Attributes)

Tüm nesneler için ortak olan özelliklerdir:

\`\`\`python
class Araba:
    # Sınıf özellikleri
    teker_sayisi = 4
    arac_tipi = "Kara Taşıtı"
    
    def __init__(self, marka, model):
        self.marka = marka
        self.model = model

# Sınıf özelliklerine erişim
print(Araba.teker_sayisi)  # 4
print(Araba.arac_tipi)     # Kara Taşıtı
\`\`\`

### 2. Instance Özellikleri (Instance Attributes)

Her nesne için özel olan özelliklerdir:

\`\`\`python
class Kullanici:
    def __init__(self, ad, email):
        # Instance özellikleri
        self.ad = ad
        self.email = email
        self.aktif = True
    
    def deaktif_et(self):
        self.aktif = False

# Her nesnenin kendi özellikleri
kullanici1 = Kullanici("Ali", "ali@mail.com")
kullanici2 = Kullanici("Veli", "veli@mail.com")

print(kullanici1.ad)    # Ali
print(kullanici2.email) # veli@mail.com
\`\`\`

## Metodlar (Methods)

Python'da üç tür metod bulunur:

### 1. Instance Metodları

En yaygın kullanılan metod türüdür. İlk parametre olarak \`self\` alır:

\`\`\`python
class BankaHesabi:
    def __init__(self, hesap_no, bakiye=0):
        self.hesap_no = hesap_no
        self.bakiye = bakiye
    
    def para_yatir(self, miktar):
        self.bakiye += miktar
        return f"{miktar} TL yatırıldı"
    
    def para_cek(self, miktar):
        if miktar <= self.bakiye:
            self.bakiye -= miktar
            return f"{miktar} TL çekildi"
        return "Yetersiz bakiye"

hesap = BankaHesabi("12345")
print(hesap.para_yatir(100))  # 100 TL yatırıldı
\`\`\`

### 2. Sınıf Metodları

Sınıf ile ilgili işlemler için kullanılır. \`@classmethod\` dekoratörü ile tanımlanır:

\`\`\`python
class Tarih:
    def __init__(self, gun, ay, yil):
        self.gun = gun
        self.ay = ay
        self.yil = yil
    
    @classmethod
    def from_string(cls, tarih_str):
        gun, ay, yil = map(int, tarih_str.split('.'))
        return cls(gun, ay, yil)
    
    def __str__(self):
        return f"{self.gun}.{self.ay}.{self.yil}"

# Sınıf metodu kullanımı
tarih = Tarih.from_string("23.05.2024")
print(tarih)  # 23.5.2024
\`\`\`

### 3. Statik Metodlar

Sınıf veya instance ile ilgili olmayan yardımcı fonksiyonlar için kullanılır:

\`\`\`python
class Matematik:
    @staticmethod
    def topla(x, y):
        return x + y
    
    @staticmethod
    def carp(x, y):
        return x * y

# Statik metod kullanımı
print(Matematik.topla(5, 3))  # 8
print(Matematik.carp(4, 2))   # 8
\`\`\`

## Sınıf Yapısı

### 1. Kapsülleme (Encapsulation)

Veri gizleme ve koruma için kullanılır:

\`\`\`python
class Calisan:
    def __init__(self, ad, maas):
        self._ad = ad      # Protected özellik
        self.__maas = maas # Private özellik
    
    def maas_goster(self):
        return f"{self._ad}'in maaşı: {self.__maas} TL"
    
    def maas_artir(self, artis):
        self.__maas += artis

calisan = Calisan("Ali", 5000)
print(calisan.maas_goster())  # Ali'in maaşı: 5000 TL
# print(calisan.__maas)       # AttributeError
\`\`\`

### 2. Property Dekoratörü

Özelliklere kontrollü erişim sağlar:

\`\`\`python
class Urun:
    def __init__(self, ad, fiyat):
        self._ad = ad
        self._fiyat = fiyat
    
    @property
    def fiyat(self):
        return self._fiyat
    
    @fiyat.setter
    def fiyat(self, yeni_fiyat):
        if yeni_fiyat > 0:
            self._fiyat = yeni_fiyat
        else:
            raise ValueError("Fiyat negatif olamaz")

urun = Urun("Laptop", 5000)
print(urun.fiyat)  # 5000
urun.fiyat = 6000  # Fiyat güncelleme
\`\`\`

## İyi Pratikler

1. **İsimlendirme Kuralları**:
   - Sınıf isimleri PascalCase kullanır (Her kelimenin ilk harfi büyük)
   - Metod ve özellik isimleri snake_case kullanır
   - Private özellikler için çift alt çizgi (__) kullanılır
   - Protected özellikler için tek alt çizgi (_) kullanılır

2. **Dokümantasyon**:
\`\`\`python
class Hesap:
    """
    Banka hesabı sınıfı.
    
    Attributes:
        hesap_no (str): Hesap numarası
        bakiye (float): Hesap bakiyesi
    
    Methods:
        para_yatir(miktar): Hesaba para yatırır
        para_cek(miktar): Hesaptan para çeker
    """
    def __init__(self, hesap_no):
        self.hesap_no = hesap_no
        self.bakiye = 0
\`\`\`

3. **Kod Organizasyonu**:
   - İlk önce sınıf özellikleri
   - Sonra __init__ metodu
   - Ardından diğer metodlar
   - En sonda yardımcı metodlar ve property'ler

## Yaygın Hatalar ve Çözümleri

1. **Self Parametresini Unutmak**:
\`\`\`python
# YANLIŞ
class Hesap:
    def bakiye_goster():  # self eksik!
        return self.bakiye

# DOĞRU
class Hesap:
    def bakiye_goster(self):
        return self.bakiye
\`\`\`

2. **Sınıf ve Instance Özelliklerini Karıştırmak**:
\`\`\`python
# YANLIŞ
class Oyuncu:
    skor = 0  # Sınıf özelliği
    def skor_ekle(self):
        Oyuncu.skor += 10  # Tüm oyuncuların skoru değişir!

# DOĞRU
class Oyuncu:
    def __init__(self):
        self.skor = 0  # Instance özelliği
    def skor_ekle(self):
        self.skor += 10  # Sadece bu oyuncunun skoru değişir
\`\`\`

3. **Gereksiz Getter/Setter Kullanımı**:
\`\`\`python
# YANLIŞ
class Kitap:
    def __init__(self, ad):
        self._ad = ad
    def get_ad(self):
        return self._ad
    def set_ad(self, ad):
        self._ad = ad

# DOĞRU (Python'da property kullan)
class Kitap:
    def __init__(self, ad):
        self._ad = ad
    @property
    def ad(self):
        return self._ad
    @ad.setter
    def ad(self, yeni_ad):
        self._ad = yeni_ad
\`\`\`
`;

const sections = [
  {
    title: "Sınıf Tanımlama",
    description: "Temel sınıf yapısı ve oluşturma",
    icon: <Code2 className="h-6 w-6" />,
    topics: [
      "Class anahtar kelimesi",
      "Sınıf gövdesi",
      "Constructor tanımlama",
      "Örnek oluşturma"
    ]
  },
  {
    title: "Özellikler",
    description: "Sınıf ve instance özellikleri",
    icon: <Settings className="h-6 w-6" />,
    topics: [
      "Class attributes",
      "Instance attributes",
      "Property dekoratörü",
      "Kapsülleme"
    ]
  },
  {
    title: "Metodlar",
    description: "Sınıf içi fonksiyonlar",
    icon: <Terminal className="h-6 w-6" />,
    topics: [
      "Instance metodları",
      "Class metodları",
      "Statik metodlar",
      "Special metodlar"
    ]
  },
  {
    title: "İyi Pratikler",
    description: "Sınıf tasarım prensipleri",
    icon: <BookOpen className="h-6 w-6" />,
    topics: [
      "İsimlendirme kuralları",
      "Dokümantasyon",
      "Kod organizasyonu",
      "Hata yönetimi"
    ]
  }
];

export default function SinifKavramiPage() {
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