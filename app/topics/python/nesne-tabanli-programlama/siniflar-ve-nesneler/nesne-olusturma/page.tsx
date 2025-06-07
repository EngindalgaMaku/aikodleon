import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowRight, Code2, Settings, Terminal, BookOpen } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python Nesne Oluşturma | AIKOD',
  description: 'Python\'da nesne oluşturma, constructor kullanımı, nesne örnekleme ve yaşam döngüsü yönetimi.',
};

const content = `
# Python'da Nesne Oluşturma

Nesne tabanlı programlamada, sınıflardan nesne örnekleri oluşturmak temel bir işlemdir. Bu bölümde, Python'da nesne oluşturmanın tüm yönlerini inceleyeceğiz.

## Constructor (__init__) Kullanımı

Constructor, bir sınıftan nesne oluşturulduğunda otomatik olarak çağrılan özel bir metoddur:

\`\`\`python
class Araba:
    def __init__(self, marka, model, yil=2024):
        self.marka = marka
        self.model = model
        self.yil = yil
        self.km = 0
        print(f"{marka} {model} oluşturuldu!")
    
    def bilgi(self):
        return f"{self.yil} {self.marka} {self.model} - {self.km} km"

# Constructor çağrılır ve nesne oluşturulur
araba1 = Araba("Toyota", "Corolla")  # Varsayılan yıl: 2024
araba2 = Araba("BMW", "X5", 2023)    # Özel yıl belirtme
\`\`\`

### Constructor Parametreleri

Constructor'da farklı parametre türleri kullanabilirsiniz:

\`\`\`python
class Urun:
    def __init__(self, ad, fiyat, *, kategori="Genel", stok=100):
        self.ad = ad                # Zorunlu parametre
        self.fiyat = fiyat          # Zorunlu parametre
        self.kategori = kategori    # Keyword-only parametre
        self.stok = stok           # Keyword-only parametre

# Farklı parametre kullanımları
urun1 = Urun("Laptop", 15000)  # Varsayılan kategori ve stok
urun2 = Urun("Telefon", 10000, kategori="Elektronik", stok=50)
\`\`\`

## Nesne Örnekleme Yöntemleri

Python'da nesneleri farklı şekillerde oluşturabiliriz:

### 1. Doğrudan Örnekleme

\`\`\`python
class Oyuncu:
    def __init__(self, isim, seviye=1):
        self.isim = isim
        self.seviye = seviye

# Doğrudan nesne oluşturma
oyuncu1 = Oyuncu("Ali")
oyuncu2 = Oyuncu(isim="Veli", seviye=5)
\`\`\`

### 2. Factory Metod Kullanımı

\`\`\`python
class Belge:
    def __init__(self, tur, icerik):
        self.tur = tur
        self.icerik = icerik
    
    @classmethod
    def pdf_olustur(cls, icerik):
        return cls("PDF", icerik)
    
    @classmethod
    def word_olustur(cls, icerik):
        return cls("WORD", icerik)

# Factory metodlar ile nesne oluşturma
pdf_belge = Belge.pdf_olustur("Python OOP Notları")
word_belge = Belge.word_olustur("Nesne Oluşturma Örnekleri")
\`\`\`

### 3. Copy ve Deepcopy

\`\`\`python
import copy

class Kisi:
    def __init__(self, ad, hobiler):
        self.ad = ad
        self.hobiler = hobiler

# Orijinal nesne
kisi1 = Kisi("Ahmet", ["Yüzme", "Kitap"])

# Shallow copy - hobiler listesi referans olarak kopyalanır
kisi2 = copy.copy(kisi1)

# Deep copy - hobiler listesi yeni bir liste olarak kopyalanır
kisi3 = copy.deepcopy(kisi1)
\`\`\`

## Nesne Yaşam Döngüsü

Python'da nesnelerin yaşam döngüsünü yönetmek için özel metodlar kullanabiliriz:

### 1. Başlatma ve Sonlandırma

\`\`\`python
class VeriTabani:
    def __init__(self):
        print("Bağlantı açılıyor...")
        self.baglanti = True
    
    def __del__(self):
        print("Bağlantı kapatılıyor...")
        self.baglanti = False

# Nesne oluşturulduğunda __init__ çağrılır
db = VeriTabani()

# Nesne silindiğinde __del__ çağrılır
del db
\`\`\`

### 2. Context Manager Kullanımı

\`\`\`python
class DosyaYoneticisi:
    def __init__(self, dosya_adi, mod):
        self.dosya_adi = dosya_adi
        self.mod = mod
    
    def __enter__(self):
        self.dosya = open(self.dosya_adi, self.mod)
        return self.dosya
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dosya.close()

# with bloğu ile otomatik kaynak yönetimi
with DosyaYoneticisi("veriler.txt", "w") as dosya:
    dosya.write("Python OOP")
# Blok sonunda dosya otomatik kapatılır
\`\`\`

## Parametre Geçme Teknikleri

Nesne oluştururken parametreleri farklı şekillerde geçebiliriz:

### 1. Pozisyonel Argümanlar

\`\`\`python
class Nokta:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Pozisyonel argümanlar
nokta1 = Nokta(5, 10)
\`\`\`

### 2. İsimli Argümanlar

\`\`\`python
class Kullanici:
    def __init__(self, *, ad, email, rol="kullanici"):
        self.ad = ad
        self.email = email
        self.rol = rol

# İsimli argümanlar (keyword arguments)
kullanici1 = Kullanici(ad="Ali", email="ali@mail.com")
kullanici2 = Kullanici(ad="Veli", email="veli@mail.com", rol="admin")
\`\`\`

### 3. Varsayılan Değerler

\`\`\`python
class Hesap:
    def __init__(self, no, bakiye=0, para_birimi="TL"):
        self.no = no
        self.bakiye = bakiye
        self.para_birimi = para_birimi

# Varsayılan değerler kullanma
hesap1 = Hesap("123")  # Bakiye=0, Para Birimi="TL"
hesap2 = Hesap("456", 1000, "USD")  # Tüm değerler özel
\`\`\`

## İyi Pratikler

1. **Constructor'ı Basit Tutun**: Constructor'da sadece temel başlatma işlemlerini yapın
2. **Varsayılan Değerleri Dikkatli Kullanın**: Değişebilir nesneleri varsayılan değer olarak kullanmaktan kaçının
3. **Type Hints Kullanın**: Parametre tiplerini belirtin
4. **Doğrulama Yapın**: Constructor'da parametre doğrulaması yapın
5. **Belgelendirme Ekleyin**: Constructor parametrelerini docstring ile açıklayın

## Yaygın Hatalar ve Çözümleri

1. **Değişebilir Varsayılan Değerler**:
\`\`\`python
# YANLIŞ
class Liste:
    def __init__(self, items=[]):  # Tehlikeli!
        self.items = items

# DOĞRU
class Liste:
    def __init__(self, items=None):
        self.items = items if items is not None else []
\`\`\`

2. **Self Parametresini Unutmak**:
\`\`\`python
# YANLIŞ
class Hesap:
    def __init__(ad, bakiye):  # self eksik!
        self.ad = ad
        self.bakiye = bakiye

# DOĞRU
class Hesap:
    def __init__(self, ad, bakiye):
        self.ad = ad
        self.bakiye = bakiye
\`\`\`

3. **Gereksiz Başlatma**:
\`\`\`python
# YANLIŞ
class Oyuncu:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.hiz = 0
        self.yon = 0
        # ... birçok değişken

# DOĞRU
class Oyuncu:
    def __init__(self, **kwargs):
        self.pozisyon = kwargs.get('pozisyon', (0, 0, 0))
        self.hareket = kwargs.get('hareket', {'hiz': 0, 'yon': 0})
\`\`\`
`;

const sections = [
  {
    title: "Constructor Kullanımı",
    description: "Nesne başlatma ve özelleştirme",
    icon: <Code2 className="h-6 w-6" />,
    topics: [
      "Constructor tanımlama",
      "Parametre kullanımı",
      "Varsayılan değerler",
      "Özel başlatma işlemleri"
    ]
  },
  {
    title: "Örnekleme Yöntemleri",
    description: "Farklı nesne oluşturma teknikleri",
    icon: <Settings className="h-6 w-6" />,
    topics: [
      "Doğrudan örnekleme",
      "Factory metodlar",
      "Copy ve deepcopy",
      "Context managers"
    ]
  },
  {
    title: "Parametre Geçme",
    description: "Constructor parametrelerini kullanma",
    icon: <Terminal className="h-6 w-6" />,
    topics: [
      "Pozisyonel argümanlar",
      "İsimli argümanlar",
      "Varsayılan değerler",
      "Type hints"
    ]
  },
  {
    title: "İyi Pratikler",
    description: "Nesne oluşturma önerileri",
    icon: <BookOpen className="h-6 w-6" />,
    topics: [
      "Constructor tasarımı",
      "Parametre doğrulama",
      "Belgelendirme",
      "Hata yönetimi"
    ]
  }
];

export default function NesneOlusturmaPage() {
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