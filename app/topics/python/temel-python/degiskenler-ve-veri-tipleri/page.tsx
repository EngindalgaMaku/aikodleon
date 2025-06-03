import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Değişkenler ve Veri Tipleri | Python Temelleri | Kodleon',
  description: "Python'da değişkenler, veri tipleri ve temel veri yapılarını öğrenin.",
};

const content = `
# Değişkenler ve Veri Tipleri

Python'da değişkenler ve veri tipleri, programlarınızda veri saklamak ve işlemek için kullanılan temel yapı taşlarıdır. Bu bölümde, Python'un temel veri tiplerini ve değişken kullanımını detaylı olarak öğreneceksiniz.

## Değişkenler

Python'da değişkenler dinamik olarak tiplendirilir, yani bir değişkenin tipini önceden belirtmenize gerek yoktur.

\`\`\`python
# Değişken tanımlama
x = 5           # integer
isim = "Ahmet"  # string
pi = 3.14       # float
aktif = True    # boolean

# Çoklu atama
a, b, c = 1, 2, 3

# Değişken tipleri
print(type(x))      # <class 'int'>
print(type(isim))   # <class 'str'>
print(type(pi))     # <class 'float'>
print(type(aktif))  # <class 'bool'>
\`\`\`

## Sayısal Veri Tipleri

### 1. Integer (int)

Tam sayıları temsil eder:

\`\`\`python
# Integer örnekleri
x = 5
y = -10
buyuk_sayi = 1_000_000  # Alt çizgi okunabilirlik için

# İşlemler
toplam = x + y
carpim = x * y
bolum = x / y          # Float sonuç
tam_bolum = x // y     # Integer sonuç
mod = x % y            # Mod alma
us = x ** 2            # Üs alma
\`\`\`

### 2. Float (float)

Ondalıklı sayıları temsil eder:

\`\`\`python
# Float örnekleri
pi = 3.14159
e = 2.71828
bilimsel = 1.23e-4  # Bilimsel gösterim

# Hassasiyet
from decimal import Decimal
hassas = Decimal('0.1') + Decimal('0.2')  # Hassas hesaplama
\`\`\`

### 3. Complex (complex)

Karmaşık sayıları temsil eder:

\`\`\`python
# Complex sayı örnekleri
z1 = 2 + 3j
z2 = complex(1, 2)

# İşlemler
toplam = z1 + z2
carpim = z1 * z2
\`\`\`

## Metin Veri Tipi (str)

String'ler metinsel verileri temsil eder:

\`\`\`python
# String tanımlama
isim = "Ahmet"
soyad = 'Yılmaz'
uzun_metin = """Bu bir
çok satırlı
metindir."""

# String işlemleri
tam_isim = isim + " " + soyad  # Birleştirme
tekrar = isim * 3              # Tekrarlama
uzunluk = len(isim)            # Uzunluk
karakter = isim[0]             # İndeksleme
dilim = isim[1:3]             # Dilimleme

# String metodları
buyuk = isim.upper()          # Büyük harfe çevirme
kucuk = isim.lower()          # Küçük harfe çevirme
baslik = isim.title()         # Başlık formatı
bosluk_sil = "  metin  ".strip()  # Boşluk silme

# f-string (format)
mesaj = f"Merhaba {isim} {soyad}!"
\`\`\`

## Liste Veri Tipi (list)

Listeler, sıralı ve değiştirilebilir koleksiyonlardır:

\`\`\`python
# Liste oluşturma
sayilar = [1, 2, 3, 4, 5]
karisik = [1, "iki", 3.0, True]
ic_ice = [1, [2, 3], [4, 5]]

# Liste işlemleri
sayilar.append(6)        # Eleman ekleme
sayilar.insert(0, 0)    # İndekse ekleme
sayilar.remove(3)       # Eleman silme
son = sayilar.pop()     # Son elemanı çıkar
sayilar.sort()          # Sıralama
sayilar.reverse()       # Ters çevirme

# Liste dilimleme
ilk_uc = sayilar[:3]    # İlk üç eleman
son_uc = sayilar[-3:]   # Son üç eleman
adim = sayilar[::2]     # İkişer adım
\`\`\`

## Tuple Veri Tipi (tuple)

Tuple'lar değiştirilemez (immutable) listelerdir:

\`\`\`python
# Tuple oluşturma
koordinat = (10, 20)
rgb = (255, 128, 0)
tek = (1,)  # Tek elemanlı tuple

# Tuple işlemleri
x, y = koordinat        # Tuple unpacking
indeks = rgb.index(128) # Eleman indeksi
sayi = rgb.count(255)   # Eleman sayısı
\`\`\`

## Dictionary Veri Tipi (dict)

Dictionary'ler anahtar-değer çiftlerini saklar:

\`\`\`python
# Dictionary oluşturma
kisi = {
    "ad": "Ahmet",
    "yas": 25,
    "sehir": "İstanbul"
}

# Dictionary işlemleri
kisi["meslek"] = "Mühendis"  # Eleman ekleme
del kisi["yas"]             # Eleman silme
deger = kisi.get("ad")      # Değer alma
anahtarlar = kisi.keys()    # Anahtarlar
degerler = kisi.values()    # Değerler
\`\`\`

## Set Veri Tipi (set)

Set'ler benzersiz elemanları saklar:

\`\`\`python
# Set oluşturma
renkler = {"kırmızı", "mavi", "yeşil"}
sayilar = set([1, 2, 2, 3, 3, 4])  # Tekrarlar elenir

# Set işlemleri
renkler.add("sarı")        # Eleman ekleme
renkler.remove("mavi")     # Eleman silme
renkler.discard("mor")     # Güvenli silme

# Küme işlemleri
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}
birlesim = A | B           # Birleşim
kesisim = A & B           # Kesişim
fark = A - B              # Fark
\`\`\`

## Tip Dönüşümleri

Python'da veri tipleri arasında dönüşüm yapabilirsiniz:

\`\`\`python
# Tip dönüşüm örnekleri
sayi = int("123")         # String'den integer'a
metin = str(123)          # Integer'dan string'e
ondalik = float("3.14")   # String'den float'a
liste = list("Python")    # String'den liste'ye
demet = tuple([1, 2, 3])  # Liste'den tuple'a
kume = set([1, 2, 2, 3])  # Liste'den set'e
\`\`\`

## Alıştırmalar

1. **Veri Tipi Dönüşümleri**
   - Farklı veri tipleri arasında dönüşümler yapın
   - Dönüşüm hatalarını yakalayın ve yönetin
   - Tip kontrolü yapan fonksiyonlar yazın

2. **String İşlemleri**
   - Bir metni tersine çevirin
   - Palindrom kontrolü yapın
   - Metindeki kelime sayısını bulun

3. **Koleksiyon İşlemleri**
   - Liste elemanlarını benzersiz yapın
   - İki listeyi birleştirip sıralayın
   - Dictionary'den liste oluşturun

## Sonraki Adımlar

- [Kontrol Yapıları](/topics/python/temel-python/kontrol-yapilari)
- [Python Veri Yapıları Dokümantasyonu](https://docs.python.org/3/tutorial/datastructures.html)
- [Python String Metodları](https://docs.python.org/3/library/stdtypes.html#string-methods)
`;

export default function PythonDataTypesPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild className="mb-6">
            <Link href="/topics/python/temel-python" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Python Temelleri
            </Link>
          </Button>
        </div>

        <div className="prose prose-lg dark:prose-invert">
          <MarkdownContent content={content} />
        </div>

        <div className="mt-16 text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} Kodleon | Python Eğitim Platformu</p>
        </div>
      </div>
    </div>
  );
} 