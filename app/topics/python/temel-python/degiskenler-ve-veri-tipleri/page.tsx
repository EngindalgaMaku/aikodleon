import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Değişkenler ve Veri Tipleri | Python Temelleri | Kodleon',
  description: "Python'da değişkenler, veri tipleri, temel veri yapıları ve bunlarla ilgili alıştırmaları öğrenin.",
};

const content = `
# Değişkenler ve Veri Tipleri

Python'da verileri saklamak, yönetmek ve işlemek için değişkenler ve veri tipleri kullanılır. Bu bölüm, Python'daki temel yapı taşlarını derinlemesine anlamanızı sağlayacak.

## Değişkenler ve Atama İşlemleri

Değişkenler, verileri hafızada saklamak için kullanılan isimlendirilmiş alanlardır. Python'da bir değişkenin tipini önceden belirtmenize gerek yoktur; tip, atanan değere göre dinamik olarak belirlenir.

### Değişken Tanımlama ve İsimlendirme Kuralları (PEP 8)
- Değişken adları harf veya alt çizgi (\`_\`) ile başlamalıdır.
- Sayı ile başlayamaz.
- Sadece alfanümerik karakterler (A-z, 0-9) ve alt çizgi içerebilir.
- Büyük/küçük harfe duyarlıdır (\`isim\` ve \`Isim\` farklı değişkenlerdir).
- Python'un anahtar kelimeleri (örn. \`if\`, \`for\`, \`class\`) kullanılamaz.
- **Öneri**: Değişken adları küçük harflerle yazılır ve kelimeler alt çizgi ile ayrılır (snake_case). Örn: \`ogrenci_adi\`.

\`\`\`python
# Temel atama işlemleri
sayi = 10                  # int
kullanici_adi = "kodleon"  # str
pi_sayisi = 3.14           # float
giris_yapildi = True       # bool

# Çoklu atama
x, y, z = 5, 10, 15

# Bir değişkene başka bir değişkeni atama
a = 5
b = a  # b'nin değeri artık 5

# Değişken tipini kontrol etme
print(type(kullanici_adi))  # <class 'str'>
\`\`\`

---

## Sayısal Veri Tipleri

Python'da sayısal veriler için üç ana tip bulunur: \`int\`, \`float\`, ve \`complex\`.

### 1. Tamsayılar (int)
Pozitif veya negatif tam sayıları temsil ederler. Boyutları sistem belleği ile sınırlıdır.

\`\`\`python
pozitif_sayi = 123
negatif_sayi = -456
buyuk_sayi = 9_000_000_000  # Alt çizgi okunabilirliği artırır

# Aritmetik işlemler
toplam = 10 + 5       # 15
fark = 10 - 5         # 5
carpim = 10 * 5       # 50
bolum = 10 / 3        # 3.333... (Her zaman float döner)
tam_bolum = 10 // 3   # 3 (Taban bölme)
mod = 10 % 3          # 1 (Kalan)
us = 2 ** 4           # 16 (Üs alma)
\`\`\`

### 2. Ondalıklı Sayılar (float)
Ondalıklı kısmı olan sayıları temsil ederler. Bilimsel gösterimle de ifade edilebilirler.

\`\`\`python
pi = 3.14
yercekimi = 9.81
bilimsel = 6.022e23  # 6.022 * 10^23

# Float ve int ile işlem
sonuc = pi + 5  # 8.14 (Sonuç float olur)
\`\`\`

> **Not:** Float sayılarla yapılan hesaplamalarda küçük hassasiyet hataları olabilir. Finansal gibi yüksek hassasiyet gerektiren durumlar için \`Decimal\` modülü kullanılmalıdır.

### 3. Karmaşık Sayılar (complex)
Matematikteki karmaşık sayıları temsil ederler (a + bj). Gerçek ve sanal kısımlardan oluşurlar.

\`\`\`python
z1 = 3 + 5j
z2 = complex(2, -3)

print(z1.real)  # 3.0 (Gerçek kısım)
print(z1.imag)  # 5.0 (Sanal kısım)
\`\`\`

---

## Metin Veri Tipi (str)

Metinsel verileri saklamak için kullanılır. Tek (\`'\`) veya çift (\`"\`) tırnak içinde tanımlanabilirler.

### Temel String İşlemleri
\`\`\`python
mesaj = "Merhaba Dünya!"

# İndeksleme (Elemanlara erişim)
print(mesaj[0])      # 'M'
print(mesaj[-1])     # '!' (Sondan birinci)

# Dilimleme (String'in bir bölümünü alma)
print(mesaj[0:7])    # 'Merhaba'
print(mesaj[8:])     # 'Dünya!'
print(mesaj[:7])     # 'Merhaba'
print(mesaj[::2])    # 'MraaDny!' (Bir atlayarak)
print(mesaj[::-1])   # '!aynüD abahreM' (Ters çevirme)

# Birleştirme ve Tekrarlama
ad = "Kod"
soyad = "Leon"
tam_ad = ad + " " + soyad  # 'Kod Leon'
cizgi = "-" * 10           # '----------'
\`\`\`

### String Metodları
String'ler, üzerinde işlem yapmayı kolaylaştıran birçok metoda sahiptir.

\`\`\`python
metin = "  Python Öğreniyorum  "
print(metin.lower())         # '  python öğreniyorum  '
print(metin.upper())         # '  PYTHON ÖĞRENIYORUM  '
print(metin.strip())         # 'Python Öğreniyorum' (Baştaki ve sondaki boşlukları siler)
print(metin.replace("Python", "Java")) # '  Java Öğreniyorum  '
print(metin.split())         # ['Python', 'Öğreniyorum'] (Boşluklara göre ayırır)
print(metin.startswith("  P")) # True
\`\`\`

### Formatlı String'ler (f-string)
Değişkenleri string içine yerleştirmenin en modern ve kolay yoludur.

\`\`\`python
isim = "Ali"
yas = 30
mesaj = f"Merhaba, benim adım {isim} ve ben {yas} yaşındayım."
print(mesaj) # 'Merhaba, benim adım Ali ve ben 30 yaşındayım.'
\`\`\`

---

## Boolean Veri Tipi (bool)
Sadece iki değer alabilir: \`True\` veya \`False\`. Mantıksal işlemlerde ve koşul kontrollerinde kullanılır.

\`\`\`python
is_active = True
is_admin = False

# Mantıksal Operatörler: and, or, not
print(is_active and is_admin)  # False
print(is_active or is_admin)   # True
print(not is_active)           # False

# Karşılaştırma operatörleri bool sonuç döndürür
print(10 > 5)   # True
print(10 == 5)  # False
\`\`\`

---

## Koleksiyon Veri Tipleri

Birden fazla veriyi bir arada tutan veri yapılarıdır.

### 1. Listeler (list)
- Sıralı ve değiştirilebilir (mutable) koleksiyonlardır.
- Farklı veri tiplerini bir arada barındırabilirler.

\`\`\`python
sayilar = [10, 20, 30, 40]
karisik = [1, "elma", 3.5, True]

# Eleman değiştirme
sayilar[0] = 5
print(sayilar)  # [5, 20, 30, 40]

# Metodlar
sayilar.append(50)      # Sona ekler -> [5, 20, 30, 40, 50]
sayilar.insert(1, 15)   # İndekse ekler -> [5, 15, 20, 30, 40, 50]
sayilar.pop()           # Son elemanı siler ve döndürür -> 50
sayilar.remove(30)      # Değere göre siler -> [5, 15, 20, 40]
sayilar.sort(reverse=True) # Büyükten küçüğe sıralar
\`\`\`

### 2. Demetler (tuple)
- Sıralı ancak değiştirilemez (immutable) koleksiyonlardır.
- Listelere göre daha hızlıdırlar ve verilerin değişmemesi gereken durumlarda kullanılırlar.

\`\`\`python
koordinat = (10.5, 20.3)
renk_kodu = (255, 165, 0) # RGB

# Değiştirme denemesi hata verir
# koordinat[0] = 5.0  # TypeError

# Tuple unpacking
x, y = koordinat
print(f"X: {x}, Y: {y}")
\`\`\`

### 3. Sözlükler (dict)
- Anahtar-değer (key-value) çiftlerinden oluşan, sırasız (Python 3.7+ sıralı) ve değiştirilebilir koleksiyonlardır.
- Anahtarlar benzersiz ve değiştirilemez olmalıdır (örn: str, int, tuple).

\`\`\`python
ogrenci = {
    "ad": "Ayşe",
    "numara": 123,
    "bolum": "Bilgisayar Mühendisliği",
    "notlar": [85, 90, 78]
}

# Değere erişim
print(ogrenci["ad"])            # 'Ayşe'
print(ogrenci.get("bolum"))     # 'Bilgisayar Mühendisliği'
print(ogrenci.get("yas", 20)) # Anahtar yoksa varsayılan değeri döndürür

# Değer ekleme / değiştirme
ogrenci["yas"] = 21
ogrenci["ad"] = "Ayşe Yılmaz"

# Anahtarları, değerleri ve çiftleri alma
print(ogrenci.keys())
print(ogrenci.values())
print(ogrenci.items())
\`\`\`

### 4. Kümeler (set)
- Sırasız, benzersiz ve değiştirilebilir elemanlardan oluşur.
- Matematiksel küme işlemleri (birleşim, kesişim vb.) için idealdir.

\`\`\`python
renkler = {"mavi", "yeşil", "kırmızı", "mavi"}
print(renkler)  # {'yeşil', 'mavi', 'kırmızı'} (Tekrarlar otomatik silinir)

# Küme işlemleri
kume_a = {1, 2, 3, 4}
kume_b = {3, 4, 5, 6}

print(kume_a.union(kume_b))         # Birleşim: {1, 2, 3, 4, 5, 6}
print(kume_a.intersection(kume_b))  # Kesişim: {3, 4}
print(kume_a.difference(kume_b))    # Fark (A'da olup B'de olmayan): {1, 2}
\`\`\`

---

## Tip Dönüşümleri (Type Casting)

Bir veri tipini başka bir veri tipine dönüştürme işlemidir.

\`\`\`python
sayi_str = "123"
sayi_int = int(sayi_str)
print(sayi_int + 1) # 124

puan = 95.7
puan_int = int(puan) # Ondalık kısım atılır
print(puan_int) # 95

liste = [1, 2, 3]
demet = tuple(liste)
kume = set(liste)

# Her dönüşüm mümkün değildir
# int("Merhaba") # ValueError
\`\`\`

---

## Alıştırmalar ve Çözümleri

### Alıştırma 1: Basit Değişken ve String İşlemleri
Kullanıcıdan adını, soyadını ve doğum yılını alan bir program yazın. Ardından, kullanıcının yaşını hesaplayarak aşağıdaki formatta bir selamlama mesajı oluşturun:
\`"Merhaba Ali Yılmaz, 2024 yılı itibariyle 34 yaşındasınız."\`

**Çözüm:**
\`\`\`python
import datetime

ad = input("Adınız: ")
soyad = input("Soyadınız: ")
dogum_yili_str = input("Doğum yılınız: ")

# Tip dönüşümü ve yaş hesaplama
dogum_yili = int(dogum_yili_str)
guncel_yil = datetime.datetime.now().year
yas = guncel_yil - dogum_yili

# f-string ile mesaj oluşturma
mesaj = f"Merhaba {ad.title()} {soyad.title()}, {guncel_yil} yılı itibariyle {yas} yaşındasınız."
print(mesaj)
\`\`\`

### Alıştırma 2: Liste ve Metodları
Bir alışveriş listesi oluşturun. Bu liste üzerinde aşağıdaki işlemleri sırasıyla yapın:
1. Listeye "ekmek", "süt", "yumurta" elemanlarını ekleyin.
2. Listenin başına "peynir" ekleyin.
3. "süt" elemanını listeden silin.
4. Listenin son halini alfabetik olarak sıralayıp ekrana yazdırın.

**Çözüm:**
\`\`\`python
alisveris_listesi = []

# 1. Elemanları ekleme
alisveris_listesi.append("ekmek")
alisveris_listesi.append("süt")
alisveris_listesi.append("yumurta")
print(f"İlk hali: {alisveris_listesi}")

# 2. Başa eleman ekleme
alisveris_listesi.insert(0, "peynir")
print(f"Peynir eklendi: {alisveris_listesi}")

# 3. Eleman silme
alisveris_listesi.remove("süt")
print(f"Süt silindi: {alisveris_listesi}")

# 4. Sıralama ve yazdırma
alisveris_listesi.sort()
print(f"Son hali (sıralı): {alisveris_listesi}")
\`\`\`

### Alıştırma 3: Sözlük ve Veri Erişimi
Bir ürün bilgilerini içeren bir sözlük oluşturun: \`id\`, \`ad\`, \`fiyat\`, \`stok_miktari\`. Ardından bu ürünün KDV dahil fiyatını hesaplayıp (KDV %20) ürün adıyla birlikte ekrana yazdırın.
Örnek: \`{"id": 1, "ad": "Laptop", "fiyat": 25000, "stok_miktari": 50}\`

**Çözüm:**
\`\`\`python
urun = {
    "id": 1,
    "ad": "Laptop",
    "fiyat": 25000,
    "stok_miktari": 50
}

KDV_ORANI = 0.20

# Fiyatı alıp KDV'li fiyatı hesaplama
fiyat = urun["fiyat"]
kdvli_fiyat = fiyat * (1 + KDV_ORANI)

# Sonucu yazdırma
urun_adi = urun["ad"]
print(f"'{urun_adi}' ürününün KDV dahil fiyatı: {kdvli_fiyat:.2f} TL")
\`\`\`

## Sonraki Adımlar
Bu temel veri tiplerini ve yapılarını anladıktan sonra, programlarınızın akışını kontrol etmeyi öğreneceğiniz [Kontrol Yapıları](/topics/python/temel-python/kontrol-yapilari) bölümüne geçebilirsiniz.
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

        <div className="prose prose-lg dark:prose-invert bg-white dark:bg-gray-850 rounded-xl p-8 shadow-md">
          <MarkdownContent content={content} />
        </div>

        <div className="mt-16 text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} Kodleon | Python Eğitim Platformu</p>
        </div>
      </div>
    </div>
  );
} 