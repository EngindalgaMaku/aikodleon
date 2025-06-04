import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Veri Yapıları | Python Temelleri | Kodleon',
  description: "Python'da listeler, demetler, sözlükler ve kümeler gibi temel veri yapılarını ve bunların kullanımını öğrenin.",
};

const content = `
# Veri Yapıları

Python'da veri yapıları, verileri organize etmek ve yönetmek için kullanılan temel yapı taşlarıdır. Bu bölümde, Python'un yerleşik veri yapılarını ve bunların kullanımını öğreneceksiniz.

## Listeler (Lists)

Listeler, sıralı ve değiştirilebilir veri koleksiyonlarıdır:

\`\`\`python
# Liste oluşturma
sayilar = [1, 2, 3, 4, 5]
meyveler = ["elma", "armut", "muz"]
karisik = [1, "elma", 3.14, True]

# İndeksleme ve dilimleme
print(sayilar[0])       # İlk eleman: 1
print(sayilar[-1])      # Son eleman: 5
print(sayilar[1:4])     # [2, 3, 4]
print(sayilar[::2])     # [1, 3, 5] (2'şer atlayarak)

# Liste metodları
sayilar.append(6)       # Sona ekleme
sayilar.insert(0, 0)    # Başa ekleme
sayilar.remove(3)       # Değere göre silme
sayilar.pop()          # Son elemanı çıkarma
sayilar.sort()         # Sıralama
sayilar.reverse()      # Ters çevirme

# Liste işlemleri
a = [1, 2, 3]
b = [4, 5, 6]
c = a + b              # Listeleri birleştirme
a.extend(b)            # Liste sonuna ekleme
print(len(a))          # Liste uzunluğu
print(2 in a)          # Eleman kontrolü
\`\`\`

## Demetler (Tuples)

Demetler, değiştirilemez (immutable) listelerdir:

\`\`\`python
# Demet oluşturma
koordinat = (3, 4)
rgb = (255, 128, 0)
tek_elemanli = (1,)    # Tek elemanlı demet için virgül gerekli

# Demet işlemleri
x, y = koordinat       # Demet çözme (tuple unpacking)
print(koordinat[0])    # İndeksleme
print(len(rgb))        # Uzunluk
print(128 in rgb)      # Eleman kontrolü

# Demetlerin kullanım alanları
def koordinat_hesapla():
    return (10, 20)    # Çoklu değer döndürme

x, y = koordinat_hesapla()

# Değiştirilemezlik
# koordinat[0] = 5     # TypeError: 'tuple' object does not support item assignment
\`\`\`

## Sözlükler (Dictionaries)

Sözlükler, anahtar-değer çiftlerini saklayan veri yapılarıdır:

\`\`\`python
# Sözlük oluşturma
kisi = {
    "ad": "Ahmet",
    "yas": 25,
    "sehir": "İstanbul"
}

# Erişim ve değiştirme
print(kisi["ad"])           # Değere erişim
kisi["yas"] = 26           # Değer güncelleme
kisi["meslek"] = "Mühendis" # Yeni çift ekleme

# Güvenli erişim
yas = kisi.get("yas", 0)    # Varsayılan değerle erişim
meslek = kisi.get("meslek", "Belirtilmedi")

# Sözlük metodları
print(kisi.keys())          # Anahtarlar
print(kisi.values())        # Değerler
print(kisi.items())         # Anahtar-değer çiftleri

# Sözlük işlemleri
kisi.update({"email": "ahmet@email.com"})  # Çoklu güncelleme
del kisi["yas"]            # Çift silme
kisi.pop("sehir")          # Çift çıkarma ve değerini döndürme
kisi.clear()               # Sözlüğü temizleme

# İç içe sözlükler
ogrenciler = {
    "101": {
        "ad": "Ayşe",
        "notlar": [85, 90, 95]
    },
    "102": {
        "ad": "Mehmet",
        "notlar": [75, 80, 85]
    }
}
\`\`\`

## Kümeler (Sets)

Kümeler, benzersiz elemanları saklayan veri yapılarıdır:

\`\`\`python
# Küme oluşturma
sayilar = {1, 2, 3, 4, 5}
meyveler = set(["elma", "armut", "muz"])
tekrar = {1, 2, 2, 3, 3, 4}  # {1, 2, 3, 4}

# Küme işlemleri
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

birlesim = A | B            # Birleşim
kesisim = A & B            # Kesişim
fark = A - B               # Fark
simetrik_fark = A ^ B      # Simetrik fark

# Küme metodları
sayilar.add(6)             # Eleman ekleme
sayilar.remove(1)          # Eleman silme (yoksa hata verir)
sayilar.discard(10)        # Eleman silme (yoksa hata vermez)
sayilar.pop()              # Rastgele eleman çıkarma
sayilar.clear()            # Kümeyi temizleme

# Küme kontrolleri
print(2 in sayilar)        # Eleman kontrolü
print(A.issubset(B))       # Alt küme kontrolü
print(A.issuperset(B))     # Üst küme kontrolü
print(A.isdisjoint(B))     # Ayrık küme kontrolü
\`\`\`

## Dize İşlemleri (String Operations)

Dizeler de bir veri yapısıdır ve zengin işlem kümesine sahiptir:

\`\`\`python
# Dize oluşturma ve biçimlendirme
ad = "Ahmet"
soyad = 'Yılmaz'
tam_ad = f"{ad} {soyad}"
uzun_metin = """Çok
satırlı
metin"""

# Dize metodları
metin = "  Python Programlama  "
print(metin.strip())       # Boşlukları temizleme
print(metin.upper())       # Büyük harfe çevirme
print(metin.lower())       # Küçük harfe çevirme
print(metin.title())       # Başlık formatı
print(metin.replace("Python", "Java"))  # Değiştirme

# Dize bölme ve birleştirme
kelimeler = metin.split()  # Boşluğa göre bölme
dosya = "resim.jpg"
ad, uzanti = dosya.split(".")  # Noktaya göre bölme
"-".join(kelimeler)        # Birleştirme

# Dize arama
cumle = "Python çok güzel bir programlama dili"
print("Python" in cumle)   # İçerik kontrolü
print(cumle.startswith("Python"))  # Başlangıç kontrolü
print(cumle.endswith("dili"))      # Bitiş kontrolü
print(cumle.find("güzel"))         # Konum bulma
\`\`\`

## Koleksiyon Modülü (Collections)

Python'un \`collections\` modülü, özel veri yapıları sunar:

\`\`\`python
from collections import Counter, defaultdict, namedtuple, deque

# Counter: Eleman sayımı
metin = "mississippi"
sayac = Counter(metin)
print(sayac)  # {'i': 4, 's': 4, 'p': 2, 'm': 1}

# defaultdict: Varsayılan değerli sözlük
d = defaultdict(list)
d["a"].append(1)  # Otomatik liste oluşturur

# namedtuple: İsimli alanları olan demet
Nokta = namedtuple("Nokta", ["x", "y"])
p = Nokta(3, 4)
print(p.x, p.y)

# deque: Çift yönlü kuyruk
kuyruk = deque([1, 2, 3])
kuyruk.appendleft(0)   # Başa ekleme
kuyruk.append(4)       # Sona ekleme
kuyruk.popleft()       # Baştan çıkarma
kuyruk.pop()           # Sondan çıkarma
\`\`\`

## Alıştırmalar

1. **Liste İşlemleri**
   - Bir listedeki tekrar eden elemanları temizleyen fonksiyon yazın
   - İki listeyi birleştirip sıralayan program yazın
   - Liste içindeki en sık geçen elemanı bulan fonksiyon yazın

2. **Sözlük İşlemleri**
   - Öğrenci not takip sistemi oluşturun
   - İç içe sözlüklerle telefon rehberi yapın
   - Sözlük kullanarak metin içindeki kelimelerin frekansını hesaplayın

3. **Küme İşlemleri**
   - İki metin arasındaki ortak kelimeleri bulan program yazın
   - Küme işlemlerini kullanarak sayı problemleri çözün
   - Öğrenci grupları arasındaki ilişkileri küme işlemleriyle analiz edin

## Sonraki Adımlar

- [Python Veri Yapıları Dokümantasyonu](https://docs.python.org/3/tutorial/datastructures.html)
- [Python Collections Modülü](https://docs.python.org/3/library/collections.html)
- [Python String Metodları](https://docs.python.org/3/library/stdtypes.html#string-methods)
`;

export default function PythonDataStructuresPage() {
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