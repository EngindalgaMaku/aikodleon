import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Veri Yapıları | Python Temelleri | Kodleon',
  description: "Python'daki listeler, demetler, sözlükler ve kümeler gibi temel yerleşik veri yapılarını, ileri düzey koleksiyonları ve pratik kullanım senaryolarını öğrenin.",
};

const content = `
# Veri Yapıları

Veri yapıları, verileri verimli bir şekilde saklamak, organize etmek ve yönetmek için kullanılan temel programlama araçlarıdır. Python, kullanımı kolay ve güçlü yerleşik veri yapıları sunar.

---

## 1. Listeler (Lists)
Listeler, Python'da en sık kullanılan, sıralı ve **değiştirilebilir (mutable)** veri koleksiyonlarıdır.

- **Özellikleri:** Sıralıdır (elemanların bir sırası vardır), değiştirilebilirdir (elemanlar eklenebilir, silinebilir, güncellenebilir), farklı veri tiplerini barındırabilir.
- **Kullanım Alanları:** Sıralı veri koleksiyonları, veritabanından gelen kayıtlar, kullanıcı listeleri.

### Temel Liste İşlemleri

\`\`\`python
# Liste oluşturma
meyveler = ["elma", "armut", "muz"]
sayilar = [1, 2, 3, 4, 5]

# Elemanlara erişim (İndeksleme)
print(meyveler[0])    # 'elma'
print(sayilar[-1])   # 5 (Sondan birinci)

# Dilimleme (Slicing)
alt_liste = sayilar[1:4]  # [2, 3, 4]
print(alt_liste)
\`\`\`

### Liste Metodları

\`\`\`python
# Eleman ekleme
sayilar.append(6)      # Sona ekler: [1, 2, 3, 4, 5, 6]
sayilar.insert(0, 0)   # Belirtilen indekse ekler: [0, 1, 2, 3, 4, 5, 6]

# Eleman silme
sayilar.remove(3)      # Değere göre ilk bulduğunu siler
son_eleman = sayilar.pop() # Son elemanı siler ve döndürür
print(f"Silinen son eleman: {son_eleman}")

# Diğer metodlar
sayilar.sort(reverse=True) # Ters sıralar
print(sayilar.count(2))    # 2 elemanının sayısını verir
print(sayilar.index(4))    # 4 elemanının indeksini verir
\`\`\`

### Liste Anlayışları (List Comprehensions)
Listeleri daha kısa ve okunaklı bir şekilde oluşturmak için kullanılır.

\`\`\`python
# 0'dan 9'a kadar olan sayıların karelerini içeren bir liste
kareler = [x**2 for x in range(10)]
print(kareler) # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Sadece çift sayıların kareleri
cift_kareler = [x**2 for x in range(10) if x % 2 == 0]
print(cift_kareler) # [0, 4, 16, 36, 64]
\`\`\`

---

## 2. Demetler (Tuples)
Demetler, sıralı ve **değiştirilemez (immutable)** veri koleksiyonlarıdır.

- **Özellikleri:** Sıralıdır, değiştirilemez (elemanları güncellenemez), listelere göre daha hızlıdır ve daha az bellek kullanır.
- **Kullanım Alanları:** Değişmemesi gereken veriler (koordinatlar, ayarlar), fonksiyonlardan birden fazla değer döndürme.

\`\`\`python
# Demet oluşturma
koordinat = (10.5, 20.3)
renk_kodu = (255, 165, 0) # RGB

# Elemanlara erişim (listelerle aynı)
print(koordinat[0])

# Demet Çözme (Tuple Unpacking)
x, y = koordinat
print(f"X: {x}, Y: {y}")

# Değiştirme denemesi hata verir
# koordinat[0] = 5.0  # TypeError fırlatır
\`\`\`

---

## 3. Sözlükler (Dictionaries)
Sözlükler, anahtar-değer (\`key-value\`) çiftlerinden oluşan, sırasız (Python 3.7+ itibarıyla eklenme sırasını korur) ve **değiştirilebilir** koleksiyonlardır.

- **Özellikleri:** Anahtarlar benzersiz ve değiştirilemez olmalıdır (örn: str, int, tuple). Değerler herhangi bir veri tipi olabilir.
- **Kullanım Alanları:** JSON verileri, ayar dosyaları, bir nesnenin özelliklerini saklama.

### Sözlük İşlemleri

\`\`\`python
ogrenci = {
    "ad": "Ayşe",
    "numara": 123,
    "bolum": "Bilgisayar Mühendisliği"
}

# Değere erişim
print(ogrenci["ad"])            # 'Ayşe'
print(ogrenci.get("bolum"))     # 'Bilgisayar Mühendisliği'

# Değer ekleme / güncelleme
ogrenci["yas"] = 21
ogrenci["ad"] = "Ayşe Yılmaz"

# Eleman silme
cikarilan_deger = ogrenci.pop("numara")
print(f"Çıkarılan numara: {cikarilan_deger}")

# Döngü ile gezinme
for anahtar, deger in ogrenci.items():
    print(f"{anahtar.title()}: {deger}")
\`\`\`

---

## 4. Kümeler (Sets)
Kümeler, sırasız, benzersiz elemanlardan oluşan ve **değiştirilebilir** koleksiyonlardır.

- **Özellikleri:** Tekrar eden elemanları barındırmazlar. Matematiksel küme işlemleri için optimize edilmişlerdir.
- **Kullanım Alanları:** Bir koleksiyondaki benzersiz elemanları bulma, üyelik kontrolü, kesişim, birleşim gibi işlemler.

\`\`\`python
# Küme oluşturma
sayilar = {1, 2, 3, 2, 4, 1}
print(sayilar) # {1, 2, 3, 4} (Tekrarlar otomatik silinir)

# Küme işlemleri
kume_a = {1, 2, 3, 4}
kume_b = {3, 4, 5, 6}

# Birleşim
print(kume_a | kume_b)  # {1, 2, 3, 4, 5, 6}
# Kesişim
print(kume_a & kume_b)  # {3, 4}
# Fark
print(kume_a - kume_b)  # {1, 2}
# Simetrik Fark (sadece birinde olanlar)
print(kume_a ^ kume_b)  # {1, 2, 5, 6}
\`\`\`

---

## Alıştırmalar ve Çözümleri

### Alıştırma 1: Liste Anlayışı ve Filtreleme
1'den 50'ye kadar olan sayılardan, 3'e ve 5'e aynı anda tam bölünebilen sayıların karelerini içeren bir listeyi, liste anlayışı (list comprehension) kullanarak oluşturun.

**Çözüm:**
\`\`\`python
# 3 ve 5'in en küçük ortak katı 15'tir.
# Dolayısıyla 15'e tam bölünen sayıları arıyoruz.
sonuc_listesi = [sayi**2 for sayi in range(1, 51) if sayi % 15 == 0]

print("1-50 arasında 3'e ve 5'e tam bölünen sayıların kareleri:")
print(sonuc_listesi) # [225, 900, 2025]
\`\`\`

### Alıştırma 2: Sözlük ile Kelime Frekansı
Verilen bir cümlenin içindeki her kelimenin kaç kez geçtiğini sayan ve sonucu bir sözlük olarak döndüren bir program yazın. Büyük/küçük harf duyarlılığı olmamalıdır.

**Çözüm:**
\`\`\`python
cumle = "Bu bir test cümlesi ve bu cümle test amaçlıdır"
kelime_frekanslari = {}

# Cümleyi küçük harfe çevirip kelimelere ayır
kelimeler = cumle.lower().split()

for kelime in kelimeler:
    # Eğer kelime sözlükte varsa, sayacını 1 artır.
    # Yoksa, sözlüğe kelimeyi ekle ve değerini 1 yap.
    kelime_frekanslari[kelime] = kelime_frekanslari.get(kelime, 0) + 1

print("Kelime frekansları:")
print(kelime_frekanslari)
# Çıktı: {'bu': 2, 'bir': 1, 'test': 2, 'cümlesi': 1, 've': 1, 'cümle': 1, 'amaçlıdır': 1}
\`\`\`

### Alıştırma 3: Kümeler ile Ortak Eleman Bulma
İki farklı öğrenci grubunun katıldığı dersleri temsil eden iki liste oluşturun. Kümeleri kullanarak her iki grubun da ortak olarak katıldığı dersleri bulun.

**Çözüm:**
\`\`\`python
grup_a_dersleri = ["Matematik", "Fizik", "Kimya", "Tarih"]
grup_b_dersleri = ["Fizik", "Biyoloji", "Edebiyat", "Tarih"]

# Listeleri kümelere dönüştür
kume_a = set(grup_a_dersleri)
kume_b = set(grup_b_dersleri)

# Kesişim işlemini kullanarak ortak dersleri bul
ortak_dersler = kume_a.intersection(kume_b)

print("İki grubun da aldığı ortak dersler:")
print(list(ortak_dersler)) # Sonucu liste olarak yazdırmak daha okunaklı olabilir
# Çıktı: ['Tarih', 'Fizik'] (Sıra değişebilir)
\`\`\`

## Sonraki Adımlar
Veri yapıları, daha karmaşık algoritmalar ve programlar oluşturmanın temelidir. Bu konuyu anladıktan sonra, kodunuzu organize etmek için [Fonksiyonlar](/topics/python/temel-python/fonksiyonlar) konusuna geçmek iyi bir adımdır.
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