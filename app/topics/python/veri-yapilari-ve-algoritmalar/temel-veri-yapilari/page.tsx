import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python Temel Veri Yapıları | Veri Yapıları ve Algoritmalar | Kodleon',
  description: 'Python\'da listeler, demetler (tuple), sözlükler (dictionary) ve kümeler (set) gibi temel veri yapılarını ve kullanım senaryolarını öğrenin.',
};

const content = `
# Python'da Temel Veri Yapıları

Python'da temel veri yapıları, verileri organize etmek, saklamak ve verimli bir şekilde işlemek için kullanılan yapı taşlarıdır. Bu bölümde, en sık kullanılan dört temel veri yapısını ve bunların temel özelliklerini, kullanım senaryolarını öğreneceğiz.

## 1. Listeler (Lists)

Listeler, Python'da en çok kullanılan, çok yönlü veri yapılarından biridir.

- **Özellikleri:**
  - **Sıralıdır (Ordered):** Elemanların eklenme sırası korunur. Her elemanın bir indeksi vardır.
  - **Değiştirilebilirdir (Mutable):** Oluşturulduktan sonra elemanları eklenebilir, silinebilir veya değiştirilebilir.
  - **Farklı Veri Tiplerini İçerebilir:** İçerisinde tamsayı, string, float ve hatta başka listeler gibi farklı veri tiplerini bir arada barındırabilir.

\`\`\`python
# Liste oluşturma ve temel işlemler
sayilar = [1, 2, 3, 4, 5]
karisik_liste = [1, "Python", 3.14, True]

# Eleman ekleme
sayilar.append(6)        # Sona eleman ekler -> [1, 2, 3, 4, 5, 6]
sayilar.insert(0, 0)     # Belirli bir konuma ekler -> [0, 1, 2, 3, 4, 5, 6]

# Eleman silme
sayilar.pop()            # Son elemanı siler ve döndürür -> 6
sayilar.remove(3)        # Değeri 3 olan ilk elemanı siler

# Elemana erişim ve değiştirme
print(sayilar[0])        # İlk eleman: 0
sayilar[0] = 100         # İlk elemanı güncelleme -> [100, 1, 2, 4, 5]

# Liste dilimleme (slicing)
ilk_iki_eleman = sayilar[:2]    # -> [100, 1]
son_iki_eleman = sayilar[-2:]   # -> [4, 5]
\`\`\`

## 2. Demetler (Tuples)

Demetler, listelere çok benzerler ancak temel bir farkları vardır: değiştirilemezler.

- **Özellikleri:**
  - **Sıralıdır (Ordered):** Tıpkı listeler gibi, elemanların sırası sabittir.
  - **Değiştirilemezdir (Immutable):** Bir demet oluşturulduktan sonra içeriği değiştirilemez. Bu özellik, demetleri verilerin sabit kalması gereken durumlar için (örn. ayarlar, koordinatlar) ideal kılar.
  - **Daha Hızlıdır:** Değiştirilemez oldukları için genellikle listelere göre daha az bellek kullanır ve daha hızlı çalışırlar.

\`\`\`python
# Tuple oluşturma
koordinat = (10.0, 20.5)
rgb_renk = (255, 165, 0) # Orange

# Elemanlara erişim (değiştirme denemesi hata verir)
print(koordinat[0])       # -> 10.0
# koordinat[0] = 5.0      # TypeError: 'tuple' object does not support item assignment

# Tuple "unpacking" (elemanları değişkenlere atama)
x, y = koordinat
print(f"X: {x}, Y: {y}")

# Tek elemanlı tuple tanımlarken sona virgül konulmalıdır
tekli_tuple = (1,)
\`\`\`

## 3. Sözlükler (Dictionaries)

Sözlükler, verileri anahtar-değer (key-value) çiftleri halinde saklayan esnek veri yapılarıdır.

- **Özellikleri:**
  - **Sırasızdır (Unordered - Python 3.7 öncesi):** Python 3.7 ve sonrası sürümlerde eklenme sırasını korurlar.
  - **Anahtar ile Erişim:** Elemanlara indeks yerine benzersiz anahtarlar (key) ile erişilir.
  - **Değiştirilebilirdir (Mutable):** Yeni anahtar-değer çiftleri eklenebilir, var olanlar güncellenebilir veya silinebilir.
  - **Anahtarlar Benzersiz ve Değiştirilemez Olmalıdır:** Anahtarlar genellikle string veya sayı gibi değiştirilemez tiplerden oluşur.

\`\`\`python
# Sözlük oluşturma
ogrenci = {
    "ad": "Ahmet",
    "soyad": "Yılmaz",
    "yas": 20,
    "dersler": ["Python", "Matematik", "Fizik"]
}

# Değere erişim ve güncelleme
print(ogrenci["ad"])                # -> "Ahmet"
ogrenci["yas"] = 21                 # Yaşı güncelleme

# Yeni anahtar-değer ekleme
ogrenci["bolum"] = "Bilgisayar Mühendisliği"

# Değer silme
del ogrenci["soyad"]

# Anahtarları, değerleri ve çiftleri alma
print(ogrenci.keys())      # Tüm anahtarları verir
print(ogrenci.values())    # Tüm değerleri verir
print(ogrenci.items())     # Tüm anahtar-değer çiftlerini verir
\`\`\`

## 4. Kümeler (Sets)

Kümeler, matematikteki küme teorisinden esinlenen, benzersiz elemanları saklayan koleksiyonlardır.

- **Özellikleri:**
  - **Sırasızdır (Unordered):** Elemanların belirli bir sırası yoktur.
  - **Benzersizdir (Unique):** Bir eleman küme içinde yalnızca bir kez bulunabilir. Tekrarlanan elemanlar otomatik olarak kaldırılır.
  - **Değiştirilebilirdir (Mutable):** Eleman eklenip çıkarılabilir.
  - **Matematiksel İşlemler:** Birleşim, kesişim, fark gibi küme operasyonlarını destekler.

\`\`\`python
# Küme oluşturma
sayilar = {1, 2, 3, 3, 4, 5, 5}
print(sayilar)  # -> {1, 2, 3, 4, 5} (Tekrarlar silinir)

# Eleman ekleme ve silme
sayilar.add(6)          # Eleman ekler
sayilar.remove(1)       # Elemanı siler (eleman yoksa hata verir)
sayilar.discard(10)     # Güvenli silme (eleman yoksa hata vermez)

# Küme işlemleri
kume_a = {1, 2, 3, 4}
kume_b = {3, 4, 5, 6}

birlesim = kume_a.union(kume_b)      # veya A | B  -> {1, 2, 3, 4, 5, 6}
kesisim = kume_a.intersection(kume_b) # veya A & B  -> {3, 4}
fark = kume_a.difference(kume_b)      # veya A - B  -> {1, 2}
\`\`\`

---

## Veri Yapısı Seçimi: Ne Zaman Hangisini Kullanmalı?

- **Liste:** Sıralı bir koleksiyona ihtiyacınız olduğunda ve içeriğini sık sık değiştirmeniz gerektiğinde (ekleme, silme, güncelleme).
- **Demet:** Birbirleriyle ilişkili ve değişmemesi gereken bir grup veriyi bir arada tutmak istediğinizde.
- **Sözlük:** Verilere hızlı bir şekilde anahtar (key) ile erişmeniz gerektiğinde ve verileriniz anahtar-değer ilişkisi şeklinde organize edilebiliyorsa.
- **Küme:** Bir koleksiyondaki elemanların benzersiz olmasını garantilemek veya matematiksel küme operasyonları yapmak istediğinizde.

## Alıştırmalar

Aşağıdaki alıştırmaları yaparak temel veri yapıları bilginizi pekiştirebilirsiniz.

### Alıştırma 1: Liste Filtreleme
**Görev:** İçinde hem pozitif hem de negatif sayılar bulunan bir listeden yalnızca pozitif olanları içeren yeni bir liste oluşturun.
\`\`\`python
# Başlangıç listesi
karma_sayilar = [1, -2, 3, -5, 8, -3, 0, 7]

# Çözümünüzü buraya yazın
pozitif_sayilar = [sayi for sayi in karma_sayilar if sayi >= 0]
print(pozitif_sayilar)
\`\`\`

### Alıştırma 2: Sözlük Birleştirme
**Görev:** İki farklı sözlüğü tek bir sözlükte birleştirin. Eğer aynı anahtar her iki sözlükte de varsa, ikinci sözlükteki değer geçerli olmalıdır.
\`\`\`python
# Başlangıç sözlükleri
sozluk1 = {'a': 1, 'b': 2, 'c': 3}
sozluk2 = {'b': 4, 'd': 5}

# Çözümünüzü buraya yazın
birlesik_sozluk = sozluk1.copy()
birlesik_sozluk.update(sozluk2)
# veya daha modern bir yöntem: birlesik_sozluk = {**sozluk1, **sozluk2}
print(birlesik_sozluk)
\`\`\`

### Alıştırma 3: Ortak Elemanları Bulma
**Görev:** İki listenin kesişimini (ortak elemanlarını) bulan bir fonksiyon yazın. Sonuç listesinde tekrar eden eleman olmamalıdır. (İpucu: Kümelerden faydalanın!)
\`\`\`python
# Başlangıç listeleri
liste1 = [1, 2, 3, 4, 5, 5]
liste2 = [4, 5, 6, 7, 8, 5]

# Çözümünüzü buraya yazın
def ortak_elemanlari_bul(l1, l2):
    return list(set(l1).intersection(set(l2)))

ortaklar = ortak_elemanlari_bul(liste1, liste2)
print(ortaklar)
\`\`\`
`;

export default function BasicDataStructuresPage() {
  return (
    <div className="container mx-auto max-w-4xl px-4 py-12">
      <div className="mb-8">
        <Button asChild variant="outline" size="sm" className="gap-2">
          <Link href="/topics/python/veri-yapilari-ve-algoritmalar">
            <ArrowLeft className="h-4 w-4" />
            Geri Dön
          </Link>
        </Button>
      </div>

      <article className="prose dark:prose-invert max-w-none">
        <MarkdownContent content={content} />
      </article>

      <div className="mt-12 text-center">
        <Button asChild variant="default" size="lg" className="gap-2">
          <Link href="/topics/python/veri-yapilari-ve-algoritmalar/ileri-veri-yapilari">
            İleri Veri Yapılarına Geç <ArrowLeft className="h-5 w-5" />
          </Link>
        </Button>
      </div>
    </div>
  );
}
