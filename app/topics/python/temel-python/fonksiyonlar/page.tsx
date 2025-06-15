import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Fonksiyonlar | Python Temelleri | Kodleon',
  description: "Python'da yeniden kullanılabilir kod blokları olan fonksiyonları tanımlamayı, parametreleri, *args, **kwargs, lambda fonksiyonlarını ve kapsam (scope) kurallarını öğrenin.",
};

const content = `
# Fonksiyonlar

Fonksiyonlar, belirli bir görevi yerine getirmek için organize edilmiş, yeniden kullanılabilir kod bloklarıdır. Kod tekrarını önler, programı daha modüler, okunaklı ve yönetilebilir hale getirirler.

---

## 1. Fonksiyon Tanımlama ve Çağırma
Python'da bir fonksiyon \`def\` anahtar kelimesi ile tanımlanır, ardından fonksiyon adı ve parantezler \`()\` gelir.

\`\`\`python
# Parametresiz bir fonksiyon
def selamla():
    """Bu fonksiyon ekrana 'Merhaba, Dünya!' yazar."""
    print("Merhaba, Dünya!")

# Fonksiyonu çağırma
selamla()
\`\`\`

### Parametreler ve Argümanlar
Fonksiyonlar, işlevlerini yerine getirmek için dışarıdan veri alabilirler. Bu verilere **parametre** denir. Fonksiyonu çağırırken bu parametrelere gönderilen değerlere ise **argüman** denir.

\`\`\`python
def kisi_selamla(isim): # 'isim' bir parametredir
    """Verilen ismi kullanarak kişiye özel selamlama yapar."""
    print(f"Merhaba, {isim}!")

kisi_selamla("Ayşe") # "Ayşe" bir argümandır
kisi_selamla("Mehmet")
\`\`\`

### \`return\` İfadesi
Bir fonksiyon, yaptığı işlemin sonucunu çağrıldığı yere geri gönderebilir. Bu, \`return\` anahtar kelimesi ile yapılır.

\`\`\`python
def topla(a, b):
    """İki sayıyı toplayıp sonucunu döndürür."""
    return a + b

toplam_sonuc = topla(5, 3)
print(f"Toplam: {toplam_sonuc}") # Çıktı: Toplam: 8
\`\`\`

---

## 2. Gelişmiş Argüman Kullanımı

### Varsayılan Argüman Değerleri
Parametrelere varsayılan değerler atayarak, fonksiyon çağrılırken bu argümanların girilmesini isteğe bağlı hale getirebilirsiniz.

\`\`\`python
def guc_al(taban, us=2):
    """Bir sayının belirtilen üssünü alır. Üs belirtilmezse karesini alır."""
    return taban ** us

print(guc_al(3))     # 9 (varsayılan üs olan 2 kullanıldı)
print(guc_al(3, 3))  # 27 (üs olarak 3 gönderildi)
\`\`\`

### Anahtar Kelime Argümanları (Keyword Arguments)
Argümanları, parametre adlarını belirterek sırasız bir şekilde gönderebilirsiniz. Bu, özellikle çok sayıda parametresi olan fonksiyonlarda kodun okunabilirliğini artırır.

\`\`\`python
def kullanici_profili(ad, soyad, yas, sehir="Bilinmiyor"):
    print(f"{ad} {soyad}, {yas} yaşında, {sehir}'de yaşıyor.")

# Anahtar kelime argümanları ile çağırma
kullanici_profili(yas=25, sehir="Ankara", ad="Ali", soyad="Veli")
\`\`\`

### Değişken Sayıda Argümanlar: \`*args\` ve \`**kwargs\`

#### \`*args\` (Konumsal Argümanlar)
Bir fonksiyonun değişken sayıda konumsal argüman almasını sağlar. Bu argümanlar fonksiyon içinde bir **demet (tuple)** olarak saklanır.

\`\`\`python
def sayilari_topla(*args):
    """Verilen tüm sayıları toplar."""
    print(f"Gelen argümanlar (tuple): {args}")
    return sum(args)

print(sayilari_topla(1, 2, 3))          # 6
print(sayilari_topla(10, 20, 30, 40))   # 100
\`\`\`

#### \`**kwargs\` (Anahtar Kelime Argümanları)
Bir fonksiyonun değişken sayıda anahtar kelime argümanı almasını sağlar. Bu argümanlar fonksiyon içinde bir **sözlük (dictionary)** olarak saklanır.

\`\`\`python
def profil_olustur(**kwargs):
    """Verilen bilgileri kullanarak bir profil yazdırır."""
    print("Kullanıcı Profili:")
    for anahtar, deger in kwargs.items():
        print(f"- {anahtar.capitalize()}: {deger}")

profil_olustur(ad="Zeynep", yas=30, meslek="Mühendis", sehir="İzmir")
\`\`\`

---

## 3. Lambda Fonksiyonları
Lambda fonksiyonları, \`lambda\` anahtar kelimesi ile oluşturulan küçük, isimsiz (anonim) fonksiyonlardır. Genellikle tek bir ifade içerirler ve başka bir fonksiyonun argümanı olarak kullanılırlar.

\`\`\`python
# Normal fonksiyon
def kare(x):
    return x * x

# Lambda eşdeğeri
kare_lambda = lambda x: x * x

print(kare_lambda(5)) # 25

# Kullanım örneği: listeyi elemanların ikinci değerine göre sıralama
liste = [(1, 5), (3, 2), (2, 8)]
liste.sort(key=lambda item: item[1])
print(liste) # [(3, 2), (1, 5), (2, 8)]
\`\`\`

---

## 4. Değişken Kapsamı (Scope)
Bir değişkenin programın hangi bölümünden erişilebilir olduğunu belirtir.

- **Yerel Kapsam (Local Scope):** Bir fonksiyon içinde tanımlanan değişkenler sadece o fonksiyon içinde geçerlidir.
- **Global Kapsam (Global Scope):** Fonksiyonların dışında, en üst seviyede tanımlanan değişkenlerdir ve programın her yerinden erişilebilirler.

\`\`\`python
x = 10 # Global değişken

def benim_fonksiyonum():
    y = 5 # Yerel değişken
    print(f"Fonksiyon içi global x: {x}")
    print(f"Fonksiyon içi yerel y: {y}")

benim_fonksiyonum()
print(f"Fonksiyon dışı global x: {x}")
# print(y) # Bu satır NameError verir çünkü 'y' yerel bir değişkendir.
\`\`\`

---

## Alıştırmalar ve Çözümleri

### Alıştırma 1: Asal Sayı Bulma
Bir sayının asal olup olmadığını kontrol eden bir fonksiyon yazın. Fonksiyon, sayı asalsa \`True\`, değilse \`False\` döndürmelidir. (1 asal değildir.)

**Çözüm:**
\`\`\`python
def asal_mi(sayi):
    """Bir sayının asal olup olmadığını kontrol eder."""
    if sayi <= 1:
        return False
    # 2'den sayının kareköküne kadar olan sayılara bölünüp bölünmediğini kontrol et
    for i in range(2, int(sayi**0.5) + 1):
        if sayi % i == 0:
            return False
    return True

# Test
print(f"17 asal mı? {asal_mi(17)}") # True
print(f"25 asal mı? {asal_mi(25)}") # False
print(f"2 asal mı? {asal_mi(2)}")   # True
\`\`\`

### Alıştırma 2: Faktöriyel Hesaplama (Özyinelemeli)
Bir sayının faktöriyelini özyinelemeli (recursive) bir fonksiyon kullanarak hesaplayın. Özyineleme, bir fonksiyonun kendi kendini çağırmasıdır.

**Çözüm:**
\`\`\`python
def faktoriyel(n):
    """Bir sayının faktöriyelini özyinelemeli olarak hesaplar."""
    # Temel durum: 0! veya 1! her zaman 1'dir.
    if n == 0 or n == 1:
        return 1
    # Özyinelemeli adım
    else:
        return n * faktoriyel(n - 1)

# Test
sayi = 5
print(f"{sayi}! = {faktoriyel(sayi)}") # 120
\`\`\`

### Alıştırma 3: Lambda ile Filtreleme
Verilen bir listedeki pozitif, negatif ve sıfır sayılarını ayrı listelere ayıran bir program yazın. \`filter()\` ve lambda fonksiyonlarını kullanın.

**Çözüm:**
\`\`\`python
sayilar = [-5, -2, 0, 1, 3, 4, -1, 0, 8]

pozitif_sayilar = list(filter(lambda x: x > 0, sayilar))
negatif_sayilar = list(filter(lambda x: x < 0, sayilar))
sifirlar = list(filter(lambda x: x == 0, sayilar))

print(f"Orijinal Liste: {sayilar}")
print(f"Pozitif Sayılar: {pozitif_sayilar}")
print(f"Negatif Sayılar: {negatif_sayilar}")
print(f"Sıfırlar: {sifirlar}")
\`\`\`

## Sonraki Adımlar
Fonksiyonlar Python'da güçlü bir araçtır. Bu temelleri anladıktan sonra, kodunuzu daha da organize etmek için [Sınıflar ve Nesneler (OOP)](/topics/python/temel-python/siniflar-ve-nesneler) konusuna geçebilirsiniz.
`;

export default function PythonFunctionsPage() {
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