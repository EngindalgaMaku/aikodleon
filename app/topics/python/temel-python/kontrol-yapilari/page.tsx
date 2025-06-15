import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Kontrol Yapıları | Python Temelleri | Kodleon',
  description: "Python'da program akışını yönetmek için kullanılan koşullu ifadeler (if/elif/else), döngüler (for, while) ve hata yönetimi (try/except) yapılarını öğrenin.",
};

const content = `
# Kontrol Yapıları

Kontrol yapıları, bir programın hangi kod bloklarının hangi koşullar altında ve kaç kez çalışacağını belirleyen temel programlama mekanizmalarıdır. Bu yapılar sayesinde programlarımıza mantık ve esneklik kazandırırız.

---

## 1. Koşullu İfadeler (if, elif, else)

Koşullu ifadeler, belirli bir koşulun doğru (\`True\`) veya yanlış (\`False\`) olmasına bağlı olarak farklı kod bloklarını çalıştırmamızı sağlar.

### if Yapısı
En temel koşul yapısıdır. Belirtilen koşul doğru ise içindeki kod bloğu çalışır.

\`\`\`python
# Örnek: Yaş kontrolü
yas = 20
if yas >= 18:
    print("Ehliyet alabilirsiniz.")
\`\`\`

### if-else Yapısı
Koşul doğru ise \`if\` bloğu, yanlış ise \`else\` bloğu çalışır.

\`\`\`python
# Örnek: Parola kontrolü
dogru_parola = "12345"
girilen_parola = input("Parolanızı girin: ")

if girilen_parola == dogru_parola:
    print("Giriş başarılı!")
else:
    print("Hatalı parola!")
\`\`\`

### if-elif-else Yapısı
Birden fazla koşulu ardı ardına kontrol etmek için kullanılır. Python'da \`switch-case\` yapısının karşılığıdır.

\`\`\`python
# Örnek: Not değerlendirme sistemi
not_ = 77

if not_ >= 90:
    harf_notu = "AA"
elif not_ >= 85:
    harf_notu = "BA"
elif not_ >= 80:
    harf_notu = "BB"
elif not_ >= 70:
    harf_notu = "CB"
elif not_ >= 60:
    harf_notu = "CC"
else:
    harf_notu = "FF"

print(f"Harf notunuz: {harf_notu}")
\`\`\`

### İç İçe (Nested) if Yapıları
Bir \`if\` bloğunun içine başka \`if\` blokları yerleştirilebilir.

\`\`\`python
# Örnek: Kullanıcı yetkilendirme
kullanici_adi = "admin"
is_admin = True

if kullanici_adi == "admin":
    if is_admin:
        print("Yönetici paneline hoş geldiniz!")
    else:
        print("Yetkiniz yok, normal kullanıcı olarak giriş yapıldı.")
else:
    print("Normal kullanıcı girişi yapıldı.")
\`\`\`

---

## 2. Döngüler

Döngüler, bir kod bloğunu belirli bir koşul sağlandığı sürece veya bir koleksiyonun her bir elemanı için tekrar tekrar çalıştırmamızı sağlar.

### for Döngüsü
Bir koleksiyon (liste, demet, sözlük, string vb.) üzerindeki her bir eleman için işlem yapmak amacıyla kullanılır.

\`\`\`python
# Liste üzerinde döngü
meyveler = ["elma", "armut", "kiraz"]
for meyve in meyveler:
    print(f"{meyve.title()} lezzetlidir.")

# range() fonksiyonu ile belirli sayıda tekrar
# 0'dan 4'e kadar olan sayıları yazdırır
for i in range(5):
    print(i)

# String üzerinde döngü
for harf in "Python":
    print(harf, end="-") # Çıktı: P-y-t-h-o-n-
\`\`\`

### while Döngüsü
Belirtilen bir koşul doğru (\`True\`) olduğu sürece çalışmaya devam eder. Koşulun döngü içinde bir noktada \`False\` yapılmasına dikkat edilmelidir, aksi takdirde sonsuz döngü oluşur.

\`\`\`python
# Örnek: 1'den 5'e kadar sayma
sayac = 1
while sayac <= 5:
    print(f"Sayı: {sayac}")
    sayac += 1 # Koşulu sonlandırmak için sayaç artırılır

# Örnek: Kullanıcıdan doğru girdi alana kadar sorma
while True:
    cevap = input("Çıkmak için 'evet' yazın: ")
    if cevap.lower() == "evet":
        print("Program sonlandırılıyor...")
        break # Döngüyü sonlandırır
\`\`\`

### Döngü Kontrol İfadeleri: \`break\` ve \`continue\`
- **\`break\`**: İçinde bulunduğu döngüyü anında sonlandırır.
- **\`continue\`**: Döngünün mevcut adımını atlar ve bir sonraki adımdan devam eder.

\`\`\`python
# break: 10'a kadar olan sayılardan 7'yi bulunca dur
for sayi in range(1, 11):
    if sayi == 7:
        print("7 bulundu, döngü sonlandırılıyor.")
        break
    print(sayi)

# continue: 10'a kadar olan sayılardan çift olanları atla
for sayi in range(1, 11):
    if sayi % 2 == 0:
        continue
    print(sayi) # Sadece tek sayıları yazdırır
\`\`\`

### Döngülerde \`else\` Bloğu
Bir döngü, \`break\` ifadesi ile sonlandırılmazsa, döngü tamamlandıktan sonra \`else\` bloğu çalışır. Bu, genellikle bir arama işlemi sonucunda bir şeyin bulunup bulunmadığını kontrol etmek için kullanılır.

\`\`\`python
# Örnek: Listede belirli bir elemanı arama
sayilar = [2, 4, 6, 8, 10]
aranan_sayi = 7

for sayi in sayilar:
    if sayi == aranan_sayi:
        print(f"{aranan_sayi} bulundu!")
        break
else:
    # Bu blok sadece döngü break ile kırılmazsa çalışır
    print(f"{aranan_sayi} listede bulunamadı.")
\`\`\`

---

## 3. Hata Yönetimi (try, except, finally)

Program çalışırken oluşabilecek beklenmedik hataları (exceptions) yönetmek ve programın çökmesini engellemek için kullanılır.

### try-except Bloğu
Hata oluşma potansiyeli olan kod \`try\` bloğuna, oluşabilecek hatayı yakalamak için ise \`except\` bloğuna yazılır.

\`\`\`python
# Örnek: Sayısal girdi ve sıfıra bölme hatası
try:
    sayi1 = int(input("Bölünecek sayıyı girin: "))
    sayi2 = int(input("Bölen sayıyı girin: "))
    sonuc = sayi1 / sayi2
    print(f"Sonuç: {sonuc}")
except ValueError:
    print("Hata: Lütfen sadece sayısal bir değer girin.")
except ZeroDivisionError:
    print("Hata: Bir sayı sıfıra bölünemez.")
except Exception as e:
    # Beklenmedik diğer tüm hataları yakalamak için
    print(f"Beklenmedik bir hata oluştu: {e}")
\`\`\`

### \`else\` ve \`finally\` Blokları
- **\`else\`**: \`try\` bloğunda hiçbir hata oluşmazsa çalışır.
- **\`finally\`**: Hata oluşsa da oluşmasa da her durumda çalışır. Genellikle kaynakları serbest bırakmak (dosya kapatma vb.) için kullanılır.

\`\`\`python
try:
    dosya = open("veriler.txt", "r")
    # ... dosya işlemleri ...
except FileNotFoundError:
    print("Dosya bulunamadı.")
else:
    print("Dosya başarıyla okundu.")
    dosya.close() # Dosyayı kapat
finally:
    print("Hata yönetimi bloğu tamamlandı.")
\`\`\`

---

## Alıştırmalar ve Çözümleri

### Alıştırma 1: Faktöriyel Hesaplama
Kullanıcıdan bir sayı alan ve bu sayının faktöriyelini hesaplayan bir program yazın. (Örn: 5! = 5 * 4 * 3 * 2 * 1 = 120). Negatif sayılar için hata mesajı verin.

**Çözüm:**
\`\`\`python
sayi_str = input("Faktöriyelini hesaplamak için bir sayı girin: ")

try:
    sayi = int(sayi_str)
    if sayi < 0:
        print("Negatif sayıların faktöriyeli hesaplanamaz.")
    elif sayi == 0:
        print("0! = 1")
    else:
        faktoriyel = 1
        for i in range(1, sayi + 1):
            faktoriyel *= i
        print(f"{sayi}! = {faktoriyel}")
except ValueError:
    print("Lütfen geçerli bir tam sayı girin.")
\`\`\`

### Alıştırma 2: Sayı Tahmin Oyunu
Program 1 ile 100 arasında rastgele bir sayı tutsun. Kullanıcıdan bu sayıyı tahmin etmesini isteyin. Kullanıcının her tahmininden sonra "Daha büyük" veya "Daha küçük" şeklinde ipuçları verin. Kullanıcı doğru sayıyı bulduğunda kaç denemede bulduğunu ekrana yazdırın.

**Çözüm:**
\`\`\`python
import random

hedef_sayi = random.randint(1, 100)
tahmin_sayisi = 0
tahmin = 0

print("1-100 arasında bir sayı tuttum. Bakalım bulabilecek misin?")

while tahmin != hedef_sayi:
    try:
        tahmin_str = input("Tahmininiz: ")
        tahmin = int(tahmin_str)
        tahmin_sayisi += 1

        if tahmin < hedef_sayi:
            print("Daha büyük bir sayı girin.")
        elif tahmin > hedef_sayi:
            print("Daha küçük bir sayı girin.")
        else:
            print(f"Tebrikler! {hedef_sayi} sayısını {tahmin_sayisi} denemede buldunuz.")
    except ValueError:
        print("Geçersiz giriş. Lütfen bir sayı girin.")
\`\`\`

### Alıştırma 3: Asal Sayı Bulma
1'den 100'e kadar olan asal sayıları bulan ve bunları bir liste olarak ekrana yazdıran bir program yazın. Döngülerde \`else\` bloğunu kullanarak bir sayının asal olup olmadığını kontrol edin.

**Çözüm:**
\`\`\`python
asal_sayilar = []
for sayi in range(2, 101):  # Asal sayılar 2'den başlar
    for i in range(2, sayi):
        if (sayi % i) == 0:
            # Tam bölündüyse asal değildir, iç döngüyü kır
            break
    else:
        # İç döngü hiç kırılmadıysa sayı asaldır
        asal_sayilar.append(sayi)

print("1-100 arasındaki asal sayılar:")
print(asal_sayilar)
\`\`\`

## Sonraki Adımlar
Kontrol yapılarını öğrendiğinize göre, kodunuzu daha modüler ve yeniden kullanılabilir hale getirmek için [Fonksiyonlar](/topics/python/temel-python/fonksiyonlar) konusuna geçebilirsiniz.
`;

export default function PythonControlStructuresPage() {
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