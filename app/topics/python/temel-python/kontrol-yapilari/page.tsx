import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Kontrol Yapıları | Python Temelleri | Kodleon',
  description: "Python'da koşullu ifadeler, döngüler ve hata yönetimi yapılarını öğrenin.",
};

const content = `
# Kontrol Yapıları

Python'da kontrol yapıları, programınızın akışını yönetmenizi sağlayan temel programlama araçlarıdır. Bu bölümde, koşullu ifadeleri, döngüleri ve hata yönetimini öğreneceksiniz.

## Koşullu İfadeler (if-elif-else)

Koşullu ifadeler, belirli koşullara göre farklı kod bloklarını çalıştırmanızı sağlar.

### Temel if Yapısı

\`\`\`python
# Basit if örneği
yas = 18

if yas >= 18:
    print("Reşitsiniz")
else:
    print("Reşit değilsiniz")

# if-elif-else örneği
not_ = 85

if not_ >= 90:
    print("AA")
elif not_ >= 85:
    print("BA")
elif not_ >= 80:
    print("BB")
elif not_ >= 75:
    print("CB")
elif not_ >= 70:
    print("CC")
else:
    print("FF")
\`\`\`

### Mantıksal Operatörler

\`\`\`python
# and, or, not operatörleri
yas = 25
gelir = 5000

if yas >= 18 and gelir >= 4000:
    print("Kredi başvurusu yapabilirsiniz")

# Çoklu koşullar
online = True
premium = False

if online and not premium:
    print("Ücretsiz içeriklere erişebilirsiniz")
elif online and premium:
    print("Tüm içeriklere erişebilirsiniz")
else:
    print("Lütfen giriş yapın")
\`\`\`

### Karşılaştırma Operatörleri

\`\`\`python
# Temel karşılaştırma operatörleri
x = 5
y = 10

print(x == y)  # Eşitlik
print(x != y)  # Eşit değil
print(x < y)   # Küçüktür
print(x > y)   # Büyüktür
print(x <= y)  # Küçük eşittir
print(x >= y)  # Büyük eşittir

# is operatörü (kimlik karşılaştırma)
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)   # True (değer karşılaştırma)
print(a is b)   # False (kimlik karşılaştırma)
print(a is c)   # True (aynı nesneyi işaret ediyor)
\`\`\`

## Döngüler

### for Döngüsü

\`\`\`python
# Liste üzerinde iterasyon
meyveler = ["elma", "armut", "muz"]
for meyve in meyveler:
    print(meyve)

# range() ile sayı aralığında döngü
for i in range(5):      # 0'dan 4'e kadar
    print(i)

for i in range(2, 8):   # 2'den 7'ye kadar
    print(i)

for i in range(0, 10, 2):  # 0'dan 9'a 2'şer artarak
    print(i)

# Enumerate kullanımı
for index, meyve in enumerate(meyveler):
    print(f"{index}: {meyve}")

# Dictionary üzerinde döngü
kisi = {
    "ad": "Ahmet",
    "yas": 25,
    "sehir": "İstanbul"
}

for anahtar in kisi:
    print(anahtar, kisi[anahtar])

for anahtar, deger in kisi.items():
    print(f"{anahtar}: {deger}")
\`\`\`

### while Döngüsü

\`\`\`python
# Temel while döngüsü
sayac = 0
while sayac < 5:
    print(sayac)
    sayac += 1

# Sonsuz döngü ve break
while True:
    cevap = input("Çıkmak için 'q' yazın: ")
    if cevap == 'q':
        break

# continue ile döngü adımını atlama
for i in range(10):
    if i % 2 == 0:
        continue  # Çift sayıları atla
    print(i)     # Sadece tek sayıları yazdır
\`\`\`

## break, continue ve else

\`\`\`python
# break örneği
for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print(f"{n} asal değil")
            break
    else:
        print(f"{n} asal sayı")

# continue örneği
for num in range(1, 11):
    if num % 3 == 0:
        continue
    print(num)

# Döngü else'i
for i in range(5):
    if i == 10:
        break
else:
    print("Döngü normal şekilde tamamlandı")
\`\`\`

## Hata Yönetimi (try-except)

\`\`\`python
# Temel try-except
try:
    sayi = int(input("Bir sayı girin: "))
    sonuc = 10 / sayi
    print(f"Sonuç: {sonuc}")
except ValueError:
    print("Geçerli bir sayı girmediniz")
except ZeroDivisionError:
    print("Sıfıra bölme hatası")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
else:
    print("İşlem başarılı")
finally:
    print("İşlem tamamlandı")

# Özel hata yönetimi
def yas_kontrol(yas):
    if yas < 0:
        raise ValueError("Yaş negatif olamaz")
    elif yas > 120:
        raise ValueError("Geçersiz yaş değeri")
    return True

try:
    yas_kontrol(-5)
except ValueError as e:
    print(f"Hata: {e}")
\`\`\`

## with İfadesi

\`\`\`python
# Dosya işlemleri için with kullanımı
with open("dosya.txt", "r") as dosya:
    icerik = dosya.read()
    print(icerik)
# Dosya otomatik olarak kapatılır

# Birden fazla kaynak yönetimi
with open("girdi.txt") as girdi, open("cikti.txt", "w") as cikti:
    veri = girdi.read()
    cikti.write(veri.upper())
\`\`\`

## Alıştırmalar

1. **Koşullu İfadeler**
   - Bir sayının pozitif, negatif veya sıfır olduğunu kontrol eden program
   - Girilen üç sayıdan en büyüğünü bulan program
   - Basit bir hesap makinesi yapın (switch-case benzeri yapı)

2. **Döngüler**
   - 1'den 100'e kadar olan sayıların toplamını hesaplayan program
   - Çarpım tablosu oluşturan program
   - Fibonacci serisinin ilk n terimini yazdıran program

3. **Hata Yönetimi**
   - Kullanıcıdan alınan verilerle basit bir bölme işlemi
   - Dosya okuma ve yazma işlemlerinde hata yönetimi
   - Özel hata sınıfı oluşturma ve kullanma

## Sonraki Adımlar

- [Fonksiyonlar](/topics/python/temel-python/fonksiyonlar)
- [Python Akış Kontrolü Dokümantasyonu](https://docs.python.org/3/tutorial/controlflow.html)
- [Python Hata ve İstisnalar](https://docs.python.org/3/tutorial/errors.html)
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