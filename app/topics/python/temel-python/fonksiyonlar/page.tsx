import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Fonksiyonlar | Python Temelleri | Kodleon',
  description: "Python'da fonksiyonlar, parametreler, dönüş değerleri ve modüller hakkında detaylı bilgi edinin.",
};

const content = `
# Fonksiyonlar

Fonksiyonlar, Python'da kod tekrarını önleyen ve programınızı modüler hale getiren yapı taşlarıdır. Bu bölümde, fonksiyonların tanımlanmasını, parametreleri, dönüş değerlerini ve modülleri öğreneceksiniz.

## Fonksiyon Tanımlama

Python'da fonksiyonlar \`def\` anahtar kelimesi ile tanımlanır:

\`\`\`python
# Basit fonksiyon tanımlama
def selamla():
    print("Merhaba!")

# Fonksiyonu çağırma
selamla()  # Çıktı: Merhaba!

# Parametreli fonksiyon
def kisi_selamla(isim):
    print(f"Merhaba, {isim}!")

kisi_selamla("Ahmet")  # Çıktı: Merhaba, Ahmet!

# Dönüş değeri olan fonksiyon
def kare_al(sayi):
    return sayi ** 2

sonuc = kare_al(5)  # sonuc = 25
\`\`\`

## Parametreler ve Argümanlar

### Varsayılan Parametreler

\`\`\`python
# Varsayılan değerli parametreler
def guc_al(taban, us=2):
    return taban ** us

print(guc_al(3))     # 9 (3^2)
print(guc_al(3, 3))  # 27 (3^3)

# Birden fazla varsayılan parametre
def kisi_bilgileri(ad, soyad, yas=None, sehir="Belirtilmedi"):
    bilgi = f"{ad} {soyad}"
    if yas:
        bilgi += f", {yas} yaşında"
    if sehir != "Belirtilmedi":
        bilgi += f", {sehir}'de yaşıyor"
    return bilgi

print(kisi_bilgileri("Ahmet", "Yılmaz"))
print(kisi_bilgileri("Ayşe", "Demir", 25, "İstanbul"))
\`\`\`

### Konumsal ve İsimli Argümanlar

\`\`\`python
def dikdortgen_alan(uzunluk, genislik):
    return uzunluk * genislik

# Konumsal argümanlar
alan1 = dikdortgen_alan(5, 3)

# İsimli argümanlar
alan2 = dikdortgen_alan(genislik=3, uzunluk=5)

# Karışık kullanım (önce konumsal, sonra isimli)
alan3 = dikdortgen_alan(5, genislik=3)
\`\`\`

### Değişken Sayıda Argümanlar

\`\`\`python
# *args: Değişken sayıda konumsal argüman
def toplam(*sayilar):
    return sum(sayilar)

print(toplam(1, 2, 3))       # 6
print(toplam(1, 2, 3, 4, 5)) # 15

# **kwargs: Değişken sayıda isimli argüman
def kisi_olustur(**bilgiler):
    for anahtar, deger in bilgiler.items():
        print(f"{anahtar}: {deger}")

kisi_olustur(ad="Ahmet", yas=25, sehir="İstanbul")

# *args ve **kwargs birlikte kullanımı
def genel_fonksiyon(*args, **kwargs):
    print("Konumsal argümanlar:", args)
    print("İsimli argümanlar:", kwargs)

genel_fonksiyon(1, 2, 3, ad="Ahmet", yas=25)
\`\`\`

## Lambda Fonksiyonları

Lambda fonksiyonları, tek satırda tanımlanan anonim fonksiyonlardır:

\`\`\`python
# Basit lambda fonksiyonu
kare = lambda x: x ** 2
print(kare(5))  # 25

# Lambda fonksiyonlarını filtreleme ile kullanma
sayilar = [1, 2, 3, 4, 5, 6]
cift_sayilar = list(filter(lambda x: x % 2 == 0, sayilar))
print(cift_sayilar)  # [2, 4, 6]

# Lambda fonksiyonlarını sıralama ile kullanma
kisiler = [("Ahmet", 25), ("Mehmet", 30), ("Ayşe", 20)]
kisiler.sort(key=lambda x: x[1])  # Yaşa göre sıralama
print(kisiler)
\`\`\`

## Fonksiyon Dekoratörleri

Dekoratörler, fonksiyonların davranışını değiştirmek için kullanılır:

\`\`\`python
# Basit dekoratör örneği
def zaman_olc(fonksiyon):
    from time import time
    
    def sarmalayici(*args, **kwargs):
        baslangic = time()
        sonuc = fonksiyon(*args, **kwargs)
        bitis = time()
        print(f"{fonksiyon.__name__} fonksiyonu {bitis - baslangic} saniye sürdü")
        return sonuc
    
    return sarmalayici

@zaman_olc
def faktoriyel(n):
    from math import factorial
    return factorial(n)

sonuc = faktoriyel(1000)
\`\`\`

## Modüller ve Import

### Modül İçe Aktarma

\`\`\`python
# Tüm modülü içe aktarma
import math
print(math.pi)  # 3.141592653589793

# Belirli fonksiyonları içe aktarma
from random import randint, choice
print(randint(1, 10))
print(choice(["elma", "armut", "muz"]))

# Takma isimle içe aktarma
import numpy as np
import pandas as pd

# Modülden her şeyi içe aktarma (önerilmez)
from math import *
\`\`\`

### Kendi Modülünüzü Oluşturma

\`\`\`python
# hesaplamalar.py
def toplama(a, b):
    return a + b

def cikarma(a, b):
    return a - b

def carpma(a, b):
    return a * b

def bolme(a, b):
    if b != 0:
        return a / b
    raise ValueError("Sıfıra bölme hatası")

# Başka bir dosyada kullanım
from hesaplamalar import toplama, cikarma
# veya
import hesaplamalar

sonuc = hesaplamalar.toplama(5, 3)
\`\`\`

## Alıştırmalar

1. **Temel Fonksiyonlar**
   - Bir sayının faktöriyelini hesaplayan fonksiyon yazın
   - Verilen bir listedeki en büyük ve en küçük sayıyı bulan fonksiyon yazın
   - Bir metindeki sesli harfleri sayan fonksiyon yazın

2. **Parametreler ve Dönüş Değerleri**
   - Değişken sayıda sayı alan ve ortalamasını hesaplayan fonksiyon
   - İsimli parametrelerle öğrenci bilgilerini alan ve sözlük döndüren fonksiyon
   - Birden fazla değer döndüren fonksiyon (tuple olarak)

3. **Dekoratörler ve Lambda**
   - Fonksiyon çağrılarını loglayan bir dekoratör yazın
   - Liste işlemlerini lambda fonksiyonlarıyla yapın
   - Özyinelemeli (recursive) fonksiyonlar için dekoratör yazın

## Sonraki Adımlar

- [Veri Yapıları](/topics/python/temel-python/veri-yapilari)
- [Python Fonksiyonlar Dokümantasyonu](https://docs.python.org/3/tutorial/controlflow.html#defining-functions)
- [Python Modüller Dokümantasyonu](https://docs.python.org/3/tutorial/modules.html)
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