---
title: "Python ile Nesneye Yönelik Programlama"
description: "Python'da nesneye yönelik programlamanın (OOP) temelleri, sınıflar, nesneler, kalıtım ve daha fazlası."
keywords: "python, oop, nesneye yönelik programlama, sınıf, nesne, kalıtım, inheritance, encapsulation, polymorphism"
---

# Python ile Nesneye Yönelik Programlama (OOP)

Python'da nesneye yönelik programlama (OOP), kodunuzu daha modüler, okunabilir ve yeniden kullanılabilir hale getiren güçlü bir programlama paradigmasıdır. Bu rehberde, OOP'nin temel kavramlarını ve Python'da nasıl uygulandığını detaylı örneklerle öğreneceksiniz.

## 1. Sınıf ve Nesne Nedir?

Sınıf (Class), nesnelerin şablonunu tanımlayan bir yapıdır. Nesne (Object) ise bu şablondan oluşturulan somut örneklerdir.

### Basit Bir Sınıf Örneği

```python
class Ogrenci:
    def __init__(self, ad, soyad, numara):
        self.ad = ad
        self.soyad = soyad
        self.numara = numara
    
    def bilgileri_goster(self):
        return f"{self.ad} {self.soyad} - {self.numara}"

# Nesne oluşturma
ogrenci1 = Ogrenci("Ahmet", "Yılmaz", "123")
print(ogrenci1.bilgileri_goster())  # Çıktı: Ahmet Yılmaz - 123
```

## 2. Yapıcı (Constructor) ve Özellikler

`__init__` metodu Python'da yapıcı (constructor) görevini görür. Nesne oluşturulduğunda otomatik olarak çağrılır.

### Özellikler ve Property Dekoratörü

```python
class Dikdortgen:
    def __init__(self, uzunluk, genislik):
        self._uzunluk = uzunluk  # protected özellik
        self._genislik = genislik
    
    @property
    def alan(self):
        return self._uzunluk * self._genislik
    
    @property
    def cevre(self):
        return 2 * (self._uzunluk + self._genislik)

dikdortgen = Dikdortgen(5, 3)
print(f"Alan: {dikdortgen.alan}")  # Çıktı: Alan: 15
print(f"Çevre: {dikdortgen.cevre}")  # Çıktı: Çevre: 16
```

## 3. Kapsülleme (Encapsulation)

Kapsülleme, sınıfın iç detaylarını dış dünyadan gizlemeyi sağlar. Python'da bu genellikle isimlendirme kurallarıyla yapılır:

```python
class BankaHesabi:
    def __init__(self, hesap_no, bakiye):
        self.__hesap_no = hesap_no  # private özellik
        self.__bakiye = bakiye
    
    def para_yatir(self, miktar):
        if miktar > 0:
            self.__bakiye += miktar
            return True
        return False
    
    def para_cek(self, miktar):
        if 0 < miktar <= self.__bakiye:
            self.__bakiye -= miktar
            return True
        return False
    
    def bakiye_goruntule(self):
        return self.__bakiye

hesap = BankaHesabi("12345", 1000)
hesap.para_yatir(500)
print(hesap.bakiye_goruntule())  # Çıktı: 1500
```

## 4. Kalıtım (Inheritance)

Kalıtım, bir sınıfın başka bir sınıftan özellik ve metotları devralmasını sağlar:

```python
class Hayvan:
    def __init__(self, isim, yas):
        self.isim = isim
        self.yas = yas
    
    def ses_cikar(self):
        pass

class Kopek(Hayvan):
    def ses_cikar(self):
        return "Hav hav!"

class Kedi(Hayvan):
    def ses_cikar(self):
        return "Miyav!"

kopek = Kopek("Karabaş", 3)
kedi = Kedi("Pamuk", 2)

print(kopek.ses_cikar())  # Çıktı: Hav hav!
print(kedi.ses_cikar())   # Çıktı: Miyav!
```

## 5. Çok Biçimlilik (Polymorphism)

Çok biçimlilik, aynı arayüzü kullanarak farklı türdeki nesnelerle çalışabilme yeteneğidir:

```python
def hayvan_konustur(hayvan):
    print(f"{hayvan.isim} diyor ki: {hayvan.ses_cikar()}")

hayvanlar = [
    Kopek("Karabaş", 3),
    Kedi("Pamuk", 2),
    Kopek("Findik", 1)
]

for hayvan in hayvanlar:
    hayvan_konustur(hayvan)
```

## 6. Soyut Sınıflar ve Metodlar

Python'da soyut sınıflar `abc` modülü kullanılarak oluşturulur:

```python
from abc import ABC, abstractmethod

class SoyutHayvan(ABC):
    @abstractmethod
    def ses_cikar(self):
        pass
    
    @abstractmethod
    def hareket_et(self):
        pass

class Kus(SoyutHayvan):
    def ses_cikar(self):
        return "Cik cik"
    
    def hareket_et(self):
        return "Uçuyor"
```

## 7. Özel Metodlar (Magic Methods)

Python'da özel metodlar, nesnelerin davranışlarını özelleştirmenizi sağlar:

```python
class Nokta:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"Nokta({self.x}, {self.y})"
    
    def __add__(self, other):
        return Nokta(self.x + other.x, self.y + other.y)

n1 = Nokta(1, 2)
n2 = Nokta(3, 4)
n3 = n1 + n2
print(n3)  # Çıktı: Nokta(4, 6)
```

## İyi Uygulama Örnekleri

1. **SOLID Prensipleri**
   - Single Responsibility (Tek Sorumluluk)
   - Open/Closed (Açık/Kapalı)
   - Liskov Substitution (Liskov Yerine Geçme)
   - Interface Segregation (Arayüz Ayrımı)
   - Dependency Inversion (Bağımlılığın Ters Çevrilmesi)

2. **Tasarım Desenleri**
   - Factory Pattern
   - Singleton Pattern
   - Observer Pattern
   - Strategy Pattern

## Öneriler ve En İyi Pratikler

1. Sınıf isimleri PascalCase ile yazılmalıdır
2. Metot ve özellik isimleri snake_case ile yazılmalıdır
3. Private özellikler için çift alt çizgi (`__`) kullanın
4. Docstring'leri kullanarak kodunuzu dokümante edin
5. Type hints kullanarak kod okunabilirliğini artırın

## Alıştırmalar

1. Basit bir banka hesabı sistemi oluşturun
2. Bir okul yönetim sistemi tasarlayın
3. Geometrik şekiller hiyerarşisi oluşturun
4. Bir oyun için karakter sınıfları tasarlayın

## Kaynaklar ve İleri Okuma

- [Python Resmi Dokümantasyonu](https://docs.python.org/3/tutorial/classes.html)
- [Real Python - OOP Tutorials](https://realpython.com/python3-object-oriented-programming/)
- [Python Design Patterns](https://python-patterns.guide/) 