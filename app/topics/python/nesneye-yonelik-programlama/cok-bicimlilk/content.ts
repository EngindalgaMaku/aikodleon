export const content = `
# Python'da Çok Biçimlilik (Polymorphism)

Çok biçimlilik, nesne yönelimli programlamanın temel prensiplerinden biridir. Aynı arayüzün (metod veya operatör) farklı sınıflarda farklı şekillerde uygulanabilmesini sağlar.
Bu sayede kodun daha esnek ve yeniden kullanılabilir olmasını sağlar.

::: info
# 🎯 Çok Biçimliliğin Avantajları

* **Esneklik:** Aynı arayüzü farklı sınıflarda farklı şekillerde uygulayabilme
* **Genişletilebilirlik:** Yeni sınıflar eklerken mevcut kodu değiştirmeden yeni davranışlar tanımlayabilme
* **Kod Tekrarını Önleme:** Ortak arayüzler sayesinde benzer işlemleri tek bir şekilde ele alabilme
* **Bakım Kolaylığı:** Değişiklikleri tek bir yerde yapabilme imkanı
:::

## Çok Biçimlilik Türleri

### 1. Ad Çok Biçimliliği (Ad-hoc Polymorphism)

Operatör veya fonksiyonların farklı veri tipleri için farklı davranışlar sergilemesi.

::: tip
# 💡 Ad Çok Biçimliliği İpuçları

* Python'da operatörler farklı tipler için farklı davranır
* Özel metodlar (\`__add__\`, \`__str__\` vb.) ile kendi sınıflarımız için operatör davranışlarını özelleştirebiliriz
* Fonksiyon aşırı yükleme (overloading) Python'da doğrudan desteklenmez, ancak tip kontrolü ile simüle edilebilir
:::

\`\`\`python
class Nokta:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        if isinstance(other, Nokta):
            # İki noktayı topla
            return Nokta(self.x + other.x, self.y + other.y)
        elif isinstance(other, (int, float)):
            # Noktayı bir sayı ile topla
            return Nokta(self.x + other, self.y + other)
        else:
            raise TypeError("Desteklenmeyen operand tipi")
    
    def __str__(self):
        return f"Nokta({self.x}, {self.y})"

# Kullanım örnekleri
p1 = Nokta(1, 2)
p2 = Nokta(3, 4)
p3 = p1 + p2          # İki nokta toplanır
print(p3)             # Nokta(4, 6)

p4 = p1 + 5          # Nokta sayı ile toplanır
print(p4)             # Nokta(6, 7)

try:
    p5 = p1 + "2"    # TypeError fırlatır
except TypeError as e:
    print(f"Hata: {e}")
\`\`\`

### 2. Parametrik Çok Biçimlilik (Generic Programming)

Aynı kodun farklı veri tipleri için çalışabilmesi.

\`\`\`python
from typing import TypeVar, Generic, List

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self):
        self.items: List[T] = []
    
    def push(self, item: T) -> None:
        self.items.append(item)
    
    def pop(self) -> T:
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Stack boş")
    
    def peek(self) -> T:
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Stack boş")
    
    def is_empty(self) -> bool:
        return len(self.items) == 0
    
    def size(self) -> int:
        return len(self.items)

# Farklı tiplerle kullanım
sayi_stack = Stack[int]()
sayi_stack.push(1)
sayi_stack.push(2)
print(sayi_stack.pop())  # 2

metin_stack = Stack[str]()
metin_stack.push("Python")
metin_stack.push("Java")
print(metin_stack.pop())  # "Java"
\`\`\`

### 3. Alt Tür Çok Biçimliliği (Subtype Polymorphism)

Alt sınıfların üst sınıf metodlarını override etmesi ve üst sınıf referansı üzerinden kullanılabilmesi.

::: warning
# ⚠️ Alt Tür Çok Biçimliliğinde Dikkat Edilecek Noktalar

* Alt sınıflar üst sınıfın arayüzüne sadık kalmalıdır (Liskov Substitution Principle)
* Override edilen metodlar aynı parametre yapısına sahip olmalıdır
* Üst sınıf metodları uygun şekilde çağrılmalıdır (\`super()\`)
:::

\`\`\`python
from abc import ABC, abstractmethod
from typing import List

class Sekil(ABC):
    def __init__(self, ad: str):
        self.ad = ad
    
    @abstractmethod
    def alan(self) -> float:
        pass
    
    @abstractmethod
    def cevre(self) -> float:
        pass
    
    def bilgi(self) -> str:
        return f"{self.ad} - Alan: {self.alan():.2f}, Çevre: {self.cevre():.2f}"

class Dikdortgen(Sekil):
    def __init__(self, genislik: float, yukseklik: float):
        super().__init__("Dikdörtgen")
        self.genislik = genislik
        self.yukseklik = yukseklik
    
    def alan(self) -> float:
        return self.genislik * self.yukseklik
    
    def cevre(self) -> float:
        return 2 * (self.genislik + self.yukseklik)

class Daire(Sekil):
    def __init__(self, yaricap: float):
        super().__init__("Daire")
        self.yaricap = yaricap
    
    def alan(self) -> float:
        return 3.14 * self.yaricap ** 2
    
    def cevre(self) -> float:
        return 2 * 3.14 * self.yaricap

class Ucgen(Sekil):
    def __init__(self, a: float, b: float, c: float):
        super().__init__("Üçgen")
        self.a = a
        self.b = b
        self.c = c
    
    def alan(self) -> float:
        # Heron formülü
        s = (self.a + self.b + self.c) / 2
        return (s * (s - self.a) * (s - self.b) * (s - self.c)) ** 0.5
    
    def cevre(self) -> float:
        return self.a + self.b + self.c

def sekilleri_isle(sekiller: List[Sekil]) -> None:
    for sekil in sekiller:
        print(sekil.bilgi())

# Kullanım örneği
sekiller = [
    Dikdortgen(4, 5),
    Daire(3),
    Ucgen(3, 4, 5)
]

sekilleri_isle(sekiller)
\`\`\`

## Duck Typing

Python'da çok biçimliliğin özel bir uygulaması olan Duck Typing, nesnelerin tipinden çok davranışlarına odaklanır.
"Eğer bir kuş gibi yürüyor ve bir kuş gibi ötüyorsa, o bir kuştur" prensibine dayanır.

::: info
# 🦆 Duck Typing'in Avantajları

* Daha esnek kod yazabilme
* Sınıflar arası bağımlılığı azaltma
* Kodun yeniden kullanılabilirliğini artırma
* Test edilebilirliği kolaylaştırma
:::

\`\`\`python
class Ordek:
    def ses_cikar(self):
        return "Vak vak!"
    
    def yuz(self):
        return "Ordek yüzüyor"

class Robot:
    def ses_cikar(self):
        return "Bip bip!"
    
    def yuz(self):
        return "Robot su üstünde ilerliyor"

def canli_sesi(canli):
    # Tip kontrolü yapmıyoruz, sadece gerekli metodun varlığını kontrol ediyoruz
    if hasattr(canli, 'ses_cikar'):
        return canli.ses_cikar()
    raise AttributeError("ses_cikar metodu bulunamadı")

def yuzdur(nesne):
    # Nesnenin tipini kontrol etmek yerine, yuz metodunun varlığını kontrol ediyoruz
    if hasattr(nesne, 'yuz'):
        return nesne.yuz()
    raise AttributeError("yuz metodu bulunamadı")

# Kullanım örnekleri
ordek = Ordek()
robot = Robot()

print(canli_sesi(ordek))  # "Vak vak!"
print(canli_sesi(robot))  # "Bip bip!"

print(yuzdur(ordek))      # "Ordek yüzüyor"
print(yuzdur(robot))      # "Robot su üstünde ilerliyor"

try:
    class BosClass:
        pass
    bos = BosClass()
    print(canli_sesi(bos))  # AttributeError fırlatır
except AttributeError as e:
    print(f"Hata: {e}")
\`\`\`

## Alıştırmalar

### 1. Medya Oynatıcı Sistemi

[Detaylı çözüm için tıklayın](/topics/python/nesneye-yonelik-programlama/cok-bicimlilk/medya-oynatici)

Farklı medya tiplerini oynatabilecek bir sistem tasarlayın:
* Temel \`MediaPlayer\` sınıfı
* \`play()\`, \`pause()\`, \`stop()\` gibi ortak metodlar
* Her medya tipi için özel oynatma davranışları
* Format dönüşümleri ve kalite ayarları

### 2. Şekil Çizim Uygulaması

[Detaylı çözüm için tıklayın](/topics/python/nesneye-yonelik-programlama/cok-bicimlilk/sekil-cizim)

Farklı şekilleri çizebilen bir grafik uygulaması geliştirin:
* Temel \`Shape\` sınıfı
* \`draw()\`, \`resize()\`, \`move()\` gibi ortak metodlar
* Her şekil için özel çizim mantığı
* Renk ve stil özellikleri

### 3. Oyun Karakter Sistemi

[Detaylı çözüm için tıklayın](/topics/python/nesneye-yonelik-programlama/cok-bicimlilk/oyun-karakter)

Bir RPG oyunu için karakter sistemi geliştirin:
* Temel \`Character\` sınıfı
* \`attack()\`, \`defend()\`, \`useAbility()\` gibi ortak metodlar
* Her karakter sınıfı için özel yetenekler ve davranışlar
* Seviye atlama ve ekipman sistemi

::: tip
# 💡 Çok Biçimlilik Kullanırken Dikkat Edilecek Noktalar

* **Arayüz Tutarlılığı:** Alt sınıflar üst sınıfın arayüzüne sadık kalmalıdır
* **Tek Sorumluluk:** Her sınıf tek bir sorumluluğa sahip olmalıdır
* **Açık/Kapalı Prensibi:** Sınıflar genişletmeye açık, değişikliğe kapalı olmalıdır
* **Bağımlılık Yönetimi:** Sınıflar arası bağımlılıklar minimize edilmelidir
* **Test Edilebilirlik:** Kod test edilebilir şekilde tasarlanmalıdır
:::

## Sonraki Adımlar

Çok biçimlilik konusunu detaylı örneklerle öğrendiniz. Şimdi soyut sınıflar (abstract classes) ve arayüzler (interfaces) konusuna geçerek, 
kodunuzu daha iyi organize etmeyi ve genişletilebilir hale getirmeyi öğrenebilirsiniz.
`; 