export const content = `
# Python OOP Terimler Sözlüğü

Bu sözlük, Nesneye Yönelik Programlama (OOP) kavramlarını anlamanıza yardımcı olacak temel terimleri içerir.

## Temel Kavramlar

### Nesne (Object)
Bir sınıfın örneği olan ve veri ile davranışları bir arada tutan yapı. Nesneler, gerçek dünyadaki varlıkların yazılımdaki temsilidir.

\`\`\`python
# Örnek bir nesne oluşturma
class Araba:
    def __init__(self, marka, model):
        self.marka = marka
        self.model = model

araba1 = Araba("Toyota", "Corolla")  # araba1 bir nesnedir
\`\`\`

### Sınıf (Class)
Nesnelerin şablonunu tanımlayan yapı. Bir sınıf, nesnelerin sahip olacağı özellikleri (attributes) ve metodları (methods) belirler.

### Örnek (Instance)
Bir sınıftan oluşturulan belirli bir nesne. Her örnek, aynı sınıftan oluşturulsa bile benzersizdir ve kendi veri değerlerine sahiptir.

### Örnekleme (Instantiation)
Bir sınıftan yeni bir nesne oluşturma işlemi. Python'da bu işlem sınıf adını fonksiyon gibi çağırarak yapılır.

### Soyutlama (Abstraction)
Karmaşık sistemleri basitleştirmek için gereksiz detayları gizleyip sadece gerekli özellikleri gösterme prensibi.

### Kapsülleme (Encapsulation)
Veri ve metodları bir arada tutup, dış dünyadan erişimi kontrol etme prensibi. Python'da \`_\` ve \`__\` önekleriyle yapılır.

\`\`\`python
class BankaHesabi:
    def __init__(self):
        self.__bakiye = 0  # private değişken
        
    def para_yatir(self, miktar):
        if miktar > 0:
            self.__bakiye += miktar
\`\`\`

### Kalıtım (Inheritance)
Bir sınıfın başka bir sınıfın özelliklerini ve davranışlarını devralması.

\`\`\`python
class Hayvan:
    def ses_cikar(self):
        pass

class Kopek(Hayvan):  # Hayvan sınıfından kalıtım
    def ses_cikar(self):
        return "Hav hav!"
\`\`\`

### Çok Biçimlilik (Polymorphism)
Aynı isimli metodların farklı sınıflarda farklı davranışlar sergileyebilmesi.

## İleri Kavramlar

### Soyut Sınıf (Abstract Class)
Doğrudan örneklenemeyen, sadece kalıtım için kullanılan sınıf. Python'da \`abc\` modülü ile oluşturulur.

### Arayüz (Interface)
Bir sınıfın uygulaması gereken metodları tanımlayan yapı. Python'da açıkça tanımlanmaz, soyut sınıflarla simüle edilir.

### Kompozisyon (Composition)
Bir sınıfın başka sınıfların nesnelerini içermesi ve onları kullanması.

### Çoklu Kalıtım (Multiple Inheritance)
Bir sınıfın birden fazla üst sınıftan kalıtım alabilmesi. Python bunu destekler.

\`\`\`python
class A:
    pass

class B:
    pass

class C(A, B):  # Çoklu kalıtım örneği
    pass
\`\`\`

## Python'a Özgü Terimler

### Pythonic
Python'un felsefesine ve idiomlarına uygun kod yazma yaklaşımı.

### Dunder Metodları
\`__init__\`, \`__str__\` gibi çift alt çizgi ile başlayıp biten özel metodlar.

### Dekoratörler
Fonksiyon ve sınıfların davranışını değiştirmek için kullanılan özel yapılar.

\`\`\`python
@property  # Bir dekoratör örneği
def tam_ad(self):
    return f"{self.ad} {self.soyad}"
\`\`\`

### Docstring
Python'da sınıf, fonksiyon ve modüllerin dokümantasyonu için kullanılan çok satırlı string'ler.

\`\`\`python
class Ogrenci:
    """
    Öğrenci bilgilerini tutan sınıf.
    
    Attributes:
        ad (str): Öğrencinin adı
        numara (int): Öğrenci numarası
    """
    pass
\`\`\`

### Property
Bir sınıf özelliğine get/set metodları tanımlamak için kullanılan dekoratör.

### Mixin
Sınıflara ek özellikler eklemek için kullanılan özel sınıflar.

### MRO (Method Resolution Order)
Çoklu kalıtımda metodların aranma sırası.

## En İyi Uygulamalar

### SOLID Prensipleri
- Single Responsibility (Tek Sorumluluk)
- Open/Closed (Açık/Kapalı)
- Liskov Substitution (Liskov Yerine Geçme)
- Interface Segregation (Arayüz Ayrımı)
- Dependency Inversion (Bağımlılığın Ters Çevrilmesi)

### DRY (Don't Repeat Yourself)
Kod tekrarından kaçınma prensibi.

### KISS (Keep It Simple, Stupid)
Basitliği koruma prensibi.

### YAGNI (You Ain't Gonna Need It)
İhtiyaç olmayan özellikleri eklememek.
` 