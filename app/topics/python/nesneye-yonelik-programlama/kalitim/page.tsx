import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Info, Lightbulb, AlertTriangle } from "lucide-react";
import Image from "next/image";

export const metadata: Metadata = {
  title: 'Python OOP: Kalıtım (Inheritance) | Kodleon',
  description: 'Python\'da kalıtım kavramını, türleri, kullanım örnekleri ve best practice\'leri ile öğrenin.',
};

const content = `
# Kalıtım (Inheritance)

Kalıtım, bir sınıfın başka bir sınıfın özelliklerini ve metodlarını miras almasını sağlayan OOP özelliğidir. Bu sayede kod tekrarını önler ve hiyerarşik bir yapı oluşturabiliriz.

## Temel Kalıtım

Bir sınıfın başka bir sınıftan kalıtım alması için, sınıf tanımında parantez içinde üst sınıfı belirtmemiz yeterlidir:

\`\`\`python
class Hayvan:
    def __init__(self, isim, yas):
        self.isim = isim
        self.yas = yas
    
    def ses_cikar(self):
        return "Ses yok"
    
    def bilgi_goster(self):
        return f"{self.isim}, {self.yas} yaşında"

class Kopek(Hayvan):
    def __init__(self, isim, yas, tur):
        super().__init__(isim, yas)  # Üst sınıfın constructor'ını çağır
        self.tur = tur
    
    def ses_cikar(self):  # Method override
        return "Hav hav!"
    
    def kuyruk_salla(self):  # Yeni method
        return "Kuyruk sallanıyor..."

# Kullanım
kopek = Kopek("Karabaş", 3, "Golden")
print(kopek.bilgi_goster())  # Çıktı: Karabaş, 3 yaşında
print(kopek.ses_cikar())     # Çıktı: Hav hav!
print(kopek.kuyruk_salla())  # Çıktı: Kuyruk sallanıyor...
\`\`\`

## Çoklu Kalıtım

Python, bir sınıfın birden fazla sınıftan kalıtım almasına izin verir:

\`\`\`python
class Canli:
    def yasam_formu(self):
        return "Canlı varlık"

class Ucabilen:
    def uc(self):
        return "Uçuyor..."

class Kus(Canli, Ucabilen):
    def __init__(self, isim):
        self.isim = isim
    
    def ses_cikar(self):
        return "Cik cik!"

# Kullanım
kus = Kus("Sarı")
print(kus.yasam_formu())  # Çıktı: Canlı varlık
print(kus.uc())          # Çıktı: Uçuyor...
print(kus.ses_cikar())   # Çıktı: Cik cik!
\`\`\`

## Method Resolution Order (MRO)

Python'da çoklu kalıtım kullanırken metodların hangi sırayla aranacağını MRO belirler:

\`\`\`python
class A:
    def metod(self):
        return "A'dan"

class B(A):
    def metod(self):
        return "B'den"

class C(A):
    def metod(self):
        return "C'den"

class D(B, C):
    pass

# MRO'yu görüntüleme
print(D.__mro__)  
# Çıktı: (<class '__main__.D'>, <class '__main__.B'>, 
#         <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)

d = D()
print(d.metod())  # Çıktı: B'den
\`\`\`

## super() Kullanımı

\`super()\` fonksiyonu, üst sınıfın metodlarına erişmemizi sağlar:

\`\`\`python
class Arac:
    def __init__(self, marka, model):
        self.marka = marka
        self.model = model
    
    def bilgi(self):
        return f"{self.marka} {self.model}"

class Otomobil(Arac):
    def __init__(self, marka, model, renk):
        super().__init__(marka, model)  # Üst sınıfın __init__ metodunu çağır
        self.renk = renk
    
    def bilgi(self):
        return f"{super().bilgi()}, Renk: {self.renk}"

# Kullanım
araba = Otomobil("Toyota", "Corolla", "Kırmızı")
print(araba.bilgi())  # Çıktı: Toyota Corolla, Renk: Kırmızı
\`\`\`

## Mixin Sınıfları

Mixin'ler, belirli işlevselliği sağlayan ve genellikle tek başına kullanılmayan sınıflardır:

\`\`\`python
class LogMixin:
    def log(self, message):
        print(f"[LOG] {message}")

class JSONMixin:
    def to_json(self):
        import json
        return json.dumps(self.__dict__)

class Kullanici(LogMixin, JSONMixin):
    def __init__(self, ad, email):
        self.ad = ad
        self.email = email
    
    def kaydet(self):
        self.log(f"Kullanıcı kaydedildi: {self.ad}")
        return self.to_json()

# Kullanım
kullanici = Kullanici("Ahmet", "ahmet@example.com")
print(kullanici.kaydet())
# Çıktı: [LOG] Kullanıcı kaydedildi: Ahmet
# {"ad": "Ahmet", "email": "ahmet@example.com"}
\`\`\`

## İyi Uygulama Örnekleri

1. **Composition vs Inheritance**
\`\`\`python
# Inheritance - Bazen uygun olmayabilir
class FileManager(dict):
    pass

# Composition - Genellikle daha iyi bir seçenek
class FileManager:
    def __init__(self):
        self._data = {}
\`\`\`

2. **Abstract Base Classes**
\`\`\`python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)
\`\`\`

3. **Interface Segregation**
\`\`\`python
# Kötü örnek
class Animal:
    def fly(self): pass
    def swim(self): pass
    def run(self): pass

# İyi örnek
class Flyable:
    def fly(self): pass

class Swimmable:
    def swim(self): pass

class Bird(Flyable):
    def fly(self):
        return "Kuş uçuyor"

class Fish(Swimmable):
    def swim(self):
        return "Balık yüzüyor"
\`\`\`

## Alıştırmalar

1. **Şekil Hiyerarşisi**
Bir Shape base class oluşturun ve bundan türeyen Circle, Rectangle, Triangle sınıfları yazın.

2. **Çalışan Sistemi**
Employee base class'ından türeyen Manager, Developer, Designer sınıfları oluşturun.

3. **Oyun Karakterleri**
Character base class'ından türeyen Warrior, Mage, Archer sınıfları yazın.

## Kaynaklar

- [Python Inheritance Documentation](https://docs.python.org/3/tutorial/classes.html#inheritance)
- [Real Python - Inheritance and Composition](https://realpython.com/inheritance-composition-python/)
- [Python MRO Guide](https://www.python.org/download/releases/2.3/mro/)
`;

export default function InheritancePage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Interactive Examples Section */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">İnteraktif Örnekler</h2>
          <Tabs defaultValue="example1">
            <TabsList>
              <TabsTrigger value="example1">Temel Kalıtım</TabsTrigger>
              <TabsTrigger value="example2">Çoklu Kalıtım</TabsTrigger>
              <TabsTrigger value="example3">Mixin Örneği</TabsTrigger>
            </TabsList>
            <TabsContent value="example1">
              <Card>
                <CardHeader>
                  <CardTitle>Hayvan Sınıf Hiyerarşisi</CardTitle>
                  <CardDescription>
                    Temel kalıtım örneği
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                    <code>{`class Hayvan:
    def __init__(self, isim):
        self.isim = isim
    
    def ses_cikar(self):
        return "Ses yok"

class Kedi(Hayvan):
    def ses_cikar(self):
        return "Miyav!"

# Kullanım
kedi = Kedi("Pamuk")
print(kedi.ses_cikar())  # Çıktı: Miyav!`}</code>
                  </pre>
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="example2">
              <Card>
                <CardHeader>
                  <CardTitle>Süper Kahraman</CardTitle>
                  <CardDescription>
                    Çoklu kalıtım örneği
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                    <code>{`class Ucabilen:
    def uc(self):
        return "Uçuyor..."

class GucluVuran:
    def vur(self):
        return "Güçlü vuruş!"

class SuperKahraman(Ucabilen, GucluVuran):
    def __init__(self, isim):
        self.isim = isim

# Kullanım
kahraman = SuperKahraman("Süperman")
print(kahraman.uc())   # Çıktı: Uçuyor...
print(kahraman.vur())  # Çıktı: Güçlü vuruş!`}</code>
                  </pre>
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="example3">
              <Card>
                <CardHeader>
                  <CardTitle>Logger Mixin</CardTitle>
                  <CardDescription>
                    Mixin kullanım örneği
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                    <code>{`class LoggerMixin:
    def log(self, message):
        print(f"[LOG] {message}")

class Veritabani(LoggerMixin):
    def kaydet(self, veri):
        self.log(f"Veri kaydedildi: {veri}")
        return "Başarılı"

# Kullanım
db = Veritabani()
db.kaydet("test")  # Çıktı: [LOG] Veri kaydedildi: test`}</code>
                  </pre>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        {/* Tips and Best Practices */}
        <div className="my-12 space-y-4">
          <h2 className="text-3xl font-bold mb-8">İpuçları ve En İyi Pratikler</h2>
          
          <Alert>
            <Info className="h-4 w-4" />
            <AlertTitle>Kalıtım vs Kompozisyon</AlertTitle>
            <AlertDescription>
              "Kalıtım yerine kompozisyon kullan" prensibini unutmayın. Eğer "is-a" ilişkisi yoksa, kompozisyonu tercih edin.
            </AlertDescription>
          </Alert>

          <Alert>
            <Lightbulb className="h-4 w-4" />
            <AlertTitle>super() Kullanımı</AlertTitle>
            <AlertDescription>
              Üst sınıfın metodlarını çağırırken her zaman super() kullanın. Bu, çoklu kalıtımda sorunları önler.
            </AlertDescription>
          </Alert>

          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Derin Kalıtım Hiyerarşisi</AlertTitle>
            <AlertDescription>
              Çok derin kalıtım hiyerarşilerinden kaçının. Genellikle 2-3 seviyeden derin olmaması önerilir.
            </AlertDescription>
          </Alert>
        </div>
      </div>
    </div>
  );
} 