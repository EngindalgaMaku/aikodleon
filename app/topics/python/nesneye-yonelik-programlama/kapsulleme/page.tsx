import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Info, Lightbulb, AlertTriangle } from "lucide-react";
import Image from "next/image";

export const metadata: Metadata = {
  title: 'Python OOP: Kapsülleme (Encapsulation) | Kodleon',
  description: 'Python\'da kapsülleme kavramını, veri gizleme yöntemlerini ve property kullanımını öğrenin.',
};

const content = `
# Kapsülleme (Encapsulation)

Kapsülleme, bir sınıfın içindeki veri ve metodların dış dünyadan gizlenmesi ve sadece belirlenen arayüzler üzerinden erişilmesini sağlayan OOP prensibidir.

## Private ve Protected Üyeler

Python'da gerçek anlamda private değişkenler yoktur, ancak isimlendirme kuralları ile gizlilik sağlanabilir:

\`\`\`python
class BankaHesabi:
    def __init__(self, hesap_no, bakiye):
        self._hesap_no = hesap_no    # Protected (tek alt çizgi)
        self.__bakiye = bakiye       # Private (çift alt çizgi)
    
    def bakiye_goruntule(self):
        return self.__bakiye
    
    def para_yatir(self, miktar):
        if miktar > 0:
            self.__bakiye += miktar
            return True
        return False

# Kullanım
hesap = BankaHesabi("123456", 1000)
print(hesap.bakiye_goruntule())  # 1000
print(hesap._hesap_no)          # Erişilebilir ama önerilmez
# print(hesap.__bakiye)         # AttributeError
print(hesap._BankaHesabi__bakiye)  # Name mangling ile erişim
\`\`\`

## Property Dekoratörü

Property dekoratörü, sınıf özelliklerine kontrollü erişim sağlar:

\`\`\`python
class Calisan:
    def __init__(self, ad, maas):
        self._ad = ad
        self._maas = maas
    
    @property
    def maas(self):
        return self._maas
    
    @maas.setter
    def maas(self, yeni_maas):
        if yeni_maas < 0:
            raise ValueError("Maaş negatif olamaz!")
        self._maas = yeni_maas
    
    @property
    def ad(self):
        return self._ad
    
    @ad.setter
    def ad(self, yeni_ad):
        if not yeni_ad.strip():
            raise ValueError("Ad boş olamaz!")
        self._ad = yeni_ad

# Kullanım
calisan = Calisan("Ahmet", 5000)
print(calisan.maas)      # 5000
calisan.maas = 6000      # Setter çağrılır
# calisan.maas = -1000   # ValueError: Maaş negatif olamaz!
\`\`\`

## Getter ve Setter Metodları

Property dekoratörü öncesinde kullanılan geleneksel yaklaşım:

\`\`\`python
class Ogrenci:
    def __init__(self, ad, not_ortalamasi):
        self.__ad = ad
        self.__not_ortalamasi = not_ortalamasi
    
    def get_not_ortalamasi(self):
        return self.__not_ortalamasi
    
    def set_not_ortalamasi(self, yeni_ortalama):
        if 0 <= yeni_ortalama <= 100:
            self.__not_ortalamasi = yeni_ortalama
        else:
            raise ValueError("Not 0-100 arasında olmalıdır!")
    
    def get_ad(self):
        return self.__ad
    
    def set_ad(self, yeni_ad):
        if len(yeni_ad) >= 2:
            self.__ad = yeni_ad
        else:
            raise ValueError("Ad en az 2 karakter olmalıdır!")

# Kullanım
ogrenci = Ogrenci("Ali", 85)
print(ogrenci.get_not_ortalamasi())  # 85
ogrenci.set_not_ortalamasi(90)       # OK
# ogrenci.set_not_ortalamasi(150)    # ValueError
\`\`\`

## Name Mangling

Python'da çift alt çizgi (\`__\`) ile başlayan özellikler için name mangling uygulanır:

\`\`\`python
class Ornek:
    def __init__(self):
        self.__gizli = "Gizli veri"
        self._yarigizli = "Yarı gizli veri"
        self.acik = "Açık veri"

nesne = Ornek()
print(nesne.acik)         # "Açık veri"
print(nesne._yarigizli)   # "Yarı gizli veri"
# print(nesne.__gizli)    # AttributeError
print(nesne._Ornek__gizli)  # "Gizli veri" (name mangling ile erişim)
\`\`\`

## İyi Uygulama Örnekleri

1. **Property Kullanımı**
\`\`\`python
class Dikdortgen:
    def __init__(self, genislik, yukseklik):
        self._genislik = genislik
        self._yukseklik = yukseklik
        self._alan = None  # Cache için
    
    @property
    def alan(self):
        # Cache kullanımı
        if self._alan is None:
            self._alan = self._genislik * self._yukseklik
        return self._alan
    
    @property
    def genislik(self):
        return self._genislik
    
    @genislik.setter
    def genislik(self, deger):
        if deger <= 0:
            raise ValueError("Genişlik pozitif olmalıdır!")
        self._genislik = deger
        self._alan = None  # Cache'i sıfırla
\`\`\`

2. **Validation ve Type Checking**
\`\`\`python
class Kullanici:
    def __init__(self, email):
        self.email = email  # Setter otomatik çağrılır
    
    @property
    def email(self):
        return self._email
    
    @email.setter
    def email(self, value):
        if '@' not in value:
            raise ValueError("Geçersiz email adresi!")
        self._email = value.lower()
\`\`\`

3. **Read-Only Property**
\`\`\`python
class Urun:
    def __init__(self, kod, fiyat):
        self._kod = kod
        self._fiyat = fiyat
    
    @property
    def kod(self):
        return self._kod
    # setter tanımlanmadığı için kod read-only olur
\`\`\`

## Alıştırmalar

1. **Banka Hesabı**
Bakiye, hesap numarası gibi hassas bilgileri kapsülleyen bir BankaHesabi sınıfı yazın.

2. **Öğrenci Bilgi Sistemi**
Notları ve kişisel bilgileri güvenli şekilde tutan bir Ogrenci sınıfı oluşturun.

3. **E-ticaret Ürün Sınıfı**
Ürün bilgilerini ve fiyat değişikliklerini kontrollü şekilde yöneten bir Urun sınıfı yazın.

## Kaynaklar

- [Python Property Documentation](https://docs.python.org/3/library/functions.html#property)
- [Real Python - OOP Properties](https://realpython.com/python-property/)
- [Python Name Mangling](https://docs.python.org/3/tutorial/classes.html#private-variables)
`;

export default function EncapsulationPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Interactive Examples Section */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">İnteraktif Örnekler</h2>
          <Tabs defaultValue="example1">
            <TabsList>
              <TabsTrigger value="example1">Property Örneği</TabsTrigger>
              <TabsTrigger value="example2">Getter/Setter</TabsTrigger>
              <TabsTrigger value="example3">Name Mangling</TabsTrigger>
            </TabsList>
            <TabsContent value="example1">
              <Card>
                <CardHeader>
                  <CardTitle>Sıcaklık Dönüşümü</CardTitle>
                  <CardDescription>
                    Property kullanarak sıcaklık dönüşümü yapan sınıf
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                    <code>{`class Sicaklik:
    def __init__(self, celsius):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return (self.celsius * 9/5) + 32

# Kullanım
s = Sicaklik(25)
print(s.fahrenheit)  # 77.0
s.celsius = 30
print(s.fahrenheit)  # 86.0`}</code>
                  </pre>
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="example2">
              <Card>
                <CardHeader>
                  <CardTitle>Banka Hesabı</CardTitle>
                  <CardDescription>
                    Geleneksel getter/setter kullanımı
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                    <code>{`class BankaHesabi:
    def __init__(self, bakiye):
        self.__bakiye = bakiye
    
    def get_bakiye(self):
        return self.__bakiye
    
    def set_bakiye(self, miktar):
        if miktar >= 0:
            self.__bakiye = miktar
        else:
            raise ValueError("Bakiye negatif olamaz!")

# Kullanım
hesap = BankaHesabi(1000)
print(hesap.get_bakiye())  # 1000
hesap.set_bakiye(2000)     # OK`}</code>
                  </pre>
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="example3">
              <Card>
                <CardHeader>
                  <CardTitle>Gizli Özellikler</CardTitle>
                  <CardDescription>
                    Name mangling örneği
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                    <code>{`class GizliVeri:
    def __init__(self):
        self.__gizli = "123456"
        self._yarigizli = "abcdef"
    
    def gizli_goster(self):
        return self.__gizli

# Kullanım
veri = GizliVeri()
print(veri._yarigizli)        # abcdef
# print(veri.__gizli)         # AttributeError
print(veri._GizliVeri__gizli) # 123456`}</code>
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
            <AlertTitle>Property Kullanımı</AlertTitle>
            <AlertDescription>
              Veri doğrulama ve dönüşüm işlemleri için property dekoratörünü tercih edin.
            </AlertDescription>
          </Alert>

          <Alert>
            <Lightbulb className="h-4 w-4" />
            <AlertTitle>Değişken İsimlendirme</AlertTitle>
            <AlertDescription>
              Protected üyeler için tek alt çizgi (_), private üyeler için çift alt çizgi (__) kullanın.
            </AlertDescription>
          </Alert>

          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Name Mangling</AlertTitle>
            <AlertDescription>
              Name mangling'e güvenmeyin. Bu bir güvenlik önlemi değil, yanlışlıkla üzerine yazılmayı önleyen bir özelliktir.
            </AlertDescription>
          </Alert>
        </div>
      </div>
    </div>
  );
} 