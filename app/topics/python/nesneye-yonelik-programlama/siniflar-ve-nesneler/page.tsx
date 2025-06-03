import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Info, Lightbulb, AlertTriangle } from "lucide-react";
import Image from "next/image";

export const metadata: Metadata = {
  title: 'Python OOP: Sınıflar ve Nesneler | Kodleon',
  description: 'Python\'da sınıf ve nesne kavramlarını, oluşturma yöntemlerini ve kullanım örneklerini öğrenin.',
};

const content = `
# Python'da Sınıflar ve Nesneler

Python'da nesneye yönelik programlamanın temel yapı taşları olan sınıflar ve nesneleri detaylı olarak öğrenelim.

## Sınıf Nedir?

Sınıf (Class), nesneler için bir şablon veya taslak görevi görür. Bir sınıf:
- Veri özellikleri (attributes)
- Metodlar (methods)
içerir.

Örnek bir sınıf tanımı:

\`\`\`python
class Ogrenci:
    # Sınıf özelliği (class attribute)
    okul = "Kodleon Akademi"
    
    # Yapıcı metod (constructor)
    def __init__(self, ad, soyad, numara):
        # Nesne özellikleri (instance attributes)
        self.ad = ad
        self.soyad = soyad
        self.numara = numara
        self.dersler = []
    
    # Nesne metodu (instance method)
    def bilgileri_goster(self):
        return f"{self.ad} {self.soyad} - {self.numara}"
    
    # Ders ekleme metodu
    def ders_ekle(self, ders):
        self.dersler.append(ders)
        return f"{ders} dersi eklendi"
\`\`\`

## Nesne Oluşturma

Sınıftan bir nesne (object) oluşturmak için sınıf adını fonksiyon gibi çağırırız:

\`\`\`python
# Yeni bir öğrenci nesnesi oluşturma
ogrenci1 = Ogrenci("Ahmet", "Yılmaz", "123")

# Nesne metodlarını kullanma
print(ogrenci1.bilgileri_goster())  # Çıktı: Ahmet Yılmaz - 123
print(ogrenci1.ders_ekle("Python"))  # Çıktı: Python dersi eklendi

# Sınıf özelliğine erişim
print(Ogrenci.okul)  # Çıktı: Kodleon Akademi
print(ogrenci1.okul)  # Çıktı: Kodleon Akademi
\`\`\`

## Constructor (\`__init__\`)

\`__init__\` metodu, sınıftan bir nesne oluşturulduğunda otomatik olarak çağrılan özel bir metoddur:

\`\`\`python
class Dikdortgen:
    def __init__(self, uzunluk, genislik):
        self.uzunluk = uzunluk
        self.genislik = genislik
        # Nesne oluşturulduğunda alan ve çevre otomatik hesaplanır
        self.alan = uzunluk * genislik
        self.cevre = 2 * (uzunluk + genislik)

# Nesne oluşturma
d1 = Dikdortgen(5, 3)
print(f"Alan: {d1.alan}")    # Çıktı: Alan: 15
print(f"Çevre: {d1.cevre}")  # Çıktı: Çevre: 16
\`\`\`

## Self Parametresi

\`self\` parametresi, nesnenin kendisini temsil eder ve nesne metodlarının ilk parametresi olmalıdır:

\`\`\`python
class Araba:
    def __init__(self, marka, model):
        self.marka = marka
        self.model = model
        self.hiz = 0
    
    def hizlan(self, artis):
        self.hiz += artis
        return f"Yeni hız: {self.hiz} km/s"
    
    def yavasla(self, azalis):
        if self.hiz - azalis >= 0:
            self.hiz -= azalis
        else:
            self.hiz = 0
        return f"Yeni hız: {self.hiz} km/s"

# Kullanım
araba = Araba("Toyota", "Corolla")
print(araba.hizlan(20))   # Çıktı: Yeni hız: 20 km/s
print(araba.yavasla(5))   # Çıktı: Yeni hız: 15 km/s
\`\`\`

## Instance vs. Class Attributes

Python'da iki tür özellik vardır:

1. **Instance Attributes (Nesne Özellikleri)**
   - Her nesne için özeldir
   - \`__init__\` içinde veya nesne metodlarında tanımlanır
   - \`self\` ile erişilir

2. **Class Attributes (Sınıf Özellikleri)**
   - Tüm nesneler için ortaktır
   - Sınıf içinde doğrudan tanımlanır
   - Sınıf adı ile erişilebilir

\`\`\`python
class Calisan:
    # Sınıf özelliği
    sirket = "Kodleon"
    calisan_sayisi = 0
    
    def __init__(self, ad, maas):
        # Nesne özellikleri
        self.ad = ad
        self.maas = maas
        # Sınıf özelliğini güncelleme
        Calisan.calisan_sayisi += 1

# Kullanım
c1 = Calisan("Ali", 5000)
c2 = Calisan("Ayşe", 6000)

print(Calisan.calisan_sayisi)  # Çıktı: 2
print(c1.sirket)  # Çıktı: Kodleon
print(c2.sirket)  # Çıktı: Kodleon

# Sınıf özelliğini değiştirme
Calisan.sirket = "Kodleon Tech"
print(c1.sirket)  # Çıktı: Kodleon Tech
print(c2.sirket)  # Çıktı: Kodleon Tech
\`\`\`

## İyi Uygulama Örnekleri

1. **Anlamlı İsimlendirme**
\`\`\`python
# İyi örnek
class OgrenciKayit:
    def __init__(self, ad, soyad):
        self.ad = ad
        self.soyad = soyad

# Kötü örnek
class X:
    def __init__(self, a, b):
        self.a = a
        self.b = b
\`\`\`

2. **Docstring Kullanımı**
\`\`\`python
class BankaHesabi:
    """
    Banka hesabı işlemlerini yöneten sınıf.
    
    Attributes:
        hesap_no (str): Hesap numarası
        bakiye (float): Hesap bakiyesi
    """
    
    def __init__(self, hesap_no, bakiye=0):
        self.hesap_no = hesap_no
        self.bakiye = bakiye
    
    def para_yatir(self, miktar):
        """
        Hesaba para yatırma işlemi.
        
        Args:
            miktar (float): Yatırılacak miktar
            
        Returns:
            float: Güncel bakiye
        """
        self.bakiye += miktar
        return self.bakiye
\`\`\`

3. **Type Hints Kullanımı**
\`\`\`python
from typing import List, Optional

class Kutuphane:
    def __init__(self, ad: str) -> None:
        self.ad: str = ad
        self.kitaplar: List[str] = []
    
    def kitap_ekle(self, kitap: str) -> None:
        self.kitaplar.append(kitap)
    
    def kitap_bul(self, ad: str) -> Optional[str]:
        return next((k for k in self.kitaplar if k == ad), None)
\`\`\`

## Alıştırmalar

1. **Basit Hesap Makinesi**
Toplama, çıkarma, çarpma ve bölme işlemlerini yapabilen bir hesap makinesi sınıfı oluşturun.

2. **Öğrenci Not Sistemi**
Öğrenci bilgilerini ve notlarını tutabilen, ortalama hesaplayabilen bir sınıf oluşturun.

3. **Kütüphane Yönetimi**
Kitap ekleme, silme, ödünç verme ve iade işlemlerini yapabilen bir kütüphane sınıfı oluşturun.

## Kaynaklar

- [Python Resmi Dokümantasyonu - Classes](https://docs.python.org/3/tutorial/classes.html)
- [Real Python - OOP in Python](https://realpython.com/python3-object-oriented-programming/)
- [Python Design Patterns](https://python-patterns.guide/)
`;

export default function ClassesAndObjectsPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <Button asChild variant="ghost" size="sm" className="gap-1">
          <Link href="/topics/python/nesneye-yonelik-programlama">
            <ArrowLeft className="h-4 w-4" />
            OOP Konularına Dön
          </Link>
        </Button>
      </div>
      
      <div className="prose prose-lg dark:prose-invert max-w-none">
        <MarkdownContent content={content} />
      </div>
      
      {/* Interactive Examples Section */}
      <div className="my-12">
        <h2 className="text-3xl font-bold mb-8">İnteraktif Örnekler</h2>
        <Tabs defaultValue="example1">
          <TabsList>
            <TabsTrigger value="example1">Örnek 1</TabsTrigger>
            <TabsTrigger value="example2">Örnek 2</TabsTrigger>
            <TabsTrigger value="example3">Örnek 3</TabsTrigger>
          </TabsList>
          <TabsContent value="example1">
            <Card>
              <CardHeader>
                <CardTitle>Basit Sınıf Örneği</CardTitle>
                <CardDescription>
                  Temel bir öğrenci sınıfı ve kullanımı
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`class Ogrenci:
    def __init__(self, ad, numara):
        self.ad = ad
        self.numara = numara
    
    def bilgi_goster(self):
        return f"{self.ad} - {self.numara}"

# Kullanım
ogrenci = Ogrenci("Ali", "123")
print(ogrenci.bilgi_goster())`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="example2">
            <Card>
              <CardHeader>
                <CardTitle>Hesap Makinesi</CardTitle>
                <CardDescription>
                  Dört işlem yapabilen bir sınıf örneği
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`class HesapMakinesi:
    def topla(self, a, b):
        return a + b
    
    def cikar(self, a, b):
        return a - b
    
    def carp(self, a, b):
        return a * b
    
    def bol(self, a, b):
        if b != 0:
            return a / b
        return "Sıfıra bölünemez"

# Kullanım
hm = HesapMakinesi()
print(hm.topla(5, 3))  # 8`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="example3">
            <Card>
              <CardHeader>
                <CardTitle>Banka Hesabı</CardTitle>
                <CardDescription>
                  Para yatırma ve çekme işlemleri yapabilen bir sınıf
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`class BankaHesabi:
    def __init__(self, bakiye=0):
        self.bakiye = bakiye
    
    def para_yatir(self, miktar):
        self.bakiye += miktar
        return f"Yeni bakiye: {self.bakiye}"
    
    def para_cek(self, miktar):
        if miktar <= self.bakiye:
            self.bakiye -= miktar
            return f"Yeni bakiye: {self.bakiye}"
        return "Yetersiz bakiye"

# Kullanım
hesap = BankaHesabi(1000)
print(hesap.para_yatir(500))  # Yeni bakiye: 1500`}</code>
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
          <AlertTitle>İsimlendirme Kuralları</AlertTitle>
          <AlertDescription>
            Sınıf isimleri PascalCase (HesapMakinesi), metod ve özellik isimleri snake_case (hesap_no) olmalıdır.
          </AlertDescription>
        </Alert>

        <Alert>
          <Lightbulb className="h-4 w-4" />
          <AlertTitle>Docstring Kullanımı</AlertTitle>
          <AlertDescription>
            Her sınıf ve metodun işlevini açıklayan docstring'ler ekleyin. Bu, kodunuzun bakımını kolaylaştırır.
          </AlertDescription>
        </Alert>

        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Dikkat Edilmesi Gerekenler</AlertTitle>
          <AlertDescription>
            Sınıf özelliklerini doğrudan erişime açık bırakmak yerine, getter ve setter metodları kullanmayı düşünün.
          </AlertDescription>
        </Alert>
      </div>

      {/* Navigasyon Linkleri */}
      <div className="mt-12 flex flex-col sm:flex-row justify-between gap-4">
        <Button variant="outline" disabled className="gap-2">
          <ArrowLeft className="h-4 w-4" />
          Önceki Konu
        </Button>
        
        <Button asChild variant="default" className="gap-2">
          <Link href="/topics/python/nesneye-yonelik-programlama/kalitim">
            Sonraki Konu: Kalıtım
            <ArrowRight className="h-4 w-4" />
          </Link>
        </Button>
      </div>
      
      <div className="mt-16 text-center text-sm text-muted-foreground">
        <p>© {new Date().getFullYear()} Kodleon | Yapay Zeka Eğitim Platformu</p>
      </div>
    </div>
  );
} 