'use client';

import Link from 'next/link';
import { ArrowLeft, ArrowRight, Code2, BookOpen } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Info, Lightbulb, AlertTriangle } from "lucide-react";
import Image from "next/image";
import { useState } from "react";
import Quiz from './components/Quiz';

const content = `
# Sınıflar ve Nesneler

Python'da nesne yönelimli programlamanın temel yapı taşları sınıflar ve nesnelerdir. Bu bölümde, sınıfların nasıl oluşturulduğunu ve nesnelerin nasıl kullanıldığını öğreneceksiniz.

## Sınıf Nedir?

Sınıf (class), nesnelerin özelliklerini ve davranışlarını tanımlayan bir şablondur. Örneğin, bir "Araba" sınıfı düşünelim:

\`\`\`python
class Araba:
    # Constructor (Yapıcı) metod
    def __init__(self, marka, model, yil):
        self.marka = marka    # Instance variable (örnek değişkeni)
        self.model = model    # Instance variable
        self.yil = yil        # Instance variable
        self.hiz = 0         # Varsayılan değer

    # Instance method (örnek metodu)
    def hizlan(self, artis):
        self.hiz += artis
        return f"Hız {self.hiz} km/s'ye yükseltildi"

    def yavasla(self, azalis):
        if self.hiz - azalis >= 0:
            self.hiz -= azalis
            return f"Hız {self.hiz} km/s'ye düşürüldü"
        return "Hız 0'ın altına düşemez"

    def bilgi_goster(self):
        return f"{self.yil} model {self.marka} {self.model}, Mevcut hız: {self.hiz} km/s"
\`\`\`

## Nesne Oluşturma ve Kullanma

Sınıftan nesne oluşturmak ve metodları kullanmak:

\`\`\`python
# Yeni bir araba nesnesi oluşturma
araba1 = Araba("Toyota", "Corolla", 2022)

# Metodları kullanma
print(araba1.bilgi_goster())  # 2022 model Toyota Corolla, Mevcut hız: 0 km/s
print(araba1.hizlan(50))      # Hız 50 km/s'ye yükseltildi
print(araba1.yavasla(20))     # Hız 30 km/s'ye düşürüldü
print(araba1.bilgi_goster())  # 2022 model Toyota Corolla, Mevcut hız: 30 km/s
\`\`\`

## Constructor (\`__init__\`) Metodu

Constructor, sınıftan bir nesne oluşturulduğunda otomatik olarak çağrılan özel bir metoddur:

\`\`\`python
class Ogrenci:
    def __init__(self, ad, soyad, numara):
        self.ad = ad
        self.soyad = soyad
        self.numara = numara
        self.dersler = []    # Boş liste ile başlatma

    def ders_ekle(self, ders):
        self.dersler.append(ders)
        return f"{ders} dersi eklendi"

    def dersleri_listele(self):
        if self.dersler:
            return f"{self.ad}'in aldığı dersler: {', '.join(self.dersler)}"
        return f"{self.ad} henüz ders almamış"

# Kullanım örneği
ogrenci1 = Ogrenci("Ahmet", "Yılmaz", "2023001")
print(ogrenci1.ders_ekle("Python"))
print(ogrenci1.ders_ekle("Veri Yapıları"))
print(ogrenci1.dersleri_listele())
\`\`\`

## Instance Metodları

Instance metodları, nesnenin durumunu değiştiren ve nesneyle ilgili işlemler yapan fonksiyonlardır:

\`\`\`python
class BankaHesabi:
    def __init__(self, hesap_no, sahip, bakiye=0):
        self.hesap_no = hesap_no
        self.sahip = sahip
        self.bakiye = bakiye
        self.islemler = []

    def para_yatir(self, miktar):
        if miktar > 0:
            self.bakiye += miktar
            self.islemler.append(f"Yatırma: +{miktar} TL")
            return f"{miktar} TL yatırıldı. Yeni bakiye: {self.bakiye} TL"
        return "Geçersiz miktar"

    def para_cek(self, miktar):
        if miktar > 0 and self.bakiye >= miktar:
            self.bakiye -= miktar
            self.islemler.append(f"Çekme: -{miktar} TL")
            return f"{miktar} TL çekildi. Yeni bakiye: {self.bakiye} TL"
        return "Yetersiz bakiye veya geçersiz miktar"

    def hesap_ozeti(self):
        ozet = f"Hesap No: {self.hesap_no}\\nSahip: {self.sahip}\\nBakiye: {self.bakiye} TL\\n"
        ozet += "\\nSon İşlemler:\\n"
        for islem in self.islemler[-5:]:  # Son 5 işlem
            ozet += f"- {islem}\\n"
        return ozet

# Kullanım örneği
hesap = BankaHesabi("123456", "Ali Veli", 1000)
print(hesap.para_yatir(500))
print(hesap.para_cek(200))
print(hesap.hesap_ozeti())
\`\`\`

## Self Parametresi

\`self\` parametresi, metodun çağrıldığı nesneyi temsil eder. Python'da instance metodlarının ilk parametresi her zaman \`self\` olmalıdır:

\`\`\`python
class Nokta:
    def __init__(self, x, y):
        self.x = x  # self.x nesnenin x koordinatını temsil eder
        self.y = y  # self.y nesnenin y koordinatını temsil eder

    def kordinatlari_goster(self):
        return f"X: {self.x}, Y: {self.y}"

    def nokta_tasi(self, delta_x, delta_y):
        self.x += delta_x  # self ile nesnenin x değerini değiştiriyoruz
        self.y += delta_y  # self ile nesnenin y değerini değiştiriyoruz
        return self.kordinatlari_goster()

# Kullanım örneği
nokta1 = Nokta(5, 10)
print(nokta1.kordinatlari_goster())  # X: 5, Y: 10
print(nokta1.nokta_tasi(3, -2))      # X: 8, Y: 8
\`\`\`
`;

const examples = [
  {
    title: "Araba Sınıfı",
    description: "Araba sınıfının nasıl oluşturulduğunu ve metodlarının nasıl kullanıldığını gösterir.",
    code: `class Araba:
    # Constructor (Yapıcı) metod
    def __init__(self, marka, model, yil):
        self.marka = marka    # Instance variable (örnek değişkeni)
        self.model = model    # Instance variable
        self.yil = yil        # Instance variable
        self.hiz = 0         # Varsayılan değer

    # Instance method (örnek metodu)
    def hizlan(self, artis):
        self.hiz += artis
        return f"Hız {self.hiz} km/s'ye yükseltildi"

    def yavasla(self, azalis):
        if self.hiz - azalis >= 0:
            self.hiz -= azalis
            return f"Hız {self.hiz} km/s'ye düşürüldü"
        return "Hız 0'ın altına düşemez"

    def bilgi_goster(self):
        return f"{self.yil} model {self.marka} {self.model}, Mevcut hız: {self.hiz} km/s"
`,
    explanation: "Araba sınıfı, marka, model ve yıl gibi özellikleri ve hizlan, yavasla ve bilgi_goster gibi metodları içerir."
  },
  {
    title: "Nesne Oluşturma ve Kullanma",
    description: "Araba sınıfından bir nesne oluşturma ve metodlarının nasıl kullanıldığını gösterir.",
    code: `# Yeni bir araba nesnesi oluşturma
araba1 = Araba("Toyota", "Corolla", 2022)

# Metodları kullanma
print(araba1.bilgi_goster())  # 2022 model Toyota Corolla, Mevcut hız: 0 km/s
print(araba1.hizlan(50))      # Hız 50 km/s'ye yükseltildi
print(araba1.yavasla(20))     # Hız 30 km/s'ye düşürüldü
print(araba1.bilgi_goster())  # 2022 model Toyota Corolla, Mevcut hız: 30 km/s
`,
    explanation: "Araba sınıfından bir nesne oluşturulur ve bilgi_goster, hizlan ve yavasla metodları kullanılarak nesnenin özellikleri ve davranışları incelenir."
  },
  {
    title: "Constructor (\`__init__\`) Metodu",
    description: "Constructor metodunun nasıl kullanıldığını gösterir.",
    code: `class Ogrenci:
    def __init__(self, ad, soyad, numara):
        self.ad = ad
        self.soyad = soyad
        self.numara = numara
        self.dersler = []    # Boş liste ile başlatma

    def ders_ekle(self, ders):
        self.dersler.append(ders)
        return f"{ders} dersi eklendi"

    def dersleri_listele(self):
        if self.dersler:
            return f"{self.ad}'in aldığı dersler: {', '.join(self.dersler)}"
        return f"{self.ad} henüz ders almamış"

# Kullanım örneği
ogrenci1 = Ogrenci("Ahmet", "Yılmaz", "2023001")
print(ogrenci1.ders_ekle("Python"))
print(ogrenci1.ders_ekle("Veri Yapıları"))
print(ogrenci1.dersleri_listele())
`,
    explanation: "Ogrenci sınıfı, ad, soyad ve numara gibi özellikleri ve ders_ekle ve dersleri_listele gibi metodları içerir. Constructor metodu, nesne oluşturulduğunda otomatik olarak çağrılır."
  },
  {
    title: "Instance Metodları",
    description: "Instance metodlarının nasıl kullanıldığını gösterir.",
    code: `class BankaHesabi:
    def __init__(self, hesap_no, sahip, bakiye=0):
        self.hesap_no = hesap_no
        self.sahip = sahip
        self.bakiye = bakiye
        self.islemler = []

    def para_yatir(self, miktar):
        if miktar > 0:
            self.bakiye += miktar
            self.islemler.append(f"Yatırma: +{miktar} TL")
            return f"{miktar} TL yatırıldı. Yeni bakiye: {self.bakiye} TL"
        return "Geçersiz miktar"

    def para_cek(self, miktar):
        if miktar > 0 and self.bakiye >= miktar:
            self.bakiye -= miktar
            self.islemler.append(f"Çekme: -{miktar} TL")
            return f"{miktar} TL çekildi. Yeni bakiye: {self.bakiye} TL"
        return "Yetersiz bakiye veya geçersiz miktar"

    def hesap_ozeti(self):
        ozet = f"Hesap No: {self.hesap_no}\\nSahip: {self.sahip}\\nBakiye: {self.bakiye} TL\\n"
        ozet += "\\nSon İşlemler:\\n"
        for islem in self.islemler[-5:]:  # Son 5 işlem
            ozet += f"- {islem}\\n"
        return ozet

# Kullanım örneği
hesap = BankaHesabi("123456", "Ali Veli", 1000)
print(hesap.para_yatir(500))
print(hesap.para_cek(200))
print(hesap.hesap_ozeti())
`,
    explanation: "BankaHesabi sınıfı, hesap_no, sahip ve bakiye gibi özellikleri ve para_yatir, para_cek ve hesap_ozeti gibi metodları içerir. Instance metodları, nesnenin durumunu değiştirebilir ve nesnenin özelliklerine erişebilir."
  },
  {
    title: "Self Parametresi",
    description: "Self parametresinin nasıl kullanıldığını gösterir.",
    code: `class Nokta:
    def __init__(self, x, y):
        self.x = x  # self.x nesnenin x koordinatını temsil eder
        self.y = y  # self.y nesnenin y koordinatını temsil eder

    def kordinatlari_goster(self):
        return f"X: {self.x}, Y: {self.y}"

    def nokta_tasi(self, delta_x, delta_y):
        self.x += delta_x  # self ile nesnenin x değerini değiştiriyoruz
        self.y += delta_y  # self ile nesnenin y değerini değiştiriyoruz
        return self.kordinatlari_goster()

# Kullanım örneği
nokta1 = Nokta(5, 10)
print(nokta1.kordinatlari_goster())  # X: 5, Y: 10
print(nokta1.nokta_tasi(3, -2))      # X: 8, Y: 8
`,
    explanation: "Nokta sınıfı, x ve y koordinatlarını içeren bir sınıftır. kordinatlari_goster ve nokta_tasi metodları, self parametresi kullanılarak nesnenin durumunu değiştirir ve nesnenin özelliklerine erişir."
  }
];

export default function ClassesAndObjects() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <Link 
            href="/topics/python/nesneye-yonelik-programlama" 
            className="inline-flex items-center text-primary hover:underline mb-4"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Nesneye Yönelik Programlama'ya Dön
          </Link>
        </div>

        <MarkdownContent content={content} />

        <section className="my-12">
          <h2 className="text-3xl font-bold mb-8">Alıştırmalar ve Örnekler</h2>
          <div className="grid gap-8">
            {examples.map((example, index) => (
              <Card key={index} className="overflow-hidden">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Code2 className="h-5 w-5 text-primary" />
                    {example.title}
                  </CardTitle>
                  <CardDescription>{example.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="mb-4">
                    <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                      <code className="text-sm">{example.code}</code>
                    </pre>
                  </div>
                  <p className="text-muted-foreground">{example.explanation}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        <section className="my-12">
          <Quiz />
        </section>

        <div className="mt-8 flex justify-between">
          <Button asChild variant="outline">
            <Link href="/topics/python/nesneye-yonelik-programlama">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Önceki: Giriş
            </Link>
          </Button>
          <Button asChild>
            <Link href="/topics/python/nesneye-yonelik-programlama/kalitim">
              Sonraki: Kalıtım
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 