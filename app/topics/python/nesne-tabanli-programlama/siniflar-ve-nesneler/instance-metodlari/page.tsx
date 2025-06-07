import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowRight, Code2, Settings, Terminal, BookOpen } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python Instance Metodları | AIKOD',
  description: 'Python\'da instance metodları, self parametresi, metod tanımlama ve çağırma teknikleri.',
};

const content = `
# Python'da Instance Metodları

Instance metodları, bir sınıfın nesneleri üzerinde çalışan ve her nesne için özel davranış gösteren metodlardır. Bu metodlar, nesnenin durumunu değiştirebilir veya nesneyle ilgili bilgileri döndürebilir.

## Instance Metod Tanımlama

Instance metodları, sınıf içinde tanımlanan ve ilk parametre olarak \`self\` alan metodlardır:

\`\`\`python
class Ogrenci:
    def __init__(self, ad, numara):
        self.ad = ad
        self.numara = numara
        self.notlar = []
    
    def not_ekle(self, not_degeri):  # Instance metod
        if 0 <= not_degeri <= 100:
            self.notlar.append(not_degeri)
            return True
        return False
    
    def ortalama_hesapla(self):  # Instance metod
        if self.notlar:
            return sum(self.notlar) / len(self.notlar)
        return 0.0

# Metodları kullanma
ogrenci = Ogrenci("Ali", "101")
ogrenci.not_ekle(85)
ogrenci.not_ekle(90)
print(ogrenci.ortalama_hesapla())  # 87.5
\`\`\`

## Self Parametresi

\`self\` parametresi, metodun çağrıldığı nesneyi temsil eder:

\`\`\`python
class BankaHesabi:
    def __init__(self, hesap_no, bakiye=0):
        self.hesap_no = hesap_no
        self.bakiye = bakiye
        self.islemler = []
    
    def para_yatir(self, miktar):
        self.bakiye += miktar
        self.islem_kaydet("Para Yatırma", miktar)
        return f"{miktar} TL yatırıldı. Yeni bakiye: {self.bakiye} TL"
    
    def para_cek(self, miktar):
        if self.bakiye >= miktar:
            self.bakiye -= miktar
            self.islem_kaydet("Para Çekme", miktar)
            return f"{miktar} TL çekildi. Yeni bakiye: {self.bakiye} TL"
        return "Yetersiz bakiye!"
    
    def islem_kaydet(self, tur, miktar):
        from datetime import datetime
        tarih = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.islemler.append(f"{tarih} - {tur}: {miktar} TL")
    
    def hesap_ozeti(self):
        return f"Hesap No: {self.hesap_no}\\nBakiye: {self.bakiye} TL\\n\\nSon İşlemler:\\n" + "\\n".join(self.islemler[-5:])

# Self parametresinin kullanımı
hesap = BankaHesabi("12345")
print(hesap.para_yatir(1000))
print(hesap.para_cek(500))
print(hesap.hesap_ozeti())
\`\`\`

## Metod Çağırma Teknikleri

Instance metodları iki şekilde çağrılabilir:

### 1. Nesne Üzerinden Çağırma (Önerilen)

\`\`\`python
class Dikdortgen:
    def __init__(self, en, boy):
        self.en = en
        self.boy = boy
    
    def alan_hesapla(self):
        return self.en * self.boy
    
    def cevre_hesapla(self):
        return 2 * (self.en + self.boy)

# Nesne üzerinden çağırma
d1 = Dikdortgen(5, 3)
print(d1.alan_hesapla())    # 15
print(d1.cevre_hesapla())   # 16
\`\`\`

### 2. Sınıf Üzerinden Çağırma

\`\`\`python
# Sınıf üzerinden çağırma (önerilmez)
d2 = Dikdortgen(4, 6)
print(Dikdortgen.alan_hesapla(d2))   # 24
print(Dikdortgen.cevre_hesapla(d2))  # 20
\`\`\`

## Zincirleme Metod Çağrıları

Metodları zincirleme şekilde çağırmak için metod sonunda \`self\` döndürün:

\`\`\`python
class Hesaplayici:
    def __init__(self):
        self.deger = 0
    
    def topla(self, sayi):
        self.deger += sayi
        return self  # self döndürerek zincirleme çağrı sağlanır
    
    def carp(self, sayi):
        self.deger *= sayi
        return self
    
    def cikar(self, sayi):
        self.deger -= sayi
        return self
    
    def sonuc(self):
        return self.deger

# Zincirleme metod çağrıları
hesap = Hesaplayici()
sonuc = hesap.topla(5).carp(2).cikar(3).sonuc()
print(sonuc)  # 7
\`\`\`

## Özel Metodlar (Magic Methods)

Python'da bazı özel isimli instance metodları vardır:

\`\`\`python
class Nokta:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"Nokta({self.x}, {self.y})"
    
    def __add__(self, other):
        return Nokta(self.x + other.x, self.y + other.y)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __len__(self):
        from math import sqrt
        return int(sqrt(self.x**2 + self.y**2))

# Özel metodların kullanımı
n1 = Nokta(3, 4)
n2 = Nokta(1, 2)

print(str(n1))          # Nokta(3, 4)
print(n1 + n2)         # Nokta(4, 6)
print(n1 == n2)        # False
print(len(n1))         # 5
\`\`\`

## Return Değerleri

Instance metodları farklı türlerde değerler döndürebilir:

\`\`\`python
class Urun:
    def __init__(self, ad, fiyat, stok):
        self.ad = ad
        self.fiyat = fiyat
        self.stok = stok
    
    def bilgi_getir(self) -> dict:
        return {
            "ad": self.ad,
            "fiyat": self.fiyat,
            "stok": self.stok
        }
    
    def stok_guncelle(self, miktar) -> bool:
        if self.stok + miktar >= 0:
            self.stok += miktar
            return True
        return False
    
    def fiyat_hesapla(self, adet) -> float:
        if adet <= self.stok:
            return self.fiyat * adet
        raise ValueError("Yetersiz stok!")
    
    def durum_mesaji(self) -> str:
        return "Stokta var" if self.stok > 0 else "Tükendi"

# Farklı return değerlerinin kullanımı
urun = Urun("Laptop", 15000, 5)
print(urun.bilgi_getir())      # {'ad': 'Laptop', 'fiyat': 15000, 'stok': 5}
print(urun.stok_guncelle(-2))  # True
print(urun.fiyat_hesapla(2))   # 30000.0
print(urun.durum_mesaji())     # Stokta var
\`\`\`

## İyi Pratikler

1. **Metod İsimlendirme**:
   - Eylem bildiren fiiller kullanın (örn: \`hesapla\`, \`ekle\`, \`sil\`)
   - snake_case kullanın (örn: \`not_ekle\`, \`bakiye_guncelle\`)
   - Anlamlı ve açıklayıcı isimler seçin

2. **Dokümantasyon**:
\`\`\`python
class Hesap:
    def para_transfer(self, hedef_hesap, miktar):
        """
        Başka bir hesaba para transferi yapar.
        
        Args:
            hedef_hesap (Hesap): Para transferi yapılacak hesap
            miktar (float): Transfer edilecek miktar
        
        Returns:
            bool: Transfer başarılı ise True, değilse False
        
        Raises:
            ValueError: Miktar negatif ise veya yetersiz bakiye varsa
        """
        if miktar <= 0:
            raise ValueError("Transfer miktarı pozitif olmalıdır")
        
        if self.bakiye >= miktar:
            self.bakiye -= miktar
            hedef_hesap.bakiye += miktar
            return True
        return False
\`\`\`

3. **Tek Sorumluluk İlkesi**:
   - Her metod tek bir işi yapmalı
   - Karmaşık işlemleri daha küçük metodlara bölün
   - Metodlar birbirini tamamlayıcı olmalı

## Yaygın Hatalar ve Çözümleri

1. **Self Parametresini Unutmak**:
\`\`\`python
# YANLIŞ
class Oyuncu:
    def puan_ekle(puan):  # self eksik!
        self.puan += puan

# DOĞRU
class Oyuncu:
    def puan_ekle(self, puan):
        self.puan += puan
\`\`\`

2. **Instance Değişkenlerine Erişim Hatası**:
\`\`\`python
# YANLIŞ
class Araba:
    def hiz_artir(self):
        hiz += 10  # hiz tanımlı değil!

# DOĞRU
class Araba:
    def __init__(self):
        self.hiz = 0
    
    def hiz_artir(self):
        self.hiz += 10
\`\`\`

3. **Gereksiz Self Kullanımı**:
\`\`\`python
# YANLIŞ
class Matematik:
    def kare_al(self, sayi):
        return self.sayi * self.sayi  # self gereksiz!

# DOĞRU
class Matematik:
    def kare_al(self, sayi):
        return sayi * sayi
\`\`\`
`;

const sections = [
  {
    title: "Metod Tanımlama",
    description: "Instance metod yapısı ve kullanımı",
    icon: <Code2 className="h-6 w-6" />,
    topics: [
      "Metod syntax",
      "Self parametresi",
      "Parametre kullanımı",
      "Return değerleri"
    ]
  },
  {
    title: "Çağırma Teknikleri",
    description: "Metodları çağırma yöntemleri",
    icon: <Terminal className="h-6 w-6" />,
    topics: [
      "Nesne üzerinden çağırma",
      "Sınıf üzerinden çağırma",
      "Zincirleme çağrılar",
      "Özel metodlar"
    ]
  },
  {
    title: "Veri İşleme",
    description: "Metod içinde veri yönetimi",
    icon: <Settings className="h-6 w-6" />,
    topics: [
      "Instance değişkenleri",
      "Parametre işleme",
      "Veri dönüşümleri",
      "Hata yönetimi"
    ]
  },
  {
    title: "İyi Pratikler",
    description: "Metod tasarım prensipleri",
    icon: <BookOpen className="h-6 w-6" />,
    topics: [
      "İsimlendirme kuralları",
      "Dokümantasyon",
      "Tek sorumluluk",
      "Kod organizasyonu"
    ]
  }
];

export default function InstanceMetodlariPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Concept Cards */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Önemli Kavramlar</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {sections.map((section, index) => (
              <Card key={index} className="bg-blue-50 hover:bg-blue-100 dark:bg-blue-950/50 dark:hover:bg-blue-950/70 transition-all duration-300">
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg text-blue-600 dark:text-blue-400">
                      {section.icon}
                    </div>
                    <CardTitle>{section.title}</CardTitle>
                  </div>
                  <CardDescription className="dark:text-gray-300">{section.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground dark:text-gray-400">
                    {section.topics.map((topic, i) => (
                      <li key={i}>{topic}</li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Back Button */}
        <div className="mt-12 flex justify-end">
          <Button asChild variant="outline" className="group">
            <Link href="/topics/python/nesne-tabanli-programlama/siniflar-ve-nesneler">
              Sınıflar ve Nesneler Sayfasına Dön
              <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 