import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowRight, Library, Building2, GraduationCap, Car } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python OOP Temel Örnekler | Kodleon',
  description: 'Python nesne tabanlı programlamanın temel kavramlarını içeren detaylı örnekler ve açıklamalar.',
};

const content = `
# Python OOP Temel Örnekler

Bu bölümde, nesne tabanlı programlamanın temel kavramlarını içeren detaylı örnekleri inceleyeceğiz. Her örnek, kullanılan OOP kavramlarının açıklamalarını ve kodun nasıl çalıştığını adım adım göstermektedir.

## 1. Kütüphane Yönetim Sistemi

Bu örnek, sınıflar, kalıtım ve kapsülleme kavramlarını kullanarak basit bir kütüphane yönetim sistemi oluşturur:

\`\`\`python
from datetime import datetime, timedelta
from typing import List, Optional

class Kitap:
    def __init__(self, isbn: str, baslik: str, yazar: str):
        self._isbn = isbn          # Protected attribute
        self.baslik = baslik
        self.yazar = yazar
        self.odunc_durumu = False  # Public attribute
    
    @property
    def isbn(self) -> str:         # Getter method
        return self._isbn
    
    def odunc_al(self) -> bool:
        if not self.odunc_durumu:
            self.odunc_durumu = True
            return True
        return False
    
    def iade_et(self) -> bool:
        if self.odunc_durumu:
            self.odunc_durumu = False
            return True
        return False
    
    def __str__(self) -> str:
        return f"{self.baslik} - {self.yazar} (ISBN: {self.isbn})"

class Uye:
    def __init__(self, id: int, ad: str, soyad: str):
        self.id = id
        self.ad = ad
        self.soyad = soyad
        self.odunc_kitaplar: List[Kitap] = []
    
    def kitap_odunc_al(self, kitap: Kitap) -> bool:
        if len(self.odunc_kitaplar) < 3 and kitap.odunc_al():
            self.odunc_kitaplar.append(kitap)
            return True
        return False
    
    def kitap_iade_et(self, kitap: Kitap) -> bool:
        if kitap in self.odunc_kitaplar and kitap.iade_et():
            self.odunc_kitaplar.remove(kitap)
            return True
        return False
    
    def __str__(self) -> str:
        return f"{self.ad} {self.soyad} (ID: {self.id})"

class Kutuphane:
    def __init__(self, ad: str):
        self.ad = ad
        self._kitaplar: List[Kitap] = []      # Protected attribute
        self._uyeler: List[Uye] = []          # Protected attribute
    
    def kitap_ekle(self, kitap: Kitap) -> None:
        self._kitaplar.append(kitap)
    
    def uye_ekle(self, uye: Uye) -> None:
        self._uyeler.append(uye)
    
    def kitap_ara(self, isbn: str) -> Optional[Kitap]:
        return next((k for k in self._kitaplar if k.isbn == isbn), None)
    
    def uye_ara(self, id: int) -> Optional[Uye]:
        return next((u for u in self._uyeler if u.id == id), None)
    
    def kitaplari_listele(self) -> None:
        print(f"\\n{self.ad} Kitap Listesi:")
        for kitap in self._kitaplar:
            durum = "Ödünç Verildi" if kitap.odunc_durumu else "Mevcut"
            print(f"- {kitap} [{durum}]")

# Kullanım örneği
def main():
    # Kütüphane oluştur
    kutuphane = Kutuphane("Kodleon Kütüphanesi")
    
    # Kitaplar ekle
    kitap1 = Kitap("123", "Python Programlama", "Ahmet Yılmaz")
    kitap2 = Kitap("456", "Veri Yapıları", "Mehmet Demir")
    kutuphane.kitap_ekle(kitap1)
    kutuphane.kitap_ekle(kitap2)
    
    # Üye ekle
    uye1 = Uye(1, "Ali", "Kaya")
    kutuphane.uye_ekle(uye1)
    
    # Kitap ödünç alma işlemi
    if uye1.kitap_odunc_al(kitap1):
        print(f"{uye1.ad} {uye1.soyad} kitabı ödünç aldı: {kitap1.baslik}")
    
    # Kitapları listele
    kutuphane.kitaplari_listele()
    
    # Kitap iade işlemi
    if uye1.kitap_iade_et(kitap1):
        print(f"{uye1.ad} {uye1.soyad} kitabı iade etti: {kitap1.baslik}")
    
    # Güncel durumu göster
    kutuphane.kitaplari_listele()

if __name__ == "__main__":
    main()
\`\`\`

### Kullanılan OOP Kavramları:

1. **Sınıflar ve Nesneler**
   - \`Kitap\`, \`Uye\` ve \`Kutuphane\` sınıfları
   - Her sınıfın kendi özellikleri ve metodları

2. **Kapsülleme**
   - Protected özellikler (\`_isbn\`, \`_kitaplar\`, \`_uyeler\`)
   - Getter metodlar (\`@property\`)
   - Public ve private metod ayrımı

3. **Tip Kontrolü**
   - Type hints kullanımı (\`List[Kitap]\`, \`Optional[Uye]\`)
   - Return type annotations

4. **Nesne İlişkileri**
   - Composition (Kutuphane sınıfı Kitap ve Uye nesnelerini içerir)
   - One-to-many ilişkiler

## 2. Banka Hesap Sistemi

Bu örnek, kalıtım ve polimorfizm kavramlarını kullanarak bir banka hesap sistemi oluşturur:

\`\`\`python
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

class Hesap(ABC):
    def __init__(self, hesap_no: str, sahip: str, bakiye: float = 0):
        self._hesap_no = hesap_no    # Protected
        self._sahip = sahip          # Protected
        self._bakiye = bakiye        # Protected
        self._islemler: List[str] = []
    
    @property
    def hesap_no(self) -> str:
        return self._hesap_no
    
    @property
    def bakiye(self) -> float:
        return self._bakiye
    
    def islem_ekle(self, islem: str) -> None:
        tarih = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._islemler.append(f"{tarih} - {islem}")
    
    def islemleri_goster(self) -> None:
        print(f"\\nHesap No: {self._hesap_no}")
        print(f"Sahip: {self._sahip}")
        print("İşlemler:")
        for islem in self._islemler:
            print(f"- {islem}")
    
    @abstractmethod
    def para_yatir(self, miktar: float) -> bool:
        pass
    
    @abstractmethod
    def para_cek(self, miktar: float) -> bool:
        pass

class VadesizHesap(Hesap):
    def para_yatir(self, miktar: float) -> bool:
        if miktar > 0:
            self._bakiye += miktar
            self.islem_ekle(f"Para yatırma: +{miktar}TL")
            return True
        return False
    
    def para_cek(self, miktar: float) -> bool:
        if 0 < miktar <= self._bakiye:
            self._bakiye -= miktar
            self.islem_ekle(f"Para çekme: -{miktar}TL")
            return True
        return False

class VadeliHesap(Hesap):
    def __init__(self, hesap_no: str, sahip: str, vade_suresi: int, bakiye: float = 0):
        super().__init__(hesap_no, sahip, bakiye)
        self.vade_suresi = vade_suresi  # Gün cinsinden
    
    def para_yatir(self, miktar: float) -> bool:
        if miktar >= 1000:  # Minimum vade miktarı
            self._bakiye += miktar
            self.islem_ekle(f"Vadeli para yatırma: +{miktar}TL")
            return True
        return False
    
    def para_cek(self, miktar: float) -> bool:
        # Vadeli hesaptan para çekilemez
        return False

class Banka:
    def __init__(self, ad: str):
        self.ad = ad
        self._hesaplar: List[Hesap] = []
    
    def hesap_ekle(self, hesap: Hesap) -> None:
        self._hesaplar.append(hesap)
    
    def hesap_bul(self, hesap_no: str) -> Optional[Hesap]:
        return next((h for h in self._hesaplar if h.hesap_no == hesap_no), None)
    
    def hesaplari_listele(self) -> None:
        print(f"\\n{self.ad} Hesap Listesi:")
        for hesap in self._hesaplar:
            print(f"- Hesap No: {hesap.hesap_no}, Bakiye: {hesap.bakiye}TL")

# Kullanım örneği
def main():
    # Banka oluştur
    banka = Banka("Kodleon Bank")
    
    # Hesaplar oluştur
    vadesiz = VadesizHesap("V123", "Ali Yılmaz")
    vadeli = VadeliHesap("V456", "Ayşe Demir", 30)  # 30 günlük vade
    
    # Hesapları bankaya ekle
    banka.hesap_ekle(vadesiz)
    banka.hesap_ekle(vadeli)
    
    # İşlemler yap
    vadesiz.para_yatir(1000)
    vadesiz.para_cek(500)
    vadeli.para_yatir(5000)
    
    # Hesap durumlarını göster
    banka.hesaplari_listele()
    vadesiz.islemleri_goster()
    vadeli.islemleri_goster()

if __name__ == "__main__":
    main()
\`\`\`

### Kullanılan OOP Kavramları:

1. **Soyut Sınıflar**
   - \`ABC\` ve \`@abstractmethod\` kullanımı
   - Soyut \`Hesap\` sınıfı

2. **Kalıtım**
   - \`VadesizHesap\` ve \`VadeliHesap\` sınıfları \`Hesap\` sınıfından türetilmiş
   - \`super().__init__()\` kullanımı

3. **Polimorfizm**
   - Farklı hesap türleri için farklı \`para_yatir\` ve \`para_cek\` davranışları
   - Aynı arayüz, farklı implementasyonlar

4. **Kapsülleme**
   - Protected özellikler
   - Getter metodlar
   - İşlem logları

## 3. Öğrenci Bilgi Sistemi

[Devamı için tıklayın →](/topics/python/nesne-tabanli-programlama/pratik-ornekler/temel-ornekler/ogrenci-sistemi)

## 4. Araç Kiralama Sistemi

[Devamı için tıklayın →](/topics/python/nesne-tabanli-programlama/pratik-ornekler/temel-ornekler/arac-kiralama)
`;

const sections = [
  {
    title: "Kütüphane Sistemi",
    description: "Temel sınıf ve nesne kavramları",
    icon: <Library className="h-6 w-6" />,
    topics: [
      "Sınıf yapısı",
      "Nesne oluşturma",
      "Metod tanımlama",
      "Özellik yönetimi"
    ]
  },
  {
    title: "Banka Sistemi",
    description: "Kalıtım ve polimorfizm örneği",
    icon: <Building2 className="h-6 w-6" />,
    topics: [
      "Soyut sınıflar",
      "Metod override",
      "Hesap yönetimi",
      "İşlem takibi"
    ]
  },
  {
    title: "Öğrenci Sistemi",
    description: "Kompozisyon ve ilişkiler",
    icon: <GraduationCap className="h-6 w-6" />,
    link: "/topics/python/nesne-tabanli-programlama/pratik-ornekler/temel-ornekler/ogrenci-sistemi",
    topics: [
      "Öğrenci kayıtları",
      "Ders yönetimi",
      "Not sistemi",
      "Raporlama"
    ]
  },
  {
    title: "Araç Kiralama",
    description: "Kompleks nesne ilişkileri",
    icon: <Car className="h-6 w-6" />,
    link: "/topics/python/nesne-tabanli-programlama/pratik-ornekler/temel-ornekler/arac-kiralama",
    topics: [
      "Araç envanteri",
      "Kiralama işlemleri",
      "Müşteri takibi",
      "Fatura sistemi"
    ]
  }
];

export default function TemelOrneklerPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Example Cards */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Örnek Projelerin Özeti</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {sections.map((section, index) => (
              <Card key={index} className="bg-green-50 hover:bg-green-100 dark:bg-green-950/50 dark:hover:bg-green-950/70 transition-all duration-300">
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg text-green-600 dark:text-green-400">
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
                {section.link && (
                  <CardFooter>
                    <Button asChild variant="outline" className="w-full group">
                      <Link href={section.link}>
                        Detaylı İncele
                        <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
                      </Link>
                    </Button>
                  </CardFooter>
                )}
              </Card>
            ))}
          </div>
        </div>

        {/* Back to Main Examples Page Button */}
        <div className="mt-12 flex justify-end">
          <Button asChild variant="outline" className="group">
            <Link href="/topics/python/nesne-tabanli-programlama/pratik-ornekler">
              Tüm Örneklere Dön
              <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 