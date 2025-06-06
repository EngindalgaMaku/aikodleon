import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowRight, GraduationCap, Book, Calculator, FileText } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python OOP Öğrenci Bilgi Sistemi Örneği | AIKOD',
  description: 'Python nesne tabanlı programlama ile öğrenci bilgi sistemi uygulaması örneği ve detaylı açıklamalar.',
};

const content = `
# Öğrenci Bilgi Sistemi Örneği

Bu örnek, nesne tabanlı programlama kavramlarını kullanarak kapsamlı bir öğrenci bilgi sistemi oluşturur. Sistem, öğrenci kayıtları, ders yönetimi, not takibi ve raporlama gibi temel özellikleri içerir.

## Sistem Bileşenleri

### 1. Öğrenci Sınıfı
- Öğrenci bilgilerini tutar
- Ders kayıt işlemlerini yönetir
- Not görüntüleme ve ortalama hesaplama

### 2. Ders Sınıfı
- Ders bilgilerini tutar
- Öğrenci listesi yönetimi
- Not girişi ve güncelleme

### 3. Öğretmen Sınıfı
- Öğretmen bilgilerini tutar
- Verdiği derslerin yönetimi
- Not girme yetkisi

### 4. Fakülte Sınıfı
- Bölüm ve ders yönetimi
- Öğrenci ve öğretmen kayıtları
- Raporlama işlemleri

## Kod Örneği

\`\`\`python
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from statistics import mean

class NotTuru(Enum):
    VIZE = "Vize"
    FINAL = "Final"
    ODEV = "Ödev"
    PROJE = "Proje"

@dataclass
class Not:
    not_turu: NotTuru
    puan: float
    tarih: datetime
    aciklama: str = ""

class Ders:
    def __init__(self, kod: str, ad: str, kredi: int):
        self.kod = kod
        self.ad = ad
        self.kredi = kredi
        self._ogrenciler: Dict[str, List[Not]] = {}  # {ogrenci_no: [notlar]}
        self.ogretmen = None
    
    def ogrenci_ekle(self, ogrenci) -> bool:
        if ogrenci.ogrenci_no not in self._ogrenciler:
            self._ogrenciler[ogrenci.ogrenci_no] = []
            return True
        return False
    
    def not_ekle(self, ogrenci_no: str, not_: Not) -> bool:
        if ogrenci_no in self._ogrenciler:
            self._ogrenciler[ogrenci_no].append(not_)
            return True
        return False
    
    def not_ortalamasi(self, ogrenci_no: str) -> Optional[float]:
        if ogrenci_no in self._ogrenciler:
            notlar = self._ogrenciler[ogrenci_no]
            if notlar:
                return mean([n.puan for n in notlar])
        return None
    
    def __str__(self) -> str:
        return f"{self.kod} - {self.ad} ({self.kredi} kredi)"

class Ogrenci:
    def __init__(self, ogrenci_no: str, ad: str, soyad: str, bolum: str):
        self.ogrenci_no = ogrenci_no
        self.ad = ad
        self.soyad = soyad
        self.bolum = bolum
        self._dersler: List[Ders] = []
    
    def ders_ekle(self, ders: Ders) -> bool:
        if ders not in self._dersler and ders.ogrenci_ekle(self):
            self._dersler.append(ders)
            return True
        return False
    
    def not_goruntule(self, ders: Ders) -> List[Not]:
        if ders in self._dersler:
            return ders._ogrenciler.get(self.ogrenci_no, [])
        return []
    
    def genel_ortalama(self) -> float:
        notlar = []
        for ders in self._dersler:
            ort = ders.not_ortalamasi(self.ogrenci_no)
            if ort is not None:
                notlar.append(ort)
        return mean(notlar) if notlar else 0.0
    
    def transkript_goruntule(self) -> str:
        result = f"\\nTranskript - {self.ad} {self.soyad} ({self.ogrenci_no})\\n"
        result += "=" * 50 + "\\n"
        
        for ders in self._dersler:
            result += f"\\n{ders}\\n"
            for not_ in self.not_goruntule(ders):
                result += f"  {not_.not_turu.value}: {not_.puan}\\n"
            ort = ders.not_ortalamasi(self.ogrenci_no)
            if ort is not None:
                result += f"  Ortalama: {ort:.2f}\\n"
        
        result += "\\n" + "=" * 50
        result += f"\\nGenel Ortalama: {self.genel_ortalama():.2f}"
        return result
    
    def __str__(self) -> str:
        return f"{self.ad} {self.soyad} ({self.ogrenci_no})"

class Ogretmen:
    def __init__(self, sicil_no: str, ad: str, soyad: str, unvan: str):
        self.sicil_no = sicil_no
        self.ad = ad
        self.soyad = soyad
        self.unvan = unvan
        self._dersler: List[Ders] = []
    
    def ders_ekle(self, ders: Ders) -> bool:
        if ders not in self._dersler:
            self._dersler.append(ders)
            ders.ogretmen = self
            return True
        return False
    
    def not_gir(self, ders: Ders, ogrenci: Ogrenci, not_: Not) -> bool:
        if ders in self._dersler:
            return ders.not_ekle(ogrenci.ogrenci_no, not_)
        return False
    
    def sinif_listesi(self, ders: Ders) -> str:
        if ders in self._dersler:
            result = f"\\n{ders} - Sınıf Listesi\\n"
            result += "=" * 50 + "\\n"
            for ogrenci_no in ders._ogrenciler:
                ort = ders.not_ortalamasi(ogrenci_no)
                result += f"{ogrenci_no}: {ort:.2f if ort else 'Not girilmemiş'}\\n"
            return result
        return "Bu ders size ait değil."
    
    def __str__(self) -> str:
        return f"{self.unvan} {self.ad} {self.soyad}"

class Fakulte:
    def __init__(self, ad: str):
        self.ad = ad
        self._dersler: List[Ders] = []
        self._ogrenciler: List[Ogrenci] = []
        self._ogretmenler: List[Ogretmen] = []
    
    def ders_ekle(self, ders: Ders) -> bool:
        if ders not in self._dersler:
            self._dersler.append(ders)
            return True
        return False
    
    def ogrenci_ekle(self, ogrenci: Ogrenci) -> bool:
        if ogrenci not in self._ogrenciler:
            self._ogrenciler.append(ogrenci)
            return True
        return False
    
    def ogretmen_ekle(self, ogretmen: Ogretmen) -> bool:
        if ogretmen not in self._ogretmenler:
            self._ogretmenler.append(ogretmen)
            return True
        return False
    
    def bolum_raporu(self, bolum: str) -> str:
        result = f"\\n{self.ad} - {bolum} Bölümü Raporu\\n"
        result += "=" * 50 + "\\n\\n"
        
        bolum_ogrencileri = [o for o in self._ogrenciler if o.bolum == bolum]
        result += f"Toplam Öğrenci Sayısı: {len(bolum_ogrencileri)}\\n"
        
        if bolum_ogrencileri:
            ort = mean([o.genel_ortalama() for o in bolum_ogrencileri])
            result += f"Bölüm Genel Ortalaması: {ort:.2f}\\n\\n"
            
            result += "Öğrenci Listesi:\\n"
            for ogrenci in sorted(bolum_ogrencileri, key=lambda x: x.genel_ortalama(), reverse=True):
                result += f"{ogrenci}: {ogrenci.genel_ortalama():.2f}\\n"
        
        return result

# Kullanım örneği
def main():
    # Fakülte oluştur
    fakulte = Fakulte("Mühendislik Fakültesi")
    
    # Dersler oluştur
    python = Ders("CSE101", "Python Programlama", 4)
    veri_yapilari = Ders("CSE102", "Veri Yapıları", 4)
    fakulte.ders_ekle(python)
    fakulte.ders_ekle(veri_yapilari)
    
    # Öğretmen oluştur
    ogretmen = Ogretmen("T123", "Ahmet", "Yılmaz", "Dr.")
    fakulte.ogretmen_ekle(ogretmen)
    ogretmen.ders_ekle(python)
    ogretmen.ders_ekle(veri_yapilari)
    
    # Öğrenciler oluştur
    ogrenci1 = Ogrenci("S101", "Ali", "Kaya", "Bilgisayar Müh.")
    ogrenci2 = Ogrenci("S102", "Ayşe", "Demir", "Bilgisayar Müh.")
    fakulte.ogrenci_ekle(ogrenci1)
    fakulte.ogrenci_ekle(ogrenci2)
    
    # Ders kayıtları
    ogrenci1.ders_ekle(python)
    ogrenci1.ders_ekle(veri_yapilari)
    ogrenci2.ders_ekle(python)
    
    # Not girişleri
    simdi = datetime.now()
    ogretmen.not_gir(python, ogrenci1, Not(NotTuru.VIZE, 85, simdi))
    ogretmen.not_gir(python, ogrenci1, Not(NotTuru.FINAL, 90, simdi))
    ogretmen.not_gir(python, ogrenci2, Not(NotTuru.VIZE, 75, simdi))
    ogretmen.not_gir(veri_yapilari, ogrenci1, Not(NotTuru.VIZE, 70, simdi))
    
    # Raporlar
    print(ogrenci1.transkript_goruntule())
    print(ogretmen.sinif_listesi(python))
    print(fakulte.bolum_raporu("Bilgisayar Müh."))

if __name__ == "__main__":
    main()
\`\`\`

## Kullanılan OOP Kavramları

1. **Sınıf ve Nesne Yapısı**
   - Öğrenci, Ders, Öğretmen ve Fakülte sınıfları
   - Her sınıfın kendi özellikleri ve metodları
   - Dataclass kullanımı (Not sınıfı)

2. **Kapsülleme**
   - Protected özellikler (\`_dersler\`, \`_ogrenciler\`, vb.)
   - Getter ve setter metodlar
   - Veri doğrulama ve kontrol

3. **Tip Kontrolü ve Enum**
   - Type hints kullanımı
   - NotTuru için Enum sınıfı
   - Optional ve List tipleri

4. **Nesne İlişkileri**
   - Composition (Fakülte -> Dersler, Öğrenciler, Öğretmenler)
   - Many-to-many ilişkiler (Öğrenci-Ders)
   - One-to-many ilişkiler (Öğretmen-Ders)

5. **Veri Yapıları**
   - Dictionary kullanımı (not takibi)
   - List kullanımı (kayıt listeleri)
   - String formatlama

## Geliştirme Önerileri

1. Veritabanı entegrasyonu eklenebilir
2. Web arayüzü oluşturulabilir
3. Dosya işlemleri eklenebilir (Excel export, PDF transkript)
4. Ders programı yönetimi eklenebilir
5. Devamsızlık takibi eklenebilir
`;

const sections = [
  {
    title: "Sınıf Yapısı",
    description: "Temel sınıflar ve ilişkileri",
    icon: <GraduationCap className="h-6 w-6" />,
    topics: [
      "Öğrenci sınıfı",
      "Ders sınıfı",
      "Öğretmen sınıfı",
      "Fakülte sınıfı"
    ]
  },
  {
    title: "Ders Yönetimi",
    description: "Ders ve not işlemleri",
    icon: <Book className="h-6 w-6" />,
    topics: [
      "Ders kayıtları",
      "Not girişi",
      "Devam durumu",
      "Ders programı"
    ]
  },
  {
    title: "Not Sistemi",
    description: "Not hesaplama ve raporlama",
    icon: <Calculator className="h-6 w-6" />,
    topics: [
      "Not türleri",
      "Ortalama hesaplama",
      "Not görüntüleme",
      "İstatistikler"
    ]
  },
  {
    title: "Raporlama",
    description: "Sistem raporları ve analizler",
    icon: <FileText className="h-6 w-6" />,
    topics: [
      "Transkript",
      "Sınıf listesi",
      "Bölüm raporu",
      "Başarı analizi"
    ]
  }
];

export default function OgrenciSistemiPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Feature Cards */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Sistem Özellikleri</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {sections.map((section, index) => (
              <Card key={index} className="bg-yellow-50 hover:bg-yellow-100 dark:bg-yellow-950/50 dark:hover:bg-yellow-950/70 transition-all duration-300">
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg text-yellow-600 dark:text-yellow-400">
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

        {/* Back to Basic Examples Button */}
        <div className="mt-12 flex justify-end">
          <Button asChild variant="outline" className="group">
            <Link href="/topics/python/nesne-tabanli-programlama/pratik-ornekler/temel-ornekler">
              Temel Örneklere Dön
              <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 