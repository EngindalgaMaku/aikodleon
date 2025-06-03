import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python OOP: Soyut Sınıflar ve Arayüzler | Kodleon',
  description: 'Python\'da soyut sınıfları, arayüzleri ve protokolleri öğrenin.',
};

const content = `
# Python'da Soyut Sınıflar ve Arayüzler

Soyut sınıflar ve arayüzler, sınıflar arasında sözleşmeler tanımlamamızı ve kodun daha düzenli olmasını sağlar.

## Soyut Sınıflar (Abstract Base Classes)

Python'da soyut sınıflar \`abc\` modülü ile oluşturulur:

\`\`\`python
from abc import ABC, abstractmethod

class Sekil(ABC):
    @abstractmethod
    def alan(self):
        pass
    
    @abstractmethod
    def cevre(self):
        pass
    
    def bilgi(self):  # Normal metod
        return f"Alan: {self.alan()}, Çevre: {self.cevre()}"

class Dikdortgen(Sekil):
    def __init__(self, genislik, yukseklik):
        self.genislik = genislik
        self.yukseklik = yukseklik
    
    def alan(self):
        return self.genislik * self.yukseklik
    
    def cevre(self):
        return 2 * (self.genislik + self.yukseklik)

# Kullanım
# sekil = Sekil()  # TypeError: Can't instantiate abstract class
d = Dikdortgen(5, 3)
print(d.bilgi())  # Alan: 15, Çevre: 16
\`\`\`

## Arayüzler (Interfaces)

Python'da resmi bir arayüz kavramı yoktur, ancak soyut sınıflar ile benzer işlevsellik sağlanabilir:

\`\`\`python
from abc import ABC, abstractmethod

class OdemeArayuzu(ABC):
    @abstractmethod
    def ode(self, miktar):
        pass
    
    @abstractmethod
    def bakiye_sorgula(self):
        pass

class KrediKarti(OdemeArayuzu):
    def __init__(self, bakiye):
        self.bakiye = bakiye
    
    def ode(self, miktar):
        if miktar <= self.bakiye:
            self.bakiye -= miktar
            return True
        return False
    
    def bakiye_sorgula(self):
        return self.bakiye

class Havale(OdemeArayuzu):
    def __init__(self, bakiye):
        self.bakiye = bakiye
    
    def ode(self, miktar):
        if miktar <= self.bakiye:
            self.bakiye -= miktar
            return True
        return False
    
    def bakiye_sorgula(self):
        return self.bakiye

# Kullanım
def odeme_yap(odeme_yontemi: OdemeArayuzu, miktar):
    if odeme_yontemi.ode(miktar):
        print(f"{miktar}TL ödeme yapıldı")
        print(f"Kalan bakiye: {odeme_yontemi.bakiye_sorgula()}TL")
    else:
        print("Yetersiz bakiye")

kart = KrediKarti(1000)
havale = Havale(500)

odeme_yap(kart, 300)    # 300TL ödeme yapıldı, Kalan bakiye: 700TL
odeme_yap(havale, 600)  # Yetersiz bakiye
\`\`\`

## Protokoller (Protocols)

Python 3.8+ ile gelen protokoller, yapısal alt tipleme sağlar:

\`\`\`python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Yazilabilir(Protocol):
    def yaz(self, veri: str) -> bool:
        ...

class DosyaYazici:
    def yaz(self, veri: str) -> bool:
        print(f"Dosyaya yazılıyor: {veri}")
        return True

class VeritabaniYazici:
    def yaz(self, veri: str) -> bool:
        print(f"Veritabanına yazılıyor: {veri}")
        return True

def veri_kaydet(hedef: Yazilabilir, veri: str):
    if hedef.yaz(veri):
        print("Veri başarıyla kaydedildi")
    else:
        print("Veri kaydedilemedi")

# Kullanım
dosya = DosyaYazici()
db = VeritabaniYazici()

veri_kaydet(dosya, "Test verisi")
veri_kaydet(db, "Test verisi")

# Protokol kontrolü
print(isinstance(dosya, Yazilabilir))  # True
print(isinstance(db, Yazilabilir))     # True
\`\`\`

## Çoklu Arayüz Uygulaması

Bir sınıf birden fazla arayüzü uygulayabilir:

\`\`\`python
class Okunabilir(Protocol):
    def oku(self) -> str:
        ...

class Yazilabilir(Protocol):
    def yaz(self, veri: str) -> bool:
        ...

class Dosya:
    def __init__(self, icerik: str = ""):
        self.icerik = icerik
    
    def oku(self) -> str:
        return self.icerik
    
    def yaz(self, veri: str) -> bool:
        self.icerik = veri
        return True

# Kullanım
def veri_kopyala(kaynak: Okunabilir, hedef: Yazilabilir):
    veri = kaynak.oku()
    return hedef.yaz(veri)

dosya1 = Dosya("Merhaba")
dosya2 = Dosya()

if veri_kopyala(dosya1, dosya2):
    print("Veri kopyalandı")
    print(f"Yeni içerik: {dosya2.oku()}")
\`\`\`

## Pratik Örnek: Veritabanı Soyutlaması

\`\`\`python
class VeritabaniArayuzu(ABC):
    @abstractmethod
    def baglan(self) -> bool:
        pass
    
    @abstractmethod
    def kaydet(self, veri: dict) -> bool:
        pass
    
    @abstractmethod
    def sorgula(self, filtre: dict) -> list:
        pass
    
    @abstractmethod
    def kapat(self) -> bool:
        pass

class SQLiteVeritabani(VeritabaniArayuzu):
    def baglan(self) -> bool:
        print("SQLite veritabanına bağlanıldı")
        return True
    
    def kaydet(self, veri: dict) -> bool:
        print(f"Veri SQLite'a kaydedildi: {veri}")
        return True
    
    def sorgula(self, filtre: dict) -> list:
        print(f"SQLite'dan sorgulandı: {filtre}")
        return [{"id": 1, "data": "test"}]
    
    def kapat(self) -> bool:
        print("SQLite bağlantısı kapatıldı")
        return True

class MongoVeritabani(VeritabaniArayuzu):
    def baglan(self) -> bool:
        print("MongoDB'ye bağlanıldı")
        return True
    
    def kaydet(self, veri: dict) -> bool:
        print(f"Veri MongoDB'ye kaydedildi: {veri}")
        return True
    
    def sorgula(self, filtre: dict) -> list:
        print(f"MongoDB'den sorgulandı: {filtre}")
        return [{"_id": 1, "data": "test"}]
    
    def kapat(self) -> bool:
        print("MongoDB bağlantısı kapatıldı")
        return True

# Kullanım
def veritabani_islemleri(db: VeritabaniArayuzu):
    db.baglan()
    db.kaydet({"name": "test"})
    sonuclar = db.sorgula({"name": "test"})
    print(f"Sorgu sonuçları: {sonuclar}")
    db.kapat()

# Her iki veritabanı ile de çalışır
sqlite_db = SQLiteVeritabani()
mongo_db = MongoVeritabani()

veritabani_islemleri(sqlite_db)
veritabani_islemleri(mongo_db)
\`\`\`

## Alıştırmalar

1. Bir \`MedyaOynatici\` arayüzü tasarlayın:
   - \`oynat\`, \`durdur\`, \`ileri_sar\`, \`geri_sar\` metodları olsun
   - \`MuzikOynatici\` ve \`VideoOynatici\` sınıfları oluşturun

2. Bir \`Bildirim\` sistemi oluşturun:
   - \`BildirimArayuzu\` tanımlayın
   - \`EmailBildirim\`, \`SMSBildirim\`, \`PushBildirim\` sınıfları yazın
   - Her bildirim türü için farklı gönderim mantığı uygulayın

3. Bir \`Raporlama\` sistemi geliştirin:
   - \`RaporArayuzu\` oluşturun
   - \`PDFRapor\`, \`ExcelRapor\`, \`HTMLRapor\` sınıfları yazın
   - Her rapor türü için farklı formatlama ve dışa aktarma işlemleri ekleyin

## Sonraki Adımlar

Soyut sınıflar ve arayüzler konusunu öğrendiniz. Şimdi pratik örnekler ve projeler konusuna geçerek, öğrendiklerinizi gerçek dünya problemlerine nasıl uygulayacağınızı görebilirsiniz.
`;

export default function AbstractClassesAndInterfacesPage() {
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
      
      {/* Navigasyon Linkleri */}
      <div className="mt-12 flex flex-col sm:flex-row justify-between gap-4">
        <Button asChild variant="outline" className="gap-2">
          <Link href="/topics/python/nesneye-yonelik-programlama/cok-bicimlilk">
            <ArrowLeft className="h-4 w-4" />
            Önceki Konu: Çok Biçimlilik
          </Link>
        </Button>
        
        <Button asChild variant="default" className="gap-2">
          <Link href="/topics/python/nesneye-yonelik-programlama/pratik-ornekler-ve-projeler">
            Sonraki Konu: Pratik Örnekler ve Projeler
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