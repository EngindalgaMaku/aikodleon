import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python OOP: Kapsülleme (Encapsulation) | Kodleon',
  description: 'Python\'da kapsülleme kavramını, private ve protected üyeleri, property dekoratörlerini öğrenin.',
};

const content = `
# Python'da Kapsülleme (Encapsulation)

Kapsülleme, bir sınıfın iç yapısını dış dünyadan gizleyerek, sınıf içi verilere kontrollü erişim sağlamamızı sağlar.

## Private ve Protected Üyeler

Python'da gerçek private üyeler yoktur, ancak isimlendirme kuralları ile gizlilik sağlanır:

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
print(hesap._hesap_no)      # Erişilebilir ama önerilmez
# print(hesap.__bakiye)     # AttributeError
print(hesap.bakiye_goruntule())  # Doğru kullanım
\`\`\`

## Property Dekoratörleri

Property'ler, sınıf özelliklerine kontrollü erişim sağlar:

\`\`\`python
class Ogrenci:
    def __init__(self, ad, not_ortalamasi):
        self._ad = ad
        self._not_ortalamasi = not_ortalamasi
    
    @property
    def ad(self):
        return self._ad
    
    @property
    def not_ortalamasi(self):
        return self._not_ortalamasi
    
    @not_ortalamasi.setter
    def not_ortalamasi(self, deger):
        if 0 <= deger <= 100:
            self._not_ortalamasi = deger
        else:
            raise ValueError("Not 0-100 arasında olmalıdır")

# Kullanım
ogrenci = Ogrenci("Ahmet", 85)
print(ogrenci.ad)  # Property olarak erişim
print(ogrenci.not_ortalamasi)
ogrenci.not_ortalamasi = 90  # Setter ile değer atama
\`\`\`

## Getter ve Setter Metodları

Property'lere alternatif olarak geleneksel getter/setter metodları:

\`\`\`python
class Dikdortgen:
    def __init__(self, genislik, yukseklik):
        self.__genislik = genislik
        self.__yukseklik = yukseklik
    
    def get_genislik(self):
        return self.__genislik
    
    def set_genislik(self, deger):
        if deger > 0:
            self.__genislik = deger
        else:
            raise ValueError("Genişlik pozitif olmalıdır")
    
    def get_yukseklik(self):
        return self.__yukseklik
    
    def set_yukseklik(self, deger):
        if deger > 0:
            self.__yukseklik = deger
        else:
            raise ValueError("Yükseklik pozitif olmalıdır")
    
    def alan(self):
        return self.__genislik * self.__yukseklik

# Kullanım
d = Dikdortgen(5, 3)
print(d.get_genislik())
d.set_genislik(10)
print(d.alan())
\`\`\`

## Property vs Getter/Setter

Property'ler daha Pythonic bir yaklaşım sunar:

\`\`\`python
class Dikdortgen:
    def __init__(self, genislik, yukseklik):
        self._genislik = genislik
        self._yukseklik = yukseklik
    
    @property
    def genislik(self):
        return self._genislik
    
    @genislik.setter
    def genislik(self, deger):
        if deger > 0:
            self._genislik = deger
        else:
            raise ValueError("Genişlik pozitif olmalıdır")
    
    @property
    def yukseklik(self):
        return self._yukseklik
    
    @yukseklik.setter
    def yukseklik(self, deger):
        if deger > 0:
            self._yukseklik = deger
        else:
            raise ValueError("Yükseklik pozitif olmalıdır")
    
    @property
    def alan(self):
        return self._genislik * self._yukseklik

# Kullanım
d = Dikdortgen(5, 3)
print(d.genislik)      # Property getter
d.genislik = 10        # Property setter
print(d.alan)          # Property olarak alan hesaplama
\`\`\`

## Pratik Örnek: Ürün Sınıfı

\`\`\`python
class Urun:
    def __init__(self, ad, fiyat, stok):
        self._ad = ad
        self._fiyat = fiyat
        self._stok = stok
    
    @property
    def ad(self):
        return self._ad
    
    @property
    def fiyat(self):
        return self._fiyat
    
    @fiyat.setter
    def fiyat(self, yeni_fiyat):
        if yeni_fiyat >= 0:
            self._fiyat = yeni_fiyat
        else:
            raise ValueError("Fiyat negatif olamaz")
    
    @property
    def stok(self):
        return self._stok
    
    def stok_ekle(self, miktar):
        if miktar > 0:
            self._stok += miktar
            return True
        return False
    
    def stok_cikar(self, miktar):
        if 0 < miktar <= self._stok:
            self._stok -= miktar
            return True
        return False

# Kullanım
urun = Urun("Laptop", 15000, 10)
print(f"Ürün: {urun.ad}")
print(f"Fiyat: {urun.fiyat} TL")
print(f"Stok: {urun.stok}")

urun.fiyat = 16000  # Fiyat güncelleme
urun.stok_ekle(5)   # Stok ekleme
print(f"Yeni stok: {urun.stok}")
\`\`\`

## Alıştırmalar

1. Bir \`KrediKarti\` sınıfı oluşturun:
   - Kart numarası ve bakiye private olsun
   - Para çekme ve yatırma işlemleri için metodlar ekleyin
   - Bakiye kontrolü yapın

2. Bir \`Kullanici\` sınıfı oluşturun:
   - Şifre private olsun
   - Şifre değiştirme metodu ekleyin
   - Şifre kontrolü için property kullanın

3. Bir \`SicaklikOlcer\` sınıfı oluşturun:
   - Celsius ve Fahrenheit dönüşümleri için property'ler ekleyin
   - Geçerli sıcaklık aralığı kontrolü yapın

## Sonraki Adımlar

Kapsülleme konusunu öğrendiniz. Şimdi çok biçimlilik (polymorphism) konusuna geçerek, aynı arayüzü farklı sınıflarda nasıl kullanacağımızı öğrenebilirsiniz.
`;

export default function EncapsulationPage() {
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
          <Link href="/topics/python/nesneye-yonelik-programlama/kalitim">
            <ArrowLeft className="h-4 w-4" />
            Önceki Konu: Kalıtım
          </Link>
        </Button>
        
        <Button asChild variant="default" className="gap-2">
          <Link href="/topics/python/nesneye-yonelik-programlama/cok-bicimlilk">
            Sonraki Konu: Çok Biçimlilik
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