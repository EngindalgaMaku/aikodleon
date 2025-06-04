'use client';

import { ArrowLeft, ArrowRight } from 'lucide-react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import Quiz from './components/Quiz';
import CodeRunner from './components/CodeRunner';

export default function ClassesAndObjects() {
  const dikdortgenCode = `# Dikdörtgen sınıfını tanımlayalım
class Dikdortgen:
    # Constructor (yapıcı) metodu
    def __init__(self, uzunluk, genislik):
        # Örnek değişkenlerini (instance variables) tanımlama
        self.uzunluk = uzunluk  # Dikdörtgenin uzunluğu
        self.genislik = genislik  # Dikdörtgenin genişliği
    
    # Alan hesaplama metodu
    def alan_hesapla(self):
        return self.uzunluk * self.genislik
    
    # Çevre hesaplama metodu
    def cevre_hesapla(self):
        return 2 * (self.uzunluk + self.genislik)

# Test edelim
d1 = Dikdortgen(5, 3)  # 5x3 boyutunda bir dikdörtgen oluştur
print(f"Dikdörtgenin alanı: {d1.alan_hesapla()}")  # Alan: 5 * 3 = 15
print(f"Dikdörtgenin çevresi: {d1.cevre_hesapla()}")  # Çevre: 2 * (5 + 3) = 16

# Farklı boyutlarda başka bir dikdörtgen oluşturalım
d2 = Dikdortgen(10, 4)
print(f"\\nYeni dikdörtgenin alanı: {d2.alan_hesapla()}")  # Alan: 10 * 4 = 40
print(f"Yeni dikdörtgenin çevresi: {d2.cevre_hesapla()}")  # Çevre: 2 * (10 + 4) = 28`;

  const ogrenciCode = `# Öğrenci sınıfını tanımlayalım
class Ogrenci:
    # Sınıf değişkeni (class variable) - tüm öğrenciler için ortak
    okul = "Kodleon Akademi"
    
    def __init__(self, ad, numara):
        # Örnek değişkenleri (instance variables) - her öğrenci için özel
        self.ad = ad
        self.numara = numara
        self.dersler = []  # Boş bir liste ile başla
        self.notlar = {}   # Boş bir sözlük ile başla
    
    def ders_ekle(self, ders):
        if ders not in self.dersler:
        self.dersler.append(ders)
            print(f"{ders} dersi başarıyla eklendi.")
        else:
            print(f"{ders} dersi zaten mevcut!")
    
    def not_ekle(self, ders, not_):
        if ders in self.dersler:
            self.notlar[ders] = not_
            print(f"{ders} dersi için {not_} notu eklendi.")
        else:
            print(f"Önce {ders} dersini eklemelisiniz!")
    
    def not_ortalamasi(self):
        if not self.notlar:  # Hiç not yoksa
            return 0
        return sum(self.notlar.values()) / len(self.notlar)
    
    def bilgileri_goster(self):
        print(f"\\nÖğrenci Bilgileri:")
        print(f"Ad: {self.ad}")
        print(f"Numara: {self.numara}")
        print(f"Okul: {self.okul}")
        print(f"Dersler: {', '.join(self.dersler)}")
        print(f"Not Ortalaması: {self.not_ortalamasi():.2f}")

# Test edelim
# Yeni bir öğrenci oluştur
ogrenci1 = Ogrenci("Ahmet Yılmaz", 101)

# Dersler ekleyelim
ogrenci1.ders_ekle("Matematik")
ogrenci1.ders_ekle("Fizik")
ogrenci1.ders_ekle("Matematik")  # Zaten var!

# Notlar ekleyelim
ogrenci1.not_ekle("Matematik", 85)
ogrenci1.not_ekle("Fizik", 90)
ogrenci1.not_ekle("Kimya", 75)  # Önce ders eklenmeli!

# Öğrenci bilgilerini göster
ogrenci1.bilgileri_goster()`;

  const bankaHesabiCode = `# Banka hesabı sınıfını tanımlayalım
class BankaHesabi:
    # Sınıf değişkeni - tüm hesaplar için ortak
    banka_adi = "Kodleon Bank"
    
    def __init__(self, hesap_no, sahip_adi, bakiye=0):
        # Örnek değişkenleri - her hesap için özel
        self.hesap_no = hesap_no
        self.sahip_adi = sahip_adi
        self.bakiye = bakiye
        self.islemler = []  # İşlem geçmişi
    
    def islem_kaydet(self, islem_tipi, miktar):
        from datetime import datetime
        tarih = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.islemler.append(f"{tarih} - {islem_tipi}: {miktar} TL")
    
    def para_yatir(self, miktar):
        if miktar > 0:
        self.bakiye += miktar
            self.islem_kaydet("Yatırma", miktar)
            return f"{miktar} TL yatırıldı. Yeni bakiye: {self.bakiye} TL"
        return "Geçersiz miktar!"
    
    def para_cek(self, miktar):
        if miktar <= 0:
            return "Geçersiz miktar!"
        if self.bakiye >= miktar:
            self.bakiye -= miktar
            self.islem_kaydet("Çekme", miktar)
            return f"{miktar} TL çekildi. Yeni bakiye: {self.bakiye} TL"
        return "Yetersiz bakiye!"
    
    def hesap_ozeti(self):
        print(f"\\nHesap Özeti:")
        print(f"Banka: {self.banka_adi}")
        print(f"Hesap No: {self.hesap_no}")
        print(f"Hesap Sahibi: {self.sahip_adi}")
        print(f"Güncel Bakiye: {self.bakiye} TL")
        print("\\nSon İşlemler:")
        for islem in self.islemler[-5:]:  # Son 5 işlem
            print(islem)

# Test edelim
# Yeni bir banka hesabı oluştur
hesap = BankaHesabi("12345", "Ali Veli", 1000)

# Çeşitli işlemler yapalım
print(hesap.para_yatir(500))    # Para yatırma
print(hesap.para_cek(2000))     # Başarısız çekme denemesi
print(hesap.para_cek(300))      # Başarılı çekme
print(hesap.para_yatir(-100))   # Geçersiz miktar

# Hesap özetini göster
hesap.hesap_ozeti()`;

  const aracCode = `# Araç sınıfını tanımlayalım
class Arac:
    # Sınıf değişkenleri
    uretici_firma = "Kodleon Motors"
    toplam_arac_sayisi = 0
    
    def __init__(self, model, yil, renk, fiyat):
        # Her yeni araç oluşturulduğunda sayacı artır
        Arac.toplam_arac_sayisi += 1
        
        # Örnek değişkenleri
        self.model = model
        self.yil = yil
        self.renk = renk
        self.fiyat = fiyat
        self.calisiyor = False
        self.hiz = 0
        self.toplam_yol = 0
    
    def calistir(self):
        if not self.calisiyor:
            self.calisiyor = True
            return f"{self.model} çalıştırıldı."
        return f"{self.model} zaten çalışıyor."
    
    def durdur(self):
        if self.calisiyor:
            self.calisiyor = False
            self.hiz = 0
            return f"{self.model} durduruldu."
        return f"{self.model} zaten durmuş durumda."
    
    def hizlan(self, artis):
        if not self.calisiyor:
            return "Önce aracı çalıştırın!"
        if artis > 0:
            self.hiz += artis
            return f"Hız {artis} km/s artırıldı. Mevcut hız: {self.hiz} km/s"
        return "Geçersiz hız artışı!"
    
    def yavasla(self, azalis):
        if not self.calisiyor:
            return "Araç zaten çalışmıyor!"
        if azalis > 0:
            yeni_hiz = max(0, self.hiz - azalis)  # Hız 0'ın altına düşemez
            azalan = self.hiz - yeni_hiz
            self.hiz = yeni_hiz
            return f"Hız {azalan} km/s azaltıldı. Mevcut hız: {self.hiz} km/s"
        return "Geçersiz hız azalışı!"
    
    def bilgileri_goster(self):
        print(f"\\nAraç Bilgileri:")
        print(f"Üretici: {self.uretici_firma}")
        print(f"Model: {self.model}")
        print(f"Yıl: {self.yil}")
        print(f"Renk: {self.renk}")
        print(f"Fiyat: {self.fiyat:,} TL")
        print(f"Durum: {'Çalışıyor' if self.calisiyor else 'Durmuş'}")
        print(f"Mevcut Hız: {self.hiz} km/s")
        print(f"Toplam Araç Sayısı: {Arac.toplam_arac_sayisi}")

# Test edelim
# Yeni bir araç oluştur
arac1 = Arac("SUV-X", 2024, "Kırmızı", 1500000)

# Çeşitli işlemler yapalım
print(arac1.calistir())      # Aracı çalıştır
print(arac1.hizlan(20))      # Hızlan
print(arac1.hizlan(30))      # Biraz daha hızlan
print(arac1.yavasla(15))     # Yavaşla
print(arac1.durdur())        # Aracı durdur

# İkinci bir araç oluştur
arac2 = Arac("Sedan-Y", 2024, "Mavi", 1200000)

# Araçların bilgilerini göster
arac1.bilgileri_goster()
arac2.bilgileri_goster()`;

  return (
    <div className="container mx-auto py-8">
      <nav className="flex flex-col sm:flex-row justify-between items-center gap-4 mb-8 bg-muted/30 p-4 rounded-lg">
        <Link href="/topics/python/nesneye-yonelik-programlama" className="w-full sm:w-auto">
          <Button variant="outline" className="w-full">
            <ArrowLeft className="mr-2 h-4 w-4" />
            <div className="flex flex-col items-start">
              <span className="text-xs text-muted-foreground">Önceki Konu</span>
              <span>Nesneye Yönelik Programlama</span>
            </div>
          </Button>
        </Link>
        <Link href="/topics/python/nesneye-yonelik-programlama/kalitim" className="w-full sm:w-auto">
          <Button variant="outline" className="w-full">
            <div className="flex flex-col items-end">
              <span className="text-xs text-muted-foreground">Sonraki Konu</span>
              <span>Kalıtım</span>
      </div>
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </Link>
      </nav>

      <h1 className="text-4xl font-bold mb-6">Sınıflar ve Nesneler</h1>

      <section className="prose prose-lg max-w-none mb-12">
        <h2 className="text-3xl font-semibold mb-4">Sınıf (Class) Nedir?</h2>
        <p>
          Sınıf (class), nesne yönelimli programlamanın temel yapı taşıdır. Bir sınıf, nesnelerin özelliklerini ve davranışlarını tanımlayan bir şablondur.
          Gerçek dünyadaki nesnelerin yazılımdaki temsilidir.
        </p>
        
        <h3 className="text-2xl font-semibold mt-6 mb-4">Sınıfların Temel Özellikleri</h3>
        <ul className="list-disc pl-6 mb-6">
          <li>
            <strong>Özellikler (Attributes):</strong> Sınıfın sahip olduğu veriler. Örneğin, bir öğrenci sınıfı için ad, numara, notlar gibi.
          </li>
          <li>
            <strong>Metodlar (Methods):</strong> Sınıfın yapabileceği işlemler. Örneğin, not ekleme, ortalama hesaplama gibi.
          </li>
          <li>
            <strong>Constructor (__init__):</strong> Sınıftan bir nesne oluşturulduğunda çağrılan özel metod.
          </li>
          <li>
            <strong>Self Parametresi:</strong> Metodlarda nesnenin kendisini temsil eden özel parametre.
          </li>
        </ul>

        <div className="bg-blue-50 dark:bg-blue-900/10 p-6 rounded-lg mb-6">
          <h4 className="text-xl font-semibold mb-3">🎯 Örnek: Dikdörtgen Sınıfı</h4>
          <p className="mb-4">
            Aşağıdaki örnekte, bir dikdörtgenin özelliklerini (uzunluk, genişlik) ve davranışlarını (alan hesaplama, çevre hesaplama)
            modelleyen bir sınıf tanımlıyoruz. Bu örnek, sınıfların temel yapısını ve kullanımını göstermektedir.
          </p>
          <CodeRunner initialCode={dikdortgenCode} />
        </div>
      </section>

      <section className="prose prose-lg max-w-none mb-12">
        <h2 className="text-3xl font-semibold mb-4">Nesne (Object) Nedir?</h2>
        <p>
          Nesne, bir sınıfın örneğidir. Yani, sınıf şablonundan yaratılan ve bellekte yer kaplayan somut bir varlıktır.
          Her nesne, oluşturulduğu sınıfın özelliklerine ve metodlarına sahiptir, ancak kendi değerlerini tutar.
        </p>

        <h3 className="text-2xl font-semibold mt-6 mb-4">Nesnelerin Özellikleri</h3>
        <ul className="list-disc pl-6 mb-6">
          <li>
            <strong>Kimlik (Identity):</strong> Her nesnenin benzersiz bir kimliği vardır.
          </li>
          <li>
            <strong>Durum (State):</strong> Nesnenin özelliklerinin o anki değerleri.
          </li>
          <li>
            <strong>Davranış (Behavior):</strong> Nesnenin metodları aracılığıyla yapabileceği işlemler.
          </li>
        </ul>

        <div className="bg-blue-50 dark:bg-blue-900/10 p-6 rounded-lg mb-6">
          <h4 className="text-xl font-semibold mb-3">🎯 Örnek: Öğrenci Sınıfı ve Nesneleri</h4>
          <p className="mb-4">
            Bu örnekte, bir öğrencinin bilgilerini ve yapabileceği işlemleri modelleyen daha kapsamlı bir sınıf tanımlıyoruz.
            Sınıf değişkenleri, örnek değişkenleri ve çeşitli metodların kullanımını göstermektedir.
          </p>
          <CodeRunner initialCode={ogrenciCode} />
        </div>
      </section>

      <section className="prose prose-lg max-w-none mb-12">
        <h2 className="text-3xl font-semibold mb-4">Sınıf ve Nesne Kavramlarının Detaylı İncelenmesi</h2>
        
        <h3 className="text-2xl font-semibold mt-6 mb-4">1. Sınıf ve Örnek Değişkenleri</h3>
        <p>
          Python'da iki tür değişken vardır: sınıf değişkenleri (class variables) ve örnek değişkenleri (instance variables).
          Sınıf değişkenleri tüm nesneler için ortakken, örnek değişkenleri her nesne için özeldir.
        </p>

        <div className="bg-blue-50 dark:bg-blue-900/10 p-6 rounded-lg mb-6">
          <h4 className="text-xl font-semibold mb-3">🎯 Örnek: Banka Hesabı Uygulaması</h4>
          <p className="mb-4">
            Bu örnekte, bir banka hesabının özelliklerini ve işlemlerini modelleyen kapsamlı bir sınıf tanımlıyoruz.
            İşlem geçmişi tutma, para yatırma/çekme işlemleri ve hesap özeti gibi gerçek hayatta karşılaşabileceğimiz
            özellikleri içermektedir.
          </p>
          <CodeRunner initialCode={bankaHesabiCode} />
      </div>

        <h3 className="text-2xl font-semibold mt-6 mb-4">2. Metodlar ve Özellikler</h3>
        <p>
          Metodlar, nesnelerin davranışlarını tanımlayan fonksiyonlardır. Özellikler ise nesnelerin durumunu
          temsil eden verilerdir. İyi tasarlanmış bir sınıfta, metodlar ve özellikler birbirleriyle uyum içinde çalışır.
        </p>

        <div className="bg-blue-50 dark:bg-blue-900/10 p-6 rounded-lg mb-6">
          <h4 className="text-xl font-semibold mb-3">🎯 Örnek: Araç Sınıfı</h4>
          <p className="mb-4">
            Bu örnekte, bir aracın özelliklerini ve davranışlarını modelleyen kapsamlı bir sınıf tanımlıyoruz.
            Sınıf değişkenleri, örnek değişkenleri, metodlar ve durum yönetimi gibi OOP kavramlarının
            pratik bir uygulamasını göstermektedir.
          </p>
          <CodeRunner initialCode={aracCode} />
        </div>
      </section>

      <section className="prose prose-lg max-w-none mb-12">
        <h2 className="text-3xl font-semibold mb-4">Önemli Noktalar ve İyi Pratikler</h2>
        
        <div className="bg-yellow-50 dark:bg-yellow-900/10 p-6 rounded-lg mb-6">
          <h3 className="text-2xl font-semibold mb-4">Sınıf Tasarımında Dikkat Edilecek Hususlar</h3>
          <ul className="list-disc pl-6">
            <li>
              <strong>Tek Sorumluluk İlkesi:</strong> Her sınıf tek bir sorumluluğa sahip olmalıdır.
            </li>
            <li>
              <strong>Kapsülleme:</strong> Sınıfın iç yapısı dışarıya karşı korunmalıdır.
            </li>
            <li>
              <strong>Uygun İsimlendirme:</strong> Sınıf ve metod isimleri açıklayıcı olmalıdır.
            </li>
            <li>
              <strong>Dokümantasyon:</strong> Karmaşık metodlar için açıklamalar eklenmelidir.
            </li>
          </ul>
      </div>
      
        <div className="bg-green-50 dark:bg-green-900/10 p-6 rounded-lg mb-6">
          <h3 className="text-2xl font-semibold mb-4">Yaygın Hatalar ve Çözümleri</h3>
          <ul className="list-disc pl-6">
            <li>
              <strong>Self Parametresini Unutmak:</strong> Metod tanımlarında self parametresi unutulmamalıdır.
            </li>
            <li>
              <strong>Yanlış Değişken Kullanımı:</strong> Sınıf ve örnek değişkenlerinin farkı iyi anlaşılmalıdır.
            </li>
            <li>
              <strong>Gereksiz Kod Tekrarı:</strong> Ortak işlevler için yardımcı metodlar kullanılmalıdır.
            </li>
            <li>
              <strong>Yetersiz Hata Kontrolü:</strong> Metodlarda gerekli kontroller yapılmalıdır.
            </li>
          </ul>
      </div>
      </section>

      <section className="my-12">
        <Quiz />
      </section>
    </div>
  );
} 