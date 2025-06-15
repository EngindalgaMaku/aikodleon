export interface Example {
    id: number;
    title: string;
    difficulty: 'Kolay' | 'Orta' | 'Zor';
    topics: string[];
    description: string;
    solution: string;
  }
  
  export const examples: Example[] = [
      // Page 1
    {
      id: 1,
      title: 'Kitap Sınıfı Oluşturma',
      difficulty: 'Kolay',
      topics: ['Sınıflar', 'Nesneler', '__init__'],
      description: 'Bir kitabı temsil eden `Kitap` adında bir sınıf oluşturun. Bu sınıf, `baslik`, `yazar` ve `sayfa_sayisi` özelliklerini almalıdır. Ayrıca, kitabın bilgilerini yazdıran bir `bilgi_goster` metodu ekleyin.',
      solution: `
  \`\`\`python
  class Kitap:
      def __init__(self, baslik, yazar, sayfa_sayisi):
          self.baslik = baslik
          self.yazar = yazar
          self.sayfa_sayisi = sayfa_sayisi
  
      def bilgi_goster(self):
          print(f"Başlık: {self.baslik}, Yazar: {self.yazar}, Sayfa: {self.sayfa_sayisi}")
  
  # Örnek Kullanım
  kitap1 = Kitap("Sefiller", "Victor Hugo", 550)
  kitap1.bilgi_goster()
  \`\`\`
      `,
    },
    {
      id: 2,
      title: 'Geometrik Şekil Kalıtımı',
      difficulty: 'Kolay',
      topics: ['Kalıtım', 'super()'],
      description: '`Sekil` adında bir temel sınıf oluşturun. Ardından bu sınıftan türeyen `Kare` ve `Daire` sınıfları oluşturun. Her sınıfın kendi alanını hesaplayan bir `alan_hesapla` metodu olsun.',
      solution: `
  \`\`\`python
  import math
  
  class Sekil:
      def __init__(self, renk):
          self.renk = renk
  
  class Kare(Sekil):
      def __init__(self, renk, kenar_uzunlugu):
          super().__init__(renk)
          self.kenar_uzunlugu = kenar_uzunlugu
  
      def alan_hesapla(self):
          return self.kenar_uzunlugu ** 2
  
  class Daire(Sekil):
      def __init__(self, renk, yaricap):
          super().__init__(renk)
          self.yaricap = yaricap
  
      def alan_hesapla(self):
          return math.pi * (self.yaricap ** 2)
  
  # Örnek Kullanım
  kare = Kare("Mavi", 5)
  print(f"Mavi Karenin Alanı: {kare.alan_hesapla()}")
  
  daire = Daire("Kırmızı", 3)
  print(f"Kırmızı Dairenin Alanı: {daire.alan_hesapla():.2f}")
  \`\`\`
      `,
    },
    {
      id: 3,
      title: 'Çalışan Maaş Kapsülleme',
      difficulty: 'Orta',
      topics: ['Kapsülleme', 'Property'],
      description: 'Bir `Calisan` sınıfı oluşturun. Maaş özelliğini `private` yapın ve maaşı değiştirmek veya görüntülemek için `property` dekoratörlerini kullanarak `getter` ve `setter` metodları yazın. Maaş negatif bir değere ayarlanamamalıdır.',
      solution: `
  \`\`\`python
  class Calisan:
      def __init__(self, ad, maas):
          self.ad = ad
          self._maas = maas  # _ ile private olduğunu belirtiyoruz
  
      @property
      def maas(self):
          return self._maas
  
      @maas.setter
      def maas(self, yeni_maas):
          if yeni_maas < 0:
              print("Hata: Maaş negatif olamaz.")
          else:
              self._maas = yeni_maas
  
  # Örnek Kullanım
  emp = Calisan("Ali Veli", 5000)
  print(f"{emp.ad} Maaşı: {emp.maas}")
  
  emp.maas = 6000
  print(f"Yeni Maaş: {emp.maas}")
  
  emp.maas = -500 # Hata mesajı vermeli
  \`\`\`
      `,
    },
    {
      id: 4,
      title: 'Hayvan Sesleri (Çok Biçimlilik)',
      difficulty: 'Orta',
      topics: ['Çok Biçimlilik', 'Kalıtım'],
      description: '`Hayvan` adında bir temel sınıf ve bu sınıftan türeyen `Kedi` ve `Kopek` sınıfları oluşturun. Her hayvanın `ses_cikar` adında bir metodu olsun ama her biri farklı bir ses çıkarsın. Bir döngü içinde farklı hayvan nesnelerinin seslerini yazdırın.',
      solution: `
  \`\`\`python
  class Hayvan:
      def ses_cikar(self):
          print("Hayvan sesi...")
  
  class Kedi(Hayvan):
      def ses_cikar(self):
          print("Miyav!")
  
  class Kopek(Hayvan):
      def ses_cikar(self):
          print("Hav hav!")
  
  def hayvan_konustur(hayvan):
      hayvan.ses_cikar()
  
  # Örnek Kullanım
  kedi = Kedi()
  kopek = Kopek()
  
  hayvanlar = [kedi, kopek]
  for hayvan in hayvanlar:
      hayvan_konustur(hayvan)
  \`\`\`
      `,
    },
      // Page 2
    {
      id: 5,
      title: 'Vektör Sınıfı ve Operatör Yükleme',
      difficulty: 'Zor',
      topics: ['Sınıflar', 'Operatör Yükleme'],
      description: 'İki boyutlu bir vektörü (`x` ve `y` koordinatları) temsil eden bir `Vektor` sınıfı yazın. İki vektörü toplamak için `+` operatörünü (`__add__` metodu) ve bir vektörün uzunluğunu bulmak için `len()` fonksiyonunu (`__len__` metodu) yeniden yükleyin.',
      solution: `
  \`\`\`python
  import math
  
  class Vektor:
      def __init__(self, x, y):
          self.x = x
          self.y = y
  
      def __add__(self, other):
          # İki vektörü toplar
          return Vektor(self.x + other.x, self.y + other.y)
  
      def __len__(self):
          # Vektörün orijine olan uzaklığını (uzunluğunu) hesaplar
          return int(math.sqrt(self.x**2 + self.y**2))
  
      def __repr__(self):
          # Nesnenin yazdırılabilir temsilini sağlar
          return f"Vektor({self.x}, {self.y})"
  
  # Örnek Kullanım
  v1 = Vektor(2, 4)
  v2 = Vektor(3, 5)
  
  v3 = v1 + v2
  print(f"{v1} + {v2} = {v3}")
  print(f"Vektör v3'ün uzunluğu: {len(v3)}")
  \`\`\`
      `,
    },
    {
      id: 6,
      title: 'Soyut Veritabanı Bağlantısı',
      difficulty: 'Zor',
      topics: ['Soyut Sınıflar', 'ABC'],
      description: '`VeritabaniBaglantisi` adında soyut bir temel sınıf oluşturun. Bu sınıf, `baglan()` ve `kes()` adında iki soyut metoda sahip olsun. Ardından, bu soyut sınıftan `PostgreSQLBaglantisi` ve `SQLiteBaglantisi` adında iki somut sınıf türetin ve metodları uygulayın.',
      solution: `
  \`\`\`python
  from abc import ABC, abstractmethod
  
  class VeritabaniBaglantisi(ABC):
      @abstractmethod
      def baglan(self):
          pass
  
      @abstractmethod
      def kes(self):
          pass
  
  class PostgreSQLBaglantisi(VeritabaniBaglantisi):
      def baglan(self):
          print("PostgreSQL veritabanına bağlanılıyor...")
  
      def kes(self):
          print("PostgreSQL bağlantısı kesiliyor...")
  
  class SQLiteBaglantisi(VeritabaniBaglantisi):
      def baglan(self):
          print("SQLite veritabanına bağlanılıyor...")
  
      def kes(self):
          print("SQLite bağlantısı kesiliyor...")
  
  # Örnek Kullanım
  pg_conn = PostgreSQLBaglantisi()
  pg_conn.baglan()
  pg_conn.kes()
  
  sqlite_conn = SQLiteBaglantisi()
  sqlite_conn.baglan()
  sqlite_conn.kes()
  \`\`\`
      `,
    },
    {
      id: 7,
      title: 'Araba ve Motor Kompozisyonu',
      difficulty: 'Orta',
      topics: ['Kompozisyon', 'Sınıflar'],
      description: '`Motor` ve `Araba` adında iki sınıf oluşturun. `Araba` sınıfı, bir `Motor` nesnesini kendi içinde barındırsın (kompozisyon). `Araba` sınıfının bir `calistir` metodu olsun ve bu metod, motorun `start` metodunu çağırsın.',
      solution: `
  \`\`\`python
  class Motor:
      def start(self):
          print("Motor çalıştı... Vroom!")
  
      def stop(self):
          print("Motor durdu.")
  
  class Araba:
      def __init__(self, marka, model):
          self.marka = marka
          self.model = model
          self.motor = Motor() # Araba bir motor nesnesi içerir
  
      def calistir(self):
          print(f"{self.marka} {self.model} çalıştırılıyor...")
          self.motor.start()
  
      def durdur(self):
          print(f"{self.marka} {self.model} durduruluyor...")
          self.motor.stop()
  
  # Örnek Kullanım
  arac = Araba("Ford", "Mustang")
  arac.calistir()
  arac.durdur()
  \`\`\`
      `,
    },
    {
      id: 8,
      title: 'Singleton Tasarım Deseni',
      difficulty: 'Zor',
      topics: ['Tasarım Desenleri', 'Sınıf Metodları'],
      description: 'Bir sınıftan sadece tek bir nesne oluşturulmasını sağlayan Singleton tasarım desenini uygulayın. Örneğin, bir `Ayarlar` sınıfı oluşturun ve program boyunca bu sınıftan sadece bir tane nesne yaratılabilsin.',
      solution: `
  \`\`\`python
  class Ayarlar:
      _instance = None  # Sınıf seviyesinde tek bir nesneyi tutacak değişken
  
      def __new__(cls, *args, **kwargs):
          if cls._instance is None:
              cls._instance = super().__new__(cls)
          return cls._instance
  
      def __init__(self):
          # Bu kısım sadece ilk oluşturmada çalışsın diye kontrol edilebilir
          if not hasattr(self, 'is_initialized'):
              self.tema = "Karanlık"
              self.dil = "Türkçe"
              self.is_initialized = True
              print("Ayarlar nesnesi oluşturuldu.")
  
  # Örnek Kullanım
  ayarlar1 = Ayarlar()
  print(f"Ayar 1: Tema={ayarlar1.tema}, Dil={ayarlar1.dil}")
  
  ayarlar2 = Ayarlar()
  print("İkinci kez nesne oluşturma denendi.")
  print(f"Ayar 2: Tema={ayarlar2.tema}, Dil={ayarlar2.dil}")
  
  # Nesnelerin aynı olduğunu kontrol et
  print(f"ayarlar1 ve ayarlar2 aynı nesne mi? {ayarlar1 is ayarlar2}")
  
  # Bir ayarı değiştirip diğerinde kontrol et
  ayarlar1.tema = "Açık"
  print(f"Ayar 1'de tema değiştirildi. Ayar 2 tema: {ayarlar2.tema}")
  \`\`\`
      `,
    },
      // Page 3
    {
      id: 9,
      title: 'Kütüphane Sistemi',
      difficulty: 'Zor',
      topics: ['Sınıflar', 'Kompozisyon', 'Listeler'],
      description: '`Kitap` ve `Kutuphane` sınıfları oluşturun. `Kutuphane` sınıfı, `Kitap` nesnelerini bir listede tutmalıdır. `kitap_ekle`, `kitap_bul` ve `tum_kitaplari_goster` gibi metodlar içermelidir.',
      solution: `
  \`\`\`python
  class Kitap:
      def __init__(self, baslik, yazar):
          self.baslik = baslik
          self.yazar = yazar
  
      def __repr__(self):
          return f"'{self.baslik}' by {self.yazar}"
  
  class Kutuphane:
      def __init__(self, ad):
          self.ad = ad
          self.kitaplar = []
  
      def kitap_ekle(self, kitap):
          self.kitaplar.append(kitap)
          print(f"{kitap} kütüphaneye eklendi.")
  
      def kitap_bul(self, baslik):
          for kitap in self.kitaplar:
              if kitap.baslik.lower() == baslik.lower():
                  return kitap
          return None
  
      def tum_kitaplari_goster(self):
          print(f"--- {self.ad} Kütüphanesi Kitap Listesi ---")
          if not self.kitaplar:
              print("Kütüphanede hiç kitap yok.")
          else:
              for kitap in self.kitaplar:
                  print(f"- {kitap}")
  
  # Örnek Kullanım
  kutuphane = Kutuphane("Şehir")
  k1 = Kitap("1984", "George Orwell")
  k2 = Kitap("Hayvan Çiftliği", "George Orwell")
  kutuphane.kitap_ekle(k1)
  kutuphane.kitap_ekle(k2)
  kutuphane.tum_kitaplari_goster()
  
  aranan_kitap = kutuphane.kitap_bul("1984")
  if aranan_kitap:
      print(f"Bulunan kitap: {aranan_kitap}")
  \`\`\`
      `,
    },
    {
      id: 10,
      title: 'E-ticaret Ürün ve Sepet Sınıfları',
      difficulty: 'Orta',
      topics: ['Sınıflar', 'Listeler', 'Kompozisyon'],
      description: 'Bir e-ticaret sistemi için `Urun` ve `Sepet` sınıflarını tasarlayın. `Urun` sınıfı ürün adı ve fiyatı tutmalıdır. `Sepet` sınıfı ise ürün ekleme, ürün çıkarma ve toplam tutarı hesaplama işlevlerine sahip olmalıdır.',
      solution: `
  \`\`\`python
  class Urun:
      def __init__(self, ad, fiyat):
          self.ad = ad
          self.fiyat = fiyat
      
      def __repr__(self):
          return f"Ürün({self.ad}, {self.fiyat} TL)"
  
  class Sepet:
      def __init__(self):
          self.urunler = []
  
      def urun_ekle(self, urun):
          self.urunler.append(urun)
          print(f"{urun.ad} sepete eklendi.")
  
      def urun_cikar(self, urun_adi):
          for urun in self.urunler:
              if urun.ad == urun_adi:
                  self.urunler.remove(urun)
                  print(f"{urun_adi} sepetten çıkarıldı.")
                  return
          print(f"{urun_adi} sepetinizde bulunamadı.")
  
      def toplam_tutar(self):
          return sum(urun.fiyat for urun in self.urunler)
  
  # Örnek Kullanım
  laptop = Urun("Laptop", 1500)
  mouse = Urun("Mouse", 50)
  
  sepetim = Sepet()
  sepetim.urun_ekle(laptop)
  sepetim.urun_ekle(mouse)
  
  print(f"Sepetteki ürünler: {sepetim.urunler}")
  print(f"Toplam Tutar: {sepetim.toplam_tutar()} TL")
  
  sepetim.urun_cikar("Laptop")
  print(f"Yeni Toplam Tutar: {sepetim.toplam_tutar()} TL")
  \`\`\`
      `,
    },
    {
      id: 11,
      title: 'Oyun Karakteri ve Yetenekleri',
      difficulty: 'Orta',
      topics: ['Kalıtım', 'Kompozisyon'],
      description: 'Bir oyun için temel bir `Karakter` sınıfı ve bundan türeyen `Savasci` ve `Sihirbaz` sınıfları oluşturun. Karakterlerin can, mana gibi özellikleri olsun. `Savasci` kılıçla saldırırken, `Sihirbaz` büyü yapsın.',
      solution: `
  \`\`\`python
  class Karakter:
      def __init__(self, ad, can, mana):
          self.ad = ad
          self.can = can
          self.mana = mana
      
      def durum_goster(self):
          print(f"{self.ad}: Can={self.can}, Mana={self.mana}")
  
  class Savasci(Karakter):
      def __init__(self, ad):
          super().__init__(ad, can=150, mana=50)
  
      def saldir(self, hedef):
          print(f"{self.ad} kılıcıyla {hedef.ad}'a saldırıyor!")
          hedef.can -= 15
  
  class Sihirbaz(Karakter):
      def __init__(self, ad):
          super().__init__(ad, can=100, mana=100)
  
      def buyu_yap(self, hedef):
          if self.mana >= 20:
              print(f"{self.ad} ateş topu büyüsüyle {hedef.ad}'a saldırıyor!")
              hedef.can -= 25
              self.mana -= 20
          else:
              print("Yeterli mana yok!")
  
  # Örnek Kullanım
  conan = Savasci("Conan")
  merlin = Sihirbaz("Merlin")
  
  conan.durum_goster()
  merlin.durum_goster()
  
  merlin.buyu_yap(conan)
  conan.saldir(merlin)
  
  conan.durum_goster()
  merlin.durum_goster()
  \`\`\`
      `,
    },
    {
      id: 12,
      title: 'Özel Metodlar: __str__ ve __repr__',
      difficulty: 'Kolay',
      topics: ['Sınıflar', 'Özel Metodlar'],
      description: 'Bir `Calisan` sınıfı oluşturun. Bu sınıf için hem `__str__` (kullanıcı dostu gösterim) hem de `__repr__` (geliştirici dostu, nesneyi yeniden oluşturabilecek gösterim) özel metodlarını tanımlayın.',
      solution: `
  \`\`\`python
  class Calisan:
      def __init__(self, ad, pozisyon, maas):
          self.ad = ad
          self.pozisyon = pozisyon
          self.maas = maas
      
      def __str__(self):
          # print() veya str() ile çağrılır. Kullanıcıya yönelik.
          return f"{self.ad} - {self.pozisyon}"
          
      def __repr__(self):
          # Nesneyi doğrudan konsola yazdığında veya repr() ile çağrılır. Geliştiriciye yönelik.
          return f"Calisan('{self.ad}', '{self.pozisyon}', {self.maas})"
  
  # Örnek Kullanım
  calisan = Calisan("Ayşe Yılmaz", "Yazılım Geliştirici", 8000)
  
  # __str__ kullanımı
  print(calisan)
  
  # __repr__ kullanımı
  print(repr(calisan))
  
  # Konsolda sadece 'calisan' yazınca da __repr__ çağrılır
  \`\`\`
      `,
    },
      // Page 4
    {
      id: 13,
      title: 'Banka ve Müşteri İlişkisi',
      difficulty: 'Zor',
      topics: ['Kompozisyon', 'Sınıflar Arası İlişki'],
      description: '`Musteri`, `Hesap` ve `Banka` sınıfları oluşturun. Bir `Banka` birden fazla `Musteri`ye sahip olabilir. Her `Musteri` birden fazla `Hesap`a sahip olabilir. Müşteri ekleme, hesap açma ve bir müşterinin tüm hesaplarının toplam bakiyesini gösterme gibi işlevler ekleyin.',
      solution: `
  \`\`\`python
  class Hesap:
      def __init__(self, hesap_no, bakiye=0):
          self.hesap_no = hesap_no
          self.bakiye = bakiye
  
  class Musteri:
      def __init__(self, ad, musteri_no):
          self.ad = ad
          self.musteri_no = musteri_no
          self.hesaplar = []
  
      def hesap_ac(self, hesap_no, baslangic_bakiye=0):
          yeni_hesap = Hesap(hesap_no, baslangic_bakiye)
          self.hesaplar.append(yeni_hesap)
          return yeni_hesap
  
      def toplam_bakiye(self):
          return sum(h.bakiye for h in self.hesaplar)
  
  class Banka:
      def __init__(self, banka_adi):
          self.banka_adi = banka_adi
          self.musteriler = []
  
      def musteri_ekle(self, ad, musteri_no):
          yeni_musteri = Musteri(ad, musteri_no)
          self.musteriler.append(yeni_musteri)
          return yeni_musteri
  
  # Örnek Kullanım
  kodleon_bank = Banka("Kodleon Bank")
  musteri1 = kodleon_bank.musteri_ekle("Ali Veli", "123")
  musteri1.hesap_ac("HESAP01", 500)
  musteri1.hesap_ac("HESAP02", 1500)
  
  musteri2 = kodleon_bank.musteri_ekle("Ayşe Yılmaz", "456")
  musteri2.hesap_ac("HESAP03", 3000)
  
  print(f"{musteri1.ad}'in toplam bakiyesi: {musteri1.toplam_bakiye()} TL")
  print(f"{musteri2.ad}'in toplam bakiyesi: {musteri2.toplam_bakiye()} TL")
  \`\`\`
      `,
    },
    {
      id: 14,
      title: 'Statik Metod Kullanımı',
      difficulty: 'Kolay',
      topics: ['Sınıf Metodları', '@staticmethod'],
      description: 'Bir `Matematik` sınıfı oluşturun. Bu sınıfın nesnesini oluşturmaya gerek kalmadan doğrudan çağrılabilecek, verilen bir sayının faktöriyelini hesaplayan `faktoriyel` adında bir statik metod ekleyin.',
      solution: `
  \`\`\`python
  class Matematik:
      @staticmethod
      def faktoriyel(n):
          if n < 0:
              return "Negatif sayılar için faktöriyel tanımsızdır."
          if n == 0:
              return 1
          sonuc = 1
          for i in range(1, n + 1):
              sonuc *= i
          return sonuc
  
  # Örnek Kullanım
  # Sınıftan bir nesne oluşturmaya gerek yok
  print(f"5'in faktöriyeli: {Matematik.faktoriyel(5)}")
  print(f"0'ın faktöriyeli: {Matematik.faktoriyel(0)}")
  \`\`\`
      `,
    },
    {
      id: 15,
      title: 'Sınıf Metodu ile Nesne Oluşturma',
      difficulty: 'Orta',
      topics: ['Sınıf Metodları', '@classmethod'],
      description: 'Bir `Tarih` sınıfı oluşturun. `__init__` metodu `gun`, `ay`, `yil` alsın. Ek olarak, "gun-ay-yil" formatında bir stringi alıp, bu stringi parse ederek bir `Tarih` nesnesi oluşturan `stringden_olustur` adında bir sınıf metodu (@classmethod) ekleyin.',
      solution: `
  \`\`\`python
  class Tarih:
      def __init__(self, gun, ay, yil):
          self.gun = gun
          self.ay = ay
          self.yil = yil
      
      def goster(self):
          print(f"{self.gun:02d}/{self.ay:02d}/{self.yil}")
  
      @classmethod
      def stringden_olustur(cls, tarih_stringi):
          # tarih_stringi "27-10-2023" formatında olmalı
          gun, ay, yil = map(int, tarih_stringi.split('-'))
          return cls(gun, ay, yil) # Yeni bir Tarih nesnesi oluşturup döndürür
  
  # Örnek Kullanım
  # Normal yolla nesne oluşturma
  tarih1 = Tarih(27, 10, 2023)
  tarih1.goster()
  
  # Sınıf metodu ile nesne oluşturma
  tarih_str = "01-05-2024"
  tarih2 = Tarih.stringden_olustur(tarih_str)
  tarih2.goster()
  \`\`\`
      `,
    },
    {
      id: 16,
      title: 'İstisna (Exception) Sınıfı Tanımlama',
      difficulty: 'Zor',
      topics: ['İstisna Yönetimi', 'Sınıflar'],
      description: 'Kendi özel istisna sınıfınızı oluşturun. `YetersizBakiyeError` adında, Python\'un `Exception` sınıfından türeyen bir sınıf tanımlayın. Banka hesabından para çekme işleminde bakiye yetersizse bu özel istisnayı fırlatın.',
      solution: `
  \`\`\`python
  # Özel istisna sınıfımız
  class YetersizBakiyeError(Exception):
      def __init__(self, bakiye, cekilmek_istenen):
          self.bakiye = bakiye
          self.cekilmek_istenen = cekilmek_istenen
          mesaj = f"Hesabınızdaki {bakiye} TL, çekmek istediğiniz {cekilmek_istenen} TL için yetersiz."
          super().__init__(mesaj)
  
  class BankaHesabi:
      def __init__(self, bakiye=0.0):
          self.bakiye = bakiye
  
      def para_cek(self, miktar):
          if miktar > self.bakiye:
              raise YetersizBakiyeError(self.bakiye, miktar)
          self.bakiye -= miktar
          print(f"{miktar} TL çekildi. Kalan bakiye: {self.bakiye}")
  
  # Örnek Kullanım
  hesap = BankaHesabi(100)
  try:
      hesap.para_cek(50)
      hesap.para_cek(80) # Bu satırda hata fırlatılacak
  except YetersizBakiyeError as e:
      print(f"Hata: {e}")
  \`\`\`
      `,
    },
      // Page 5
    {
      id: 17,
      title: 'Veri Sınıfları (Data Classes)',
      difficulty: 'Orta',
      topics: ['dataclasses', 'Sınıflar'],
      description: 'Python 3.7+ ile gelen `dataclasses` modülünü kullanarak basit bir `Kisi` sınıfı oluşturun. Bu modül, `__init__`, `__repr__`, `__eq__` gibi özel metodları otomatik olarak oluşturur.',
      solution: `
  \`\`\`python
  from dataclasses import dataclass
  
  @dataclass
  class Kisi:
      ad: str
      soyad: str
      yas: int
      aktif: bool = True
  
  # Örnek Kullanım
  kisi1 = Kisi("Ahmet", "Çelik", 30)
  kisi2 = Kisi("Ahmet", "Çelik", 30)
  kisi3 = Kisi("Mehmet", "Yılmaz", 45, aktif=False)
  
  # Otomatik oluşturulan __repr__ metodu sayesinde güzel çıktı
  print(kisi1)
  print(kisi3)
  
  # Otomatik oluşturulan __eq__ metodu sayesinde nesne karşılaştırması
  print(f"kisi1 ve kisi2 eşit mi? {kisi1 == kisi2}")
  print(f"kisi1 ve kisi3 eşit mi? {kisi1 == kisi3}")
  \`\`\`
      `,
    },
    {
      id: 18,
      title: 'Basit Bir Blog Sistemi Modeli',
      difficulty: 'Zor',
      topics: ['Sınıflar Arası İlişki', 'Kompozisyon'],
      description: '`Yazar`, `Gonderi` ve `Yorum` sınıflarını içeren bir blog sistemi modeli tasarlayın. Bir `Yazar` birden fazla `Gonderi` yazabilir. Bir `Gonderi` birden fazla `Yorum` alabilir. Sınıflar arasındaki ilişkileri kurun ve örnek verilerle sistemi test edin.',
      solution: `
  \`\`\`python
  class Yorum:
      def __init__(self, kullanici, icerik):
          self.kullanici = kullanici
          self.icerik = icerik
  
  class Gonderi:
      def __init__(self, yazar, baslik, icerik):
          self.yazar = yazar
          self.baslik = baslik
          self.icerik = icerik
          self.yorumlar = []
  
      def yorum_ekle(self, yorum):
          self.yorumlar.append(yorum)
  
      def gonderiyi_goster(self):
          print(f"--- {self.baslik} ---")
          print(f"Yazar: {self.yazar.ad}")
          print(f"İçerik: {self.icerik}")
          print("\\nYorumlar:")
          if not self.yorumlar:
              print("Henüz yorum yok.")
          for yorum in self.yorumlar:
              print(f"- {yorum.kullanici}: {yorum.icerik}")
  
  class Yazar:
      def __init__(self, ad):
          self.ad = ad
  
  # Örnek Kullanım
  yazar1 = Yazar("Ali Can")
  gonderi1 = Gonderi(yazar1, "Python OOP Harika!", "Nesne tabanlı programlama...")
  
  yorum1 = Yorum("Ayşe", "Harika bir yazı!")
  yorum2 = Yorum("Mehmet", "Teşekkürler.")
  
  gonderi1.yorum_ekle(yorum1)
  gonderi1.yorum_ekle(yorum2)
  
  gonderi1.gonderiyi_goster()
  \`\`\`
      `,
    },
    {
      id: 19,
      title: 'Çoklu Kalıtım ve MRO',
      difficulty: 'Zor',
      topics: ['Kalıtım', 'MRO'],
      description: '`A`, `B`, `C` ve `D` adında dört sınıf oluşturun. `D` sınıfı, `B` ve `C` sınıflarından; `B` ve `C` sınıfları ise `A` sınıfından kalıtım alsın (Elmas Problemi - Diamond Problem). `D` sınıfının Metod Çözümleme Sırasını (Method Resolution Order - MRO) yazdırın.',
      solution: `
  \`\`\`python
  class A:
      def kimim_ben(self):
          print("Ben A sınıfıyım")
  
  class B(A):
      def kimim_ben(self):
          print("Ben B sınıfıyım")
  
  class C(A):
      def kimim_ben(self):
          print("Ben C sınıfıyım")
  
  class D(B, C):
      pass
  
  # Örnek Kullanım
  d_nesnesi = D()
  d_nesnesi.kimim_ben() # Python'un MRO'suna göre B'deki metod çağrılır
  
  # MRO'yu göster
  print("\\nD sınıfının Metod Çözümleme Sırası (MRO):")
  print(D.mro())
  # veya print(D.__mro__)
  \`\`\`
      `,
    },
    {
      id: 20,
      title: 'Özellikleri Dinamik Olarak Ayarlama',
      difficulty: 'Orta',
      topics: ['Sınıflar', '__dict__', 'setattr'],
      description: 'Bir `Config` sınıfı oluşturun. Bu sınıfa bir sözlük (`dict`) vererek nesnenin özelliklerini dinamik olarak ayarlamanızı sağlayan bir mekanizma kurun. Örneğin, `{"tema": "koyu", "font_boyutu": 14}` sözlüğü, nesnenin `tema` ve `font_boyutu` özelliklerini oluşturmalıdır.',
      solution: `
  \`\`\`python
  class Config:
      def __init__(self, ayarlar_sozlugu=None):
          if ayarlar_sozlugu:
              for anahtar, deger in ayarlar_sozlugu.items():
                  # setattr fonksiyonu ile dinamik olarak özellik ata
                  setattr(self, anahtar, deger)
  
      def __repr__(self):
          # Nesnenin özelliklerini göster
          return f"Config({self.__dict__})"
  
  # Örnek Kullanım
  ayar_verisi = {
      "tema": "koyu",
      "font_boyutu": 14,
      "kullanici_adi": "kodleon"
  }
  
  config = Config(ayar_verisi)
  print(config)
  
  # Özelliklere erişim
  print(f"Tema: {config.tema}")
  print(f"Kullanıcı Adı: {config.kullanici_adi}")
  \`\`\`
      `,
    },
    {
      id: 21,
      title: 'Decorator Tasarım Deseni',
      difficulty: 'Zor',
      topics: ['Tasarım Desenleri', 'Decorator', 'Fonksiyonlar'],
      description: 'Bir temel kahve sınıfı ve bu kahveyi "dekore eden" (üzerine ek özellikler ekleyen) süt ve şeker gibi sınıflar oluşturun. Bu yapı, bir nesneye dinamik olarak yeni sorumluluklar eklemenizi sağlar.',
      solution: `
\`\`\`python
from abc import ABC, abstractmethod

# Component Arayüzü
class Kahve(ABC):
    @abstractmethod
    def get_aciklama(self):
        pass

    @abstractmethod
    def get_maliyet(self):
        pass

# Concrete Component
class SadeKahve(Kahve):
    def get_aciklama(self):
        return "Sade Kahve"
    
    def get_maliyet(self):
        return 5.0

# Decorator Temel Sınıfı
class KahveDecorator(Kahve):
    def __init__(self, kahve: Kahve):
        self._kahve = kahve

    @abstractmethod
    def get_aciklama(self):
        pass

    @abstractmethod
    def get_maliyet(self):
        pass

# Concrete Decorators
class SutEklentisi(KahveDecorator):
    def get_aciklama(self):
        return self._kahve.get_aciklama() + ", Sütlü"
    
    def get_maliyet(self):
        return self._kahve.get_maliyet() + 1.5

class SekerEklentisi(KahveDecorator):
    def get_aciklama(self):
        return self._kahve.get_aciklama() + ", Şekerli"
    
    def get_maliyet(self):
        return self._kahve.get_maliyet() + 0.5

# Örnek Kullanım
kahvem = SadeKahve()
print(f"{kahvem.get_aciklama()}: {kahvem.get_maliyet()} TL")

sutlu_kahve = SutEklentisi(kahvem)
print(f"{sutlu_kahve.get_aciklama()}: {sutlu_kahve.get_maliyet()} TL")

sutlu_sekerli_kahve = SekerEklentisi(sutlu_kahve)
print(f"{sutlu_sekerli_kahve.get_aciklama()}: {sutlu_sekerli_kahve.get_maliyet()} TL")
\`\`\`
    `,
  },
  {
    id: 22,
    title: 'Context Manager Oluşturma',
    difficulty: 'Orta',
    topics: ['Context Manager', '__enter__', '__exit__', 'with'],
    description: 'Bir dosya işlemini yöneten bir context manager sınıfı (`DosyaYoneticisi`) oluşturun. Bu sınıf, `with` bloğuna girildiğinde dosyayı açmalı ve bloktan çıkıldığında (hata olsa bile) dosyayı otomatik olarak kapatmalıdır.',
    solution: `
\`\`\`python
class DosyaYoneticisi:
    def __init__(self, dosya_adi, mod):
        self.dosya_adi = dosya_adi
        self.mod = mod
        self.dosya = None
        print("init metodu çağrıldı.")

    def __enter__(self):
        print("enter metodu çağrıldı.")
        self.dosya = open(self.dosya_adi, self.mod)
        return self.dosya

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("exit metodu çağrıldı.")
        if self.dosya:
            self.dosya.close()
        # Eğer bir istisna oluştuysa, burada ele alabilirsiniz.
        # True dönerseniz istisna bastırılır, False dönerseniz yükseltilir.
        return False

# Örnek Kullanım
with DosyaYoneticisi('test.txt', 'w') as f:
    f.write('Merhaba Kodleon!')
    print("with bloğu içindeyim.")

print("with bloğundan çıkıldı.")
# Bu noktada dosya otomatik olarak kapatılmıştır.
\`\`\`
    `,
  },
  {
    id: 23,
    title: 'Enum ile Sabitleri Yönetme',
    difficulty: 'Kolay',
    topics: ['Enum', 'Sabitler'],
    description: 'Haftanın günlerini temsil eden bir `HaftaninGunleri` adında bir `Enum` oluşturun. Enum kullanarak günlerin sıralamasını ve adlarını güvenli bir şekilde yönetin.',
    solution: `
\`\`\`python
from enum import Enum

class HaftaninGunleri(Enum):
    PAZARTESI = 1
    SALI = 2
    CARSAMBA = 3
    PERSEMBE = 4
    CUMA = 5
    CUMARTESI = 6
    PAZAR = 7

# Örnek Kullanım
bugun = HaftaninGunleri.CARSAMBA

print(f"Bugün: {bugun}")
print(f"Bugünün adı: {bugun.name}")
print(f"Bugünün değeri: {bugun.value}")

if bugun == HaftaninGunleri.CUMARTESI or bugun == HaftaninGunleri.PAZAR:
    print("Hafta sonu!")
else:
    print("Hafta içi.")

# Enum'lar üzerinde döngü kurma
for gun in HaftaninGunleri:
    print(f"{gun.value}: {gun.name}")
\`\`\`
    `,
  },
  {
    id: 24,
    title: 'Factory Tasarım Deseni',
    difficulty: 'Zor',
    topics: ['Tasarım Desenleri', 'Factory'],
    description: 'Farklı türde hayvanlar (`Kedi`, `Kopek`) oluşturan bir `HayvanFactory` sınıfı oluşturun. Bu factory, kendisine verilen bir stringe göre ilgili hayvan nesnesini yaratıp döndürmelidir. Bu, nesne oluşturma mantığını merkezileştirir.',
    solution: `
\`\`\`python
from abc import ABC, abstractmethod

class Hayvan(ABC):
    @abstractmethod
    def ses_cikar(self):
        pass

class Kedi(Hayvan):
    def ses_cikar(self):
        return "Miyav!"

class Kopek(Hayvan):
    def ses_cikar(self):
        return "Hav hav!"

class HayvanFactory:
    @staticmethod
    def hayvan_olustur(hayvan_turu: str) -> Hayvan:
        if hayvan_turu.lower() == 'kedi':
            return Kedi()
        elif hayvan_turu.lower() == 'kopek':
            return Kopek()
        else:
            raise ValueError(f"Bilinmeyen hayvan türü: {hayvan_turu}")

# Örnek Kullanım
factory = HayvanFactory()

hayvan1 = factory.hayvan_olustur("kedi")
print(f"Kedi sesi: {hayvan1.ses_cikar()}")

hayvan2 = factory.hayvan_olustur("kopek")
print(f"Köpek sesi: {hayvan2.ses_cikar()}")

try:
    factory.hayvan_olustur("kus")
except ValueError as e:
    print(e)
\`\`\`
    `,
  },
    // Page 6
  {
    id: 25,
    title: 'İleri Kalıtım: Mixin Kullanımı',
    difficulty: 'Zor',
    topics: ['Kalıtım', 'Mixin'],
    description: 'JSON formatına serileştirme yeteneği kazandıran bir `JSONMixin` sınıfı oluşturun. Bu mixin\'i farklı sınıflarla (`Calisan`, `Urun`) kullanarak kod tekrarı yapmadan onlara `to_json` metodu ekleyin.',
    solution: `
\`\`\`python
import json

class JSONMixin:
    def to_json(self):
        # Nesnenin __dict__'ini (özelliklerini) JSON string'ine çevirir
        return json.dumps(self.__dict__, indent=4, sort_keys=True)

class Calisan(JSONMixin):
    def __init__(self, ad, sicil_no):
        self.ad = ad
        self.sicil_no = sicil_no

class Urun(JSONMixin):
    def __init__(self, ad, fiyat, stok):
        self.ad = ad
        self.fiyat = fiyat
        self.stok = stok

# Örnek Kullanım
calisan = Calisan("Zeynep Kaya", "12345")
print("--- Çalışan JSON ---")
print(calisan.to_json())

urun = Urun("Akıllı Telefon", 4500, 150)
print("\\n--- Ürün JSON ---")
print(urun.to_json())
\`\`\`
    `,
  },
  {
    id: 26,
    title: 'Bellek Optimizasyonu: __slots__',
    difficulty: 'Orta',
    topics: ['__slots__', 'Optimizasyon'],
    description: 'Milyonlarca küçük nesne oluşturmanız gereken bir senaryo düşünün. `__slots__` kullanarak bir sınıfın bellek kullanımını nasıl optimize edeceğinizi gösterin. `__slots__` olmayan bir sınıf ile karşılaştırın.',
    solution: `
\`\`\`python
import sys

class NoktaNormal:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class NoktaSlotlu:
    __slots__ = ['x', 'y']  # Sadece bu özelliklere izin verilir
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Normal nesne
normal_nesne = NoktaNormal(1, 2)
print(f"Normal nesne boyutu: {sys.getsizeof(normal_nesne)} bytes")
print(f"Normal nesnenin __dict__'i: {normal_nesne.__dict__}")

# Slotlu nesne
slotlu_nesne = NoktaSlotlu(1, 2)
print(f"Slotlu nesne boyutu: {sys.getsizeof(slotlu_nesne)} bytes")
# print(slotlu_nesne.__dict__) # AttributeError: 'NoktaSlotlu' object has no attribute '__dict__'

# Slotlu nesneye yeni özellik eklemeyi dene
try:
    slotlu_nesne.z = 3
except AttributeError as e:
    print(f"Yeni özellik ekleme hatası: {e}")
\`\`\`
    `,
  },
  {
    id: 27,
    title: 'İskambil Destesi Sınıfı',
    difficulty: 'Orta',
    topics: ['Sınıflar', 'Listeler', 'Kompozisyon', '__len__', '__getitem__'],
    description: 'Standart bir 52 kartlık iskambil destesini temsil eden bir `Deste` sınıfı oluşturun. Deste, `Kart` nesnelerinden oluşmalıdır. Desteyi karıştırma, kart çekme gibi metodlar ekleyin. Ayrıca `len()` ve `deste[i]` gibi işlemlere izin verin.',
    solution: `
\`\`\`python
import random

class Kart:
    def __init__(self, deger, tur):
        self.deger = deger
        self.tur = tur
    
    def __repr__(self):
        return f"{self.tur} {self.deger}"

class Deste:
    DEGERLER = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    TURLER = ['Kupa', 'Karo', 'Maça', 'Sinek']

    def __init__(self):
        self.kartlar = [Kart(d, t) for t in self.TURLER for d in self.DEGERLER]
    
    def __repr__(self):
        return f"Deste ({len(self.kartlar)} kart)"

    def __len__(self):
        return len(self.kartlar)

    def __getitem__(self, pozisyon):
        return self.kartlar[pozisyon]
    
    def karistir(self):
        random.shuffle(self.kartlar)
        print("Deste karıştırıldı.")
        
    def kart_cek(self):
        if len(self.kartlar) == 0:
            return None
        return self.kartlar.pop()

# Örnek Kullanım
deste = Deste()
print(deste)
print(f"Destenin ilk kartı: {deste[0]}")
deste.karistir()
print(f"Karıştırıldıktan sonraki ilk kart: {deste[0]}")

print("\\n5 kart çekiliyor:")
for _ in range(5):
    print(f"- {deste.kart_cek()}")

print(f"\\nDestede kalan kart sayısı: {len(deste)}")
\`\`\`
    `,
  },
  {
    id: 28,
    title: 'Basit RPG Savaş Simülasyonu',
    difficulty: 'Zor',
    topics: ['Sınıflar', 'Kalıtım', 'Kompozisyon', 'Döngüler'],
    description: 'Bir `Kahraman` ve bir `Canavar` sınıfı oluşturun. Her ikisi de `Varlik` adında bir temel sınıftan türesin. Sırayla birbirlerine saldırdıkları basit bir savaş döngüsü yazın. Savaş, birinin canı sıfırın altına düşünceye kadar devam etsin.',
    solution: `
\`\`\`python
import random
import time

class Varlik:
    def __init__(self, ad, can, guc):
        self.ad = ad
        self.can = can
        self.guc = guc
    
    def saldir(self, hedef):
        hasar = random.randint(self.guc // 2, self.guc)
        print(f"{self.ad}, {hedef.ad}'a saldırıyor ve {hasar} hasar veriyor!")
        hedef.can -= hasar
    
    @property
    def hayatta_mi(self):
        return self.can > 0

class Kahraman(Varlik):
    pass

class Canavar(Varlik):
    pass

# Savaş Simülasyonu
kahraman = Kahraman("Aragon", 100, 20)
canavar = Canavar("Ork", 80, 15)

tur = 1
while kahraman.hayatta_mi and canavar.hayatta_mi:
    print(f"\\n--- TUR {tur} ---")
    print(f"{kahraman.ad}: {kahraman.can} CAN | {canavar.ad}: {canavar.can} CAN")
    
    # Kahraman saldırır
    kahraman.saldir(canavar)
    if not canavar.hayatta_mi:
        break
    
    time.sleep(1) # Savaşın akışını görmek için bekle
    
    # Canavar saldırır
    canavar.saldir(kahraman)
    
    time.sleep(1)
    tur += 1

print("\\n--- SAVAŞ SONUCU ---")
if kahraman.hayatta_mi:
    print(f"🎉 {kahraman.ad} savaşı kazandı!")
else:
    print(f"☠️ {canavar.ad} savaşı kazandı!")
\`\`\`
    `,
  }
];