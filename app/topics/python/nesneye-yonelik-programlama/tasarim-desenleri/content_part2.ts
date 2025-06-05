export const content_part2 = `
## Yapısal Desenler (Structural Patterns)

Yapısal desenler, sınıfların ve nesnelerin daha büyük yapılar oluşturmak için nasıl bir araya getirileceğiyle ilgilenir. Bu desenler, sistemin yapısını daha esnek ve verimli hale getirmeye yardımcı olur.

### 1. Adapter (Adaptör) Deseni

Adapter deseni, uyumsuz arayüzlere sahip sınıfların birlikte çalışmasını sağlar. Bu desen, var olan bir sınıfın arayüzünü başka bir arayüze dönüştürerek, istemcinin beklediği arayüzü sağlar.

\`\`\`python
# Hedef arayüz
class OdemeIslemcisi:
    def odeme_yap(self, miktar):
        pass

# Adaptee: Uyarlanacak sınıf
class PayPalAPI:
    def paypal_odeme(self, tutar, para_birimi="TRY"):
        print(f"PayPal ile {tutar} {para_birimi} ödeme yapıldı")
        return True

# Adapter: PayPal'ı OdemeIslemcisi arayüzüne uyarlar
class PayPalAdapter(OdemeIslemcisi):
    def __init__(self, paypal_api):
        self.paypal = paypal_api
    
    def odeme_yap(self, miktar):
        return self.paypal.paypal_odeme(miktar)

# Adaptee: Uyarlanacak başka bir sınıf
class StripeAPI:
    def stripe_ile_ode(self, tutar_cent):
        # Stripe kuruş cinsinden çalışır
        print(f"Stripe ile {tutar_cent/100} TL ödeme yapıldı")
        return {"success": True, "transaction_id": "123456"}

# Adapter: Stripe'ı OdemeIslemcisi arayüzüne uyarlar
class StripeAdapter(OdemeIslemcisi):
    def __init__(self, stripe_api):
        self.stripe = stripe_api
    
    def odeme_yap(self, miktar):
        # TL'yi kuruşa çevir
        cent_amount = int(miktar * 100)
        result = self.stripe.stripe_ile_ode(cent_amount)
        return result["success"]

# İstemci kodu
def siparis_tamamla(odeme_islemcisi, miktar):
    print(f"{miktar} TL tutarında sipariş işleniyor...")
    if odeme_islemcisi.odeme_yap(miktar):
        print("Ödeme başarılı, sipariş tamamlandı")
    else:
        print("Ödeme başarısız")

# Kullanım
paypal = PayPalAdapter(PayPalAPI())
stripe = StripeAdapter(StripeAPI())

siparis_tamamla(paypal, 100)
siparis_tamamla(stripe, 200)
\`\`\`

### 2. Bridge (Köprü) Deseni

Bridge deseni, soyutlama ile uygulamayı birbirinden ayırarak, ikisinin bağımsız olarak değişebilmesini sağlar. Bu desen, "soyutlamadan uygulamaya" olan bağımlılığı azaltır.

\`\`\`python
from abc import ABC, abstractmethod

# Implementor: Uygulama arayüzü
class RenderMotoru(ABC):
    @abstractmethod
    def ciz_metin(self, metin, pozisyon):
        pass
    
    @abstractmethod
    def ciz_sekil(self, sekil_tipi, pozisyon, boyut):
        pass

# Concrete Implementors: Somut uygulama sınıfları
class OpenGLRenderer(RenderMotoru):
    def ciz_metin(self, metin, pozisyon):
        print(f"OpenGL ile '{metin}' metni {pozisyon} konumuna çizildi")
    
    def ciz_sekil(self, sekil_tipi, pozisyon, boyut):
        print(f"OpenGL ile {sekil_tipi} şekli {pozisyon} konumunda {boyut} boyutunda çizildi")

class DirectXRenderer(RenderMotoru):
    def ciz_metin(self, metin, pozisyon):
        print(f"DirectX ile '{metin}' metni {pozisyon} konumuna çizildi")
    
    def ciz_sekil(self, sekil_tipi, pozisyon, boyut):
        print(f"DirectX ile {sekil_tipi} şekli {pozisyon} konumunda {boyut} boyutunda çizildi")

# Abstraction: Soyutlama
class UIBilesen(ABC):
    def __init__(self, render_motoru):
        self.render_motoru = render_motoru
    
    @abstractmethod
    def ciz(self):
        pass

# Refined Abstractions: İyileştirilmiş soyutlamalar
class Buton(UIBilesen):
    def __init__(self, render_motoru, metin, pozisyon):
        super().__init__(render_motoru)
        self.metin = metin
        self.pozisyon = pozisyon
    
    def ciz(self):
        self.render_motoru.ciz_sekil("dikdörtgen", self.pozisyon, (100, 30))
        self.render_motoru.ciz_metin(self.metin, (self.pozisyon[0] + 10, self.pozisyon[1] + 10))

class Pencere(UIBilesen):
    def __init__(self, render_motoru, baslik, pozisyon, boyut):
        super().__init__(render_motoru)
        self.baslik = baslik
        self.pozisyon = pozisyon
        self.boyut = boyut
    
    def ciz(self):
        self.render_motoru.ciz_sekil("dikdörtgen", self.pozisyon, self.boyut)
        self.render_motoru.ciz_metin(self.baslik, (self.pozisyon[0] + 5, self.pozisyon[1] + 5))

# Kullanım
opengl = OpenGLRenderer()
directx = DirectXRenderer()

buton_opengl = Buton(opengl, "Tamam", (100, 100))
buton_directx = Buton(directx, "Tamam", (100, 100))

pencere_opengl = Pencere(opengl, "Ana Pencere", (50, 50), (400, 300))
pencere_directx = Pencere(directx, "Ana Pencere", (50, 50), (400, 300))

buton_opengl.ciz()
buton_directx.ciz()
pencere_opengl.ciz()
pencere_directx.ciz()
\`\`\`

### 3. Composite (Bileşik) Deseni

Composite deseni, nesneleri ağaç yapıları halinde düzenleyerek, tek nesneler ve nesne kompozisyonlarına aynı şekilde davranılmasını sağlar. Bu desen, parça-bütün hiyerarşilerini temsil etmek için kullanılır.

\`\`\`python
from abc import ABC, abstractmethod

# Component: Temel bileşen arayüzü
class DosyaSistemiOgesi(ABC):
    def __init__(self, isim):
        self.isim = isim
    
    @abstractmethod
    def boyut(self):
        pass
    
    @abstractmethod
    def yazdir(self, seviye=0):
        pass

# Leaf: Basit bileşenler (yapraklar)
class Dosya(DosyaSistemiOgesi):
    def __init__(self, isim, boyut_bayt):
        super().__init__(isim)
        self._boyut = boyut_bayt
    
    def boyut(self):
        return self._boyut
    
    def yazdir(self, seviye=0):
        print("  " * seviye + f"- {self.isim} ({self._boyut} bayt)")

# Composite: Bileşik nesneler
class Klasor(DosyaSistemiOgesi):
    def __init__(self, isim):
        super().__init__(isim)
        self.cocuklar = []
    
    def ekle(self, ogesi):
        self.cocuklar.append(ogesi)
    
    def cikar(self, ogesi):
        self.cocuklar.remove(ogesi)
    
    def boyut(self):
        toplam_boyut = 0
        for cocuk in self.cocuklar:
            toplam_boyut += cocuk.boyut()
        return toplam_boyut
    
    def yazdir(self, seviye=0):
        print("  " * seviye + f"+ {self.isim} ({self.boyut()} bayt)")
        for cocuk in self.cocuklar:
            cocuk.yazdir(seviye + 1)

# Kullanım
# Dosyalar oluştur
belge1 = Dosya("rapor.docx", 2048)
belge2 = Dosya("sunum.pptx", 4096)
resim1 = Dosya("logo.png", 1024)
resim2 = Dosya("arkaplan.jpg", 3072)

# Klasörler oluştur
belgeler = Klasor("Belgeler")
belgeler.ekle(belge1)
belgeler.ekle(belge2)

resimler = Klasor("Resimler")
resimler.ekle(resim1)
resimler.ekle(resim2)

proje = Klasor("Proje")
proje.ekle(belgeler)
proje.ekle(resimler)

# Dosya yapısını görüntüle
proje.yazdir()
\`\`\`

### 4. Decorator (Dekoratör) Deseni

Decorator deseni, bir nesneye dinamik olarak ek sorumluluklar eklemek için kullanılır. Bu desen, alt sınıflama yapmadan nesnelerin işlevselliğini genişletmeyi sağlar.

Python'da dekoratör deseni, Python'un yerleşik dekoratör sözdizimi (@decorator) ile karıştırılmamalıdır, ancak aynı temel prensibe dayanır.

\`\`\`python
from abc import ABC, abstractmethod

# Component: Temel arayüz
class Metin(ABC):
    @abstractmethod
    def formatla(self):
        pass

# ConcreteComponent: Temel sınıf
class BasitMetin(Metin):
    def __init__(self, icerik):
        self.icerik = icerik
    
    def formatla(self):
        return self.icerik

# Decorator: Temel dekoratör sınıfı
class MetinDekorator(Metin):
    def __init__(self, dekorlanmis_metin):
        self.dekorlanmis_metin = dekorlanmis_metin
    
    def formatla(self):
        return self.dekorlanmis_metin.formatla()

# ConcreteDecorator: Somut dekoratörler
class KalinMetin(MetinDekorator):
    def formatla(self):
        return f"<b>{self.dekorlanmis_metin.formatla()}</b>"

class ItalikMetin(MetinDekorator):
    def formatla(self):
        return f"<i>{self.dekorlanmis_metin.formatla()}</i>"

class AltCiziliMetin(MetinDekorator):
    def formatla(self):
        return f"<u>{self.dekorlanmis_metin.formatla()}</u>"

# Kullanım
basit_metin = BasitMetin("Merhaba Dünya")
kalin_metin = KalinMetin(basit_metin)
italik_metin = ItalikMetin(basit_metin)
kalin_ve_italik = ItalikMetin(KalinMetin(basit_metin))
tum_formatlama = AltCiziliMetin(ItalikMetin(KalinMetin(basit_metin)))

print(basit_metin.formatla())        # "Merhaba Dünya"
print(kalin_metin.formatla())        # "<b>Merhaba Dünya</b>"
print(italik_metin.formatla())       # "<i>Merhaba Dünya</i>"
print(kalin_ve_italik.formatla())    # "<i><b>Merhaba Dünya</b></i>"
print(tum_formatlama.formatla())     # "<u><i><b>Merhaba Dünya</b></i></u>"
\`\`\`

::: tip
Python'un kendi dekoratör sözdizimi, işlev ve sınıf davranışını değiştirmek için kullanılır ve bu desenle benzer fikre dayanır.
:::

### 5. Facade (Ön Yüz) Deseni

Facade deseni, karmaşık bir alt sistemin daha basit bir arayüzünü sağlar. Bu desen, istemcilerin alt sistemle doğrudan etkileşimini azaltarak karmaşıklığı gizler.

\`\`\`python
# Alt sistem sınıfları
class VideoOkuyucu:
    def dosya_ac(self, dosya_yolu):
        print(f"Video dosyası açıldı: {dosya_yolu}")
        return True
    
    def video_bilgisi_al(self):
        return {"codec": "H.264", "çözünürlük": "1920x1080", "fps": 30}

class SesIslemci:
    def sesi_ayikla(self, video_yolu):
        print(f"{video_yolu} dosyasından ses ayıklandı")
        return "ses_temp.mp3"
    
    def ses_formatla(self, ses_yolu, format):
        print(f"{ses_yolu} dosyası {format} formatına dönüştürüldü")
        return f"ses.{format}"

class VideoIslemci:
    def video_kenarlik_ekle(self, video_yolu, genislik=10):
        print(f"{video_yolu} videosuna {genislik}px kenarlık eklendi")
    
    def video_filtre_uygula(self, video_yolu, filtre):
        print(f"{video_yolu} videosuna {filtre} filtresi uygulandı")

class AltyaziIslemci:
    def altyazi_ekle(self, video_yolu, altyazi_yolu):
        print(f"{video_yolu} videosuna {altyazi_yolu} altyazısı eklendi")

class VideoKayitci:
    def kaydet(self, video_yolu, cikti_yolu, kalite="yüksek"):
        print(f"{video_yolu} işlendi ve {cikti_yolu} dosyasına {kalite} kalitede kaydedildi")
        return cikti_yolu

# Facade: Ön yüz sınıfı
class VideoEditoru:
    def __init__(self):
        self.okuyucu = VideoOkuyucu()
        self.ses_islemci = SesIslemci()
        self.video_islemci = VideoIslemci()
        self.altyazi_islemci = AltyaziIslemci()
        self.kayitci = VideoKayitci()
    
    def video_isleme(self, video_yolu, cikti_yolu, filtre=None, altyazi=None, ses_format="mp3"):
        """Tüm video işleme adımlarını basit bir arayüz ile sunar"""
        print("Video işleme başlatıldı...")
        
        # Video aç
        self.okuyucu.dosya_ac(video_yolu)
        video_bilgisi = self.okuyucu.video_bilgisi_al()
        print(f"Video bilgisi: {video_bilgisi}")
        
        # Video işle
        if filtre:
            self.video_islemci.video_filtre_uygula(video_yolu, filtre)
        
        # Altyazı ekle
        if altyazi:
            self.altyazi_islemci.altyazi_ekle(video_yolu, altyazi)
        
        # Ses işle
        ses_yolu = self.ses_islemci.sesi_ayikla(video_yolu)
        islenenmis_ses = self.ses_islemci.ses_formatla(ses_yolu, ses_format)
        
        # Kaydet
        sonuc = self.kayitci.kaydet(video_yolu, cikti_yolu)
        print(f"Video işleme tamamlandı: {sonuc}")
        return sonuc

# Kullanım
editor = VideoEditoru()

# Karmaşık işlemleri basit bir arayüz ile çağırma
editor.video_isleme(
    video_yolu="input.mp4",
    cikti_yolu="output.mp4",
    filtre="vintage",
    altyazi="altyazi.srt"
)
\`\`\`

### 6. Flyweight (Sineksiklet) Deseni

Flyweight deseni, çok sayıda benzer nesnenin verimli bir şekilde kullanılmasını sağlar. Bu desen, nesneler arasında ortak durumları paylaştırarak bellek kullanımını azaltır.

\`\`\`python
import random

# Flyweight: Paylaşılan nesne
class Karakter:
    def __init__(self, karakter, font, size):
        self.karakter = karakter
        self.font = font
        self.size = size
        # Karakter glifinin render edilmesi gibi maliyetli işlemler burada yapılır
        print(f"'{karakter}' karakteri '{font}' fontu ile {size}pt boyutunda oluşturuldu")
    
    def render(self, x, y, renk):
        print(f"'{self.karakter}' karakteri ({x},{y}) konumunda {renk} renkte gösteriliyor")

# FlyweightFactory: Flyweight nesnelerini yönetir ve paylaştırır
class KarakterFactory:
    def __init__(self):
        self.karakterler = {}
    
    def get_karakter(self, karakter, font="Arial", size=12):
        # Anahtar oluştur
        key = f"{karakter}-{font}-{size}"
        
        # Eğer karakter daha önce oluşturulmadıysa, oluştur ve depola
        if key not in self.karakterler:
            self.karakterler[key] = Karakter(karakter, font, size)
        
        return self.karakterler[key]
    
    def get_karakter_sayisi(self):
        return len(self.karakterler)

# Client: Dış durum (extrinsic state) ile Flyweight'leri kullanır
class MetinEditoru:
    def __init__(self):
        self.karakter_factory = KarakterFactory()
        self.karakter_konumlari = []
    
    def metin_ekle(self, metin, x, y, font="Arial", size=12):
        renk_secenekleri = ["siyah", "mavi", "kırmızı", "yeşil"]
        
        for i, karakter_deger in enumerate(metin):
            # Her karakterin rengi dış durumdur (extrinsic)
            renk = random.choice(renk_secenekleri)
            
            # Flyweight nesnesini al
            karakter = self.karakter_factory.get_karakter(karakter_deger, font, size)
            
            # Dış durumu kaydet
            self.karakter_konumlari.append({
                "karakter": karakter,
                "x": x + i * 10,  # Her karakteri yanyana yerleştir
                "y": y,
                "renk": renk
            })
    
    def render(self):
        for konum in self.karakter_konumlari:
            konum["karakter"].render(konum["x"], konum["y"], konum["renk"])
    
    def istatistikler(self):
        toplam_karakter = len(self.karakter_konumlari)
        paylasilan_karakter = self.karakter_factory.get_karakter_sayisi()
        tasarruf = (1 - paylasilan_karakter / toplam_karakter) * 100 if toplam_karakter > 0 else 0
        
        print(f"Toplam karakter sayısı: {toplam_karakter}")
        print(f"Benzersiz karakter sayısı: {paylasilan_karakter}")
        print(f"Bellek tasarrufu: %{tasarruf:.2f}")

# Kullanım
editor = MetinEditoru()

# Metin ekle
editor.metin_ekle("Merhaba Dünya", 10, 10)
editor.metin_ekle("Python Tasarım Desenleri", 10, 30)
editor.metin_ekle("Flyweight Deseni", 10, 50)

# Render et
editor.render()

# İstatistikler
editor.istatistikler()
\`\`\`

### 7. Proxy (Vekil) Deseni

Proxy deseni, başka bir nesneye erişimi kontrol etmek için kullanılır. Bu desen, ek işlevsellik sağlamak veya erişimi düzenlemek için araya girer.

\`\`\`python
from abc import ABC, abstractmethod

# Subject: Ortak arayüz
class GorselNesne(ABC):
    @abstractmethod
    def goruntule(self):
        pass

# RealSubject: Gerçek nesne
class YuksekCozunurlukluResim(GorselNesne):
    def __init__(self, dosya_yolu):
        self.dosya_yolu = dosya_yolu
        self.yukle()
    
    def yukle(self):
        # Yüksek çözünürlüklü resim yükleme simülasyonu
        print(f"Yüksek çözünürlüklü resim yükleniyor: {self.dosya_yolu}")
        # Gerçek uygulamada bu işlem çok bellek ve zaman alabilir
        print("Resim yüklendi.")
    
    def goruntule(self):
        print(f"Yüksek çözünürlüklü resim gösteriliyor: {self.dosya_yolu}")

# Proxy: Vekil nesne
class ResimProxy(GorselNesne):
    def __init__(self, dosya_yolu):
        self.dosya_yolu = dosya_yolu
        self.resim = None
        self.yuklenme_durumu = "Yüklenmedi"
    
    def goruntule(self):
        # Lazy loading (tembel yükleme): Resim sadece gerektiğinde yüklenir
        if self.resim is None:
            print("Önce düşük çözünürlüklü önizleme gösteriliyor...")
            print(f"Tam çözünürlüklü resim arka planda yükleniyor: {self.dosya_yolu}")
            self.yuklenme_durumu = "Yükleniyor"
            
            # Gerçek resmi yükle
            self.resim = YuksekCozunurlukluResim(self.dosya_yolu)
            self.yuklenme_durumu = "Yüklendi"
        
        # Gerçek nesnenin metodunu çağır
        self.resim.goruntule()

# Koruyucu Proxy örneği
class KorunmusResimProxy(GorselNesne):
    def __init__(self, dosya_yolu, kullanici, rol):
        self.dosya_yolu = dosya_yolu
        self.kullanici = kullanici
        self.rol = rol
        self.resim = None
    
    def goruntule(self):
        # Erişim kontrolü
        if self.rol not in ["admin", "editor"]:
            print(f"Erişim reddedildi. '{self.kullanici}' kullanıcısının '{self.dosya_yolu}' dosyasına erişim yetkisi yok.")
            return
        
        # Lazy loading
        if self.resim is None:
            self.resim = YuksekCozunurlukluResim(self.dosya_yolu)
        
        # Gerçek nesnenin metodunu çağır
        self.resim.goruntule()

# Kullanım
def galeri_gez(gorsel_nesneler):
    print("Galeri görüntüleniyor...")
    for i, nesne in enumerate(gorsel_nesneler):
        print(f"\nResim {i+1}:")
        nesne.goruntule()
        print("-" * 40)

# Normal Proxy kullanımı
resimler = [
    ResimProxy("resim1.jpg"),
    ResimProxy("resim2.jpg"),
    ResimProxy("resim3.jpg")
]

galeri_gez(resimler)

# Koruyucu Proxy kullanımı
korunmus_resimler = [
    KorunmusResimProxy("gizli_resim1.jpg", "kullanici1", "misafir"),
    KorunmusResimProxy("gizli_resim2.jpg", "kullanici2", "admin"),
    KorunmusResimProxy("gizli_resim3.jpg", "kullanici3", "editor")
]

galeri_gez(korunmus_resimler)
\`\`\`
`; 