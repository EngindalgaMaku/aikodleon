export const content = `
# Python'da Tasarım Desenleri

Tasarım desenleri, yazılım geliştirmede karşılaşılan yaygın problemlere yönelik test edilmiş ve kanıtlanmış çözüm şablonlarıdır. Bu desenler, kodun sürdürülebilirliğini, okunabilirliğini ve esnekliğini artırır.

## Tasarım Desenleri Nedir?

Tasarım desenleri, yazılım geliştirme sürecinde karşılaşılan yaygın problemlere yönelik yeniden kullanılabilir çözümlerdir. Bu desenler, "Dörtlü Çete" (Gang of Four - GoF) olarak bilinen Erich Gamma, Richard Helm, Ralph Johnson ve John Vlissides tarafından sistematize edilmiştir.

Tasarım desenleri üç ana kategoriye ayrılır:

1. **Yaratımsal Desenler (Creational Patterns)**: Nesne yaratma mekanizmalarıyla ilgilenir.
2. **Yapısal Desenler (Structural Patterns)**: Sınıfların ve nesnelerin bir araya getirilmesiyle ilgilenir.
3. **Davranışsal Desenler (Behavioral Patterns)**: Nesneler arasındaki iletişim ve sorumluluk dağılımıyla ilgilenir.

## Yaratımsal Desenler (Creational Patterns)

Yaratımsal desenler, nesne oluşturma sürecini soyutlaştırarak sistemin hangi nesneleri nasıl oluşturduğundan bağımsız hale gelmesini sağlar.

### 1. Singleton (Tekil) Deseni

Singleton deseni, bir sınıfın yalnızca bir örneğinin olmasını ve bu örneğe global erişim noktası sağlamayı amaçlar. Veritabanı bağlantıları, log yöneticileri veya yapılandırma yöneticileri gibi sistem genelinde tek bir örnek olması gereken nesneler için kullanılır.

\`\`\`python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls)
            cls._instance.value = None
        return cls._instance

# Kullanım
s1 = Singleton()
s1.value = "Singleton Örneği"
s2 = Singleton()
print(s1.value)  # "Singleton Örneği"
print(s2.value)  # "Singleton Örneği"
print(s1 is s2)  # True - Aynı nesne
\`\`\`

Singleton deseninin modern Python'da alternatif bir uygulaması:

\`\`\`python
class Singleton:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

class VeriTabaniBaglantisi(Singleton):
    def __init__(self, host=None, port=None, user=None, password=None):
        # __init__ her nesne oluşumunda çağrılır, bu nedenle
        # değerlerin sadece bir kez atanmasını sağlamak için kontrol ederiz
        if not hasattr(self, 'initialized'):
            self.host = host or "localhost"
            self.port = port or 5432
            self.user = user or "postgres"
            self.password = password or "123456"
            self.connection = None
            self.initialized = True
    
    def connect(self):
        if not self.connection:
            print(f"Bağlanılıyor: {self.host}:{self.port}")
            # Gerçek bağlantı kodu burada olacak
            self.connection = True
        return self.connection

# Kullanım
db1 = VeriTabaniBaglantisi(host="database.example.com")
db2 = VeriTabaniBaglantisi()  # Parametreler görmezden gelinecek

print(db1.host)      # "database.example.com"
print(db2.host)      # "database.example.com" (aynı nesne)
print(db1 is db2)    # True
\`\`\`

::: warning
Thread-safe (iş parçacığı güvenli) Singleton oluşturmak için ek önlemler gerekebilir.
:::

### 2. Factory Method (Fabrika Metodu) Deseni

Factory Method deseni, bir nesne oluşturma işlemini alt sınıflara devrederek, hangi sınıfın örnekleneceğine alt sınıfların karar vermesini sağlar. Bu desen, nesne oluşturma mantığını kullanım mantığından ayırmak için kullanılır.

\`\`\`python
from abc import ABC, abstractmethod

# Ürün arayüzü
class Belge(ABC):
    @abstractmethod
    def olustur(self):
        pass

# Somut ürünler
class PDFBelge(Belge):
    def olustur(self):
        return "PDF belgesi oluşturuldu"

class WordBelge(Belge):
    def olustur(self):
        return "Word belgesi oluşturuldu"

class ExcelBelge(Belge):
    def olustur(self):
        return "Excel belgesi oluşturuldu"

# Creator sınıfı
class BelgeOlusturucu(ABC):
    @abstractmethod
    def belge_olustur(self) -> Belge:
        pass
    
    def islem(self):
        belge = self.belge_olustur()
        return belge.olustur()

# Concrete Creator sınıfları
class PDFOlusturucu(BelgeOlusturucu):
    def belge_olustur(self) -> Belge:
        return PDFBelge()

class WordOlusturucu(BelgeOlusturucu):
    def belge_olustur(self) -> Belge:
        return WordBelge()

class ExcelOlusturucu(BelgeOlusturucu):
    def belge_olustur(self) -> Belge:
        return ExcelBelge()

# Kullanım
def belge_olusturma_islemi(olusturucu: BelgeOlusturucu):
    return olusturucu.islem()

# Farklı belge oluşturucular ile kullanım
pdf_sonuc = belge_olusturma_islemi(PDFOlusturucu())
word_sonuc = belge_olusturma_islemi(WordOlusturucu())
excel_sonuc = belge_olusturma_islemi(ExcelOlusturucu())

print(pdf_sonuc)    # "PDF belgesi oluşturuldu"
print(word_sonuc)   # "Word belgesi oluşturuldu"
print(excel_sonuc)  # "Excel belgesi oluşturuldu"
\`\`\`

### 3. Abstract Factory (Soyut Fabrika) Deseni

Abstract Factory deseni, somut sınıflarını belirtmeden ilişkili nesne ailelerini oluşturmak için bir arayüz sağlar. Bu desen, ürün ailelerinin bir arada çalışması gerektiğinde kullanışlıdır.

\`\`\`python
from abc import ABC, abstractmethod

# Soyut ürünler
class Buton(ABC):
    @abstractmethod
    def render(self):
        pass

class Pencere(ABC):
    @abstractmethod
    def render(self):
        pass

# Windows ürün ailesi
class WindowsButon(Buton):
    def render(self):
        return "Windows butonu render edildi"

class WindowsPencere(Pencere):
    def render(self):
        return "Windows penceresi render edildi"

# MacOS ürün ailesi
class MacOSButon(Buton):
    def render(self):
        return "MacOS butonu render edildi"

class MacOSPencere(Pencere):
    def render(self):
        return "MacOS penceresi render edildi"

# Soyut fabrika
class UIFactory(ABC):
    @abstractmethod
    def buton_olustur(self) -> Buton:
        pass
    
    @abstractmethod
    def pencere_olustur(self) -> Pencere:
        pass

# Somut fabrikalar
class WindowsFactory(UIFactory):
    def buton_olustur(self) -> Buton:
        return WindowsButon()
    
    def pencere_olustur(self) -> Pencere:
        return WindowsPencere()

class MacOSFactory(UIFactory):
    def buton_olustur(self) -> Buton:
        return MacOSButon()
    
    def pencere_olustur(self) -> Pencere:
        return MacOSPencere()

# Kullanım
def uygulama_olustur(factory: UIFactory):
    buton = factory.buton_olustur()
    pencere = factory.pencere_olustur()
    return {
        "buton": buton.render(),
        "pencere": pencere.render()
    }

# İşletim sistemine göre uygun fabrikayı seç
import platform

def fabrika_sec():
    os_name = platform.system()
    if os_name == "Windows":
        return WindowsFactory()
    elif os_name in ["Darwin", "macOS"]:
        return MacOSFactory()
    else:
        # Varsayılan olarak Windows
        return WindowsFactory()

# Kullanım
fabrika = fabrika_sec()
sonuclar = uygulama_olustur(fabrika)
print(sonuclar["buton"])
print(sonuclar["pencere"])
\`\`\`

### 4. Builder (İnşaatçı) Deseni

Builder deseni, karmaşık nesnelerin adım adım oluşturulmasını sağlar. Bu desen, bir nesnenin farklı gösterimlerini oluşturmak ve oluşturma sürecini kontrol etmek için kullanılır.

\`\`\`python
class Bilgisayar:
    def __init__(self):
        self.islemci = None
        self.bellek = None
        self.grafik_karti = None
        self.depolama = None
    
    def __str__(self):
        return f"Bilgisayar: {self.islemci} İşlemci, {self.bellek} RAM, {self.grafik_karti} Grafik, {self.depolama} Depolama"

class BilgisayarBuilder:
    def __init__(self):
        self.bilgisayar = Bilgisayar()
    
    def islemci_ekle(self, islemci):
        self.bilgisayar.islemci = islemci
        return self
    
    def bellek_ekle(self, bellek):
        self.bilgisayar.bellek = bellek
        return self
    
    def grafik_karti_ekle(self, grafik_karti):
        self.bilgisayar.grafik_karti = grafik_karti
        return self
    
    def depolama_ekle(self, depolama):
        self.bilgisayar.depolama = depolama
        return self
    
    def build(self):
        return self.bilgisayar

# Kullanım
bilgisayar = BilgisayarBuilder() \\
    .islemci_ekle("Intel i7") \\
    .bellek_ekle("16GB") \\
    .grafik_karti_ekle("NVIDIA RTX 3060") \\
    .depolama_ekle("1TB SSD") \\
    .build()

print(bilgisayar)  # "Bilgisayar: Intel i7 İşlemci, 16GB RAM, NVIDIA RTX 3060 Grafik, 1TB SSD Depolama"
\`\`\`

::: tip
Builder deseninde zincirleme metodlar (method chaining) kullanmak, okunabilirliği artırır.
:::

### 5. Prototype (Prototip) Deseni

Prototype deseni, mevcut nesnelerin kopyalarını oluşturarak yeni nesneler oluşturmayı sağlar. Bu desen, nesne oluşturmanın maliyetli olduğu durumlarda kullanışlıdır.

\`\`\`python
import copy

class Prototip:
    def __init__(self):
        self.nesneler = {}
    
    def kaydet(self, isim, nesne):
        self.nesneler[isim] = nesne
    
    def sil(self, isim):
        del self.nesneler[isim]
    
    def kopyala(self, isim, **attrs):
        """Varolan bir nesnenin derin kopyasını oluşturur ve istenirse özelliklerini günceller."""
        if isim not in self.nesneler:
            raise ValueError(f"{isim} isimli nesne bulunamadı")
        
        # Derin kopya oluştur
        kopya = copy.deepcopy(self.nesneler[isim])
        
        # Özellikleri güncelle
        for key, value in attrs.items():
            setattr(kopya, key, value)
        
        return kopya

class Belge:
    def __init__(self, baslik="", icerik="", stil=None):
        self.baslik = baslik
        self.icerik = icerik
        self.stil = stil or {}
    
    def __str__(self):
        return f"Belge: {self.baslik}\\nİçerik: {self.icerik}\\nStil: {self.stil}"

# Kullanım
prototip_yonetici = Prototip()

# Temel belge şablonları oluştur
rapor_sablonu = Belge(
    baslik="Rapor",
    icerik="Rapor içeriği...",
    stil={"font": "Arial", "size": 12, "margins": "2.5cm"}
)

makale_sablonu = Belge(
    baslik="Makale",
    icerik="Makale içeriği...",
    stil={"font": "Times New Roman", "size": 11, "margins": "2cm"}
)

# Şablonları kaydet
prototip_yonetici.kaydet("rapor", rapor_sablonu)
prototip_yonetici.kaydet("makale", makale_sablonu)

# Şablonlardan yeni belgeler oluştur
yeni_rapor = prototip_yonetici.kopyala("rapor", baslik="2023 Yılsonu Raporu", icerik="2023 yılı finansal sonuçları...")
yeni_makale = prototip_yonetici.kopyala("makale", baslik="Python ve Tasarım Desenleri", icerik="Python'da tasarım desenleri...")

print(yeni_rapor)
print("\\n")
print(yeni_makale)
\`\`\`
`; 