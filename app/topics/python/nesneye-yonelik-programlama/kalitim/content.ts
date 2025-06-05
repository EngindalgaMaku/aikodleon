export const content = `
# Python'da Kalıtım (Inheritance)

Kalıtım, nesne yönelimli programlamanın temel prensiplerinden biridir. Bir sınıfın başka bir sınıfın özelliklerini ve davranışlarını miras almasını sağlar.
Bu sayede kod tekrarını önler ve sınıflar arasında hiyerarşik bir ilişki kurulmasını sağlar.

::: info
# 🎯 Kalıtımın Avantajları

* **Kod Tekrarını Önleme:** Ortak özellikleri temel sınıfta tanımlayarak kod tekrarını önler.
* **Hiyerarşik Yapı:** Sınıflar arasında mantıksal bir hiyerarşi oluşturur.
* **Kodun Yeniden Kullanılabilirliği:** Var olan kodun yeni sınıflarda kullanılmasını sağlar.
* **Genişletilebilirlik:** Mevcut sınıfları değiştirmeden yeni özellikler eklenebilir.
:::

## Temel Kalıtım

Bir sınıftan türetme yapmak için, yeni sınıf tanımında parantez içinde temel sınıfı belirtiriz. Temel kalıtımda, bir alt sınıf (derived class) bir üst sınıftan (base class) türetilir.

::: tip
# 💡 Temel Kalıtım İpuçları

* Alt sınıf, üst sınıfın tüm public özelliklerine erişebilir
* Alt sınıf, üst sınıfın metodlarını override edebilir
* Alt sınıf, kendi özel metodlarını ve özelliklerini ekleyebilir
:::

### Örnek: Hayvan Sınıfı Hiyerarşisi

\`\`\`python
class Hayvan:
    def __init__(self, isim, yas):
        self.isim = isim
        self.yas = yas
    
    def ses_cikar(self):
        pass
    
    def bilgi_goster(self):
        return f"{self.isim} {self.yas} yaşında"

class Kopek(Hayvan):
    def __init__(self, isim, yas, tur):
        super().__init__(isim, yas)  # Üst sınıfın constructor'ını çağır
        self.tur = tur  # Köpeğe özel özellik
    
    def ses_cikar(self):
        return "Hav hav!"
    
    def bilgi_goster(self):
        return f"{super().bilgi_goster()} ve {self.tur} türünde bir köpek"

class Kedi(Hayvan):
    def __init__(self, isim, yas, renk):
        super().__init__(isim, yas)
        self.renk = renk  # Kediye özel özellik
    
    def ses_cikar(self):
        return "Miyav!"
    
    def bilgi_goster(self):
        return f"{super().bilgi_goster()} ve {self.renk} renkli bir kedi"

# Kullanım örneği
kopek = Kopek("Karabaş", 3, "Golden")
kedi = Kedi("Pamuk", 2, "Beyaz")

print(kopek.bilgi_goster())  # "Karabaş 3 yaşında ve Golden türünde bir köpek"
print(kedi.bilgi_goster())   # "Pamuk 2 yaşında ve Beyaz renkli bir kedi"
\`\`\`

## super() Fonksiyonu

\`super()\` fonksiyonu, üst sınıfın metodlarını çağırmak için kullanılır. Bu fonksiyon özellikle constructor'ları ve override edilen metodları çağırırken çok kullanışlıdır.

::: info
# 🔍 super() Fonksiyonunun Avantajları

* Üst sınıfın metodlarına kolay erişim
* Kod tekrarını önleme
* Bakımı kolay kod yapısı
* Çoklu kalıtımda doğru metod çağrısı
:::

### Örnek: Öğrenci Bilgi Sistemi

\`\`\`python
class Ogrenci:
    def __init__(self, ad, soyad):
        self.ad = ad
        self.soyad = soyad
        self.dersler = []
    
    def ders_ekle(self, ders):
        self.dersler.append(ders)
    
    def bilgi_goster(self):
        return f"{self.ad} {self.soyad}"

class LiseOgrencisi(Ogrenci):
    def __init__(self, ad, soyad, sinif):
        super().__init__(ad, soyad)  # Üst sınıfın __init__ metodunu çağır
        self.sinif = sinif
        self.kulup = None
    
    def kulube_katil(self, kulup):
        self.kulup = kulup
    
    def bilgi_goster(self):
        temel_bilgi = super().bilgi_goster()  # Üst sınıfın metodunu çağır
        return f"{temel_bilgi} - {self.sinif}. Sınıf"

# Kullanım örneği
ogrenci = LiseOgrencisi("Ahmet", "Yılmaz", 10)
ogrenci.ders_ekle("Matematik")
ogrenci.ders_ekle("Fizik")
ogrenci.kulube_katil("Satranç Kulübü")

print(ogrenci.bilgi_goster())  # "Ahmet Yılmaz - 10. Sınıf"
print(f"Dersler: {', '.join(ogrenci.dersler)}")  # "Dersler: Matematik, Fizik"
print(f"Kulüp: {ogrenci.kulup}")  # "Kulüp: Satranç Kulübü"
\`\`\`

## Çoklu Kalıtım

Python'da bir sınıf birden fazla sınıftan türetilebilir. Bu özellik, farklı sınıfların özelliklerini tek bir sınıfta birleştirmemizi sağlar.

### Örnek: Cihaz Özellikleri

\`\`\`python
class WiFiCihaz:
    def __init__(self):
        self.wifi_bagli = False
    
    def wifi_baglan(self):
        self.wifi_bagli = True
        return "WiFi'ya bağlandı"
    
    def wifi_durum(self):
        return "Bağlı" if self.wifi_bagli else "Bağlı değil"

class BluetoothCihaz:
    def __init__(self):
        self.bluetooth_acik = False
        self.eslesme_listesi = []
    
    def bluetooth_ac(self):
        self.bluetooth_acik = True
        return "Bluetooth açıldı"
    
    def cihaz_esle(self, cihaz):
        if self.bluetooth_acik:
            self.eslesme_listesi.append(cihaz)
            return f"{cihaz} ile eşleşildi"
        return "Önce Bluetooth'u açın"

class AkilliTelefon(WiFiCihaz, BluetoothCihaz):
    def __init__(self, model):
        WiFiCihaz.__init__(self)
        BluetoothCihaz.__init__(self)
        self.model = model
    
    def durum_goster(self):
        return f"""
Model: {self.model}
WiFi: {self.wifi_durum()}
Bluetooth: {"Açık" if self.bluetooth_acik else "Kapalı"}
Eşleşilen Cihazlar: {", ".join(self.eslesme_listesi) if self.eslesme_listesi else "Yok"}
"""

# Kullanım örneği
telefon = AkilliTelefon("Galaxy S23")
print(telefon.wifi_baglan())
print(telefon.bluetooth_ac())
print(telefon.cihaz_esle("Kablosuz Kulaklık"))
print(telefon.durum_goster())
\`\`\`

::: warning
# ⚠️ Çoklu Kalıtımda Dikkat Edilecek Noktalar

* **Elmas Problemi:** Aynı metodun farklı üst sınıflarda farklı şekillerde tanımlanması durumu.
* **Karmaşıklık:** Çok sayıda üst sınıf kullanımı kodun anlaşılmasını zorlaştırabilir.
* **MRO (Method Resolution Order):** Python'ın metod arama sırasını anlamak önemlidir.
:::

## Method Resolution Order (MRO)

Python'da çoklu kalıtımda metodların aranma sırası MRO ile belirlenir. MRO, bir metodun hangi sınıftan çağrılacağını belirleyen sıralamadır.

### Örnek: MRO ve Elmas Problemi

\`\`\`python
class A:
    def kim(self):
        return "A"
    
    def selamla(self):
        return f"Selam, ben {self.kim()}"

class B(A):
    def kim(self):
        return "B"

class C(A):
    def kim(self):
        return "C"

class D(B, C):
    pass

d = D()
print(D.mro())  # MRO sırasını gösterir: [<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>]
print(d.kim())  # "B" (soldan sağa arama yapılır)
print(d.selamla())  # "Selam, ben B"
\`\`\`

## isinstance() ve issubclass()

Nesne ve sınıf ilişkilerini kontrol etmek için kullanılan bu fonksiyonlar, özellikle tip kontrolü ve kalıtım hiyerarşisini doğrulamada kullanışlıdır.

::: info
# 🔍 Tip Kontrolü İpuçları

* \`isinstance()\`: Bir nesnenin belirli bir sınıfın örneği olup olmadığını kontrol eder
* \`issubclass()\`: Bir sınıfın başka bir sınıfın alt sınıfı olup olmadığını kontrol eder
* Her iki fonksiyon da kalıtım hiyerarşisini takip eder
:::

### Örnek: Tip Kontrolü ve Güvenli Tür Dönüşümü

\`\`\`python
def hayvan_bilgisi(hayvan):
    if not isinstance(hayvan, Hayvan):
        raise TypeError("Bu nesne bir Hayvan değil!")
    
    bilgi = hayvan.bilgi_goster()
    
    if isinstance(hayvan, Kopek):
        bilgi += f"\\nKöpek türü: {hayvan.tur}"
    elif isinstance(hayvan, Kedi):
        bilgi += f"\\nKedi rengi: {hayvan.renk}"
    
    return bilgi

# Kullanım örnekleri
kopek = Kopek("Karabaş", 3, "Golden")
kedi = Kedi("Pamuk", 2, "Beyaz")
try:
    print(hayvan_bilgisi(kopek))
    print(hayvan_bilgisi(kedi))
    print(hayvan_bilgisi("Bu bir hayvan değil"))  # TypeError fırlatır
except TypeError as e:
    print(f"Hata: {e}")
\`\`\`

## Alıştırmalar

### 1. Çalışan Yönetim Sistemi

[Detaylı çözüm için tıklayın](/topics/python/nesneye-yonelik-programlama/kalitim/calisan-yonetim-sistemi)

Bir şirketin çalışan yönetim sistemini modelleyin:
* \`Calisan\` temel sınıfı
* \`Muhendis\`, \`Yonetici\`, \`Pazarlamaci\` gibi alt sınıflar
* Maaş hesaplama, izin takibi, proje atama gibi özellikler
* Departman bazlı raporlama sistemi

### 2. Oyun Karakter Sistemi

[Detaylı çözüm için tıklayın](/topics/python/nesneye-yonelik-programlama/kalitim/oyun-karakter-sistemi)

Bir RPG oyunu için karakter sistemi geliştirin:
* \`Karakter\` temel sınıfı
* \`Savasci\`, \`Buyucu\`, \`Okcu\` gibi alt sınıflar
* Yetenek sistemi ve seviye atlama
* Envanter yönetimi ve ekipman sistemi

### 3. Medya Oynatıcı Sistemi

[Detaylı çözüm için tıklayın](/topics/python/nesneye-yonelik-programlama/kalitim/medya-oynatici-sistemi)

Farklı medya türlerini destekleyen bir oynatıcı sistemi oluşturun:
* \`MedyaOynatici\` temel sınıfı
* \`MuzikOynatici\`, \`VideoOynatici\`, \`PodcastOynatici\` alt sınıfları
* Çalma listesi yönetimi
* Format dönüştürme ve kalite ayarları

::: tip
# 💡 Kalıtım Kullanırken Dikkat Edilecek Noktalar

* **IS-A İlişkisi:** Kalıtım kullanırken "is-a" ilişkisinin varlığından emin olun.
* **Kompozisyon vs Kalıtım:** Bazen kalıtım yerine kompozisyon kullanmak daha uygun olabilir.
* **Liskov Substitution Prensibi:** Alt sınıflar, üst sınıfların yerine kullanılabilmelidir.
* **DRY Prensibi:** Kendini tekrar eden kodları ortak bir üst sınıfa taşıyın.
* **SOLID Prensipleri:** Kalıtım hiyerarşisini tasarlarken SOLID prensiplerine uyun.
:::

## Sonraki Adımlar

Kalıtım konusunu detaylı örneklerle öğrendiniz. Şimdi kapsülleme (encapsulation) konusuna geçerek, sınıf içi verileri nasıl koruyacağımızı ve erişimi nasıl kontrol edeceğimizi öğrenebilirsiniz.
`; 