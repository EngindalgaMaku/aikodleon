export const content = `
# Python'da Kapsülleme (Encapsulation)

Kapsülleme, nesneye yönelik programlamanın temel prensiplerinden biridir. Bu kavram, verilerin ve bu veriler üzerinde işlem yapan metodların bir arada tutulması ve dış dünyadan erişimin kontrollü bir şekilde sağlanması anlamına gelir.

## Kapsülleme Nedir?

Kapsülleme iki temel amaca hizmet eder:

1. **Veri Gizleme (Data Hiding)**: Sınıfın iç yapısını dış dünyadan gizleyerek, sınıfın detaylarını kullanıcılardan saklar.
2. **Erişim Kontrolü**: Verilere erişimi kontrol ederek, istenmeyen değişiklikleri engeller.

::: info
Kapsülleme, bir sınıfın içindeki verilerin güvenliğini sağlar ve sınıfın nasıl kullanılması gerektiğini belirler. Bu sayede kodun bakımı kolaylaşır ve hata riski azalır.
:::

## Python'da Kapsülleme Nasıl Uygulanır?

Python'da kapsülleme için üç farklı erişim belirleyici (access modifier) kullanılır:

### 1. Public Üyeler
- Varsayılan olarak tüm üyeler public'tir
- Herhangi bir özel işaret kullanılmaz
- Sınıf dışından doğrudan erişilebilir

\`\`\`python
class Ogrenci:
    def __init__(self, ad, numara):
        self.ad = ad          # Public değişken
        self.numara = numara  # Public değişken
    
    def bilgi_goster(self):   # Public metod
        return f"{self.ad} - {self.numara}"

# Kullanım
ogrenci = Ogrenci("Ahmet", 101)
print(ogrenci.ad)  # Doğrudan erişim mümkün
print(ogrenci.bilgi_goster())
\`\`\`

### 2. Protected Üyeler
- Tek alt çizgi (_) ile başlar
- Sınıf dışından erişilebilir ama erişilmemesi önerilir
- Alt sınıflardan erişilebilir

\`\`\`python
class Calisan:
    def __init__(self, ad, maas):
        self._ad = ad      # Protected değişken
        self._maas = maas  # Protected değişken
    
    def _maas_hesapla(self):  # Protected metod
        return self._maas * 1.1

class Mudur(Calisan):
    def maas_goster(self):
        # Protected üyelere alt sınıftan erişim
        return f"{self._ad} - {self._maas_hesapla()}"
\`\`\`

::: warning
Protected üyeler Python'da gerçek bir erişim kısıtlaması sağlamaz. Bu sadece bir konvansiyondur ve programcılara "bu üyeyi doğrudan kullanmayın" mesajı verir.
:::

### 3. Private Üyeler
- Çift alt çizgi (__) ile başlar
- Sınıf dışından doğrudan erişilemez
- Name mangling ile gizlenir

\`\`\`python
class BankaHesabi:
    def __init__(self):
        self.__bakiye = 0  # Private değişken
    
    def para_yatir(self, miktar):
        if miktar > 0:
            self.__bakiye += miktar
            return True
        return False
    
    def bakiye_goruntule(self):
        return self.__bakiye

# Kullanım
hesap = BankaHesabi()
hesap.para_yatir(1000)
print(hesap.bakiye_goruntule())  # Doğru kullanım
# print(hesap.__bakiye)  # Hata! Private değişkene doğrudan erişilemez
\`\`\`

## Property Dekoratörü ile Kapsülleme

Python'da \`@property\` dekoratörü, private değişkenlere kontrollü erişim sağlamak için kullanılır. Bu yöntem, Java'daki getter ve setter metodlarının daha Pythonic bir versiyonudur.

\`\`\`python
class Personel:
    def __init__(self, ad, yas):
        self.__ad = ad
        self.__yas = yas
    
    @property
    def yas(self):
        """Yaş bilgisini döndürür"""
        return self.__yas
    
    @yas.setter
    def yas(self, yeni_yas):
        """Yaş bilgisini kontrollü bir şekilde günceller"""
        if 18 <= yeni_yas <= 65:
            self.__yas = yeni_yas
        else:
            raise ValueError("Yaş 18-65 arasında olmalıdır")

# Kullanım
p1 = Personel("Ali", 30)
print(p1.yas)      # Getter gibi kullanım
p1.yas = 35        # Setter gibi kullanım
# p1.yas = 15      # ValueError: Yaş 18-65 arasında olmalıdır
\`\`\`

::: tip
Property dekoratörü, nesne değişkenlerine erişimi ve değişikliği kontrol altında tutmanın en elegant yoludur. Ayrıca kodun okunabilirliğini artırır ve bakımını kolaylaştırır.
:::

## Kapsüllemenin Faydaları

1. **Veri Güvenliği**: Veriler istenmeyen değişikliklerden korunur.
2. **Kontrollü Erişim**: Verilere erişim ve değişiklik kontrol altında tutulur.
3. **Esneklik**: İç yapı değiştirilse bile dış arayüz aynı kalabilir.
4. **Bakım Kolaylığı**: Kodun bakımı ve güncellenmesi kolaylaşır.

## En İyi Uygulamalar

1. **Minimum Açıklık Prensibi**: Mümkün olan en kısıtlı erişim seviyesini kullanın.
2. **Property Kullanımı**: Veri doğrulama veya hesaplama gerektiren durumlar için property kullanın.
3. **Dokümantasyon**: Protected ve private üyelerin neden bu şekilde tanımlandığını açıklayın.
4. **Tutarlılık**: Benzer veriler için benzer erişim seviyeleri kullanın.

::: warning
Python'da gerçek private değişkenler yoktur. Name mangling (\`__değişken\`) bile değişkene erişimi tamamen engellemez, sadece zorlaştırır. Bu nedenle güvenlik kritik verileri saklamak için başka yöntemler kullanılmalıdır.
:::

## Pratik Örnek: Kütüphane Sistemi

Aşağıdaki örnek, kapsüllemenin gerçek bir uygulamada nasıl kullanılabileceğini gösterir:

\`\`\`python
class Kitap:
    def __init__(self, baslik, yazar, isbn):
        self.__baslik = baslik
        self.__yazar = yazar
        self.__isbn = isbn
        self.__odunc_durumu = False
    
    @property
    def baslik(self):
        return self.__baslik
    
    @property
    def odunc_durumu(self):
        return "Ödünç Verildi" if self.__odunc_durumu else "Rafta"
    
    def odunc_al(self):
        if not self.__odunc_durumu:
            self.__odunc_durumu = True
            return True
        return False
    
    def iade_et(self):
        if self.__odunc_durumu:
            self.__odunc_durumu = False
            return True
        return False

class Kutuphane:
    def __init__(self):
        self.__kitaplar = {}
    
    def kitap_ekle(self, kitap):
        if isinstance(kitap, Kitap):
            self.__kitaplar[kitap.baslik] = kitap
            return True
        return False
    
    def kitap_ara(self, baslik):
        return self.__kitaplar.get(baslik)

# Kullanım
kutuphane = Kutuphane()
kitap = Kitap("Python Programming", "John Smith", "123-456-789")
kutuphane.kitap_ekle(kitap)

aranan_kitap = kutuphane.kitap_ara("Python Programming")
if aranan_kitap:
    print(f"Kitap durumu: {aranan_kitap.odunc_durumu}")
    if aranan_kitap.odunc_al():
        print("Kitap başarıyla ödünç alındı")
    print(f"Yeni durum: {aranan_kitap.odunc_durumu}")
\`\`\`

Bu örnekte:
- Private değişkenler (\`__baslik\`, \`__odunc_durumu\` vb.) ile veri gizleme
- Property dekoratörü ile kontrollü erişim
- Metodlar aracılığıyla veri değişikliği
- Sınıflar arası etkileşimde kapsülleme

gösterilmiştir.

## Özet

Kapsülleme, nesneye yönelik programlamanın önemli bir prensibidir ve Python'da çeşitli yöntemlerle uygulanabilir. Doğru kullanıldığında:
- Kod organizasyonu iyileşir
- Hata riski azalır
- Bakım kolaylaşır
- Kodun yeniden kullanılabilirliği artar

::: tip
Kapsülleme kullanırken aşırıya kaçmamak önemlidir. Her değişkeni private yapmak yerine, gerçekten gizlenmesi gereken verileri belirleyip ona göre karar vermek daha doğru bir yaklaşımdır.
:::
`; 