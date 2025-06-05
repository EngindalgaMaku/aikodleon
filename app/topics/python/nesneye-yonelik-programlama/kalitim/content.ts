export const content = `
# Python'da Kalıtım (Inheritance)

Kalıtım, nesne yönelimli programlamanın temel prensiplerinden biridir. Bir sınıfın başka bir sınıfın özelliklerini ve davranışlarını miras almasını sağlar.
Bu sayede kod tekrarını önler ve sınıflar arasında hiyerarşik bir ilişki kurulmasını sağlar.

<div class="bg-blue-50 dark:bg-blue-900/10 p-6 rounded-lg mb-8">
  <h3 class="text-2xl font-semibold mb-4">🎯 Kalıtımın Avantajları</h3>
  <ul class="list-disc pl-6">
    <li><strong>Kod Tekrarını Önleme:</strong> Ortak özellikleri temel sınıfta tanımlayarak kod tekrarını önler.</li>
    <li><strong>Hiyerarşik Yapı:</strong> Sınıflar arasında mantıksal bir hiyerarşi oluşturur.</li>
    <li><strong>Kodun Yeniden Kullanılabilirliği:</strong> Var olan kodun yeni sınıflarda kullanılmasını sağlar.</li>
    <li><strong>Genişletilebilirlik:</strong> Mevcut sınıfları değiştirmeden yeni özellikler eklenebilir.</li>
  </ul>
</div>

## Temel Kalıtım

Bir sınıftan türetme yapmak için, yeni sınıf tanımında parantez içinde temel sınıfı belirtiriz:

\`\`\`python
class Hayvan:
    def __init__(self, isim, yas):
        self.isim = isim
        self.yas = yas
    
    def ses_cikar(self):
        pass

class Kopek(Hayvan):
    def ses_cikar(self):
        return "Hav hav!"

class Kedi(Hayvan):
    def ses_cikar(self):
        return "Miyav!"
\`\`\`

## super() Fonksiyonu

\`super()\` fonksiyonu, üst sınıfın metodlarını çağırmak için kullanılır:

\`\`\`python
class Ogrenci:
    def __init__(self, ad, soyad):
        self.ad = ad
        self.soyad = soyad

class LiseOgrencisi(Ogrenci):
    def __init__(self, ad, soyad, sinif):
        super().__init__(ad, soyad)  # Üst sınıfın __init__ metodunu çağır
        self.sinif = sinif
\`\`\`

## Çoklu Kalıtım

Python'da bir sınıf birden fazla sınıftan türetilebilir:

\`\`\`python
class A:
    def metod_a(self):
        return "A sınıfından"

class B:
    def metod_b(self):
        return "B sınıfından"

class C(A, B):  # C sınıfı hem A hem B'den türetildi
    pass

c = C()
print(c.metod_a())  # "A sınıfından"
print(c.metod_b())  # "B sınıfından"
\`\`\`

<div class="bg-yellow-50 dark:bg-yellow-900/10 p-6 rounded-lg mb-8">
  <h3 class="text-2xl font-semibold mb-4">⚠️ Çoklu Kalıtımda Dikkat Edilecek Noktalar</h3>
  <ul class="list-disc pl-6">
    <li><strong>Elmas Problemi:</strong> Aynı metodun farklı üst sınıflarda farklı şekillerde tanımlanması durumu.</li>
    <li><strong>Karmaşıklık:</strong> Çok sayıda üst sınıf kullanımı kodun anlaşılmasını zorlaştırabilir.</li>
    <li><strong>MRO (Method Resolution Order):</strong> Python'ın metod arama sırasını anlamak önemlidir.</li>
  </ul>
</div>

## Method Resolution Order (MRO)

Python'da çoklu kalıtımda metodların aranma sırası MRO ile belirlenir:

\`\`\`python
class A:
    def kim(self):
        return "A"

class B(A):
    def kim(self):
        return "B"

class C(A):
    def kim(self):
        return "C"

class D(B, C):
    pass

d = D()
print(D.mro())  # MRO sırasını gösterir
print(d.kim())  # "B" (soldan sağa arama yapılır)
\`\`\`

## isinstance() ve issubclass()

Nesne ve sınıf ilişkilerini kontrol etmek için kullanılan fonksiyonlar:

\`\`\`python
kopek = Kopek("Karabaş", 3)
print(isinstance(kopek, Kopek))      # True
print(isinstance(kopek, Hayvan))     # True
print(issubclass(Kopek, Hayvan))     # True
\`\`\`

## Alıştırmalar

1. **Çalışan Yönetim Sistemi**
   
   [Detaylı çözüm için tıklayın](/topics/python/nesneye-yonelik-programlama/kalitim/calisan-yonetim-sistemi)
   
   - Bir şirketin çalışan yönetim sistemini modelleyin:
     - \`Calisan\` temel sınıfı
     - \`Muhendis\`, \`Yonetici\`, \`Pazarlamaci\` gibi alt sınıflar
     - Maaş hesaplama, izin takibi, proje atama gibi özellikler
     - Departman bazlı raporlama sistemi

2. **Oyun Karakter Sistemi**
   
   [Detaylı çözüm için tıklayın](/topics/python/nesneye-yonelik-programlama/kalitim/oyun-karakter-sistemi)
   
   - Bir RPG oyunu için karakter sistemi geliştirin:
     - \`Karakter\` temel sınıfı
     - \`Savasci\`, \`Buyucu\`, \`Okcu\` gibi alt sınıflar
     - Yetenek sistemi ve seviye atlama
     - Envanter yönetimi ve ekipman sistemi

3. **Medya Oynatıcı Sistemi**
   
   [Detaylı çözüm için tıklayın](/topics/python/nesneye-yonelik-programlama/kalitim/medya-oynatici-sistemi)
   
   - Farklı medya türlerini destekleyen bir oynatıcı sistemi oluşturun:
     - \`MedyaOynatici\` temel sınıfı
     - \`MuzikOynatici\`, \`VideoOynatici\`, \`PodcastOynatici\` alt sınıfları
     - Çalma listesi yönetimi
     - Format dönüştürme ve kalite ayarları

<div class="bg-purple-50 dark:bg-purple-900/10 p-6 rounded-lg mb-8">
  <h3 class="text-2xl font-semibold mb-4">💡 Kalıtım Kullanırken Dikkat Edilecek Noktalar</h3>
  <ul class="list-disc pl-6">
    <li><strong>IS-A İlişkisi:</strong> Kalıtım kullanırken "is-a" ilişkisinin varlığından emin olun.</li>
    <li><strong>Kompozisyon vs Kalıtım:</strong> Bazen kalıtım yerine kompozisyon kullanmak daha uygun olabilir.</li>
    <li><strong>Liskov Substitution Prensibi:</strong> Alt sınıflar, üst sınıfların yerine kullanılabilmelidir.</li>
    <li><strong>DRY Prensibi:</strong> Kendini tekrar eden kodları ortak bir üst sınıfa taşıyın.</li>
    <li><strong>SOLID Prensipleri:</strong> Kalıtım hiyerarşisini tasarlarken SOLID prensiplerine uyun.</li>
  </ul>
</div>

## Sonraki Adımlar

Kalıtım konusunu detaylı örneklerle öğrendiniz. Şimdi kapsülleme (encapsulation) konusuna geçerek, sınıf içi verileri nasıl koruyacağımızı ve erişimi nasıl kontrol edeceğimizi öğrenebilirsiniz.
`; 