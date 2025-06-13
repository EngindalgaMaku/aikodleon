---
title: "Method Overriding (Metot Ezme)"
description: "Python'da Method Overriding, alt sınıfların üst sınıflarda tanımlanmış olan metotları kendi ihtiyaçlarına göre yeniden tanımlamasına olanak tanır. Bu, polimorfizmin temel taşlarından biridir."
---

## Method Overriding Nedir?

**Method Overriding**, nesne yönelimli programlamada (OOP), bir alt sınıfın (child class), üst sınıfından (parent class) miras aldığı bir metodun implementasyonunu değiştirmesi veya yeniden tanımlaması işlemidir. Yani, alt sınıf, üst sınıftaki aynı isme, aynı parametrelere ve aynı dönüş tipine sahip bir metodu kendi içinde tekrar tanımlayarak, o metoda yeni bir davranış kazandırır.

Bu mekanizma, bir alt sınıfın, miras aldığı genel bir davranışı özelleştirmesi gerektiğinde kullanılır.

### Temel Özellikleri:

*   **Aynı Metot İmzası:** Ezilen (overridden) ve ezen (overriding) metotların isimleri ve parametre sayıları/tipleri aynı olmalıdır.
*   **Miras Alma İlişkisi:** Method overriding, yalnızca miras alma (`inheritance`) ilişkisi olan sınıflar arasında gerçekleşebilir.
*   **Polimorfizm:** Overriding, polimorfik davranışın temelini oluşturur. Aynı metot çağrısının, nesnenin ait olduğu sınıfa göre farklı davranışlar sergilemesini sağlar.

## Neden Method Overriding Kullanılır?

1.  **Özelleştirme (Customization):** Üst sınıftaki genel bir davranışı, alt sınıfın özel ihtiyaçlarına göre uyarlamak için kullanılır. Örneğin, bir `Hayvan` sınıfının `ses_cikar()` metodu genel bir "ses" çıktısı verirken, `Kedi` alt sınıfı bu metodu override ederek "Miyav" çıktısı verebilir.
2.  **Genişletilebilirlik (Extensibility):** Alt sınıflar, üst sınıfın metodunu tamamen değiştirmek yerine, `super()` fonksiyonunu kullanarak orijinal davranışı çağırıp üzerine yeni işlevler ekleyebilir.
3.  **Arayüz Uyumu (Interface Consistency):** Bir grup ilişkili sınıfın aynı arayüze (metot isimlerine) sahip olmasını sağlar, bu da kodun daha okunabilir ve yönetilebilir olmasına yardımcı olur.

## `super()` Fonksiyonu ile Üst Sınıf Metodunu Çağırma

Bir metodu override ettiğinizde, bazen üst sınıfın orijinal metodundaki işlevselliği tamamen kaybetmek istemezsiniz. `super()` fonksiyonu, bu gibi durumlarda üst sınıfın metoduna erişmenizi sağlar. Bu, mevcut davranışı genişletmek için çok kullanışlıdır.

### Kod Örneği

```python
class Hayvan:
    def __init__(self, ad):
        self.ad = ad

    def ses_cikar(self):
        return "Bir hayvan sesi"

    def bilgi_goster(self):
        print(f"Ben bir hayvanım. Adım: {self.ad}")

class Kedi(Hayvan):
    def __init__(self, ad, tuy_rengi):
        # Üst sınıfın __init__ metodunu çağırarak 'ad' özelliğini ayarlıyoruz
        super().__init__(ad)
        self.tuy_rengi = tuy_rengi

    # ses_cikar metodunu override ediyoruz
    def ses_cikar(self):
        return "Miyav!"

    # bilgi_goster metodunu override ediyor ve genişletiyoruz
    def bilgi_goster(self):
        # Önce üst sınıfın orijinal metodunu çağırıyoruz
        super().bilgi_goster()
        # Sonra kendi özel işlevselliğimizi ekliyoruz
        print(f"Ayrıca bir kediyim ve tüy rengim: {self.tuy_rengi}")

# Nesneleri oluşturalım
hayvan = Hayvan("Leo")
kedi = Kedi("Boncuk", "Sarı")

print(f"{hayvan.ad} diyor ki: {hayvan.ses_cikar()}")  # Çıktı: Leo diyor ki: Bir hayvan sesi
print(f"{kedi.ad} diyor ki: {kedi.ses_cikar()}")      # Çıktı: Boncuk diyor ki: Miyav!

print("\n--- Bilgi Gösterimi ---")
hayvan.bilgi_goster()
# Çıktı:
# Ben bir hayvanım. Adım: Leo

print("-" * 20)
kedi.bilgi_goster()
# Çıktı:
# Ben bir hayvanım. Adım: Boncuk
# Ayrıca bir kediyim ve tüy rengim: Sarı
```

Bu örnekte:
*   `Kedi` sınıfı, `Hayvan` sınıfının `ses_cikar` metodunu tamamen değiştirerek kendine özgü bir davranış sergiliyor.
*   `Kedi` sınıfının `bilgi_goster` metodu ise `super().bilgi_goster()` diyerek önce `Hayvan` sınıfındaki orijinal metodu çalıştırıyor, ardından kendi ek bilgisini yazdırıyor. Bu, metot genişletmeye güzel bir örnektir. 