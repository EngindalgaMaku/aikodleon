'use client';

import { ArrowLeft, ArrowRight } from 'lucide-react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import Quiz from './components/Quiz';

export default function ClassesAndObjects() {
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

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-4">Sınıf Nedir?</h2>
        <p className="mb-4">
          Sınıf (class), nesne yönelimli programlamada nesnelerin özelliklerini ve davranışlarını tanımlayan bir şablondur.
          Bir sınıf, verileri (özellikler) ve bu veriler üzerinde işlem yapan fonksiyonları (metodlar) bir arada tutar.
        </p>
        <pre className="bg-gray-100 p-4 rounded-lg mb-4">
          <code className="language-python">{`class Dikdortgen:
    def __init__(self, uzunluk, genislik):
        self.uzunluk = uzunluk
        self.genislik = genislik
    
    def alan_hesapla(self):
        return self.uzunluk * self.genislik
    
    def cevre_hesapla(self):
        return 2 * (self.uzunluk + self.genislik)`}</code>
        </pre>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-4">Nesne Nedir?</h2>
        <p className="mb-4">
          Nesne (object), bir sınıfın örneğidir. Sınıf şablonundan yaratılan ve bellekte yer kaplayan yapılardır.
          Her nesne, sınıfta tanımlanan özelliklere ve metodlara sahiptir.
        </p>
        <pre className="bg-gray-100 p-4 rounded-lg mb-4">
          <code className="language-python">{`# Dikdortgen sınıfından bir nesne oluşturma
d1 = Dikdortgen(5, 3)

# Nesnenin metodlarını kullanma
print(f"Alan: {d1.alan_hesapla()}")  # Çıktı: Alan: 15
print(f"Çevre: {d1.cevre_hesapla()}")  # Çıktı: Çevre: 16`}</code>
        </pre>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-4">Constructor (__init__)</h2>
        <p className="mb-4">
          Constructor, bir sınıftan nesne oluşturulduğunda otomatik olarak çağrılan özel bir metoddur.
          Python'da constructor metodu __init__ olarak adlandırılır ve nesnenin başlangıç değerlerini ayarlamak için kullanılır.
        </p>
        <pre className="bg-gray-100 p-4 rounded-lg mb-4">
          <code className="language-python">{`class Ogrenci:
    def __init__(self, ad, numara):
        self.ad = ad
        self.numara = numara
        self.dersler = []
    
    def ders_ekle(self, ders):
        self.dersler.append(ders)

# Öğrenci nesnesi oluşturma
ogrenci1 = Ogrenci("Ahmet", 101)
ogrenci1.ders_ekle("Matematik")
print(f"Öğrenci: {ogrenci1.ad}, Numara: {ogrenci1.numara}")`}</code>
        </pre>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-4">Self Parametresi</h2>
        <p className="mb-4">
          Python'da self parametresi, bir sınıf içindeki metodlarda nesnenin kendisini temsil eder.
          self sayesinde nesnenin özelliklerine ve metodlarına erişebiliriz.
        </p>
        <pre className="bg-gray-100 p-4 rounded-lg mb-4">
          <code className="language-python">{`class BankaHesabi:
    def __init__(self, hesap_no, bakiye=0):
        self.hesap_no = hesap_no
        self.bakiye = bakiye
    
    def para_yatir(self, miktar):
        self.bakiye += miktar
        return f"{miktar} TL yatırıldı. Yeni bakiye: {self.bakiye} TL"
    
    def para_cek(self, miktar):
        if self.bakiye >= miktar:
            self.bakiye -= miktar
            return f"{miktar} TL çekildi. Yeni bakiye: {self.bakiye} TL"
        return "Yetersiz bakiye!"

# Hesap oluşturma ve işlemler
hesap = BankaHesabi("123456", 1000)
print(hesap.para_yatir(500))  # 500 TL yatırıldı. Yeni bakiye: 1500 TL
print(hesap.para_cek(2000))  # Yetersiz bakiye!`}</code>
        </pre>
      </section>

      <section className="my-12">
        <Quiz />
      </section>
    </div>
  );
} 