export interface Example {
  id: number;
  title: string;
  difficulty: 'Kolay' | 'Orta' | 'Zor';
  topics: string[];
  description: string;
  solution: string;
}

export const examples: Example[] = [
  {
    id: 1,
    title: 'Sayı Tahmin Oyunu',
    difficulty: 'Kolay',
    topics: ['Döngüler', 'Koşullu İfadeler', 'Random Modülü'],
    description: `
Kullanıcıdan 1 ile 100 arasında bir sayıyı tahmin etmesini isteyen bir oyun yazın. Program rastgele bir sayı tutmalı ve kullanıcının her tahmininden sonra "Daha Yüksek" veya "Daha Düşük" şeklinde ipuçları vermelidir. Kullanıcı doğru sayıyı bulduğunda, kaç denemede bulduğunu ekrana yazdırın.
    `,
    solution: `
\`\`\`python
import random

def sayi_tahmin_oyunu():
    hedef_sayi = random.randint(1, 100)
    tahmin = 0
    deneme_sayisi = 0

    print("1 ile 100 arasında bir sayı tuttum. Bakalım bulabilecek misin?")

    while tahmin != hedef_sayi:
        try:
            tahmin = int(input("Tahminin: "))
            deneme_sayisi += 1

            if tahmin < hedef_sayi:
                print("Daha Yüksek!")
            elif tahmin > hedef_sayi:
                print("Daha Düşük!")
            else:
                print(f"🎉 Tebrikler! {hedef_sayi} sayısını {deneme_sayisi} denemede buldun.")
        except ValueError:
            print("Lütfen geçerli bir sayı girin.")

sayi_tahmin_oyunu()
\`\`\`
    `,
  },
  {
    id: 2,
    title: 'Basit Hesap Makinesi',
    difficulty: 'Kolay',
    topics: ['Fonksiyonlar', 'Koşullu İfadeler', 'Sözlükler'],
    description: `
Kullanıcıdan iki sayı ve bir işlem (+, -, *, /) alan bir hesap makinesi fonksiyonu yazın. Fonksiyon, verilen işleme göre sonucu hesaplayıp döndürmelidir. Geçersiz bir işlem girilirse veya sıfıra bölme hatası olursa uygun bir mesaj verin.
    `,
    solution: `
\`\`\`python
def hesap_makinesi(sayi1, sayi2, islem):
    if islem == '+':
        return sayi1 + sayi2
    elif islem == '-':
        return sayi1 - sayi2
    elif islem == '*':
        return sayi1 * sayi2
    elif islem == '/':
        if sayi2 == 0:
            return "Hata: Sıfıra bölme yapılamaz."
        return sayi1 / sayi2
    else:
        return "Hata: Geçersiz işlem."

# Kullanım
num1 = float(input("Birinci sayıyı girin: "))
num2 = float(input("İkinci sayıyı girin: "))
op = input("İşlemi girin (+, -, *, /): ")

sonuc = hesap_makinesi(num1, num2, op)
print(f"Sonuç: {sonuc}")
\`\`\`
    `,
  },
  {
    id: 3,
    title: 'Metin Analizi Aracı',
    difficulty: 'Orta',
    topics: ['String İşlemleri', 'Sözlükler', 'Fonksiyonlar'],
    description: `
Bir metin dosyasını okuyan ve içindeki kelime sayısını, cümle sayısını ve en sık kullanılan 5 kelimeyi bulan bir fonksiyon yazın. Cümlelerin nokta, soru işareti veya ünlem işareti ile bittiğini varsayabilirsiniz.
    `,
    solution: `
\`\`\`python
import re
from collections import Counter

def metin_analizi(dosya_yolu):
    try:
        with open(dosya_yolu, 'r', encoding='utf-8') as dosya:
            icerik = dosya.read()
    except FileNotFoundError:
        return "Hata: Dosya bulunamadı."

    # Cümle sayısını bul
    cumleler = re.split(r'[.?!]', icerik)
    cumle_sayisi = len([c for c in cumleler if c.strip()])

    # Kelimeleri temizle ve say
    kelimeler = re.findall(r'\\b\\w+\\b', icerik.lower())
    kelime_sayisi = len(kelimeler)

    # En sık geçen 5 kelimeyi bul
    kelime_frekanslari = Counter(kelimeler)
    en_sik_5 = kelime_frekanslari.most_common(5)

    print(f"--- Metin Analizi Sonuçları ---")
    print(f"Kelime Sayısı: {kelime_sayisi}")
    print(f"Cümle Sayısı: {cumle_sayisi}")
    print("En Sık Geçen 5 Kelime:")
    for kelime, sayi in en_sik_5:
        print(f"- {kelime}: {sayi} kez")

# Örnek kullanım (ornek.txt adında bir dosya oluşturup içine metin yazın)
# metin_analizi('ornek.txt')
\`\`\`
    `,
  },
  {
    id: 4,
    title: 'Öğrenci Not Sistemi',
    difficulty: 'Orta',
    topics: ['Sözlükler', 'Döngüler', 'Fonksiyonlar'],
    description: `
Öğrenci bilgilerini (ad, numara) ve notlarını saklayan bir sistem yazın. Aşağıdaki işlevleri yerine getiren fonksiyonlar oluşturun:
1.  Yeni öğrenci ekleme.
2.  Öğrenciye not ekleme.
3.  Bir öğrencinin not ortalamasını hesaplama.
4.  Tüm öğrencileri ve not ortalamalarını listeleme.
Verileri iç içe bir sözlük yapısında tutun.
    `,
    solution: `
\`\`\`python
ogrenci_sistemi = {}

def ogrenci_ekle(numara, ad):
    if numara not in ogrenci_sistemi:
        ogrenci_sistemi[numara] = {'ad': ad, 'notlar': []}
        print(f"{ad} sisteme eklendi.")
    else:
        print("Hata: Bu numarada bir öğrenci zaten var.")

def not_ekle(numara, not_degeri):
    if numara in ogrenci_sistemi:
        ogrenci_sistemi[numara]['notlar'].append(not_degeri)
        print(f"{numara} numaralı öğrenciye {not_degeri} notu eklendi.")
    else:
        print("Hata: Öğrenci bulunamadı.")

def ortalama_hesapla(numara):
    if numara in ogrenci_sistemi:
        notlar = ogrenci_sistemi[numara]['notlar']
        if not notlar:
            return 0
        return sum(notlar) / len(notlar)
    return "Öğrenci bulunamadı."

def tum_ogrencileri_listele():
    print("\\n--- Öğrenci Listesi ---")
    for numara, bilgiler in ogrenci_sistemi.items():
        ortalama = ortalama_hesapla(numara)
        print(f"Numara: {numara}, Ad: {bilgiler['ad']}, Ortalama: {ortalama:.2f}")

# Kullanım
ogrenci_ekle(101, "Ali Veli")
not_ekle(101, 85)
not_ekle(101, 90)

ogrenci_ekle(102, "Ayşe Fatma")
not_ekle(102, 70)
not_ekle(102, 75)

tum_ogrencileri_listele()
\`\`\`
    `,
  },
  {
    id: 5,
    title: 'Palindrom Kontrolcüsü',
    difficulty: 'Kolay',
    topics: ['String İşlemleri', 'Fonksiyonlar'],
    description: `
Bir kelimenin veya cümlenin palindrom olup olmadığını kontrol eden bir fonksiyon yazın. Palindrom, tersten okunduğunda da aynı olan kelime veya cümledir. Kontrol yapılırken boşlukları ve büyük/küçük harf farkını göz ardı edin.
    `,
    solution: `
\`\`\`python
import re

def is_palindrome(text):
    # Metni küçük harfe çevir ve sadece harf/rakamları al
    cleaned_text = re.sub(r'[^a-z0-9]', '', text.lower())
    
    # Metni tersiyle karşılaştır
    return cleaned_text == cleaned_text[::-1]

# Test
print(f"'A Man, A Plan, A Canal: Panama' bir palindrom mu? {is_palindrome('A Man, A Plan, A Canal: Panama')}") # True
print(f"'Kodleon' bir palindrom mu? {is_palindrome('Kodleon')}") # False
print(f"'kayak' bir palindrom mu? {is_palindrome('kayak')}") # True
\`\`\`
    `,
  },
  {
    id: 6,
    title: 'Fibonacci Dizisi Oluşturucu',
    difficulty: 'Orta',
    topics: ['Döngüler', 'Listeler', 'Fonksiyonlar'],
    description: `
Verilen bir 'n' sayısına kadar olan Fibonacci dizisini oluşturan bir fonksiyon yazın. Fibonacci dizisi, her sayının kendinden önceki iki sayının toplamı olduğu bir seridir (0, 1, 1, 2, 3, 5, 8, ...).
    `,
    solution: `
\`\`\`python
def fibonacci_dizisi(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    dizi = [0, 1]
    while len(dizi) < n:
        sonraki_sayi = dizi[-1] + dizi[-2]
        dizi.append(sonraki_sayi)
        
    return dizi

# Test
adet = 10
print(f"İlk {adet} Fibonacci sayısı: {fibonacci_dizisi(adet)}")
# Çıktı: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
\`\`\`
    `,
  },
  {
    id: 7,
    title: 'Basit Dosya Yedekleme Scripti',
    difficulty: 'Zor',
    topics: ['Dosya İşlemleri', 'os Modülü', 'datetime Modülü'],
    description: `
Belirtilen bir klasördeki tüm dosyaları, o anki tarih ve saat bilgisiyle adlandırılmış yeni bir klasöre kopyalayarak yedekleyen bir Python scripti yazın.
    `,
    solution: `
\`\`\`python
import os
import shutil
from datetime import datetime

def yedekle(kaynak_klasor, hedef_ana_klasor):
    # Yedek klasör adını oluştur (örn: 'yedek_2023-10-27_15-30-00')
    zaman_damgasi = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hedef_klasor_yolu = os.path.join(hedef_ana_klasor, f"yedek_{zaman_damgasi}")

    try:
        # Hedef klasörü oluştur
        os.makedirs(hedef_klasor_yolu)
        print(f"Yedek klasörü oluşturuldu: {hedef_klasor_yolu}")

        # Kaynak klasördeki dosyaları listele ve kopyala
        for dosya_adi in os.listdir(kaynak_klasor):
            kaynak_dosya = os.path.join(kaynak_klasor, dosya_adi)
            hedef_dosya = os.path.join(hedef_klasor_yolu, dosya_adi)
            
            if os.path.isfile(kaynak_dosya):
                shutil.copy2(kaynak_dosya, hedef_dosya)
        
        print(f"Yedekleme tamamlandı. {len(os.listdir(kaynak_klasor))} dosya kopyalandı.")

    except FileNotFoundError:
        print(f"Hata: Kaynak klasör '{kaynak_klasor}' bulunamadı.")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

# Kullanım:
# 1. 'yedeklenecek_dosyalar' adında bir klasör oluşturun ve içine birkaç dosya atın.
# 2. 'yedekler' adında bir ana klasör oluşturun.
# kaynak = 'yedeklenecek_dosyalar'
# hedef = 'yedekler'
# yedekle(kaynak, hedef)
\`\`\`
    `,
  },
  {
    id: 8,
    title: 'URL Kısaltma Simülasyonu',
    difficulty: 'Zor',
    topics: ['Sözlükler', 'hashlib Modülü', 'Fonksiyonlar'],
    description: `
URL kısaltma servisi gibi çalışan bir sistem simüle edin. İki fonksiyon yazın:
1.  \`kisalt(url)\`: Uzun bir URL'i alıp, ona karşılık gelen kısa bir kod (örn: 6 karakterlik hash) üretir ve bir sözlükte saklar.
2.  \`uzun_adresi_getir(kod)\`: Kısa kodu alıp, orijinal URL'i döndürür.
    `,
    solution: `
\`\`\`python
import hashlib

# Veritabanını simüle etmek için sözlük
url_veritabani = {}

def kisalt(url):
    # URL'den bir hash oluştur ve ilk 6 karakterini al
    hash_nesnesi = hashlib.md5(url.encode())
    kisa_kod = hash_nesnesi.hexdigest()[:6]
    
    # Kodu ve orijinal URL'i sözlükte sakla
    url_veritabani[kisa_kod] = url
    
    print(f"URL kısaltıldı: {url} -> {kisa_kod}")
    return kisa_kod

def uzun_adresi_getir(kod):
    # Koda karşılık gelen URL'i döndür
    return url_veritabani.get(kod, "Hata: Bu kodla eşleşen bir URL bulunamadı.")

# Kullanım
orijinal_url = "https://www.kodleon.com/topics/python/temel-python"
kisa_kodumuz = kisalt(orijinal_url)

print(f"Kısa kod '{kisa_kodumuz}' için orijinal URL: {uzun_adresi_getir(kisa_kodumuz)}")
print(f"Geçersiz kod denemesi: {uzun_adresi_getir('abcdef')}")
\`\`\`
    `,
  },
  {
    id: 9,
    title: 'Komut Satırı Yapılacaklar Listesi',
    difficulty: 'Orta',
    topics: ['Listeler', 'Döngüler', 'Fonksiyonlar', 'Kullanıcı Girdisi'],
    description: `
Kullanıcının görev ekleyebileceği, görevleri listeleyebileceği ve bir görevi tamamlandı olarak işaretleyebileceği basit bir komut satırı tabanlı yapılacaklar listesi uygulaması oluşturun. Görevleri bir liste içinde sözlükler olarak tutun (örn: \`[{'görev': 'Python öğren', 'tamamlandi': False}]\`).
    `,
    solution: `
\`\`\`python
yapilacaklar = []

def gorevleri_goster():
    print("\\n--- YAPILACAKLAR LİSTESİ ---")
    if not yapilacaklar:
        print("Listeniz boş.")
    else:
        for i, gorev_obj in enumerate(yapilacaklar):
            durum = "✓" if gorev_obj['tamamlandi'] else "✗"
            print(f"{i + 1}. [{durum}] {gorev_obj['görev']}")
    print("--------------------------")

def gorev_ekle(yeni_gorev):
    yapilacaklar.append({'görev': yeni_gorev, 'tamamlandi': False})
    print(f"'{yeni_gorev}' eklendi.")

def gorev_tamamla(gorev_numarasi):
    try:
        indeks = int(gorev_numarasi) - 1
        if 0 <= indeks < len(yapilacaklar):
            yapilacaklar[indeks]['tamamlandi'] = True
            print(f"'{yapilacaklar[indeks]['görev']}' tamamlandı olarak işaretlendi.")
        else:
            print("Hata: Geçersiz görev numarası.")
    except ValueError:
        print("Hata: Lütfen bir sayı girin.")

def ana_menu():
    while True:
        print("\\nMenü: 1-Listele, 2-Ekle, 3-Tamamla, 4-Çıkış")
        secim = input("Seçiminiz: ")
        if secim == '1':
            gorevleri_goster()
        elif secim == '2':
            yeni = input("Yeni görev: ")
            gorev_ekle(yeni)
        elif secim == '3':
            numara = input("Tamamlanacak görev numarası: ")
            gorev_tamamla(numara)
        elif secim == '4':
            print("Görüşmek üzere!")
            break
        else:
            print("Geçersiz seçim.")

# ana_menu()
\`\`\`
    `,
  },
  {
    id: 10,
    title: 'Sezar Şifreleme',
    difficulty: 'Orta',
    topics: ['String İşlemleri', 'Fonksiyonlar', 'Sözlükler'],
    description: `
Sezar şifrelemesi tekniğini kullanarak bir metni şifreleyen ve şifresini çözen bir fonksiyon yazın. Fonksiyon, bir metin ve bir kaydırma anahtarı (shift key) almalıdır. Örneğin, 3'lük bir kaydırma ile 'a', 'd'ye dönüşür.
    `,
    solution: `
\`\`\`python
def sezar_sifrele(metin, anahtar, mod='sifrele'):
    alfabe = 'abcdefghijklmnopqrstuvwxyz'
    sonuc = ''

    if mod == 'coz':
        anahtar = -anahtar

    for harf in metin.lower():
        if harf in alfabe:
            yeni_indeks = (alfabe.find(harf) + anahtar) % len(alfabe)
            sonuc += alfabe[yeni_indeks]
        else:
            sonuc += harf # Harf değilse (boşluk, noktalama vb.) olduğu gibi bırak
            
    return sonuc

# Test
orijinal_metin = "hello world"
kaydirma_anahtari = 3

sifreli_metin = sezar_sifrele(orijinal_metin, kaydirma_anahtari, 'sifrele')
print(f"Orijinal: {orijinal_metin}")
print(f"Şifreli: {sifreli_metin}") # khoor zruog

cozulmus_metin = sezar_sifrele(sifreli_metin, kaydirma_anahtari, 'coz')
print(f"Çözülmüş: {cozulmus_metin}") # hello world
\`\`\`
    `,
  },
  {
    id: 11,
    title: 'Belirli Aralıktaki Asal Sayıları Bulma',
    difficulty: 'Orta',
    topics: ['Döngüler', 'Fonksiyonlar', 'Verimlilik'],
    description: `
İki sayı (başlangıç ve bitiş) arasında yer alan tüm asal sayıları bulan bir fonksiyon yazın. Daha önce yazdığınız \`asal_mi\` fonksiyonunu burada kullanabilirsiniz.
    `,
    solution: `
\`\`\`python
def asal_mi(sayi):
    if sayi <= 1:
        return False
    for i in range(2, int(sayi**0.5) + 1):
        if sayi % i == 0:
            return False
    return True

def araliktaki_asallar(baslangic, bitis):
    asal_listesi = []
    for sayi in range(baslangic, bitis + 1):
        if asal_mi(sayi):
            asal_listesi.append(sayi)
    return asal_listesi

# Test
start = 10
end = 50
print(f"{start} ve {end} arasındaki asal sayılar:")
print(araliktaki_asallar(start, end))
# Çıktı: [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
\`\`\`
    `,
  },
  {
    id: 12,
    title: 'Basit Banka Hesabı Sınıfı',
    difficulty: 'Zor',
    topics: ['OOP', 'Sınıflar', 'Metodlar'],
    description: `
Bir banka hesabını temsil eden bir \`BankaHesabi\` sınıfı oluşturun. Bu sınıf aşağıdaki özelliklere sahip olmalıdır:
- Başlangıç bakiyesi ile bir hesap oluşturma.
- Para yatırma (\`para_yatir\`) metodu.
- Para çekme (\`para_cek\`) metodu (yetersiz bakiye kontrolü yapmalı).
- Güncel bakiyeyi gösteren (\`bakiye_goster\`) metodu.
    `,
    solution: `
\`\`\`python
class BankaHesabi:
    def __init__(self, sahip, bakiye=0.0):
        self.sahip = sahip
        self.bakiye = bakiye
        print(f"Hoşgeldiniz, {self.sahip}! Hesap oluşturuldu.")

    def bakiye_goster(self):
        print(f"Güncel Bakiye: {self.bakiye:.2f} TL")

    def para_yatir(self, miktar):
        if miktar > 0:
            self.bakiye += miktar
            print(f"{miktar:.2f} TL yatırıldı.")
            self.bakiye_goster()
        else:
            print("Yatırılacak miktar pozitif olmalıdır.")

    def para_cek(self, miktar):
        if miktar > 0:
            if self.bakiye >= miktar:
                self.bakiye -= miktar
                print(f"{miktar:.2f} TL çekildi.")
                self.bakiye_goster()
            else:
                print("Yetersiz bakiye!")
        else:
            print("Çekilecek miktar pozitif olmalıdır.")

# Kullanım
hesabim = BankaHesabi("Ahmet Yılmaz", 1000)
hesabim.bakiye_goster()
hesabim.para_yatir(500)
hesabim.para_cek(200)
hesabim.para_cek(1500) # Yetersiz bakiye uyarısı vermeli
\`\`\`
    `,
  },
  {
    id: 13,
    title: 'Tic-Tac-Toe Oyunu',
    difficulty: 'Zor',
    topics: ['Listeler', 'Fonksiyonlar', 'Oyun Mantığı'],
    description: `
İki kişilik bir Tic-Tac-Toe (XOX) oyunu yapın. Oyun tahtasını 3x3'lük bir liste matrisi ile temsil edin. Oyunun durumu (kazanan, berabere, devam ediyor) her hamleden sonra kontrol edilmeli ve tahta ekrana çizilmelidir.
    `,
    solution: `
\`\`\`python
tahta = [' ' for _ in range(9)] # 3x3'lük tahtayı tek boyutlu liste ile temsil et

def tahtayi_ciz():
    print(f" {tahta[0]} | {tahta[1]} | {tahta[2]} ")
    print("---|---|---")
    print(f" {tahta[3]} | {tahta[4]} | {tahta[5]} ")
    print("---|---|---")
    print(f" {tahta[6]} | {tahta[7]} | {tahta[8]} ")

def kazanan_kontrol(oyuncu):
    # Kazanma koşulları
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # Yatay
        [0, 3, 6], [1, 4, 7], [2, 5, 8], # Dikey
        [0, 4, 8], [2, 4, 6]             # Çapraz
    ]
    for condition in win_conditions:
        if tahta[condition[0]] == tahta[condition[1]] == tahta[condition[2]] == oyuncu:
            return True
    return False

def oyun():
    aktif_oyuncu = 'X'
    while True:
        tahtayi_ciz()
        try:
            hamle = int(input(f"Oyuncu '{aktif_oyuncu}', hamlenizi girin (1-9): ")) - 1
            if 0 <= hamle < 9 and tahta[hamle] == ' ':
                tahta[hamle] = aktif_oyuncu
                if kazanan_kontrol(aktif_oyuncu):
                    tahtayi_ciz()
                    print(f"🎉 Oyuncu '{aktif_oyuncu}' kazandı!")
                    break
                if ' ' not in tahta:
                    tahtayi_ciz()
                    print("Oyun berabere bitti!")
                    break
                aktif_oyuncu = 'O' if aktif_oyuncu == 'X' else 'X'
            else:
                print("Geçersiz hamle. Tekrar deneyin.")
        except ValueError:
            print("Lütfen 1-9 arasında bir sayı girin.")
            
# oyun()
\`\`\`
    `,
  },
  {
    id: 14,
    title: 'CSV Dosyası İşleyici',
    difficulty: 'Orta',
    topics: ['Dosya İşlemleri', 'csv Modülü', 'Sözlükler'],
    description: `
\`notlar.csv\` adında bir dosya olduğunu varsayın. Bu dosya öğrenci adlarını ve notlarını içerir (örn: \`isim,not\`). Dosyayı okuyan, her öğrencinin not ortalamasını hesaplayan ve sonucu ekrana yazdıran bir program yazın. Bir öğrencinin birden fazla notu olabilir.
    `,
    solution: `
\`\`\`python
import csv
from collections import defaultdict

# Örnek bir notlar.csv dosyası oluşturun:
# isim,not
# Ali,85
# Veli,90
# Ali,95
# Ayşe,100
# Veli,75

def notlari_isle(dosya_adi):
    ogrenci_notlari = defaultdict(list)
    try:
        with open(dosya_adi, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                ogrenci_notlari[row['isim']].append(int(row['not']))
        
        print("--- Not Ortalamaları ---")
        for isim, notlar in ogrenci_notlari.items():
            ortalama = sum(notlar) / len(notlar)
            print(f"{isim}: {ortalama:.2f}")

    except FileNotFoundError:
        print(f"Hata: '{dosya_adi}' dosyası bulunamadı.")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

# notlari_isle('notlar.csv')
\`\`\`
    `,
  },
  {
    id: 15,
    title: 'Basamak Toplamı (Özyinelemeli)',
    difficulty: 'Kolay',
    topics: ['Fonksiyonlar', 'Özyineleme', 'Matematik'],
    description: `
Bir sayının basamaklarının toplamını özyinelemeli (recursive) bir fonksiyon kullanarak bulan bir program yazın. Örneğin, 123 için sonuç 1 + 2 + 3 = 6 olmalıdır.
    `,
    solution: `
\`\`\`python
def basamak_toplami(n):
    # Negatif sayılar için de çalışmasını sağlayalım
    n = abs(n)
    
    # Temel durum: Sayı tek basamaklı ise kendisini döndür
    if n < 10:
        return n
    # Özyinelemeli adım: Son basamağı al ve kalanı fonksiyona geri gönder
    else:
        return (n % 10) + basamak_toplami(n // 10)

# Test
print(f"123 sayısının basamakları toplamı: {basamak_toplami(123)}")   # 6
print(f"9876 sayısının basamakları toplamı: {basamak_toplami(9876)}") # 30
print(f"5 sayısının basamakları toplamı: {basamak_toplami(5)}")     # 5
\`\`\`
    `,
  },
  {
    id: 16,
    title: 'Basit Web Kazıyıcı (Scraper)',
    difficulty: 'Zor',
    topics: ['requests', 'BeautifulSoup', 'Hata Yönetimi'],
    description: `
\`requests\` ve \`BeautifulSoup4\` kütüphanelerini kullanarak belirli bir web sayfasının başlığını (\`<title>\`) ve tüm ana başlıklarını (\`<h1>\`) çeken bir fonksiyon yazın. Not: Bu kütüphaneleri yüklemeniz gerekir (\`pip install requests beautifulsoup4\`).
    `,
    solution: `
\`\`\`python
import requests
from bs4 import BeautifulSoup

def web_kaziyici(url):
    try:
        # Web sayfasına istek gönder
        response = requests.get(url, timeout=10)
        # HTTP hatalarını kontrol et (örn: 404 Not Found)
        response.raise_for_status() 

        # Sayfa içeriğini parse et
        soup = BeautifulSoup(response.text, 'html.parser')

        # Sayfa başlığını al
        sayfa_basligi = soup.title.string if soup.title else "Başlık Bulunamadı"
        print(f"Sayfa Başlığı: {sayfa_basligi.strip()}")

        # Tüm h1 etiketlerini bul ve yazdır
        print("\\n--- H1 Başlıkları ---")
        h1_etiketleri = soup.find_all('h1')
        if not h1_etiketleri:
            print("Sayfada H1 başlığı bulunamadı.")
        else:
            for h1 in h1_etiketleri:
                print(f"- {h1.get_text(strip=True)}")

    except requests.exceptions.RequestException as e:
        print(f"Hata: Web sayfasına erişilemedi. {e}")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

# Test (Erişilebilir bir web sitesi ile deneyin)
# web_kaziyici("http://info.cern.ch/") # İlk web sitesi
\`\`\`
    `,
  },
  {
    id: 17,
    title: 'JSON Verisi İşleme',
    difficulty: 'Orta',
    topics: ['Dosya İşlemleri', 'json Modülü', 'Sözlükler'],
    description: `
Bir JSON dosyasını okuyan, içindeki verileri (örneğin, bir ürün listesi) işleyen ve belirli bir kritere uyan (örneğin, fiyatı 50'den yüksek olan) ürünleri listeleyen bir fonksiyon yazın.
    `,
    solution: `
\`\`\`python
import json

# Örnek 'urunler.json' dosyası içeriği:
# [
#   {"isim": "Laptop", "fiyat": 1500},
#   {"isim": "Mouse", "fiyat": 45},
#   {"isim": "Klavye", "fiyat": 75},
#   {"isim": "Monitör", "fiyat": 800}
# ]

def pahali_urunleri_bul(dosya_yolu, esik_fiyat):
    try:
        with open(dosya_yolu, 'r', encoding='utf-8') as f:
            urunler = json.load(f)
        
        print(f"Fiyatı {esik_fiyat} TL'den yüksek olan ürünler:")
        bulunanlar = 0
        for urun in urunler:
            if urun.get('fiyat', 0) > esik_fiyat:
                print(f"- {urun['isim']} ({urun['fiyat']} TL)")
                bulunanlar += 1
        
        if bulunanlar == 0:
            print("Bu kritere uyan ürün bulunamadı.")

    except FileNotFoundError:
        print(f"Hata: '{dosya_yolu}' dosyası bulunamadı.")
    except json.JSONDecodeError:
        print(f"Hata: '{dosya_yolu}' geçerli bir JSON dosyası değil.")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

# pahali_urunleri_bul('urunler.json', 50)
\`\`\`
    `,
  },
  {
    id: 18,
    title: 'Basit Geri Sayım Zamanlayıcısı',
    difficulty: 'Kolay',
    topics: ['time Modülü', 'Döngüler', 'Kullanıcı Girdisi'],
    description: `
Kullanıcıdan saniye cinsinden bir süre alan ve bu süreden geriye doğru saniye saniye sayan bir zamanlayıcı yapın. Her saniyede kalan süreyi ekrana yazdırın.
    `,
    solution: `
\`\`\`python
import time

def geri_sayim(saniye):
    while saniye > 0:
        # divmod saniyeyi dakika ve saniyeye böler
        dakika, sn = divmod(saniye, 60)
        # {:02d} formatı, sayıyı 2 haneli olacak şekilde 0 ile doldurur
        zaman_formati = f"{dakika:02d}:{sn:02d}"
        # print içinde \\r kullanarak satır başına dön ve üzerine yaz
        print(zaman_formati, end='\\r')
        time.sleep(1)
        saniye -= 1
    
    print("Süre doldu!   ")

# try:
#     sure = int(input("Geri sayım için saniye girin: "))
#     geri_sayim(sure)
# except ValueError:
#     print("Lütfen geçerli bir sayı girin.")
\`\`\`
    `,
  },
  {
    id: 19,
    title: 'İki Klasörü Senkronize Etme',
    difficulty: 'Zor',
    topics: ['os Modülü', 'shutil Modülü', 'Dosya İşlemleri'],
    description: `
İki klasör alan (bir kaynak, bir hedef) ve kaynak klasörde olup hedef klasörde olmayan dosyaları hedefe kopyalayarak "tek yönlü senkronizasyon" yapan bir fonksiyon yazın.
    `,
    solution: `
\`\`\`python
import os
import shutil

def klasor_senkronize_et(kaynak_dir, hedef_dir):
    print(f"'{kaynak_dir}' -> '{hedef_dir}' senkronizasyonu başlıyor...")
    
    # Klasörlerin var olup olmadığını kontrol et
    if not os.path.isdir(kaynak_dir) or not os.path.isdir(hedef_dir):
        print("Hata: Kaynak veya hedef klasör bulunamadı.")
        return

    kaynak_dosyalar = set(os.listdir(kaynak_dir))
    hedef_dosyalar = set(os.listdir(hedef_dir))

    kopyalanacak_dosyalar = kaynak_dosyalar - hedef_dosyalar
    
    if not kopyalanacak_dosyalar:
        print("Klasörler zaten senkronize.")
        return

    print(f"{len(kopyalanacak_dosyalar)} dosya kopyalanacak...")
    for dosya_adi in kopyalanacak_dosyalar:
        kaynak_yol = os.path.join(kaynak_dir, dosya_adi)
        hedef_yol = os.path.join(hedef_dir, dosya_adi)
        
        if os.path.isfile(kaynak_yol):
            shutil.copy2(kaynak_yol, hedef_yol)
            print(f"- {dosya_adi} kopyalandı.")

    print("Senkronizasyon tamamlandı.")

# Kullanım:
# 1. 'kaynak' ve 'hedef' adında iki klasör oluşturun.
# 2. 'kaynak' içine birkaç dosya atın.
# 3. 'hedef' klasörüne kaynakta olmayan farklı dosyalar veya hiçbir şey koymayın.
# klasor_senkronize_et('kaynak', 'hedef')
\`\`\`
    `,
  },
  {
    id: 20,
    title: 'Tekrarlanan Elemanları Temizle',
    difficulty: 'Kolay',
    topics: ['Listeler', 'Kümeler', 'Fonksiyonlar'],
    description: `
Bir liste alan ve içindeki tekrar eden elemanları temizleyerek yeni bir liste döndüren bir fonksiyon yazın. Elemanların sırasını korumak bir bonus özelliktir.
    `,
    solution: `
\`\`\`python
# Yöntem 1: Sırayı korumadan (En basit yol)
def temizle_hizli(liste):
    return list(set(liste))

# Yöntem 2: Sırayı koruyarak
def temizle_sirali(liste):
    gorulenler = set()
    sonuc = []
    for eleman in liste:
        if eleman not in gorulenler:
            gorulenler.add(eleman)
            sonuc.append(eleman)
    return sonuc

# Test
orijinal_liste = [1, 5, 2, 1, 9, 1, 5, 8, 8]
print(f"Orijinal Liste: {orijinal_liste}")
print(f"Hızlı Temizleme (Sırasız): {temizle_hizli(orijinal_liste)}")
print(f"Sıralı Temizleme: {temizle_sirali(orijinal_liste)}")
# Çıktı: [1, 5, 2, 9, 8]
\`\`\`
    `,
  },
]; 