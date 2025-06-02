---
title: Yarasa Algoritması (Bat Algorithm - BA)
description: Yarasa Algoritması (BA), yarasaların ekolokasyon (echolocation) davranışlarından esinlenerek geliştirilmiş bir metasezgisel optimizasyon algoritmasıdır.
---

## Yarasa Algoritması (BA) Nedir?

Yarasa Algoritması (BA), 2010 yılında Xin-She Yang tarafından geliştirilen, doğadan ilham alan bir metasezgisel optimizasyon algoritmasıdır. Algoritma, mikro yarasaların avlarını bulmak ve engellerden kaçınmak için kullandıkları ekolokasyon yeteneklerini taklit eder.

Yarasalar, yüksek frekanslı ses dalgaları (darbeler) yayar ve çevrelerindeki nesnelerden yansıyan yankıları dinlerler. Bu yankıların zamanlaması ve karakteristiği, onlara avlarının yeri, büyüklüğü, türü ve hareket yönü hakkında detaylı bilgi sağlar.

BA, bu ekolokasyon davranışını idealize edilmiş kurallarla basitleştirir:

1.  Tüm yarasalar, mesafeyi algılamak ve av/yiyecek kaynakları ile arka plandaki engeller arasındaki farkı "bilmek" için ekolokasyon kullanır.
2.  Yarasalar, \( x_i \) pozisyonunda \( v_i \) hızıyla rastgele uçar, sabit bir \( f_{min} \) frekansında (veya dalga boyunda \( \lambda \)), değişen bir \( A_0 \) ses yüksekliğinde (loudness) \( r \) darbe emisyon oranında av ararlar. Frekanslarını (veya dalga boylarını) otomatik olarak ayarlayabilir ve hedeflerine olan yakınlıklarına bağlı olarak darbe emisyon oranını \( [0, 1] \) aralığında ayarlayabilirler.
3.  Ses yüksekliğinin (loudness) büyük bir pozitif değerden (\( A_0 \)) minimum bir sabit değere (\( A_{min} \)) doğru değiştiği varsayılır.

## BA Algoritmasının Adımları

BA algoritmasının temel adımları şunlardır:

1.  **Başlatma (Initialization):**
    *   Yarasa popülasyonu (çözümler) rastgele oluşturulur.
    *   Her yarasa için pozisyon (\( x_i \)), hız (\( v_i \)), frekans (\( f_i \)), darbe emisyon oranı (\( r_i \)) ve ses yüksekliği (\( A_i \)) başlatılır.
    *   Amaç fonksiyonu değerleri (uygunluk) her yarasa için hesaplanır.
    *   Global en iyi çözüm (\( x^* \)) belirlenir.

2.  **Yeni Çözümler Üretme (Global Arama):**
    *   Her yarasa `i` için pozisyonu, hızı ve frekansı güncellenir:
        \[ f_i = f_{min} + (f_{max} - f_{min}) \beta \]
        \[ v_i^{t+1} = v_i^t + (x_i^t - x^*) f_i \]
        \[ x_i^{t+1} = x_i^t + v_i^{t+1} \]
        Burada:
        *   \( \beta \), `[0, 1]` aralığında üniform bir dağılımdan gelen rastgele bir vektördür.
        *   \( f_{min} \) ve \( f_{max} \), frekans aralığını belirler.
        *   \( x^* \), mevcut global en iyi çözümdür.
        *   Bu adımlar, yarasaların global en iyi çözüme doğru hareket etmesini sağlar ve frekans değişimi arama davranışını çeşitlendirir.

3.  **Yerel Arama (Local Search):**
    *   Her yarasa için, eğer darbe emisyon oranı (\( r_i \)) rastgele bir sayıdan (\( \text{rand} \)) büyükse, mevcut en iyi çözümlerden biri etrafında yerel bir arama yapılır. Yeni bir çözüm (\( x_{new} \)) rastgele bir yürüyüşle üretilir:
        \[ x_{new} = x_{old} + \epsilon A^t \]
        Burada:
        *   \( x_{old} \), popülasyondaki mevcut en iyi çözümlerden rastgele seçilmiş bir çözümdür.
        *   \( \epsilon \), `[-1, 1]` aralığında rastgele bir sayıdır.
        *   \( A^t \), o anki tüm yarasaların ortalama ses yüksekliğidir.
        *   Bu adım, mevcut iyi çözümlerin etrafında daha hassas bir sömürü (exploitation) sağlar.

4.  **Değerlendirme ve Güncelleme:**
    *   Eğer üretilen yeni çözüm (\( x_{new} \)) mevcut yarasanın çözümünden daha iyiyse ve rastgele bir sayı (\( \text{rand} \)) yarasanın ses yüksekliğinden (\( A_i \)) küçükse, yeni çözüm kabul edilir.
    *   Bu durumda, yarasanın darbe emisyon oranı (\( r_i \)) artırılır ve ses yüksekliği (\( A_i \)) azaltılır:
        \[ r_i^{t+1} = r_i^0 [1 - \exp(-\gamma t)] \]
        \[ A_i^{t+1} = \alpha A_i^t \]
        Burada:
        *   \( r_i^0 \), yarasanın başlangıçtaki darbe emisyon oranıdır.
        *   \( \alpha \) ve \( \gamma \), sabitlerdir (örneğin, \( \alpha \) genellikle 0.9 civarında, \( \gamma \) pozitif bir değerdir). \( \alpha \) ses yüksekliğinin azalma oranını, \( \gamma \) ise darbe emisyon oranının artış hızını kontrol eder.
    *   Global en iyi çözüm (\( x^* \)) güncellenir.

5.  **Durdurma Koşulu (Termination Condition):**
    *   Algoritma, maksimum iterasyon sayısına ulaşıldığında veya başka bir durdurma kriteri karşılandığında sona erer. Aksi takdirde 2. adıma geri döner.

## BA'nın Avantajları

*   **İyi Keşif ve Sömürü Dengesi:** Algoritma, frekans ayarlama mekanizması ile global keşfi ve yerel arama adımı ile sömürüyü dengeler.
*   **Hızlı Yakınsama:** Birçok problemde, özellikle başlangıç aşamalarında hızlı bir yakınsama gösterebilir.
*   **Otomatik Parametre Kontrolü:** Ses yüksekliği ve darbe emisyon oranının otomatik olarak ayarlanması, algoritmanın arama ilerledikçe davranışını adapte etmesine yardımcı olur.
*   **Basitlik:** Diğer bazı gelişmiş metasezgisel algoritmalara kıyasla konsepti ve uygulaması nispeten basittir.

## BA'nın Dezavantajları

*   **Parametre Ayarı:** Algoritmanın performansı \( f_{min}, f_{max}, A_0, r^0, \alpha, \gamma \) gibi birkaç parametrenin doğru seçimine bağlıdır.
*   **Erken Yakınsama Riski:** Global en iyi çözüme çok güçlü bir şekilde yönelme, bazı durumlarda erken yakınsamaya ve çeşitlilik kaybına yol açabilir.
*   **Yerel Arama Stratejisi:** Yerel arama stratejisinin etkinliği probleme bağlı olabilir ve bazen daha sofistike yerel arama yöntemleri gerekebilir.

## Uygulama Alanları

BA, çeşitli mühendislik ve bilimsel optimizasyon problemlerinde başarıyla uygulanmıştır:

*   **Sürekli ve Ayrık Fonksiyon Optimizasyonu**
*   **Mühendislik Tasarımı (örneğin, yapısal optimizasyon, elektromanyetik cihaz tasarımı)**
*   **Görüntü İşleme (örneğin, görüntü eşikleme, sıkıştırma)**
*   **Makine Öğrenmesi (örneğin, kümeleme, sınıflandırma, özellik seçimi)**
*   **Çizelgeleme Problemleri**
*   **Ekonomik Dağıtım Problemleri**

## BA Varyasyonları

Temel BA'nın performansını artırmak ve sınırlamalarını gidermek için birçok varyasyon önerilmiştir:

*   **Kaotik Yarasa Algoritması:** Rastgelelik ve keşif yeteneklerini artırmak için kaotik haritalar kullanır.
*   **Ayrık Yarasa Algoritması:** Kombinatoryal optimizasyon problemlerini çözmek için uyarlanmıştır.
*   **Çok Amaçlı Yarasa Algoritması (MOBA):** Birden fazla amacı aynı anda optimize etmek için geliştirilmiştir.
*   **Yönlendirilmiş Yarasa Algoritması:** Yarasaların hareketini daha etkili yönlendirmek için ek bilgiler kullanır.
*   **Hibrit Yarasa Algoritmaları:** BA'yı diğer optimizasyon teknikleriyle (örneğin, Diferansiyel Gelişim, Simüle Edilmiş Tavlama) birleştirir.

## Sonuç

Yarasa Algoritması, yarasaların sofistike ekolokasyon yeteneklerinden ilham alan, frekans ayarlama, ses yüksekliği ve darbe emisyon oranı gibi ilginç özellikleri bir araya getiren etkili bir metasezgisel optimizasyon tekniğidir. Keşif ve sömürü arasında iyi bir denge kurma potansiyeli sayesinde geniş bir problem yelpazesine uygulanabilir ve optimizasyon alanında aktif bir araştırma konusu olmaya devam etmektedir. 