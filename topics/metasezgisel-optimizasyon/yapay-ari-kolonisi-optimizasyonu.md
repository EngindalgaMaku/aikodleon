---
title: Yapay Arı Kolonisi Optimizasyonu (Artificial Bee Colony - ABC)
description: Yapay Arı Kolonisi (ABC) algoritması, bal arılarının yiyecek arama davranışlarından esinlenerek Derviş Karaboğa tarafından 2005 yılında geliştirilmiş bir metasezgisel optimizasyon algoritmasıdır.
---

## Yapay Arı Kolonisi (ABC) Nedir?

Yapay Arı Kolonisi (ABC) algoritması, özellikle sayısal optimizasyon problemleri için tasarlanmış, popülasyon tabanlı bir arama algoritmasıdır. Algoritma, bir arı kolonisindeki yiyecek arama davranışlarını taklit eder. Kolonideki arılar üç gruba ayrılır:

1.  **Görevli Arılar (Employed Bees):** Bu arılar belirli bir yiyecek kaynağına (çözüme) atanmıştır ve bu kaynağın etrafında yeni yiyecek kaynakları (komşu çözümler) ararlar. Daha iyi bir kaynak bulurlarsa, bu yeni bilgiyi kovana getirirler.
2.  **Gözlemci Arılar (Onlooker Bees):** Kovanda bekleyen bu arılar, görevli arıların getirdiği bilgilere (danslarına) göre yiyecek kaynaklarını seçerler. Genellikle daha fazla nektar (daha iyi çözüm kalitesi) içeren kaynakları tercih etme olasılıkları daha yüksektir.
3.  **Kaşif Arılar (Scout Bees):** Bir yiyecek kaynağı tükendiğinde (belirli bir iyileştirme eşiğini geçemediğinde), o kaynağa atanan görevli arı bir kaşif arıya dönüşür ve rastgele yeni bir yiyecek kaynağı aramaya başlar.

## ABC Algoritmasının Adımları

ABC algoritmasının temel adımları şunlardır:

1.  **Başlatma (Initialization):**
    *   Yiyecek kaynaklarının (çözümlerin) başlangıç popülasyonu rastgele oluşturulur.
    *   Her yiyecek kaynağı, çözüm uzayında bir konumu temsil eder ve nektar miktarı (uygunluk değeri) hesaplanır.
    *   Görevli arı sayısı genellikle yiyecek kaynağı sayısına eşittir ve her görevli arı bir yiyecek kaynağına atanır.

2.  **Görevli Arı Aşaması (Employed Bee Phase):**
    *   Her görevli arı, mevcut yiyecek kaynağının komşuluğunda yeni bir yiyecek kaynağı (aday çözüm) üretir.
    *   Yeni kaynağın nektar miktarı (uygunluk değeri) hesaplanır.
    *   Eğer yeni kaynak mevcut kaynaktan daha iyiyse, görevli arı yeni kaynağı benimser ve eski kaynağı unutur. Aksi takdirde mevcut kaynağında kalır.

3.  **Gözlemci Arı Aşaması (Onlooker Bee Phase):**
    *   Gözlemci arılar, görevli arıların bulduğu yiyecek kaynakları arasından seçim yapar. Seçim olasılığı, kaynağın nektar miktarıyla (uygunluk değeriyle) orantılıdır. Genellikle rulet tekerleği seçimi gibi bir yöntem kullanılır.
    *   Seçilen kaynağa giden bir gözlemci arı, o kaynağın komşuluğunda yeni bir yiyecek kaynağı üretir (görevli arıların yaptığı gibi).
    *   Eğer yeni kaynak daha iyiyse, gözlemci arı bu bilgiyi günceller ve görevli arı yeni kaynağı benimser.

4.  **Kaşif Arı Aşaması (Scout Bee Phase):**
    *   Eğer bir yiyecek kaynağı belirli bir sayıda denemeden sonra (LIMIT parametresi) iyileştirilemezse, o kaynak terk edilmiş sayılır.
    *   Bu kaynağa atanan görevli arı bir kaşif arıya dönüşür.
    *   Kaşif arı, çözüm uzayında rastgele yeni bir yiyecek kaynağı üretir.

5.  **Durdurma Koşulu (Termination Condition):**
    *   Algoritma, maksimum iterasyon sayısı veya kabul edilebilir bir çözüm bulunduğunda sona erer. Aksi takdirde 2. adıma geri döner.

## ABC'nin Formülasyonu

**Görevli Arı Aşaması - Yeni Kaynak Üretimi:**

Görevli arının `i` kaynağı için yeni bir aday kaynak \( v_{ij} \) üretme formülü:

\[ v_{ij} = x_{ij} + \phi_{ij} (x_{ij} - x_{kj}) \]

Burada:
*   \( x_{ij} \), mevcut `i` kaynağının `j` boyutundaki değeridir.
*   \( x_{kj} \), rastgele seçilmiş bir komşu `k` kaynağının `j` boyutundaki değeridir (\( k \neq i \)).
*   \( \phi_{ij} \), `[-1, 1]` aralığında rastgele bir sayıdır. Bu, arama adımının büyüklüğünü ve yönünü kontrol eder.

**Gözlemci Arı Aşaması - Kaynak Seçim Olasılığı:**

Bir gözlemci arının `i` kaynağını seçme olasılığı \( p_i \):

\[ p_i = \frac{fit_i}{\sum_{n=1}^{SN} fit_n} \]

Burada:
*   \( fit_i \), `i` kaynağının uygunluk değeridir (genellikle nektar miktarıyla orantılıdır).
*   \( SN \), toplam yiyecek kaynağı sayısıdır (görevli arı sayısına eşittir).

## ABC'nin Avantajları

*   **Basitlik:** Algoritmanın konsepti ve uygulaması nispeten basittir.
*   **Daha Az Kontrol Parametresi:** Diğer birçok metasezgisel algoritmalara kıyasla daha az sayıda kontrol parametresine (koloni büyüklüğü, LIMIT, maksimum iterasyon sayısı) sahiptir.
*   **Yerel Optimumlardan Kaçınma Yeteneği:** Kaşif arı mekanizması, algoritmanın yerel optimumlara takılmasını önlemeye yardımcı olur.
*   **İyi Keşif ve Sömürü Dengesi:** Görevli ve gözlemci arılar sömürü (exploitation) yaparken, kaşif arılar keşif (exploration) yapar.

## ABC'nin Dezavantajları

*   **Yakınsama Hızı:** Bazı karmaşık problemlerde yakınsama hızı yavaş olabilir.
*   **LIMIT Parametresinin Ayarlanması:** `LIMIT` parametresinin uygun şekilde ayarlanması önemlidir. Küçük bir değer erken terk etmeye, büyük bir değer ise gereksiz hesaplamalara yol açabilir.

## Uygulama Alanları

ABC algoritması, çeşitli mühendislik ve bilim alanlarındaki optimizasyon problemlerinde başarıyla kullanılmıştır:

*   **Sayısal Fonksiyon Optimizasyonu**
*   **Makine Öğrenmesi Model Eğitimi (örneğin, sinir ağları ağırlıklarının optimizasyonu)**
*   **Veri Kümeleme**
*   **Görüntü İşleme**
*   **Çizelgeleme Problemleri**
*   **Mühendislik Tasarım Optimizasyonu**

## Sonuç

Yapay Arı Kolonisi algoritması, arıların zeki yiyecek arama davranışlarını modelleyerek güçlü bir optimizasyon aracı sunar. Basitliği, az sayıda kontrol parametresine sahip olması ve keşif ile sömürü arasında iyi bir denge kurabilmesi sayesinde geniş bir problem yelpazesine uygulanabilir. 