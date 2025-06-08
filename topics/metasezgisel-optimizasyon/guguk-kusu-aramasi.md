---
title: Guguk Kuşu Araması (Cuckoo Search - CS)
description: Guguk Kuşu Araması (Cuckoo Search - CS), guguk kuşlarının kuluçka parazitizmi davranışlarından ve Lévy uçuşlarından esinlenerek geliştirilmiş bir metasezgisel optimizasyon algoritmasıdır.
image: "/blog-images/gugukkusu.jpg"
date: "2023-06-28"
---

## Guguk Kuşu Araması (Cuckoo Search - CS) Nedir?

Guguk Kuşu Araması (Cuckoo Search - CS), 2009 yılında Xin-She Yang ve Suash Deb tarafından geliştirilen, doğadan ilham alan bir optimizasyon algoritmasıdır. Algoritma, bazı guguk kuşu türlerinin agresif üreme stratejisi olan kuluçka parazitizminden ve bazı kuşların ve meyve sineklerinin karakteristik özelliği olan Lévy uçuşlarından (Lévy flights) esinlenmiştir.

Algoritmanın temelini oluşturan üç idealize edilmiş kural şunlardır:

1.  Her guguk kuşu bir seferde bir yumurta bırakır ve bunu rastgele seçilmiş bir yuvaya bırakır.
2.  En iyi yumurtalara (çözümlere) sahip en iyi yuvalar bir sonraki nesle taşınacaktır.
3.  Mevcut yuva sayısı sabittir ve bir ev sahibi kuş tarafından bir guguk kuşu yumurtasının keşfedilme olasılığı \( p_a \) dır. Bu durumda, ev sahibi kuş ya guguk kuşunun yumurtasını atabilir ya da yuvayı terk edip tamamen yeni bir yuva inşa edebilir.

Basitlik açısından, son varsayım, \( p_a \) olasılığıyla mevcut yuvaların bir kısmının terk edilip yerine yeni yuvaların (yeni rastgele çözümlerin) inşa edilmesiyle yaklaşık olarak ifade edilebilir.

## CS Algoritmasının Adımları

CS algoritmasının temel adımları şunlardır:

1.  **Başlatma (Initialization):**
    *   `n` adet ev sahibi yuvasından oluşan bir başlangıç popülasyonu rastgele oluşturulur. Her yuva bir çözümü temsil eder.
    *   Amaç fonksiyonu değerleri (uygunluk) her yuva için hesaplanır.
    *   Algoritma parametreleri belirlenir: keşif olasılığı (\( p_a \)), adım boyutu ölçekleme faktörü (α).

2.  **Yeni Çözümler Üretme (Lévy Uçuşları ile):**
    *   Rastgele bir guguk kuşu (bir yuva `i`) seçilir.
    *   Bu guguk kuşu için Lévy uçuşları gerçekleştirilerek yeni bir çözüm (\( x_i^{t+1} \)) üretilir. Lévy uçuşları, nadir uzun sıçramalarla birlikte birçok kısa adımdan oluşan bir rastgele yürüyüş türüdür. Yeni çözüm genellikle şu şekilde üretilir:
        \[ x_i^{t+1} = x_i^t + \alpha \oplus \text{Lévy}(\lambda) \]
        Burada:
        *   \( x_i^t \), `t` anındaki `i` yuvasının (çözümünün) konumudur.
        *   \( \alpha > 0 \), adım boyutuyla ilgili bir ölçekleme faktörüdür. Genellikle \( \alpha = O(L/10) \) alınır, burada L problemin karakteristik ölçeğidir.
        *   \( \oplus \), giriş bazlı çarpma anlamına gelir (entry-wise multiplication).
        *   \( \text{Lévy}(\lambda) \), Lévy dağılımından çekilen bir rastgele sayıdır. \( \lambda \) (1 < \( \lambda \) ≤ 3) dağılımın bir parametresidir.
            Lévy uçuşları, arama uzayının hem yerel hem de küresel olarak verimli bir şekilde keşfedilmesini sağlar.

3.  **Değerlendirme ve Seçim:**
    *   Yeni üretilen çözümün (\( x_i^{t+1} \)) uygunluğu hesaplanır.
    *   Rastgele başka bir yuva `j` seçilir.
    *   Eğer yeni çözüm \( x_i^{t+1} \), yuva `j`'deki çözümden daha iyiyse, `j` yuvasındaki çözüm \( x_i^{t+1} \) ile değiştirilir.

4.  **Yuvaların Terk Edilmesi ve Yenilerinin İnşa Edilmesi:**
    *   Popülasyondaki en kötü yuvaların bir kısmı (\( p_a \) olasılığıyla) terk edilir.
    *   Bu terk edilen yuvaların yerine, arama uzayında rastgele yeni yuvalar (yeni çözümler) oluşturulur.

5.  **En İyi Çözümü Saklama:**
    *   Mevcut en iyi çözüm saklanır.

6.  **Durdurma Koşulu (Termination Condition):**
    *   Algoritma, maksimum iterasyon sayısı veya kabul edilebilir bir çözüm bulunduğunda sona erer. Aksi takdirde 2. adıma geri döner.

## Lévy Uçuşları (Lévy Flights)

Lévy uçuşları, CS algoritmasının önemli bir bileşenidir ve keşif yeteneğini artırır. Bu rastgele yürüyüş türü, adım uzunluklarının ağır kuyruklu bir olasılık dağılımı olan Lévy dağılımını izlediği anlamına gelir. Bu, algoritmanın zaman zaman büyük adımlar atarak yerel optimumlardan kaçmasına ve arama uzayının uzak bölgelerini keşfetmesine olanak tanır.

Pratik uygulamalarda, Lévy uçuşlarından rastgele sayılar üretmek için Mantegna algoritması gibi algoritmalar kullanılabilir.

## CS'nin Avantajları

*   **Basitlik ve Az Parametre:** Algoritma nispeten basittir ve Ateşböceği Algoritması veya Parçacık Sürü Optimizasyonu gibi diğer metasezgisel algoritmalara göre daha az ayarlanması gereken parametreye sahiptir (genellikle sadece `n`, `p_a` ve `α`).
*   **Etkili Keşif:** Lévy uçuşları sayesinde hem yerel hem de küresel arama yeteneği dengelidir, bu da algoritmanın yerel optimumlara takılma olasılığını azaltır.
*   **İyi Yakınsama Oranı:** Birçok optimizasyon probleminde diğer algoritmalara göre daha hızlı veya karşılaştırılabilir bir yakınsama oranı gösterdiği rapor edilmiştir.
*   **Geniş Uygulanabilirlik:** Sürekli ve ayrık optimizasyon problemlerine kolayca uyarlanabilir.

## CS'nin Dezavantajları

*   **Parametre Hassasiyeti:** Az sayıda olmasına rağmen, \( p_a \) ve \( \alpha \) parametrelerinin seçimi performansı etkileyebilir.
*   **Lévy Uçuşlarının Hesaplanması:** Lévy uçuşlarından rastgele sayılar üretmek, basit üniform veya Gauss dağılımlarına göre biraz daha karmaşık olabilir.

## Uygulama Alanları

CS, başlangıcından bu yana çeşitli optimizasyon problemlerinde başarıyla uygulanmıştır:

*   **Mühendislik Tasarım Problemleri (örneğin, yay tasarımı, kaynaklı kiriş tasarımı)**
*   **Sayısal Fonksiyon Optimizasyonu**
*   **Makine Öğrenmesi (örneğin, özellik seçimi, sinir ağı eğitimi)**
*   **Enerji Sistemleri (örneğin, enerji üretim planlaması)**
*   **Görüntü İşleme**
*   **Çizelgeleme Problemleri**

## CS Varyasyonları

Temel CS algoritmasının performansını daha da artırmak için çeşitli varyasyonlar önerilmiştir:

*   **Ayrık Guguk Kuşu Araması (Discrete Cuckoo Search):** Kombinatoryal problemler için uyarlanmıştır.
*   **Çok Amaçlı Guguk Kuşu Araması (Multi-Objective Cuckoo Search - MOCS):** Birden fazla amacı aynı anda optimize etmek için geliştirilmiştir.
*   **Kaotik Guguk Kuşu Araması:** Lévy uçuşlarındaki rastgeleliği artırmak veya parametreleri ayarlamak için kaotik haritalar kullanır.
*   **Uyarlanabilir Parametreli CS:** Algoritma parametrelerini arama süreci boyunca dinamik olarak ayarlar.

## Sonuç

Guguk Kuşu Araması, basitliği, az sayıda parametresi ve Lévy uçuşları sayesinde sağladığı etkili keşif mekanizması ile güçlü bir metasezgisel optimizasyon algoritmasıdır. Birçok farklı alandaki optimizasyon problemine başarıyla uygulanmış ve umut verici sonuçlar vermiştir. Devam eden araştırmalar, CS'nin yeteneklerini daha da genişletmekte ve onu optimizasyon araç kutusunda değerli bir seçenek haline getirmektedir. 