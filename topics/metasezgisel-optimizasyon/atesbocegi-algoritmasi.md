---
title: Ateşböceği Algoritması (Firefly Algorithm - FA)
description: Ateşböceği Algoritması (FA), ateşböceklerinin yanıp sönen ışıklarını kullanarak birbirlerini çekme davranışlarından esinlenerek geliştirilmiş bir metasezgisel optimizasyon algoritmasıdır.
image: "/blog-images/firefly.jpg"
date: "2023-06-22"
---

## Ateşböceği Algoritması (FA) Nedir?

Ateşböceği Algoritması (FA), 2008 yılında Xin-She Yang tarafından geliştirilen, doğadan ilham alan bir metasezgisel optimizasyon algoritmasıdır. Algoritma, ateşböceklerinin ışık yoğunlukları ve çekicilikleri arasındaki ilişkiyi modelleyerek optimizasyon problemlerini çözer. Temel fikir, daha parlak (daha iyi çözüme sahip) ateşböceklerinin daha az parlak olanları kendilerine çekmesidir.

Algoritmanın üç temel kuralı vardır:

1.  Tüm ateşböcekleri cinsel olarak belirsizdir, bu nedenle bir ateşböceği diğer tüm ateşböceklerine (cinsiyetten bağımsız olarak) çekilebilir.
2.  Çekicilik, parlaklıkla orantılıdır. İki ateşböceği için, daha az parlak olan, daha parlak olana doğru hareket edecektir. Parlaklık, ateşböcekleri arasındaki mesafe arttıkça azalır. Eğer hiçbiri diğerinden daha parlak değilse, rastgele hareket ederler.
3.  Bir ateşböceğinin parlaklığı, amaç fonksiyonunun değeri tarafından belirlenir veya bu değerle ilişkilidir.

## FA Algoritmasının Adımları

FA algoritmasının temel adımları şunlardır:

1.  **Başlatma (Initialization):**
    *   Ateşböceklerinin (çözümlerin) başlangıç popülasyonu rastgele oluşturulur.
    *   Her ateşböceğinin parlaklığı (I), amaç fonksiyonunun değeriyle ilişkilendirilir (genellikle maksimizasyon problemleri için I = f(x), minimizasyon problemleri için I = 1/f(x) veya benzer bir dönüşüm).
    *   Algoritma parametreleri belirlenir: ışık absorpsiyon katsayısı (γ), maksimum çekicilik (β₀), rastgelelik parametresi (α).

2.  **Parlaklık ve Mesafe Hesaplama:**
    *   Her ateşböceği çifti için aralarındaki mesafe (r) hesaplanır.
    *   Işık yoğunluğu, kaynaktan uzaklaştıkça azalır. Bu, ışık absorpsiyon katsayısı (γ) ile modellenir.

3.  **Ateşböceklerinin Hareketi (Movement):**
    *   Her bir ateşböceği `i` için, popülasyondaki diğer tüm daha parlak ateşböcekleri `j`'ye doğru hareket eder.
    *   `i` ateşböceğinin `j` ateşböceğine doğru hareketi şu formülle belirlenir:
        \[ x_i^{t+1} = x_i^t + \beta_0 e^{-\gamma r_{ij}^2} (x_j^t - x_i^t) + \alpha \epsilon_i^t \]
        Burada:
        *   \( x_i^t \), `t` anında `i` ateşböceğinin konumudur.
        *   İlk terim mevcut konumdur.
        *   İkinci terim, `j` ateşböceğinin çekiciliğinden kaynaklanan harekettir.
            *   \( \beta_0 \), \( r=0 \) 'daki maksimum çekiciliktir.
            *   \( e^{-\gamma r_{ij}^2} \), ışık absorpsiyonunu ve mesafeye bağlı çekicilik azalmasını modelleyen Gauss benzeri bir formdur.
            *   \( r_{ij} \), `i` ve `j` ateşböcekleri arasındaki mesafedir.
        *   Üçüncü terim, rastgele bir hareketi temsil eder.
            *   \( \alpha \), rastgelelik parametresidir (adım büyüklüğü ölçekleme faktörü).
            *   \( \epsilon_i^t \), genellikle Gauss veya üniform dağılımdan gelen rastgele bir vektördür.
    *   Eğer popülasyonda daha parlak bir ateşböceği yoksa, ateşböceği rastgele hareket eder.

4.  **Yeni Çözümleri Değerlendirme ve Güncelleme:**
    *   Hareket eden ateşböceklerinin yeni konumlarındaki amaç fonksiyonu değerleri (ve dolayısıyla parlaklıkları) hesaplanır.
    *   En iyi çözüm güncellenir.

5.  **Durdurma Koşulu (Termination Condition):**
    *   Algoritma, maksimum iterasyon sayısı veya kabul edilebilir bir çözüm bulunduğunda sona erer. Aksi takdirde 2. adıma geri döner.

## Çekicilik (Attractiveness)

Bir ateşböceğinin çekiciliği (β), parlaklığıyla doğru orantılıdır. Ancak, bu çekicilik gözlemci ile ateşböceği arasındaki mesafeye (r) bağlı olarak azalır. Çekicilik fonksiyonu genellikle şu şekilde ifade edilir:

\[ \beta(r) = \beta_0 e^{-\gamma r^2} \]

Burada:
*   \( \beta_0 \), \( r=0 \) anındaki (maksimum) çekiciliktir.
*   \( \gamma \), ışık absorpsiyon katsayısıdır ve çekiciliğin azalma hızını kontrol eder. \( \gamma \rightarrow 0 \) ise çekicilik sabittir (\( \beta_0 \)), \( \gamma \rightarrow \infty \) ise çekicilik neredeyse sıfırdır, bu da ateşböceklerinin birbirini görmediği ve rastgele hareket ettiği anlamına gelir.

## FA'nın Avantajları

*   **Otomatik Alt Bölümleme:** Algoritma, popülasyonu otomatik olarak alt gruplara ayırabilir ve her alt grup belirli bir yerel optimum etrafında toplanabilir. Bu, multimodal problemler için faydalıdır.
*   **Farklı Çekiciliklere Duyarlılık:** Farklı ateşböcekleri, farklı \( \gamma \) değerleri kullanarak farklı mesafelerde çekicilik gösterebilir, bu da arama davranışını çeşitlendirir.
*   **Basitlik:** Temel FA'nın uygulanması nispeten kolaydır.
*   **Doğrudan Etkileşim:** Parçacık Sürü Optimizasyonu (PSO) gibi algoritmalarda en iyi birey bilgisi dolaylı olarak kullanılırken, FA'da ateşböcekleri doğrudan birbirlerini etkiler.

## FA'nın Dezavantajları

*   **Parametre Ayarı:** Algoritmanın performansı \( \beta_0, \gamma, \alpha \) gibi parametrelerin doğru seçimine bağlıdır.
*   **Erken Yakınsama:** Popülasyondaki en parlak ateşböceği çok güçlüyse, diğer tüm ateşböcekleri hızla ona doğru yönelebilir ve bu da erken yakınsamaya ve yerel optimuma takılmaya neden olabilir.
*   **Mesafe Hesaplama Maliyeti:** Her iterasyonda tüm ateşböceği çiftleri arasındaki mesafelerin hesaplanması \( O(N^2) \) karmaşıklığına sahiptir (N popülasyon büyüklüğü).

## Uygulama Alanları

FA, çeşitli optimizasyon problemlerinde kullanılmıştır:

*   **Mühendislik Tasarım Problemleri**
*   **Görüntü İşleme ve Örüntü Tanıma**
*   **Çizelgeleme ve Rotalama Problemleri**
*   **Ekonomik Yük Dağıtımı**
*   **Veri Kümeleme ve Sınıflandırma**

## FA Varyasyonları

Temel FA'nın performansını artırmak ve sınırlamalarını gidermek için birçok varyasyon önerilmiştir:

*   **Kaotik Ateşböceği Algoritması (CFA):** Rastgelelik parametrelerini iyileştirmek için kaotik haritalar kullanır.
*   **Ayrık Ateşböceği Algoritması (DFA):** Kombinatoryal optimizasyon problemlerini çözmek için uyarlanmıştır.
*   **Çok Amaçlı Ateşböceği Algoritması (MOFA):** Birden fazla amacı aynı anda optimize etmek için geliştirilmiştir.
*   **Uyarlanabilir Parametreli FA:** Algoritma parametrelerini arama süreci boyunca dinamik olarak ayarlar.

## Sonuç

Ateşböceği Algoritması, ateşböceklerinin büyüleyici ışık sinyallerinden ilham alan etkili bir optimizasyon tekniğidir. Özellikle multimodal ve doğrusal olmayan optimizasyon problemlerinde umut verici sonuçlar göstermiştir. Parametre ayarı ve erken yakınsama gibi zorlukları olsa da, devam eden araştırmalar ve çeşitli varyasyonları ile FA, optimizasyon alanında değerli bir araç olmaya devam etmektedir. 