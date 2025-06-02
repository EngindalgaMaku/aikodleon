---
title: Balina Optimizasyon Algoritması (Whale Optimization Algorithm - WOA)
description: Balina Optimizasyon Algoritması (WOA), kambur balinaların "kabarcık ağı" avlanma tekniğinden esinlenerek geliştirilmiş bir metasezgisel optimizasyon algoritmasıdır.
---

## Balina Optimizasyon Algoritması (WOA) Nedir?

Balina Optimizasyon Algoritması (WOA), 2016 yılında Seyedali Mirjalili ve Andrew Lewis tarafından önerilen, doğadan ilham alan bir metasezgisel optimizasyon algoritmasıdır. Algoritma, kambur balinaların (Megaptera novaeangliae) benzersiz avlanma davranışı olan "kabarcık ağı" (bubble-net feeding) stratejisini taklit eder.

Kambur balinalar, avlarını (genellikle kril veya küçük balık sürüleri) çevrelemek için spiral şeklinde kabarcıklar oluşturarak bir ağ yaratırlar. Bu davranış, optimizasyon problemlerinde keşif ve sömürü aşamalarını modellemek için ilham kaynağı olmuştur.

WOA'nın temel avlanma davranışları şunlardır:

1.  **Avı Çevreleme (Encircling Prey):** Kambur balinalar avın yerini tespit eder ve etrafını sarar. WOA'da, mevcut en iyi çözüm (av) hedeflenir ve diğer arama ajanları (balinalar) bu hedefe doğru güncellenir.
2.  **Kabarcık Ağı Saldırısı (Bubble-net Attacking Method - Sömürü Aşaması):** Bu aşamada iki yaklaşım modellenir:
    *   **Küçülen Çember Mekanizması (Shrinking Encircling Mechanism):** Balinalar, avı çevreleyen çemberi daraltarak saldırır.
    *   **Spiral Güncelleme Pozisyonu (Spiral Updating Position):** Balinalar, avlarına doğru spiral bir yörünge izleyerek yüzerler.
3.  **Av Arama (Search for Prey - Keşif Aşaması):** Balinalar, rastgele bir şekilde av ararlar. Bu, algoritmanın küresel arama yapmasını sağlar.

## WOA Algoritmasının Adımları

WOA algoritmasının temel adımları şunlardır:

1.  **Başlatma (Initialization):**
    *   Balina popülasyonu (çözümler) rastgele oluşturulur.
    *   Her balinanın uygunluk değeri hesaplanır.
    *   En iyi çözüm (\( \vec{X}^* \)) belirlenir.
    *   Parametreler (örneğin, `a`, `A`, `C`, `l`, `p`) ayarlanır.

2.  **Avı Çevreleme (Encircling Prey):**
    *   Balinalar avın (en iyi çözümün) etrafını sarar. Bu davranış şu şekilde modellenir:
        \[ \vec{D} = |\vec{C} \cdot \vec{X}^*(t) - \vec{X}(t)| \]
        \[ \vec{X}(t+1) = \vec{X}^*(t) - \vec{A} \cdot \vec{D} \]
        Burada:
        *   `t`, mevcut iterasyonu gösterir.
        *   \( \vec{X}^* \), en iyi çözümün (avın) pozisyon vektörüdür.
        *   \( \vec{X} \), bir balinanın pozisyon vektörüdür.
        *   \( \vec{A} \) ve \( \vec{C} \) katsayı vektörleridir:
            \[ \vec{A} = 2\vec{a} \cdot \vec{r} - \vec{a} \]
            \[ \vec{C} = 2 \cdot \vec{r} \]
            *   \( \vec{a} \) bileşenleri, iterasyonlar boyunca lineer olarak 2'den 0'a düşürülür.
            *   \( \vec{r} \), `[0, 1]` aralığında rastgele bir vektördür.

3.  **Kabarcık Ağı Saldırısı (Sömürü Aşaması - Bubble-net Attacking Method):**
    *   Bu aşama, bir `p` olasılığına göre seçilen iki mekanizmayı içerir:
        *   **Küçülen Çember Mekanizması (Shrinking Encircling Mechanism) (Eğer \( p < 0.5 \) ve \( |A| < 1 \)):**
            Burada `a` değeri (dolayısıyla `A` değeri) azaltılarak çember daraltılır. \( \vec{X}(t+1) \) yukarıdaki "Avı Çevreleme" denklemleriyle güncellenir.
        *   **Spiral Güncelleme Pozisyonu (Spiral Updating Position) (Eğer \( p \geq 0.5 \)):**
            Balina ile av arasındaki mesafe hesaplanır ve logaritmik bir spiral yörünge oluşturulur:
            \[ \vec{D}' = |\vec{X}^*(t) - \vec{X}(t)| \]
            \[ \vec{X}(t+1) = \vec{D}' \cdot e^{bl} \cdot \cos(2\pi l) + \vec{X}^*(t) \]
            Burada:
            *   \( \vec{D}' \), balinanın ava olan mesafesidir.
            *   `b`, logaritmik spiralin şeklini tanımlayan bir sabittir (genellikle 1).
            *   `l`, `[-1, 1]` aralığında rastgele bir sayıdır.
    *   Balinaların aynı anda hem küçülen bir çemberde yüzdüğünü hem de spiral bir yol izlediğini varsaymak için, bu iki yaklaşımı birleştirmek üzere %50 olasılıkla birini seçmek için `p` kullanılır.

4.  **Av Arama (Keşif Aşaması - Search for Prey):**
    *   Bu aşama, \( |A| \geq 1 \) olduğunda gerçekleşir.
    *   Balinalar, en iyi çözüm yerine rastgele seçilmiş bir balinaya göre pozisyonlarını güncellerler. Bu, algoritmanın küresel arama yapmasını ve yerel optimumlardan kaçmasını sağlar.
        \[ \vec{D} = |\vec{C} \cdot \vec{X}_{rand} - \vec{X}| \]
        \[ \vec{X}(t+1) = \vec{X}_{rand} - \vec{A} \cdot \vec{D} \]
        Burada \( \vec{X}_{rand} \), mevcut popülasyondan rastgele seçilmiş bir balinanın pozisyon vektörüdür.

5.  **Durdurma Koşulu (Termination Condition):**
    *   Algoritma, maksimum iterasyon sayısına ulaşıldığında veya başka bir durdurma kriteri karşılandığında sona erer. Aksi halde, balinaların uygunlukları yeniden hesaplanır, \( \vec{X}^* \) güncellenir, `a`, `A`, `C`, `l`, `p` güncellenir ve 2. adıma geri dönülür.

## WOA'nın Avantajları

*   **Basit ve Az Parametreli:** Algoritmanın yapısı nispeten basittir ve ayarlanması gereken az sayıda temel parametreye sahiptir.
*   **İyi Keşif ve Sömürü Dengesi:** `A` katsayısının adaptif değişimi (dolayısıyla `a` parametresinin azalması) ve rastgele arama mekanizması, keşif ve sömürü arasında iyi bir denge sağlar.
*   **Yerel Optimumlardan Kaçınma:** Keşif aşaması, algoritmanın yerel optimumlara takılma riskini azaltmaya yardımcı olur.
*   **Kolay Uygulanabilirlik:** Diğer bazı karmaşık metasezgisel algoritmalara göre uygulaması daha kolaydır.

## WOA'nın Dezavantajları

*   **Yakınsama Hızı:** Bazı karmaşık veya yüksek boyutlu problemlerde yakınsama hızı yavaş olabilir.
*   **Parametre Ayarı:** Performansı, `b` sabiti ve `l` parametresinin aralığı gibi bazı parametrelerin ayarına duyarlı olabilir, ancak temel parametreler (`a` ve dolayısıyla `A`, `C`) genellikle standart bir şekilde güncellenir.
*   **Çeşitlilik Kaybı:** İterasyonlar ilerledikçe, tüm balinalar en iyi çözüme çok yaklaşabilir ve bu da popülasyon çeşitliliğinin erken kaybına yol açabilir.

## Uygulama Alanları

WOA, geliştirilmesinden bu yana çeşitli optimizasyon problemlerinde umut verici sonuçlar göstermiştir:

*   **Mühendislik Tasarım Problemleri**
*   **Sayısal Fonksiyon Optimizasyonu**
*   **Makine Öğrenmesi (örneğin, özellik seçimi, sinir ağı eğitimi, kümeleme)**
*   **Görüntü İşleme**
*   **Ekonomik Yük Dağıtımı ve Güç Sistemleri**
*   **Çizelgeleme Problemleri**

## WOA Varyasyonları

Temel WOA'nın performansını ve uygulanabilirliğini artırmak için çeşitli varyasyonlar ve melezleştirmeler önerilmiştir:

*   **Ayrık Balina Optimizasyon Algoritması (Discrete WOA):** Kombinatoryal ve ayrık problemler için uyarlanmıştır.
*   **Çok Amaçlı Balina Optimizasyon Algoritması (Multi-Objective WOA - MOWOA):** Birden fazla amacı aynı anda optimize etmek için.
*   **Kaotik Balina Optimizasyon Algoritması:** Algoritmanın rastgelelik ve keşif yeteneklerini geliştirmek için kaotik haritalar kullanır.
*   **Geliştirilmiş WOA (Enhanced WOA):** Keşif-sömürü dengesini iyileştirmek, çeşitliliği korumak veya yerel optimumlardan kaçış mekanizmalarını güçlendirmek için modifikasyonlar içerir.
*   **Diğer Algoritmalarla Melezleştirme:** Örneğin, Diferansiyel Gelişim (DE) veya Yerçekimi Arama Algoritması (GSA) gibi diğer metasezgisel algoritmalarla birleştirilmiş versiyonları.

## Sonuç

Balina Optimizasyon Algoritması, kambur balinaların zeki ve etkili avlanma stratejilerinden ilham alan modern bir metasezgisel optimizasyon tekniğidir. Basit yapısı, az sayıda parametresi ve keşif ile sömürü arasında doğal bir denge kurma yeteneği sayesinde çeşitli optimizasyon problemlerine başarıyla uygulanmaktadır. Optimizasyon literatüründe giderek daha fazla ilgi gören ve geliştirilen bir algoritmadır. 