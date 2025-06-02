---
title: Gri Kurt Optimizasyonu (Grey Wolf Optimizer - GWO)
description: Gri Kurt Optimizasyonu (GWO), gri kurtların sosyal hiyerarşisi ve avlanma davranışlarından esinlenerek geliştirilmiş bir metasezgisel optimizasyon algoritmasıdır.
---

## Gri Kurt Optimizasyonu (GWO) Nedir?

Gri Kurt Optimizasyonu (GWO), 2014 yılında Seyedali Mirjalili, Seyed Mohammad Mirjalili ve Andrew Lewis tarafından önerilen, sürü zekasına dayalı bir metasezgisel algoritmadır. Algoritma, gri kurtların (Canis lupus) doğal liderlik hiyerarşisini ve avlanma mekanizmalarını taklit eder.

Gri kurtlar genellikle 5-12 bireyden oluşan sürüler halinde yaşar ve katı bir sosyal baskınlık hiyerarşisine sahiptirler. Bu hiyerarşi şu şekilde sıralanır:

1.  **Alfa (α):** Sürünün lideridir. Avlanma, uyuma yeri, uyanma zamanı gibi konularda kararları verir. Alfa, en iyi çözümü temsil eder.
2.  **Beta (β):** İkinci seviyedeki kurtlardır ve alfaya yardımcı olurlar. Alfa öldüğünde veya çok yaşlandığında beta onun yerini alabilir. Beta, ikinci en iyi çözümü temsil eder.
3.  **Delta (δ):** Betalardan sonra gelirler ve alfa ile betaya itaat ederken, omega üzerinde baskındırlar. Kaşifler, gözcüler, yaşlılar, avcılar ve bakıcılar gibi rolleri üstlenirler. Delta, üçüncü en iyi çözümü temsil eder.
4.  **Omega (ω):** Hiyerarşinin en altındadır. Genellikle günah keçisi rolündedirler ve en son yemek yerler. Diğer tüm baskın kurtlara itaat etmek zorundadırlar. Omega kurtları, kalan çözümleri temsil eder.

Optimizasyonda bu hiyerarşi, en iyi üç çözümün (alfa, beta, delta) avın (optimum çözümün) konumu hakkında daha fazla bilgiye sahip olduğu varsayımıyla kullanılır. Diğer kurtlar (omegala), bu üç lider kurdun pozisyonlarına göre kendi pozisyonlarını güncellerler.

## GWO Algoritmasının Adımları

GWO algoritmasının temel adımları şunlardır:

1.  **Başlatma (Initialization):**
    *   Gri kurt popülasyonu (çözümler) rastgele oluşturulur.
    *   Her kurdun uygunluk değeri hesaplanır.
    *   En iyi üç çözüm alfa (α), beta (β), ve delta (δ) olarak belirlenir.
    *   Parametreler (örneğin, `a` katsayısı) ayarlanır.

2.  **Avı Çevreleme (Encircling Prey):**
    *   Kurtlar avın etrafını sarar. Bu davranış matematiksel olarak şu şekilde modellenir:
        \[ \vec{D} = |\vec{C} \cdot \vec{X}_p(t) - \vec{X}(t)| \]
        \[ \vec{X}(t+1) = \vec{X}_p(t) - \vec{A} \cdot \vec{D} \]
        Burada:
        *   `t`, mevcut iterasyonu gösterir.
        *   \( \vec{X}_p \), avın pozisyon vektörüdür.
        *   \( \vec{X} \), bir gri kurdun pozisyon vektörüdür.
        *   \( \vec{A} \) ve \( \vec{C} \) katsayı vektörleridir:
            \[ \vec{A} = 2\vec{a} \cdot \vec{r}_1 - \vec{a} \]
            \[ \vec{C} = 2 \cdot \vec{r}_2 \]
            *   \( \vec{a} \) bileşenleri, iterasyonlar boyunca lineer olarak 2'den 0'a düşürülür.
            *   \( \vec{r}_1 \) ve \( \vec{r}_2 \), `[0, 1]` aralığında rastgele vektörlerdir.

3.  **Avlanma (Hunting):**
    *   Alfa, beta ve delta kurtlarının avın potansiyel konumu hakkında daha iyi bilgiye sahip olduğu varsayılır.
    *   Bu nedenle, ilk üç en iyi çözüm (α, β, δ) saklanır ve diğer arama ajanları (omega kurtları), bu en iyi çözümlere göre pozisyonlarını güncellemeye zorlanır.
    *   Omega kurtlarının pozisyon güncelleme denklemleri şunlardır:
        \[ \vec{D}_\alpha = |\vec{C}_1 \cdot \vec{X}_\alpha - \vec{X}|, \quad \vec{D}_\beta = |\vec{C}_2 \cdot \vec{X}_\beta - \vec{X}|, \quad \vec{D}_\delta = |\vec{C}_3 \cdot \vec{X}_\delta - \vec{X}| \]
        \[ \vec{X}_1 = \vec{X}_\alpha - \vec{A}_1 \cdot \vec{D}_\alpha, \quad \vec{X}_2 = \vec{X}_\beta - \vec{A}_2 \cdot \vec{D}_\beta, \quad \vec{X}_3 = \vec{X}_\delta - \vec{A}_3 \cdot \vec{D}_\delta \]
        \[ \vec{X}(t+1) = \frac{\vec{X}_1 + \vec{X}_2 + \vec{X}_3}{3} \]
        Burada \( \vec{X}_\alpha, \vec{X}_\beta, \vec{X}_\delta \) sırasıyla alfa, beta ve delta kurtlarının pozisyonlarıdır ve \( \vec{A}_1, \vec{A}_2, \vec{A}_3 \) ile \( \vec{C}_1, \vec{C}_2, \vec{C}_3 \) yukarıda tanımlanan \( \vec{A} \) ve \( \vec{C} \) gibi hesaplanır.

4.  **Avı Arama (Keşif) ve Saldırı (Sömürü) (Attacking Prey - Exploitation vs. Exploration):**
    *   \( \vec{A} \) vektörünün değeri, keşif ve sömürü dengesini sağlar.
    *   \( \vec{a} \) değeri iterasyonlar boyunca 2'den 0'a azaldığı için, \( \vec{A} \) değeri de `[-a, a]` aralığında değişir.
    *   Eğer \( |A| > 1 \) ise, kurtlar avdan uzaklaşmaya (keşif yapmaya) zorlanır.
    *   Eğer \( |A| < 1 \) ise, kurtlar ava saldırmaya (sömürü yapmaya) zorlanır.
    *   \( \vec{C} \) vektörü de `[0, 2]` aralığında rastgele değerler içerir ve avın yerel optimumlarda takılıp kalmasını önlemek için rastgele ağırlıklar sağlar.

5.  **Durdurma Koşulu (Termination Condition):**
    *   Algoritma, maksimum iterasyon sayısına ulaşıldığında veya başka bir durdurma kriteri karşılandığında sona erer. Aksi halde, kurtların uygunlukları yeniden hesaplanır, alfa, beta ve delta güncellenir ve 2. adıma geri dönülür.

## GWO'nun Avantajları

*   **Basit ve Anlaşılır:** Algoritmanın temel konsepti ve uygulaması nispeten basittir.
*   **Az Parametre:** Ayarlanması gereken az sayıda parametreye sahiptir (temel olarak popülasyon büyüklüğü ve maksimum iterasyon sayısı).
*   **İyi Keşif ve Sömürü Dengesi:** `a` parametresinin adaptif ayarı, keşif ve sömürü arasında doğal bir geçiş sağlar.
*   **Yerel Optimumlardan Kaçınma:** Hiyerarşik yapı ve rastgelelik unsurları, yerel optimumlara takılma riskini azaltır.
*   **Sürü Liderliği:** En iyi çözümlerin (alfa, beta, delta) arama sürecini yönlendirmesi, yakınsamayı hızlandırabilir.

## GWO'nun Dezavantajları

*   **Ayrık Problemlere Doğrudan Uygulanamaz:** Temel GWO, sürekli optimizasyon problemleri için tasarlanmıştır. Ayrık problemler için modifikasyon gerektirir.
*   **Yüksek Boyutlu Problemlerde Performans:** Çok yüksek boyutlu problemlerde performansı düşebilir ("curse of dimensionality").
*   **Çeşitlilik Kaybı:** İterasyonlar ilerledikçe, kurtlar en iyi çözümlere çok yaklaşabilir ve bu da popülasyon çeşitliliğinin kaybına yol açabilir.

## Uygulama Alanları

GWO, çeşitli optimizasyon problemlerinde etkili bir şekilde kullanılmıştır:

*   **Mühendislik Tasarım Optimizasyonu**
*   **Makine Öğrenmesi (örneğin, özellik seçimi, kümeleme, sinir ağı eğitimi)**
*   **Görüntü İşleme**
*   **Ekonomik Yük Dağıtımı ve Güç Sistemleri Optimizasyonu**
*   **Kontrol Sistemleri Tasarımı**
*   **Robotik ve Yol Planlama**

## GWO Varyasyonları

Temel GWO'nun performansını artırmak ve farklı problem türlerine uyarlamak için birçok varyasyon geliştirilmiştir:

*   **Ayrık Gri Kurt Optimizasyonu (Discrete GWO):** Kombinatoryal ve ayrık optimizasyon problemleri için.
*   **Çok Amaçlı Gri Kurt Optimizasyonu (Multi-Objective GWO - MOGWO):** Birden fazla amacı aynı anda optimize etmek için.
*   **Kaotik Gri Kurt Optimizasyonu:** Algoritmanın rastgelelik mekanizmalarını iyileştirmek için kaotik haritalar kullanır.
*   **Geliştirilmiş Gri Kurt Optimizasyonu (Enhanced GWO):** Keşif ve sömürü dengesini veya kaçış mekanizmalarını geliştiren modifikasyonlar içerir.

## Sonuç

Gri Kurt Optimizasyonu, doğadaki etkileyici bir sosyal yapıyı ve avlanma stratejisini modelleyerek güçlü bir optimizasyon aracı sunar. Basitliği, az parametreye sahip olması ve doğal keşif/sömürü mekanizması sayesinde geniş bir problem yelpazesinde başarılı sonuçlar vermiştir. Metasezgisel optimizasyon alanında popüler ve sıkça kullanılan bir algoritmadır. 