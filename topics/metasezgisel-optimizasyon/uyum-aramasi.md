# Uyum Araması (Harmony Search - HS)

![Uyum Araması Konsepti](https://images.pexels.com/photos/3785147/pexels-photo-3785147.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2) <!-- Placeholder image -->

## Uyum Araması Nedir?

**Uyum Araması (Harmony Search - HS)**, Zong Woo Geem, Joong Hoon Kim ve G. V. Loganathan tarafından 2001 yılında geliştirilen, müzik performansındaki doğaçlama sürecinden esinlenen bir metasezgisel optimizasyon algoritmasıdır. Müzisyenlerin bir grup içinde daha iyi bir uyum (harmoni) yakalamak için enstrümanlarının perdelerini ayarlamalarına benzer şekilde, HS algoritması da bir çözüm vektörünün değişkenlerini yinelemeli olarak ayarlayarak daha iyi çözümler arar.

HS, basit konsepti, az sayıda matematiksel gereksinimi ve kolay uygulanabilirliği ile bilinir. Özellikle ayrık ve sürekli değişkenlere sahip optimizasyon problemlerinde etkili olabilir.

## Temel Kavramlar

*   **Uyum Hafızası (Harmony Memory - HM):** Popülasyon tabanlı algoritmalardaki popülasyona benzer. HM, arama uzayından bir dizi çözüm vektörünü (uyumları) saklar. Her satır bir çözüm vektörünü temsil eder.
*   **Uyum Hafızası Boyutu (Harmony Memory Size - HMS):** HM'de saklanacak çözüm vektörlerinin sayısı.
*   **Uyum Hafızası Dikkate Alma Oranı (Harmony Memory Considering Rate - HMCR):** Yeni bir çözüm vektörünün bir bileşeninin HM'deki mevcut değerlerden mi seçileceğini yoksa rastgele bir değer mi alacağını belirleyen bir olasılık (0 ile 1 arasında, örneğin 0.7-0.95).
*   **Perde Ayarlama Oranı (Pitch Adjustment Rate - PAR):** HM'den seçilen bir bileşenin, komşu bir değere ayarlanıp ayarlanmayacağını belirleyen bir olasılık (0 ile 1 arasında, örneğin 0.1-0.5).
*   **Bant Genişliği (Bandwidth - bw) veya Mesafe (distance - Fret Width):** Perde ayarlaması yapılırken mevcut değerden ne kadar uzaklaşılabileceğini belirleyen bir parametre. Sürekli değişkenler için küçük bir değerdir, ayrık değişkenler için genellikle 1'dir.
*   **Doğaçlama (Improvisation):** Yeni bir çözüm vektörü (yeni bir uyum) oluşturma süreci. Bu süreç HMCR ve PAR parametreleri kullanılarak yönlendirilir.

## Uyum Araması Adımları

1.  **Başlangıç:**
    *   Optimizasyon problemini ve amaç fonksiyonunu tanımlayın.
    *   HS parametrelerini belirleyin: `HMS`, `HMCR`, `PAR`, `bw` (veya `fw`) ve maksimum doğaçlama sayısı (veya durma kriteri).
    *   Uyum Hafızasını (HM), `HMS` adet rastgele çözüm vektörüyle doldurun ve her birinin amaç fonksiyonu değerini hesaplayın.

2.  **Yeni Bir Uyum Doğaçlama (Improvise a New Harmony):**
    *   Yeni bir çözüm vektörü `x_new = (x_new_1, x_new_2, ..., x_new_N)` oluşturulur (N, değişken sayısıdır). Her bir `x_new_i` bileşeni aşağıdaki gibi belirlenir:
        *   `rand1 = U(0,1)` (0 ile 1 arasında rastgele bir sayı) üretilir.
        *   Eğer `rand1 < HMCR` ise (Uyum Hafızasından Seçim):
            *   `x_new_i`, HM'deki `i`-inci değişkene ait mevcut değerlerden rastgele biri olarak seçilir (`x_new_i ← x_i^k` burada `k`, `{1, ..., HMS}` arasından rastgele seçilir).
            *   `rand2 = U(0,1)` üretilir.
            *   Eğer `rand2 < PAR` ise (Perde Ayarlama):
                *   `x_new_i`, mevcut değerine küçük bir miktar eklenerek veya çıkarılarak ayarlanır. Sürekli değişkenler için: `x_new_i ← x_new_i ± rand3 * bw` (`rand3 = U(0,1)`). Ayrık değişkenler için komşu bir değer seçilir.
        *   Eğer `rand1 ≥ HMCR` ise (Rastgele Seçim):
            *   `x_new_i`, izin verilen aralıktan rastgele bir değer olarak seçilir.

3.  **Uyum Hafızasını Güncelleme (Update Harmony Memory):**
    *   Yeni doğaçlanan uyumun (`x_new`) amaç fonksiyonu değeri hesaplanır.
    *   Eğer `x_new`, HM'deki en kötü uyumdan daha iyiyse, `x_new` en kötü uyumla yer değiştirir. HM, uyumların uygunluk değerlerine göre sıralı tutulabilir.

4.  **Durma Kriterini Kontrol Etme:**
    *   Eğer maksimum doğaçlama sayısına ulaşıldıysa veya başka bir durma kriteri sağlandıysa, algoritma sonlanır. Aksi takdirde, Adım 2'ye geri dönülür.

5.  **Sonuç:** Algoritma durduğunda, HM'deki en iyi uyum, problemin en iyi çözümü olarak kabul edilir.

## Avantajları

*   **Basitlik ve Kavramsal Kolaylık:** Algoritmanın anlaşılması ve uygulanması nispeten kolaydır.
*   **Az Sayıda Parametre:** Diğer bazı metasezgisel algoritmalara göre daha az sayıda ayarlanacak parametreye sahiptir.
*   **Hem Sürekli Hem Ayrık Problemlere Uygulanabilirlik:** Doğası gereği hem sürekli hem de ayrık değişkenli problemleri ele alabilir.
*   **Global ve Yerel Arama Dengesi:** HMCR ve PAR parametreleri, global keşif ve yerel iyileştirme arasında bir denge kurmaya yardımcı olur.
*   **Türevsiz Olması:** Amaç fonksiyonunun türevini gerektirmez, bu da onu karmaşık, türevlenemeyen veya gürültülü fonksiyonlar için uygun hale getirir.

## Dezavantajları

*   **Parametre Ayarı:** Performansı HMCR, PAR ve bw (veya fw) parametrelerinin seçimine duyarlı olabilir. Bu parametreler probleme özgü olabilir.
*   **Erken Yakınsama:** Bazı durumlarda, özellikle çeşitlilik mekanizmaları yetersizse, suboptimal çözümlere erken yakınsayabilir.
*   **Yavaş Yakınsama Hızı:** Çok boyutlu veya karmaşık problemlerde yakınsama hızı yavaş olabilir.
*   **En Kötü Uyumun Değiştirilmesi Stratejisi:** Her zaman en kötü uyumu değiştirmek, bazen popülasyon çeşitliliğini azaltabilir. Bazı varyantlar farklı güncelleme stratejileri kullanır.

## Uygulama Alanları

*   **Su Kaynakları Yönetimi:** Boru ağı tasarımı, rezervuar işletimi.
*   **Yapısal Tasarım:** Kafes kiriş sistemlerinin optimizasyonu, bina tasarımı.
*   **Müzik Besteleme ve Doğaçlama:** Algoritmanın esin kaynağı olan alanda da uygulamaları vardır.
*   **Makine Öğrenimi:** Özellik seçimi, kümeleme, sinir ağı eğitimi.
*   **Lojistik ve Ulaştırma:** Araç rotalama problemleri, çizelgeleme.
*   **Enerji Sistemleri:** Enerji santrali yerleşimi optimizasyonu.
*   **Veri Madenciliği:** Kural keşfi.

## Sonuç

Uyum Araması, müzikal doğaçlama sürecinden ilham alan, anlaşılması ve uygulanması kolay, güçlü bir metasezgisel optimizasyon algoritmasıdır. Çeşitli mühendislik ve bilimsel problemlerde başarılı bir şekilde uygulanmıştır. Diğer metasezgisel yöntemler gibi, performansı parametre ayarlarına ve problemin yapısına bağlı olabilir, ancak basitliği ve esnekliği onu birçok optimizasyon görevi için çekici bir seçenek haline getirir. 