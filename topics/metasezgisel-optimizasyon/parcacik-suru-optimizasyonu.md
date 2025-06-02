# Parçacık Sürü Optimizasyonu (Particle Swarm Optimization - PSO)

![Parçacık Sürü Optimizasyonu Konsept İllüstrasyonu](https://images.pexels.com/photos/1089842/pexels-photo-1089842.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2)

Parçacık Sürü Optimizasyonu (PSO), 1995 yılında James Kennedy ve Russell Eberhart tarafından geliştirilen, popülasyon tabanlı bir stokastik optimizasyon tekniğidir. PSO, kuş sürülerinin veya balık okullarının yiyecek arama gibi kolektif zeki davranışlarından esinlenmiştir. Algoritma, bir grup parçacığın (potansiyel çözümlerin) arama uzayında hareket ederek en iyi çözümü bulmaya çalışmasını simüle eder.

## Parçacık Sürü Optimizasyonu Nedir?

PPSO'da, her bir "parçacık" optimizasyon probleminin bir potansiyel çözümünü temsil eder. Her parçacık, arama uzayındaki mevcut konumunu, hızını ve şimdiye kadar bulduğu en iyi kişisel konumunu ("pbest" - personal best) hatırlar. Ayrıca, tüm sürüdeki herhangi bir parçacığın şimdiye kadar bulduğu en iyi genel konumu ("gbest" - global best) da bilinir. Parçacıklar, bu pbest ve gbest bilgilerini kullanarak hızlarını ve dolayısıyla konumlarını güncellerler, böylece daha umut verici bölgelere doğru hareket ederler.

## Temel Kavramlar

*   **Parçacık (Particle):** Arama uzayında bir potansiyel çözümü temsil eder. Her parçacığın bir konumu ve hızı vardır.
*   **Sürü (Swarm):** Parçacıkların oluşturduğu topluluktur.
*   **Konum (Position):** Parçacığın arama uzayındaki mevcut koordinatlarıdır.
*   **Hız (Velocity):** Parçacığın bir sonraki adımda ne kadar ve hangi yönde hareket edeceğini belirler.
*   **Uygunluk Fonksiyonu (Fitness Function):** Bir parçacığın (çözümün) mevcut konumundaki kalitesini değerlendirir.
*   **Kişisel En İyi (pbest - Personal Best):** Bir parçacığın geçmiş hareketleri boyunca ulaştığı en iyi (en yüksek uygunluk değerine sahip) konumdur.
*   **Global En İyi (gbest - Global Best):** Tüm sürüdeki parçacıklar arasında, algoritmanın başlangıcından itibaren ulaşılan en iyi konumdur. (Farklı topolojilerde "yerel en iyi" - lbest de kullanılabilir.)
*   **Atalet Ağırlığı (Inertia Weight - w):** Parçacığın önceki hızının mevcut hız üzerindeki etkisini kontrol eder. Genellikle arama sürecinin başında büyük, sonunda küçük değerler alır.
*   **Bilişsel Katsayı (Cognitive Coefficient - c1):** Parçacığın kendi kişisel en iyi konumuna (pbest) doğru ne kadar yöneleceğini belirler.
*   **Sosyal Katsayı (Social Coefficient - c2):** Parçacığın sürünün global en iyi konumuna (gbest) doğru ne kadar yöneleceğini belirler.

## PSO Algoritma Adımları

Tipik bir PSO algoritması aşağıdaki adımları izler:

1.  **Başlangıç:**
    *   Sürüdeki parçacık sayısı belirlenir.
    *   Her parçacık için arama uzayında rastgele başlangıç konumları ve hızları atanır.
    *   Her parçacığın uygunluk değeri hesaplanır ve pbest değeri başlangıç konumu olarak ayarlanır.
    *   Sürüdeki en iyi pbest, gbest olarak belirlenir.
2.  **İterasyon (Döngü Başlangıcı):** Belirli bir durdurma kriteri (maksimum iterasyon sayısı, çözümde yeterli iyileşme olmaması vb.) karşılanana kadar aşağıdaki adımlar her parçacık için tekrarlanır:
    *   **a. Hız Güncelleme:** Her parçacığın hızı, mevcut hızı (atalet etkisi), pbest'e olan uzaklığı (bilişsel bileşen) ve gbest'e olan uzaklığı (sosyal bileşen) dikkate alınarak güncellenir. Hız güncelleme formülü genellikle şöyledir:
        `v(t+1) = w * v(t) + c1 * rand() * (pbest - x(t)) + c2 * rand() * (gbest - x(t))`
        Burada `v(t)` mevcut hız, `x(t)` mevcut konum, `rand()` 0-1 arasında rastgele bir sayıdır.
    *   **b. Konum Güncelleme:** Her parçacığın konumu, yeni hesaplanan hızı kullanılarak güncellenir:
        `x(t+1) = x(t) + v(t+1)`
    *   **c. Uygunluk Değerlendirmesi:** Her parçacığın yeni konumundaki uygunluk değeri hesaplanır.
    *   **d. pbest Güncelleme:** Eğer yeni konumdaki uygunluk değeri, parçacığın mevcut pbest değerinden daha iyiyse, pbest yeni konumla güncellenir.
    *   **e. gbest Güncelleme:** Eğer herhangi bir parçacığın yeni pbest değeri, mevcut gbest değerinden daha iyiyse, gbest bu yeni pbest değeriyle güncellenir.
3.  **Döngü Sonu:** Durdurma kriteri karşılandığında, gbest değeri problemin en iyi çözümü olarak sunulur.

## Avantajları

*   Uygulaması görece basittir, az sayıda parametreye sahiptir.
*   Türev bilgisine ihtiyaç duymaz.
*   Genetik Algoritmalara göre genellikle daha hızlı yakınsar.
*   Hem sürekli hem de ayrık optimizasyon problemlerine uyarlanabilir.
*   İyi bir keşif (exploration) ve sömürü (exploitation) dengesi sunabilir.

## Dezavantajları

*   Yerel optimumlara takılma riski vardır, özellikle karmaşık ve çok modlu problemler için.
*   Parametrelerin (w, c1, c2) seçimi performansı etkileyebilir.
*   Problem boyutu arttıkça yakınsama süresi artabilir.

## Uygulama Alanları

PPSO, çeşitli optimizasyon problemlerinde başarılı bir şekilde kullanılmıştır:

*   **Fonksiyon Optimizasyonu:** Matematiksel test fonksiyonlarının optimizasyonu.
*   **Makine Öğrenmesi:** Sinir ağı eğitimi (ağırlıkların optimizasyonu), hiperparametre optimizasyonu, özellik seçimi.
*   **Kontrol Sistemleri:** Kontrolcü parametrelerinin ayarlanması.
*   **Görüntü ve Sinyal İşleme:** Filtre tasarımı, örüntü tanıma.
*   **Robotik:** Rota planlama.
*   **Ekonomik Dağıtım Problemleri:** Güç sistemlerinde maliyet optimizasyonu.

## Sonuç

Parçacık Sürü Optimizasyonu, basitliği, hızı ve etkinliği nedeniyle popüler bir metasezgisel optimizasyon algoritmasıdır. Doğru parametre ayarları ve bazen hibrit yaklaşımlarla birleştirildiğinde, çok çeşitli karmaşık optimizasyon problemlerine güçlü çözümler sunabilir. 