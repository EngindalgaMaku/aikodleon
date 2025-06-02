# Tavlama Benzetimi (Simulated Annealing - SA)

![Tavlama Benzetimi Algoritması](https://images.pexels.com/photos/2088205/pexels-photo-2088205.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2)

## Tavlama Benzetimi Nedir?

**Tavlama Benzetimi (Simulated Annealing - SA)**, metalürjideki tavlama işleminden esinlenilmiş, olasılıksal bir metasezgisel optimizasyon algoritmasıdır. Tavlama, bir malzemenin kontrollü bir şekilde ısıtılıp yavaşça soğutularak daha düşük enerjili ve daha kararlı bir duruma getirilmesi işlemidir. SA, bu fiziksel süreci taklit ederek global optimumu bulmayı amaçlar. Özellikle büyük ve karmaşık arama uzaylarında yerel optimumlardan kaçınmada etkilidir.

Algoritma, başlangıçta daha yüksek bir "sıcaklıkta" arama yapar, bu da daha kötü çözümleri kabul etme olasılığının daha yüksek olduğu anlamına gelir. Zamanla "sıcaklık" düşürülür (soğutma programı), bu da algoritmanın daha iyi çözümlere doğru yakınsamasını sağlar.

## Temel Kavramlar

*   **Çözüm (Solution):** Optimizasyon probleminin olası bir cevabıdır.
*   **Maliyet Fonksiyonu (Cost Function):** Bir çözümün ne kadar iyi olduğunu değerlendiren fonksiyondur. Amaç genellikle bu fonksiyonu minimize etmek veya maksimize etmektir.
*   **Komşu Çözüm (Neighboring Solution):** Mevcut çözüme küçük bir değişiklik yapılarak elde edilen yeni bir çözümdür.
*   **Sıcaklık (Temperature - T):** Algoritmanın arama davranışını kontrol eden bir parametredir. Yüksek sıcaklıkta, daha kötü çözümlerin kabul edilme olasılığı artar, bu da arama uzayının daha geniş bir şekilde keşfedilmesini sağlar. Düşük sıcaklıkta ise algoritma daha çok mevcut en iyi çözüme odaklanır.
*   **Kabul Olasılığı (Acceptance Probability):** Yeni bulunan bir çözümün, mevcut çözümden daha kötü olsa bile kabul edilme olasılığıdır. Genellikle `exp(-ΔE / T)` formülü ile hesaplanır; burada `ΔE` maliyet farkı, `T` ise mevcut sıcaklıktır.
*   **Soğutma Programı (Cooling Schedule):** Sıcaklığın zamanla nasıl azaltılacağını belirleyen stratejidir. (Örn: `T_new = T_old * α`, burada `α` soğutma oranıdır ve genellikle 1'e yakın bir değerdir, örn. 0.95).

## Tavlama Benzetimi Adımları

1.  **Başlangıç:**
    *   Rastgele bir başlangıç çözümü (`S_current`) oluşturulur.
    *   Bu çözümün maliyeti (`E_current`) hesaplanır.
    *   Başlangıç sıcaklığı (`T_initial`) ve soğutma programı belirlenir.
    *   En iyi çözüm (`S_best`) ve en iyi maliyet (`E_best`) olarak `S_current` ve `E_current` atanır.

2.  **Döngü (Sıcaklık Düşene veya Durma Kriteri Sağlanana Kadar):**
    *   a. **Komşu Üretme:** Mevcut çözüm (`S_current`) üzerinden rastgele bir komşu çözüm (`S_new`) üretilir.
    *   b. **Maliyet Hesaplama:** Yeni çözümün maliyeti (`E_new`) hesaplanır.
    *   c. **Karar Verme:**
        *   Eğer `E_new < E_current` (yeni çözüm daha iyiyse), `S_current = S_new` ve `E_current = E_new` olarak güncellenir.
        *   Eğer `E_new >= E_current` (yeni çözüm daha kötüyse veya eşitse), çözüm `exp(-(E_new - E_current) / T)` olasılığı ile kabul edilir. Eğer kabul edilirse, `S_current = S_new` ve `E_current = E_new` olur.
    *   d. **En İyiyi Güncelleme:** Eğer `E_current < E_best` ise, `S_best = S_current` ve `E_best = E_current` olarak güncellenir.
    *   e. **Sıcaklığı Düşürme:** Sıcaklık (`T`) soğutma programına göre düşürülür.

3.  **Sonuç:** `S_best` en iyi çözüm olarak döndürülür.

## Avantajları

*   **Yerel Optimumlardan Kaçınma:** Daha kötü çözümleri belirli bir olasılıkla kabul etme yeteneği sayesinde yerel minimumlara takılıp kalma riskini azaltır.
*   **Uygulama Kolaylığı:** Kavramsal olarak basit ve uygulaması nispeten kolaydır.
*   **Esneklik:** Farklı türdeki problemlere ve maliyet fonksiyonlarına uyarlanabilir.
*   **Global Optima Yakınsama:** Yeterince yavaş bir soğutma programı ile global optimuma yakınsadığı teorik olarak kanıtlanmıştır (ancak pratikte bu çok uzun sürebilir).

## Dezavantajları

*   **Parametre Ayarı:** Performansı, başlangıç sıcaklığı, soğutma programı ve durma kriterleri gibi parametrelerin doğru ayarlanmasına oldukça bağlıdır. Bu ayarlar probleme özgü olabilir ve deneme yanılma gerektirebilir.
*   **Yavaş Yakınsama:** Özellikle çok büyük arama uzaylarında veya yavaş soğutma programlarında yakınsama yavaş olabilir.
*   **"İyi" Çözüm Garantisi Yok:** Metasezgisel bir algoritma olduğu için her zaman en iyi çözümü bulacağı garanti edilmez, ancak genellikle kabul edilebilir kalitede çözümler sunar.

## Uygulama Alanları

*   **Gezgin Satıcı Problemi (TSP):** En kısa turu bulma.
*   **Devre Tasarımı (VLSI):** Bileşenlerin yerleşimi ve bağlantılarının optimizasyonu.
*   **Görüntü İşleme:** Görüntü restorasyonu ve segmentasyonu.
*   **Makine Öğrenimi:** Hiperparametre optimizasyonu, özellik seçimi.
*   **Çizelgeleme Problemleri:** Görevlerin ve kaynakların zamanlanması.
*   **Protein Katlanması:** Proteinlerin en kararlı üç boyutlu yapılarının bulunması.

## Sonuç

Tavlama Benzetimi, basitliği ve yerel optimumlardan kaçabilme yeteneği nedeniyle geniş bir yelpazede optimizasyon problemlerine başarıyla uygulanmış güçlü bir metasezgisel tekniktir. Algoritmanın etkinliği büyük ölçüde parametre ayarlarına ve problemin doğasına bağlı olsa da, karmaşık optimizasyon görevlerinde değerli bir araç olmaya devam etmektedir. 