# Genetik Algoritmalar (Genetic Algorithms - GA)

![Genetik Algoritmalar Konsept İllüstrasyonu](/images/genetic_algoritm_top.jpg)

Genetik Algoritmalar (GA), Charles Darwin'in doğal seçilim ve evrim teorisinden esinlenerek geliştirilmiş, arama ve optimizasyon problemlerinin çözümü için kullanılan metasezgisel bir yöntemdir. John Holland tarafından 1960'larda temelleri atılan bu algoritmalar, özellikle karmaşık ve geleneksel yöntemlerle çözümü zor olan problemler için güçlü bir alternatif sunar.

## Genetik Algoritmalar Nedir?

Genetik Algoritmalar, bir problemi çözmek için potansiyel çözümlerden oluşan bir popülasyonu (kromozomlar topluluğu) evrimsel süreçlere (seçilim, çaprazlama, mutasyon) tabi tutarak daha iyi çözümlere doğru iteratif bir şekilde yakınsamayı hedefler. Her bir potansiyel çözüm, bir "kromozom" (genellikle bir bit dizisi veya başka bir veri yapısı) ile temsil edilir ve bu kromozomun "uygunluk değeri" (fitness value), çözümün probleme ne kadar iyi uyduğunu gösterir.

## Temel Kavramlar

*   **Popülasyon (Population):** Probleme ait potansiyel çözümlerin (kromozomların) oluşturduğu kümedir.
*   **Kromozom (Chromosome):** Bir potansiyel çözümü temsil eden veri yapısıdır. Genlerden oluşur.
*   **Gen (Gene):** Kromozomun bir parçasıdır ve çözümün belirli bir özelliğini temsil eder.
*   **Uygunluk Fonksiyonu (Fitness Function):** Bir kromozomun (çözümün) probleme ne kadar iyi uyduğunu değerlendiren fonksiyondur. Yüksek uygunluk değeri, daha iyi bir çözümü ifade eder.
*   **Doğal Seçilim (Selection):** Daha iyi uygunluk değerlerine sahip kromozomların bir sonraki nesle aktarılma olasılığının daha yüksek olduğu süreçtir. (Örn: Rulet Tekerleği Seçimi, Turnuva Seçimi)
*   **Çaprazlama (Crossover / Recombination):** İki ebeveyn kromozomdan genetik materyalin birleştirilerek yeni çocuk (yavru) kromozomlar oluşturulmasıdır. Bu, çözüm uzayında yeni noktaların keşfedilmesini sağlar. (Örn: Tek Noktalı Çaprazlama, İki Noktalı Çaprazlama)
*   **Mutasyon (Mutation):** Bir kromozomdaki bir veya daha fazla genin rastgele değiştirilmesi işlemidir. Bu, popülasyona çeşitlilik katar ve yerel optimumlara takılmayı önlemeye yardımcı olur.

## Genetik Algoritma Adımları

Tipik bir Genetik Algoritma aşağıdaki adımları izler:

1.  **Başlangıç Popülasyonu Oluşturma:** Rastgele veya belirli bir yöntemle potansiyel çözümlerden (kromozomlardan) oluşan bir başlangıç popülasyonu oluşturulur.
2.  **Uygunluk Değerlendirmesi:** Popülasyondaki her bir kromozomun uygunluk fonksiyonu kullanılarak uygunluk değeri hesaplanır.
3.  **Döngü Başlangıcı (Yeni Nesil Üretimi):** Belirli bir durdurma kriteri (maksimum nesil sayısı, yeterince iyi bir çözüm bulunması vb.) karşılanana kadar aşağıdaki adımlar tekrarlanır:
    *   **a. Seçilim:** Mevcut popülasyondan, genellikle daha yüksek uygunluk değerine sahip olanlar tercih edilerek, bir sonraki nesli oluşturacak ebeveyn kromozomlar seçilir.
    *   **b. Çaprazlama:** Seçilen ebeveyn çiftlerine belirli bir olasılıkla çaprazlama operatörü uygulanarak yeni yavru kromozomlar üretilir.
    *   **c. Mutasyon:** Yavru kromozomlara (veya bazen tüm popülasyona) düşük bir olasılıkla mutasyon operatörü uygulanarak genlerinde küçük değişiklikler yapılır.
    *   **d. Yeni Popülasyon Oluşturma:** Eski popülasyon, üretilen yavru kromozomlarla (ve bazen elitizm prensibiyle en iyi ebeveynlerle) güncellenerek yeni bir popülasyon oluşturulur.
    *   **e. Uygunluk Değerlendirmesi:** Yeni popülasyondaki kromozomların uygunluk değerleri hesaplanır.
4.  **Döngü Sonu:** Durdurma kriteri karşılandığında, popülasyondaki en iyi uygunluk değerine sahip kromozom, problemin çözümü olarak sunulur.


![Genetik Algoritma Akış Şeması](/images/genetik_algoritm.jpg)


### 🚀 Python ile Pratik Uygulamalar

Genetik algoritmaların gücünü Python ile yazılmış detaylı kod örnekleriyle keşfedin! Temel fonksiyon optimizasyonundan daha karmaşık problemlere kadar çeşitli uygulamaları inceleyerek bu heyecan verici optimizasyon tekniğini derinlemesine anlayın.

[➡️ Genetik Algoritma Örnekleri (Python)](/topics/metasezgisel-optimizasyon/genetik-algoritmalar/genetik-algoritma-ornekleri)

---

## Avantajları

*   Karmaşık ve yüksek boyutlu arama uzaylarında etkilidirler.
*   Türev bilgisine ihtiyaç duymazlar.
*   Paralel işlemeye uygundurlar.
*   Yerel optimumlara takılma olasılıkları bazı geleneksel yöntemlere göre daha düşüktür.
*   Çok çeşitli problemlere uygulanabilirler.

## Dezavantajları

*   En iyi çözümü bulmayı garanti etmezler (sezgiseldirler).
*   Parametre ayarlaması (popülasyon büyüklüğü, çaprazlama ve mutasyon oranları vb.) performansı önemli ölçüde etkileyebilir ve zaman alabilir.
*   Uygunluk fonksiyonunun tasarımı kritik öneme sahiptir ve bazen zor olabilir.
*   Basit problemler için hesaplama maliyeti yüksek olabilir.
*   Yakınsama hızı yavaş olabilir.

## Genetik Algoritmaların Uygulama Alanları

Genetik algoritmalar, geniş bir yelpazede optimizasyon ve arama problemlerine uygulanabilir. Başlıca uygulama alanları şunlardır:

*   **Optimizasyon:** Fonksiyon optimizasyonu, parametre ayarlama, mühendislik tasarımı.
*   **Makine Öğrenmesi:** Özellik seçimi, sinir ağı eğitimi, kural tabanlı sistemlerin geliştirilmesi.
*   **Planlama ve Çizelgeleme:** Üretim planlama, rota optimizasyonu, görev zamanlama.
*   **Ekonomi ve Finans:** Portföy optimizasyonu, ticaret stratejileri.
*   **Biyoinformatik:** Gen dizileme, protein katlanması.
*   **Robotik:** Hareket planlama, robot kontrolü.

## Sonuç

Genetik algoritmalar, karmaşık ve zorlu optimizasyon problemlerini çözmek için güçlü ve esnek bir metasezgisel yaklaşımdır. Doğadan esinlenen bu yöntem, sürekli gelişmekte ve yeni uygulama alanları bulmaktadır. Temel prensiplerini ve çalışma mekanizmalarını anlamak, yapay zeka ve optimizasyon alanında çalışanlar için değerli bir beceridir. 