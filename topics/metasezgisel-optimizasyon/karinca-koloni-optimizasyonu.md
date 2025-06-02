---
title: Karınca Koloni Optimizasyonu (Ant Colony Optimization - ACO)
description: Karınca Koloni Optimizasyonu (ACO), karıncaların yiyecek arama davranışlarından esinlenerek geliştirilmiş olasılıksal bir metasezgisel optimizasyon algoritmasıdır.
---

## Karınca Koloni Optimizasyonu (ACO) Nedir?

Karınca Koloni Optimizasyonu (ACO), ilk olarak 1990'ların başında Marco Dorigo tarafından önerilen, özellikle kombinatoryal optimizasyon problemlerini çözmek için kullanılan popülasyon tabanlı bir metasezgisel algoritmadır. Algoritma, gerçek karıncaların yuvalarından yiyecek kaynaklarına en kısa yolu bulma yeteneklerinden ilham alır.

Karıncalar yiyecek ararken feromon adı verilen kimyasal bir iz bırakırlar. Diğer karıncalar bu feromon izlerini takip etme eğilimindedir. Başlangıçta yollar rastgele seçilirken, zamanla daha kısa yollar daha sık kullanıldığı için bu yollardaki feromon miktarı artar. Bu pozitif geri besleme mekanizması, koloninin en kısa yolu bulmasına yardımcı olur.

## ACO Algoritmasının Adımları

ACO algoritmasının temel adımları şunlardır:

1.  **Başlatma (Initialization):**
    *   Feromon izleri belirli bir başlangıç değerine ayarlanır.
    *   Bir grup yapay karınca (çözüm üreteçleri) oluşturulur ve başlangıç noktalarına yerleştirilir.

2.  **Çözüm Oluşturma (Solution Construction):**
    *   Her karınca, feromon izlerini ve sezgisel bilgileri (örneğin, mesafeyi) kullanarak adım adım bir çözüm oluşturur.
    *   Bir sonraki adıma geçme olasılığı, o yoldaki feromon miktarı ve sezgisel çekicilik ile doğru orantılıdır.

3.  **Feromon Güncelleme (Pheromone Update):**
    *   Tüm karıncalar çözümlerini tamamladıktan sonra feromon izleri güncellenir.
    *   Bu güncelleme iki aşamada gerçekleşir:
        *   **Buharlaşma (Evaporation):** Tüm yollardaki feromon miktarı belirli bir oranda azaltılır. Bu, eski ve daha az tercih edilen yolların etkisinin azalmasına yardımcı olur.
        *   **Takviye (Reinforcement):** Karıncaların kullandığı yollara, özellikle de iyi çözümler üreten karıncaların kullandığı yollara, feromon eklenir. Eklenen feromon miktarı genellikle çözümün kalitesiyle orantılıdır.

4.  **Durdurma Koşulu (Termination Condition):**
    *   Algoritma, önceden belirlenmiş bir durdurma kriteri (örneğin, maksimum iterasyon sayısı, çözümde belirli bir iyileşme sağlanamaması) karşılanana kadar 2. ve 3. adımları tekrar eder.

## ACO'nun Formülasyonu

Bir `i` noktasından `j` noktasına hareket eden bir karıncanın olasılığı genellikle şu şekilde formüle edilir:

\[ P_{ij}^k = \frac{[\tau_{ij}]^\alpha \cdot [\eta_{ij}]^\beta}{\sum_{l \in N_i^k} [\tau_{il}]^\alpha \cdot [\eta_{il}]^\beta} \]

Burada:
*   \( P_{ij}^k \), `k` adlı karıncanın `i`'den `j`'ye gitme olasılığıdır.
*   \( \tau_{ij} \), `i` ile `j` arasındaki yoldaki feromon miktarıdır.
*   \( \eta_{ij} \), `i` ile `j` arasındaki yolun sezgisel çekiciliğidir (genellikle \( 1/d_{ij} \) olarak alınır, \( d_{ij} \) `i` ile `j` arasındaki mesafedir).
*   \( \alpha \), feromonun göreceli önemini belirleyen bir parametredir.
*   \( \beta \), sezgisel bilginin göreceli önemini belirleyen bir parametredir.
*   \( N_i^k \), `k` adlı karıncanın `i` noktasındayken gidebileceği komşu noktalar kümesidir.

Feromon güncellemesi genellikle şu şekildedir:

\[ \tau_{ij} \leftarrow (1-\rho) \cdot \tau_{ij} + \sum_{k} \Delta \tau_{ij}^k \]

Burada:
*   \( \rho \), feromon buharlaşma oranıdır (0 < \( \rho \) < 1).
*   \( \Delta \tau_{ij}^k \), `k` adlı karıncanın `(i,j)` yolunu kullanarak çözüme yaptığı katkıdır. Genellikle, eğer karınca bu yolu kullandıysa ve iyi bir çözüm bulduysa pozitif bir değer, aksi halde 0 olur.

## ACO'nun Avantajları

*   **Doğal Paralellik:** Algoritma, birden fazla karıncanın aynı anda çözüm aramasına dayandığı için paralel hesaplamaya uygundur.
*   **Pozitif Geri Besleme:** İyi çözümlerin keşfedilmesi, gelecekteki aramaları bu bölgelere yönlendirir.
*   **Dağıtılmış Hesaplama:** Merkezi bir kontrol mekanizmasına ihtiyaç duymaz.
*   **Yerel Optimumlardan Kaçınma:** Olasılıksal yapısı sayesinde yerel optimumlara takılma olasılığı düşüktür.

## ACO'nun Dezavantajları

*   **Yavaş Yakınsama:** Diğer bazı metasezgisel yöntemlere göre yakınsama hızı yavaş olabilir.
*   **Parametre Ayarı:** Algoritmanın performansı \( \alpha, \beta, \rho \) gibi parametrelerin seçimine duyarlıdır ve bu parametrelerin en iyi değerlerini bulmak zor olabilir.
*   **Teorik Analiz Zorluğu:** Algoritmanın stokastik ve karmaşık doğası nedeniyle teorik analizi zordur.

## Uygulama Alanları

ACO, çeşitli optimizasyon problemlerinde başarıyla uygulanmıştır:

*   **Gezgin Satıcı Problemi (TSP):** ACO'nun ilk ve en bilinen uygulama alanıdır.
*   **Araç Rotalama Problemleri (VRP)**
*   **İş Çizelgeleme Problemleri**
*   **Ağ Yönlendirme**
*   **Veri Kümeleme**
*   **Özellik Seçimi**

## Örnek Problem: Gezgin Satıcı Problemi (TSP)

TSP'de amaç, bir grup şehri tam olarak bir kez ziyaret edip başlangıç şehrine en kısa toplam mesafeyle dönmektir. ACO bu probleme şu şekilde uygulanır:

1.  **Karıncalar ve Şehirler:** Her karınca bir gezgin satıcıyı temsil eder. Şehirler, grafiğin düğümleri, şehirler arası yollar ise kenarlarıdır.
2.  **Çözüm Oluşturma:** Her karınca bir başlangıç şehrinden başlar ve ziyaret edilmemiş şehirler arasından olasılıksal bir seçim yaparak bir tur oluşturur. Seçim, feromon seviyeleri ve şehirler arası mesafelere göre yapılır.
3.  **Feromon Güncelleme:** Tüm karıncalar turlarını tamamladıktan sonra, daha kısa turlarda kullanılan yollardaki feromon miktarı artırılır ve tüm yollardaki feromonlar bir miktar buharlaştırılır.
4.  **Tekrarlama:** Bu süreç, tatmin edici bir çözüm bulunana veya maksimum iterasyon sayısına ulaşılana kadar tekrarlanır.

ACO, özellikle karmaşık ve büyük ölçekli kombinatoryal optimizasyon problemlerinde etkili bir çözüm sunar. 