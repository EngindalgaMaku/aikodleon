# Diferansiyel Gelişim (Differential Evolution - DE)

![Diferansiyel Gelişim Konsepti](https://images.pexels.com/photos/3769021/pexels-photo-3769021.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2) <!-- Placeholder image -->

## Diferansiyel Gelişim Nedir?

**Diferansiyel Gelişim (Differential Evolution - DE)**, Rainer Storn ve Kenneth Price tarafından 1990'larda geliştirilen, sürekli uzaylardaki optimizasyon problemleri için tasarlanmış, popülasyon tabanlı bir metasezgisel arama algoritmasıdır. DE, basitliği, etkinliği ve az sayıda kontrol parametresine sahip olmasıyla bilinir. Genetik Algoritmalara benzer şekilde evrimsel prensiplere dayanır, ancak özellikle vektör farklarını kullanarak yeni aday çözümler üretme şekliyle onlardan ayrılır.

DE'nin temel fikri, popülasyondaki mevcut çözümler (vektörler) arasındaki farkları kullanarak yeni aday çözümler (deneme vektörleri) oluşturmaktır. Bu deneme vektörleri daha sonra mevcut çözümlerle karşılaştırılır ve daha iyi olanlar bir sonraki nesle aktarılır.

## Temel Kavramlar

*   **Popülasyon (Population):** Bir dizi potansiyel çözüm vektöründen oluşur. Her vektör, problemin bir aday çözümünü temsil eder.
*   **Hedef Vektör (Target Vector):** Popülasyondaki, mutasyona uğratılacak olan mevcut bir çözüm vektörü.
*   **Mutasyon (Mutation):** Popülasyondan rastgele seçilen üç farklı vektör kullanılarak bir "fark vektörü" oluşturulur. Bu fark vektörü, ölçeklendirme faktörü (F) ile çarpılarak başka bir rastgele seçilmiş vektöre (temel vektör) eklenir ve böylece bir "gürültülü vektör" (noisy vector) veya "mutant vektör" elde edilir. En yaygın mutasyon stratejilerinden biri `DE/rand/1` olarak bilinir: `mutant_vector = vector_r1 + F * (vector_r2 - vector_r3)`.
*   **Ölçeklendirme Faktörü (Scaling Factor - F):** Fark vektörünün genliğini kontrol eden bir parametredir (genellikle 0 ile 1 arasında, örneğin 0.5).
*   **Çaprazlama (Crossover) / Rekombinasyon (Recombination):** Mutant vektör ile hedef vektör arasında bileşenlerin (genlerin) olasılıksal olarak değiştirilmesiyle bir "deneme vektörü" (trial vector) oluşturulur. Bu işlem, çaprazlama oranı (CR) ile kontrol edilir. Her bileşen için, rastgele bir sayı CR'den küçükse veya bileşen rastgele seçilen bir indeksle eşleşiyorsa, mutant vektörden alınır; aksi takdirde hedef vektörden alınır (binomial çaprazlama).
*   **Çaprazlama Oranı (Crossover Rate - CR):** Deneme vektörüne mutant vektörden ne kadar bileşen alınacağını belirleyen bir olasılık değeridir (genellikle 0 ile 1 arasında, örneğin 0.9).
*   **Seçilim (Selection):** Oluşturulan deneme vektörünün uygunluk değeri, karşılık gelen hedef vektörün uygunluk değeriyle karşılaştırılır. Eğer deneme vektörü daha iyiyse (veya eşitse ve daha iyiyse), bir sonraki nesilde hedef vektörün yerini alır. Aksi takdirde, hedef vektör bir sonraki nesle değişmeden aktarılır (açgözlü seçim).

## Diferansiyel Gelişim Adımları

1.  **Başlangıç:**
    *   Popülasyon boyutu (`NP`) ve problem boyutları (D) belirlenir.
    *   `NP` adet çözüm vektörü arama uzayında rastgele olarak başlatılır.
    *   Kontrol parametreleri belirlenir: ölçeklendirme faktörü (`F`) ve çaprazlama oranı (`CR`).

2.  **Döngü (Belirli Bir Durma Kriteri Sağlanana Kadar - örn. maksimum nesil sayısı):**
    *   Her bir hedef vektör (`x_i`) için popülasyonda:
        *   **a. Mutasyon:** Popülasyondan `x_i`'den farklı rastgele üç vektör (`x_r1`, `x_r2`, `x_r3`) seçilir. Bir mutant vektör (`v_i`) oluşturulur (örn: `v_i = x_r1 + F * (x_r2 - x_r3)`). Sınır ihlalleri kontrol edilir ve gerekirse düzeltilir.
        *   **b. Çaprazlama:** Hedef vektör (`x_i`) ve mutant vektör (`v_i`) kullanılarak bir deneme vektörü (`u_i`) oluşturulur. Her `j` boyutu için:
            `u_ij = v_ij` eğer (`rand_j[0,1) < CR` veya `j == j_rand`)
            `u_ij = x_ij` aksi takdirde.
            (`j_rand`, `[1, D]` arasında rastgele seçilmiş bir indekstir ve en az bir bileşenin mutant vektörden gelmesini sağlar.)
        *   **c. Seçilim:** Deneme vektörünün (`u_i`) uygunluğu, hedef vektörün (`x_i`) uygunluğu ile karşılaştırılır. Eğer `u_i` daha iyiyse, bir sonraki nesilde `x_i`'nin yerini alır. Aksi takdirde `x_i` korunur.

3.  **Sonuç:** Belirlenen durma kriteri sağlandığında, popülasyondaki en iyi uygunluk değerine sahip vektör çözüm olarak döndürülür.

## Avantajları

*   **Basitlik ve Uygulama Kolaylığı:** Az sayıda adımdan oluşur ve anlaşılması, uygulanması kolaydır.
*   **Az Kontrol Parametresi:** Sadece üç ana kontrol parametresi vardır (`NP`, `F`, `CR`).
*   **Güçlü Global Optimizasyon Yeteneği:** Genellikle birçok farklı problem türünde iyi global arama performansı gösterir.
*   **Paralelleştirme Kolaylığı:** Popülasyondaki bireylerin değerlendirilmesi ve güncellenmesi büyük ölçüde paralel olarak yapılabilir.
*   **Sağlamlık (Robustness):** Farklı başlangıç koşullarına ve problem özelliklerine karşı genellikle dirençlidir.

## Dezavantajları

*   **Parametre Ayarı:** Performansı `NP`, `F` ve `CR` parametrelerinin seçimine duyarlı olabilir ve bu parametreler probleme özgü olabilir.
*   **Erken Yakınsama:** Bazı durumlarda, özellikle uygun olmayan parametrelerle veya küçük popülasyonlarla erken yakınsayabilir.
*   **Ayrık Problemler İçin Doğrudan Uygun Değil:** Temelde sürekli optimizasyon için tasarlanmıştır. Ayrık problemlere uygulanması için modifikasyonlar gerektirir.

## Uygulama Alanları

*   **Sayısal Fonksiyon Optimizasyonu:** Çok çeşitli sürekli optimizasyon test problemleri.
*   **Mühendislik Optimizasyonu:** Tasarım optimizasyonu, parametre tahmini.
*   **Makine Öğrenimi:** Sinir ağı ağırlıklarının eğitimi, hiperparametre optimizasyonu.
*   **Kimya ve Fizik:** Moleküler yerleştirme, parametre uydurma.
*   **Ekonomi ve Finans:** Model kalibrasyonu, portföy optimizasyonu.
*   **Sinyal İşleme:** Filtre tasarımı.

## Sonuç

Diferansiyel Gelişim, basitliği, etkinliği ve geniş uygulanabilirliği sayesinde optimizasyon alanında yaygın olarak kullanılan güçlü bir evrimsel algoritmadır. Doğru parametre seçimi ve bazen probleme özgü varyantlarının kullanılmasıyla birçok zorlu sürekli optimizasyon problemini başarıyla çözebilir. 