# Yasaklı Arama (Tabu Search)

![Yasaklı Arama Konsepti](https://images.pexels.com/photos/163064/play-ground-rainbow-colors-163064.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2) <!-- Placeholder image, more relevant one can be found -->

## Yasaklı Arama Nedir?

**Yasaklı Arama (Tabu Search - TS)**, Fred Glover tarafından 1986 yılında önerilen, yerel arama tekniklerinin karşılaştığı yerel optimumlara takılma sorununu aşmak için tasarlanmış bir metasezgisel optimizasyon algoritmasıdır. Temel fikir, arama sürecinde daha önce ziyaret edilen çözümlere veya bu çözümlere götüren belirli hareketlere bir süre için "yasak" (tabu) koyarak arama uzayının daha geniş alanlarının keşfedilmesini sağlamaktır. Bu yasaklar, bir "yasaklı listesi" (tabu list) içerisinde tutulur ve algoritmanın döngüsel davranışlardan kaçınmasına yardımcı olur.

Yasaklı Arama, kısa vadeli bir hafıza mekanizması (yasaklı listesi) kullanarak arama sürecini yönlendirir. Bazen, daha uzun vadeli hafıza mekanizmaları da (frekans tabanlı hafıza gibi) arama sürecini çeşitlendirmek veya yoğunlaştırmak için kullanılabilir.

## Temel Kavramlar

*   **Çözüm (Solution):** Optimize edilmeye çalışılan problemin olası bir cevabı.
*   **Komşuluk (Neighborhood):** Mevcut bir çözüme uygulanabilecek küçük değişikliklerle (hareketlerle) ulaşılabilecek çözümler kümesi.
*   **Hareket (Move):** Mevcut bir çözümden bir komşu çözüme geçiş.
*   **Yasaklı Listesi (Tabu List):** Son zamanlarda yapılan hareketlerin veya ziyaret edilen çözümlerin özelliklerini tutan kısa vadeli bir hafızadır. Bu listedeki hareketler veya özellikler belirli bir süre (tabu tenure) için yasaklanır.
*   **Tabu Süresi (Tabu Tenure):** Bir hareketin veya özelliğin yasaklı listesinde ne kadar süre kalacağını belirler. Statik veya dinamik olabilir.
*   **Aday Listesi (Candidate List):** Mevcut çözümün komşuluğundan seçilen ve değerlendirilen potansiyel bir sonraki çözümler kümesi.
*   **İzin Kriteri (Aspiration Criterion):** Yasaklı bir hareketin, belirli koşullar altında (örneğin, şimdiye kadar bulunan en iyi çözümden daha iyi bir çözüme yol açıyorsa) yasağının kaldırılıp seçilmesine izin veren bir kuraldır.
*   **Yoğunlaştırma (Intensification):** Arama uzayının umut verici bölgelerinde daha detaylı arama yapma stratejisi. Genellikle iyi çözümlerin bulunduğu bölgelere odaklanılır.
*   **Çeşitlendirme (Diversification):** Aramayı daha önce keşfedilmemiş bölgelere yönlendirerek arama uzayının geniş bir şekilde taranmasını sağlama stratejisi.

## Yasaklı Arama Adımları

1.  **Başlangıç:**
    *   Rastgele veya bir sezgisel ile bir başlangıç çözümü (`S_current`) oluşturulur.
    *   `S_best` (en iyi çözüm) olarak `S_current` atanır.
    *   Yasaklı listesi boşaltılır.

2.  **Döngü (Belirli Bir Durma Kriteri Sağlanana Kadar - örn. maksimum iterasyon sayısı):**
    *   a. **Komşuluk Oluşturma:** `S_current` çözümünün komşuluğundaki tüm olası hareketler (veya bir alt kümesi) belirlenerek aday çözümler (`N(S_current)`) oluşturulur.
    *   b. **Aday Değerlendirme:** Her bir aday çözüm değerlendirilir.
    *   c. **En İyi Komşuyu Seçme:**
        *   Yasaklı olmayan veya izin kriterini karşılayan adaylar arasından en iyi maliyete sahip olan komşu çözüm (`S_next`) seçilir.
        *   Eğer tüm komşular yasaklıysa ve hiçbiri izin kriterini karşılamıyorsa, farklı stratejiler (örn. en az kötü yasaklı komşuyu seçme) uygulanabilir.
    *   d. **Çözümü ve En İyiyi Güncelleme:**
        *   `S_current = S_next` olarak güncellenir.
        *   Eğer `S_current` çözümünün maliyeti `S_best` çözümünün maliyetinden daha iyiyse, `S_best = S_current` olur.
    *   e. **Yasaklı Listesini Güncelleme:** `S_current`'e ulaşmak için yapılan hareket (veya çözümün bir özelliği) yasaklı listesine eklenir. Eğer liste doluysa, en eski yasaklı eleman çıkarılır. Tabu süresi de bu adımda yönetilir.

3.  **Sonuç:** `S_best` en iyi çözüm olarak döndürülür.

## Avantajları

*   **Yerel Optimumlardan Kaçınma:** Yasaklı listesi sayesinde, algoritmanın daha önce ziyaret ettiği çözümlere hemen geri dönmesi engellenir, bu da yerel optimumlara takılma riskini azaltır.
*   **Esneklik:** Birçok farklı optimizasyon problemine uyarlanabilir. Yasakların ve komşuluk yapılarının probleme özgü tanımlanması mümkündür.
*   **Kavramsal Basitlik:** Temel mekanizması (hafıza ile yerel arama) anlaşılması kolaydır.
*   **Kaliteli Çözümler:** Genellikle birçok problem için yüksek kaliteli çözümler üretebilir.

## Dezavantajları

*   **Parametre Ayarı:** Performansı, tabu süresi, komşuluk yapısı, izin kriterleri gibi parametrelerin doğru ayarlanmasına oldukça bağlıdır. Bu ayarlar probleme özgü olabilir.
*   **Yasaklı Listesinin Yönetimi:** Etkili bir yasaklı listesi tasarımı ve yönetimi önemlidir. Çok kısa tabu süreleri döngülere, çok uzun süreler ise iyi çözümlerin engellenmesine yol açabilir.
*   **Hesaplama Maliyeti:** Geniş komşulukların değerlendirilmesi bazı durumlarda hesaplama yükünü artırabilir.

## Uygulama Alanları

*   **Çizelgeleme Problemleri:** İş atama, makine zamanlama, proje çizelgeleme.
*   **Araç Rotalama Problemleri (VRP):** Araçların en uygun rotalarını belirleme.
*   **Gezgin Satıcı Problemi (TSP).**
*   **Atama Problemleri:** Kaynakların görevlere atanması (örn. Quadratic Assignment Problem - QAP).
*   **Telekomünikasyon:** Ağ tasarımı, frekans atama.
*   **Finans:** Portföy optimizasyonu.
*   **Makine Öğrenimi:** Özellik seçimi.

## Sonuç

Yasaklı Arama, yerel arama yöntemlerini hafıza kullanarak geliştiren güçlü bir metasezgisel tekniktir. Doğru parametre ayarları ve probleme uygun stratejilerle birleştirildiğinde, karmaşık optimizasyon problemlerinde etkili ve kaliteli çözümler bulma potansiyeline sahiptir. 