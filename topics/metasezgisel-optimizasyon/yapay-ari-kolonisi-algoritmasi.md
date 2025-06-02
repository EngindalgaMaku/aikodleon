# Yapay Arı Kolonisi Algoritması (ABC)

![Yapay Arı Kolonisi Konsepti](https://images.pexels.com/photos/798366/pexels-photo-798366.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2)

## Yapay Arı Kolonisi Algoritması Nedir?

**Yapay Arı Kolonisi (Artificial Bee Colony - ABC) Algoritması**, Derviş Karaboğa tarafından 2005 yılında geliştirilen, bal arısı sürülerinin yiyecek arama davranışlarını taklit eden bir metasezgisel optimizasyon algoritmasıdır. ABC algoritması, özellikle sayısal optimizasyon problemleri için tasarlanmıştır ve basitliği, esnekliği ve az sayıda kontrol parametresine sahip olması nedeniyle popülerlik kazanmıştır.

Algoritma, üç tür yapay arıdan oluşur:
*   **İşçi Arılar (Employed Bees):** Belirli bir yiyecek kaynağına (çözüme) bağlıdırlar ve bu kaynağın komşuluğunda yeni yiyecek kaynakları (yeni çözümler) ararlar. Kaynakla ilgili bilgiyi (konum, nektar miktarı) diğer arılarla paylaşırlar.
*   **Gözcü Arılar (Onlooker Bees):** Kovandaki işçi arıların topladığı bilgilere (genellikle nektar miktarına orantılı olasılıkla) göre bir yiyecek kaynağı seçerler ve bu kaynağın komşuluğunda arama yaparlar.
*   **Kaşif Arılar (Scout Bees):** Bir yiyecek kaynağı belirli bir deneme sayısı (limit) sonunda iyileştirilemezse, işçi arı bu kaynağı terk eder ve yeni bir yiyecek kaynağını rastgele arayan bir kaşif arıya dönüşür. Bu mekanizma, algoritmanın yerel optimumlara takılmasını engellemeye yardımcı olur.

## Temel Kavramlar

*   **Yiyecek Kaynağı (Food Source):** Optimizasyon probleminde bir potansiyel çözümü temsil eder. Her kaynağın bir "nektar miktarı" vardır, bu da çözümün uygunluk değerine karşılık gelir.
*   **Nektar Miktarı (Nectar Amount):** Bir yiyecek kaynağının kalitesini (uygunluk değerini) gösterir.
*   **İşçi Arı Sayısı:** Genellikle popülasyondaki yiyecek kaynağı sayısına eşittir.
*   **Gözcü Arı Sayısı:** Genellikle işçi arı sayısına eşit veya yakın bir değerdedir.
*   **Kaşif Arı:** Terk edilmiş bir kaynağın işçi arısının dönüşmüş halidir.
*   **Limit Parametresi:** Bir yiyecek kaynağının, iyileştirilmeden terk edilmeden önce kaç kez denenebileceğini belirleyen bir eşik değeridir.
*   **Popülasyon:** Yiyecek kaynakları (çözümler) kümesidir.

## ABC Algoritma Adımları

1.  **Başlangıç:**
    *   Popülasyondaki yiyecek kaynağı sayısı (SN), işçi arı ve gözcü arı sayısı belirlenir.
    *   SN adet yiyecek kaynağı (çözüm) rastgele olarak arama uzayında oluşturulur.
    *   Her bir yiyecek kaynağının nektar miktarı (uygunluk değeri) hesaplanır.
    *   Limit parametresi belirlenir.

2.  **Döngü (Belirli Bir Durma Kriteri Sağlanana Kadar - örn. maksimum iterasyon sayısı):**
    *   **a. İşçi Arı Fazı:**
        *   Her işçi arı, mevcut yiyecek kaynağının komşuluğunda yeni bir yiyecek kaynağı (aday çözüm) üretir. Aday çözüm genellikle şu formülle üretilir: `v_ij = x_ij + φ_ij * (x_ij - x_kj)` burada `x_i` mevcut kaynak, `x_k` rastgele seçilmiş farklı bir kaynak, `j` rastgele seçilmiş bir boyut ve `φ_ij` [-1, 1] arasında rastgele bir sayıdır.
        *   Yeni kaynağın nektar miktarı hesaplanır.
        *   Eğer yeni kaynak mevcut kaynaktan daha iyiyse (daha fazla nektara sahipse), işçi arı yeni kaynağı ezberler ve eskisini unutur. Aksi takdirde eski kaynağı tutar ve kaynağın "deneme sayacı" bir artırılır.
    *   **b. Gözcü Arı Fazı:**
        *   Tüm işçi arılar bilgilerini (kaynakların nektar miktarları) paylaştıktan sonra, her gözcü arı, nektar miktarıyla orantılı bir olasılıkla bir yiyecek kaynağı seçer (rulet tekerleği seçimi gibi).
        *   Seçilen kaynağın komşuluğunda, işçi arı fazındaki gibi yeni bir aday kaynak üretir.
        *   Yeni kaynağın nektar miktarı hesaplanır.
        *   Açgözlü bir seçimle (greedy selection), eğer yeni kaynak daha iyiyse, gözcü arı yeni kaynağı ezberler ve eskisini unutur. Aksi takdirde eski kaynağı tutar ve kaynağın "deneme sayacı" bir artırılır.
    *   **c. Kaşif Arı Fazı:**
        *   Eğer bir yiyecek kaynağının "deneme sayacı" önceden tanımlanmış `limit` değerini aşarsa, bu kaynak terk edilmiş sayılır.
        *   Bu kaynağa bağlı olan işçi arı bir kaşif arıya dönüşür ve arama uzayında tamamen rastgele yeni bir yiyecek kaynağı üretir. Bu yeni kaynağın deneme sayacı sıfırlanır.
    *   **d. En İyiyi Kaydetme:** Döngü boyunca bulunan en iyi yiyecek kaynağı (en yüksek nektar miktarına sahip çözüm) kaydedilir.

3.  **Sonuç:** Belirlenen durma kriteri sağlandığında, bulunan en iyi yiyecek kaynağı döndürülür.

## Avantajları

*   **Basitlik ve Anlaşılırlık:** Algoritmanın temel prensipleri ve uygulanışı nispeten basittir.
*   **Az Kontrol Parametresi:** Diğer birçok metasezgisel algoritmaya göre daha az sayıda ayarlanması gereken kontrol parametresi vardır (genellikle popülasyon boyutu ve limit).
*   **İyi Keşif ve Sömürü Dengesi:** İşçi ve gözcü arılar sömürü (exploitation) sağlarken, kaşif arılar keşif (exploration) yeteneğini artırır.
*   **Yerel Optimumlardan Kaçınma Yeteneği:** Kaşif arı mekanizması, algoritmanın yerel optimumlara takılıp kalma riskini azaltmaya yardımcı olur.
*   **Farklı Problem Türlerine Uygulanabilirlik:** Özellikle sürekli optimizasyon problemlerinde etkilidir, ancak ayrık problemlere de uyarlanabilir.

## Dezavantajları

*   **Yakınsama Hızı:** Bazı karmaşık problemlerde yakınsama hızı diğer bazı algoritmalara göre daha yavaş olabilir.
*   **Parametre Hassasiyeti:** Az sayıda olsa da, `limit` parametresinin ve popülasyon boyutunun performansa etkisi olabilir ve probleme göre ayarlanması gerekebilir.
*   **Yüksek Boyutlu Problemler:** Çok yüksek boyutlu arama uzaylarında performansı düşebilir.

## Uygulama Alanları

*   **Sayısal Fonksiyon Optimizasyonu:** Standart test fonksiyonları üzerinde iyi performans gösterir.
*   **Mühendislik Tasarım Problemleri:** Yapısal tasarım, elektronik devre optimizasyonu.
*   **Makine Öğrenimi:** Sinir ağı eğitimi, kümeleme, özellik seçimi.
*   **Görüntü İşleme:** Görüntü segmentasyonu, eşikleme.
*   **Çizelgeleme Problemleri.**
*   **Veri Madenciliği.**

## Sonuç

Yapay Arı Kolonisi algoritması, arıların zeki kolektif davranışlarından ilham alan, güçlü ve esnek bir optimizasyon tekniğidir. Basit yapısı, az sayıda parametresi ve iyi bir keşif-sömürü dengesi sunması nedeniyle geniş bir problem yelpazesinde etkili bir şekilde kullanılmaktadır. 