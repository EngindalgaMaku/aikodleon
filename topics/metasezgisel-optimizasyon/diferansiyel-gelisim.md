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

## Matematiksel Formülasyon

Diferansiyel Gelişim algoritmasının temel adımları aşağıdaki matematiksel denklemlerle ifade edilebilir:

1.  **Mutasyon:**
    Her bir hedef vektör için popülasyondan rastgele seçilen üç farklı vektör (`Xr1`, `Xr2`, `Xr3`) kullanılarak bir mutant vektör (`Xv,yeni`) oluşturulur.
    \[ X_{v,yeni} = X_{r1} + F \cdot (X_{r2} - X_{r3}) \quad (Denklem\ 2.3) \]
    Burada `F`, mutasyon ölçek faktörüdür (`F ∈ [0, 2]`).

2.  **Çaprazlama (Rekombinasyon):**
    Mutant vektör (`Xv,yeni`) ile hedef vektör (`Xi`) arasında bileşenlerin olasılıksal olarak değiştirilmesiyle bir deneme vektörü (`Xu,yeni`) oluşturulur.
    \[ X_{u,yeni,j} = \begin{cases} X_{v,yeni,j} & \text{if } rand_j \leq CR \text{ veya } j = j_{rand} \\ X_{i,j} & \text{aksi takdirde} \end{cases} \quad (Denklem\ 2.8) \]
    Burada `CR`, çaprazlama oranıdır (`CR ∈ [0, 1]`) ve `j_rand`, en az bir bileşenin mutant vektörden gelmesini sağlayan rastgele bir indekstir.

3.  **Seçilim:**
    Oluşturulan deneme vektörünün uygunluk değeri, hedef vektörün uygunluk değeriyle karşılaştırılır. Eğer deneme vektörü daha iyiyse, bir sonraki nesilde hedef vektörün yerini alır.
    \[ X_{i,yeni} = \begin{cases} X_{u,yeni} & \text{if } f(X_{u,yeni}) \leq f(X_i) \\ X_i & \text{aksi takdirde} \end{cases} \quad (Denklem\ 2.9) \]

### Parametreler ve Bileşenler

| Sembol | Açıklama |
| :--- | :--- |
| **Optimizasyon Bileşenleri** | |
| \(X_{yeni}\) | Çaprazlama sonrası oluşturulan yeni aday çözüm (deneme vektörü) |
| \(X_{v,yeni}\) | Mutasyon sonrası oluşturulan aday çözüm (mutant vektör) |
| \(X_p\) | Başlangıç matrisinden rastgele seçilen p. kromozom |
| \(q, r\) | Başlangıç matrisinden rastgele seçilen q. ve r. kromozomlar |
| **Algoritma Parametreleri** | |
| `nc` | Kromozom sayısı |
| `F` | Ağırlıklandırma faktörü (`F ∈ [0, 2]`) |
| `CR` | Çaprazlama olasılığı (`CR ∈ [0, 1]`) |
| **Fonksiyonlar** | |
| `rand()` | [0, 1] aralığında rastgele sayı üreten fonksiyon |
| `ceil()` | Kendisine eşit veya kendisinden büyük pozitif tam sayıya yuvarlayan fonksiyon |

## Uygulama Örnekleri

### MATLAB Kodu

```matlab
% İTERASYON SÜRECİ
for dongu = 1:durma_kriteri
    
    % BAŞLANGIÇ MATRİSİNDEN RASSAL OLARAK ÜÇ FARKLI ÇÖZÜMÜN SEÇİLMESİ
    for kr = 1:nc
        p = ceil(rand() * nc);
        q = ceil(rand() * nc);
        r = ceil(rand() * nc);
        
        % X(yeni,1) tasarım değişkeni için mutasyona uğrayan (yeni) aday çözüm değeri
        X1yeni = OPT(1,p) + F * (OPT(1,q) - OPT(1,r));
        
        % X(yeni,2) tasarım değişkeni için mutasyona uğrayan (yeni) aday çözüm değeri
        X2yeni = OPT(2,p) + F * (OPT(2,q) - OPT(2,r));
    end
    
    % ÇAPRAZLAMA AŞAMASI (Bknz. Denklem 2.7)
    rand_kr = ceil(rand() * D); % D: Değişken sayısı
    
    if (rand() < CR) || (kr == rand_kr)
        X1 = X1yeni;
    else
        X1 = OPT(1, kr);
    end
    
    % SEÇİLİM AŞAMASI (Bknz. Denklem 2.9)
    % ... uygunluk karşılaştırması ve güncelleme ...
    
end
```

### Python Kodu

```python
import math
from random import random

# İTERASYON SÜRECİ
for dongu in range(durma_kriteri):
    
    # BAŞLANGIÇ MATRİSİNDEN RASSAL OLARAK ÜÇ FARKLI ÇÖZÜMÜN SEÇİLMESİ
    for kr in range(nc):
        p = math.ceil(random() * nc) - 1
        q = math.ceil(random() * nc) - 1
        r = math.ceil(random() * nc) - 1
        
        # X1yeni: 1. tasarım değişkeni için mutasyona uğrayan yeni aday çözüm değeri
        X1yeni = OPT[p][0] + F * (OPT[q][0] - OPT[r][0])
        
        # X2yeni: 2. tasarım değişkeni için mutasyona uğrayan yeni aday çözüm değeri
        X2yeni = OPT[p][1] + F * (OPT[q][1] - OPT[r][1])

        # ÇAPRAZLAMA AŞAMASI (Bknz. Denklem 2.7)
        rand_kr_idx = math.ceil(random() * D) - 1 # D: Değişken sayısı
        
        X1 = 0
        if (random() < CR) or (kr == rand_kr_idx):
            X1 = X1yeni
        else:
            X1 = OPT[kr][0]
            
        # ... Diğer değişkenler için de çaprazlama ...

        # SEÇİLİM AŞAMASI (Bknz. Denklem 2.9)
        # ... uygunluk karşılaştırması ve yeni bireyin popülasyona eklenmesi ...

# İterasyon sonu
# En iyi çözümün bulunması
```

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