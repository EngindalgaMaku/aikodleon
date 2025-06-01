[Üst Sayfaya Dön](../../topics/neural-networks/)

# Tekrarlayan Sinir Ağları (RNN)

Tekrarlayan Sinir Ağları (Recurrent Neural Networks - RNN), sıralı verileri işlemek ve bu verilerdeki zamansal bağımlılıkları yakalamak için tasarlanmış bir yapay sinir ağı türüdür. Geleneksel ileri beslemeli sinir ağlarının aksine, RNN'ler önceki adımlardan gelen bilgileri "hatırlamalarını" sağlayan geri besleme döngülerine sahiptir. Bu özellikleri sayesinde, bir sonraki adımı tahmin etmek veya bir dizideki örüntüleri anlamak için geçmiş bilgileri kullanabilirler.

## RNN Mimarisi ve Çalışma Prensibi

Bir RNN'nin temel yapısı, bir giriş katmanı, bir veya daha fazla gizli katman ve bir çıkış katmanından oluşur. İleri beslemeli ağlardan temel farkı, gizli katmandaki nöronların kendi çıktılarını bir sonraki zaman adımında tekrar girdi olarak almasıdır. Bu geri besleme döngüsü, ağın bir "belleğe" sahip olmasını sağlar.

**Çalışma Prensibi:**

1.  **Girdi (Input):** Sıralı verinin her bir elemanı (örneğin, bir cümlenin her bir kelimesi veya bir zaman serisinin her bir veri noktası) RNN'e adım adım verilir.
2.  **Gizli Durum (Hidden State):** Her zaman adımında, RNN mevcut girdiyi ve bir önceki zaman adımındaki gizli durumu (belleği) kullanarak yeni bir gizli durum hesaplar. Bu gizli durum, o ana kadar işlenen sekans hakkındaki bilgiyi özetler.
3.  **Çıktı (Output):** Mevcut gizli durum kullanılarak bir çıktı üretilir. Bu çıktı, bir sonraki elemanın tahmini, bir sınıflandırma etiketi veya başka bir sıralı veri olabilir.

Matematiksel olarak, bir RNN'in temel denklemleri şu şekilde ifade edilebilir:

*   **Gizli Durum Güncellemesi:**  `h_t = f(W_hh * h_{t-1} + W_xh * x_t + b_h)`
*   **Çıktı Hesaplaması:** `y_t = g(W_hy * h_t + b_y)`

Burada:
*   `h_t`: Mevcut zaman adımındaki gizli durum
*   `h_{t-1}`: Bir önceki zaman adımındaki gizli durum
*   `x_t`: Mevcut zaman adımındaki girdi
*   `y_t`: Mevcut zaman adımındaki çıktı
*   `W_hh`, `W_xh`, `W_hy`: Ağırlık matrisleri (öğrenilen parametreler)
*   `b_h`, `b_y`: Bias vektörleri (öğrenilen parametreler)
*   `f` ve `g`: Aktivasyon fonksiyonları (genellikle `tanh` veya `ReLU` gibi doğrusal olmayan fonksiyonlar)

RNN'ler, "zamanla geri yayılım" (Backpropagation Through Time - BPTT) algoritması kullanılarak eğitilir. Bu algoritmada, hata sinyalleri ağın katmanları boyunca ve aynı zamanda zaman adımları boyunca geriye doğru yayılır.

## RNN Varyantları

Standart RNN'lerin bazı sınırlamaları (özellikle uzun süreli bağımlılıkları öğrenmedeki zorluklar) nedeniyle çeşitli varyantları geliştirilmiştir:

### 1. Uzun Kısa Süreli Bellek (Long Short-Term Memory - LSTM)

*   LSTM'ler, RNN'lerin "kaybolan gradyan" (vanishing gradient) ve "patlayan gradyan" (exploding gradient) sorunlarını çözmek için tasarlanmıştır. Bu sorunlar, uzun dizilerde öğrenmeyi zorlaştırır.
*   LSTM hücreleri, bilgiyi ne zaman saklayacağını, ne zaman sileceğini ve ne zaman okuyacağını kontrol eden **kapı (gate)** mekanizmalarına sahiptir:
    *   **Unutma Kapısı (Forget Gate):** Hücre durumundan hangi bilgilerin atılacağına karar verir.
    *   **Giriş Kapısı (Input Gate):** Hücre durumuna hangi yeni bilgilerin ekleneceğine karar verir.
    *   **Çıkış Kapısı (Output Gate):** Hücre durumundan hangi bilgilerin çıktı olarak verileceğine karar verir.
*   Bu kapılar sayesinde LSTM'ler, uzun süreli bağımlılıkları daha etkili bir şekilde öğrenebilirler.

### 2. Kapılı Tekrarlayan Birim (Gated Recurrent Unit - GRU)

*   GRU'lar, LSTM'lere benzer bir şekilde kapı mekanizmalarını kullanır ancak daha basit bir yapıya sahiptirler.
*   GRU'larda iki temel kapı bulunur:
    *   **Güncelleme Kapısı (Update Gate):** Önceki gizli durumun ne kadarının korunacağını ve yeni gizli durumun ne kadarının ekleneceğini belirler.
    *   **Sıfırlama Kapısı (Reset Gate):** Önceki gizli durumun ne kadarının unutulacağına karar verir.
*   GRU'lar, LSTM'lere göre daha az parametreye sahiptir ve bazı durumlarda benzer performans sunarken daha hızlı eğitilebilirler.

### 3. Çift Yönlü RNN'ler (Bidirectional RNNs - BiRNN)

*   Standart RNN'ler, diziyi sadece ileri yönde işler. Ancak bazı durumlarda, bir elemanı anlamak için hem geçmiş hem de gelecek bağlam önemlidir (örneğin, bir cümledeki bir kelimenin anlamı).
*   Çift yönlü RNN'ler, diziyi iki farklı yönde (ileri ve geri) işleyen iki ayrı RNN katmanından oluşur. Bu iki katmanın çıktıları birleştirilerek daha zengin bir bağlamsal temsil elde edilir.
*   BiRNN'ler, genellikle BiLSTM veya BiGRU olarak LSTM veya GRU hücreleriyle birlikte kullanılır.

## RNN Uygulama Alanları

RNN'ler, sıralı verilerin önemli olduğu birçok alanda yaygın olarak kullanılmaktadır:

*   **Doğal Dil İşleme (NLP):**
    *   Makine çevirisi
    *   Metin üretimi (hikaye, şiir vb.)
    *   Duygu analizi
    *   Konuşma tanıma
    *   Soru cevaplama
    *   Adlandırılmış Varlık Tanıma (NER)
*   **Zaman Serisi Analizi:**
    *   Hisse senedi fiyat tahmini
    *   Hava durumu tahmini
    *   Anomali tespiti (örneğin, dolandırıcılık tespiti)
*   **Konuşma Sentezi (Text-to-Speech)**
*   **Müzik Üretimi**
*   **El Yazısı Tanıma**
*   **Video Analizi ve Açıklama Üretimi**
*   **Robot Kontrolü**

## RNN'lerin Avantajları ve Dezavantajları

**Avantajları:**

*   Sıralı verilerdeki zamansal bağımlılıkları modelleyebilirler.
*   Değişken uzunluktaki girdileri işleyebilirler.
*   Model boyutu, girdi dizisinin uzunluğuna bağlı olarak artmaz.
*   Zaman içinde paylaşılan ağırlıklar sayesinde daha az parametreye ihtiyaç duyabilirler.

**Dezavantajları:**

*   Eğitimleri genellikle yavaştır.
*   Uzun süreli bağımlılıkları öğrenmekte zorlanabilirler (özellikle standart RNN'ler için "kaybolan/patlayan gradyan" sorunu).
*   Paralel işleme yetenekleri sınırlıdır, çünkü her bir zaman adımının hesaplanması bir önceki adıma bağlıdır.
*   Mevcut bir durumu işlerken gelecekteki girdileri doğrudan dikkate alamazlar (BiRNN'ler bu sorunu kısmen çözer).

## Sonuç

Tekrarlayan Sinir Ağları, sıralı verilerin modellenmesinde önemli bir atılım sağlamıştır. LSTM ve GRU gibi gelişmiş varyantları, uzun süreli bağımlılıkları öğrenme yeteneklerini artırmış ve birçok karmaşık görevin çözülmesine olanak tanımıştır. Transformer gibi daha yeni mimariler bazı alanlarda RNN'lerin yerini almaya başlasa da, RNN'ler hala birçok uygulama için değerli ve etkili bir araç olmaya devam etmektedir. 