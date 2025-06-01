# Temel Sinir Ağı Mimarileri

Yapay sinir ağları (YSA), insan beyninin öğrenme şeklinden esinlenerek geliştirilmiş bilgi işlem sistemleridir. Karmaşık problemleri çözmek, örüntüleri tanımak ve verilerden öğrenmek için kullanılırlar. Temelde, birbirine bağlı ve "nöron" adı verilen işlem birimlerinden oluşurlar.

## Sinir Ağı Bileşenleri

Bir sinir ağının temel bileşenleri şunlardır:

*   **Nöronlar (Düğümler):** Ağa gelen bilgiyi işleyen ve bir çıktı üreten temel hesaplama birimleridir.
*   **Bağlantılar (Ağırlıklar):** Nöronlar arasındaki bağlantılardır. Her bağlantının bir "ağırlığı" vardır. Bu ağırlık, bir nörondan diğerine aktarılan sinyalin gücünü belirler. Öğrenme süreci temel olarak bu ağırlıkların ayarlanmasıdır.
*   **Katmanlar:** Nöronlar genellikle katmanlar halinde düzenlenir:
    *   **Giriş Katmanı:** Dış dünyadan veriyi alır.
    *   **Gizli Katman(lar):** Giriş katmanından gelen veriyi işler ve karmaşık özellikleri çıkarır. Bir sinir ağında bir veya daha fazla gizli katman bulunabilir. "Derin öğrenme" terimi genellikle çok sayıda gizli katmana sahip ağları ifade eder.
    *   **Çıkış Katmanı:** Ağın son çıktısını üretir (örneğin, bir sınıflandırma sonucu veya bir tahmin).
*   **Aktivasyon Fonksiyonu:** Bir nöronun çıktısını belirler. Genellikle doğrusal olmayan fonksiyonlardır ve ağın karmaşık örüntüleri öğrenmesine olanak tanır. Yaygın aktivasyon fonksiyonlarına örnek olarak Sigmoid, ReLU ve Tanh verilebilir.
*   **Toplama Fonksiyonu (Birleştirme Fonksiyonu):** Bir nörona gelen tüm ağırlıklı girdileri birleştirir.

## Yaygın Sinir Ağı Mimarileri

Farklı görevler ve veri türleri için çeşitli sinir ağı mimarileri geliştirilmiştir. İşte en yaygın olanlardan bazıları:

### 1. İleri Beslemeli Sinir Ağları (Feedforward Neural Networks - FNN)

*   En basit sinir ağı türüdür.
*   Bilgi, giriş katmanından çıkış katmanına doğru tek yönde akar. Geriye doğru bağlantılar (döngüler) yoktur.
*   Genellikle sınıflandırma ve regresyon problemleri için kullanılırlar.
*   **Perceptron:** Tek katmanlı bir ileri beslemeli ağdır ve doğrusal olarak ayrılabilen problemleri çözebilir.
*   **Çok Katmanlı Perceptron (Multilayer Perceptron - MLP):** Bir veya daha fazla gizli katmana sahip ileri beslemeli ağlardır. Doğrusal olmayan problemleri çözebilirler ve daha karmaşık görevler için kullanılırlar.

### 2. Evrişimli Sinir Ağları (Convolutional Neural Networks - CNN)

*   Özellikle görüntü tanıma, video analizi ve doğal dil işleme gibi alanlarda çok başarılıdırlar.
*   Görüntülerdeki mekansal hiyerarşileri (örneğin, kenarlar, şekiller, nesneler) öğrenmek için evrişim katmanları, havuzlama katmanları ve tam bağlı katmanlar gibi özel katman türleri kullanırlar.
*   Parametre paylaşımı sayesinde daha az parametre ile büyük veri setlerinde etkili bir şekilde çalışabilirler.

### 3. Tekrarlayan Sinir Ağları (Recurrent Neural Networks - RNN)

*   Sıralı verileri (örneğin, zaman serileri, metin, konuşma) işlemek için tasarlanmıştır.
*   Ağın önceki adımlardan gelen bilgileri "hatırlamasını" sağlayan geri besleme döngülerine sahiptirler. Bu sayede bağlamsal bilgiyi yakalayabilirler.
*   Doğal dil işleme (makine çevirisi, metin üretimi), konuşma tanıma ve zaman serisi tahmini gibi görevlerde yaygın olarak kullanılırlar.
*   **Uzun Kısa Süreli Bellek (Long Short-Term Memory - LSTM)** ve **Kapılı Tekrarlayan Birim (Gated Recurrent Unit - GRU)** gibi varyantları, RNN'lerin "kaybolan gradyan" problemini çözmeye yardımcı olur ve uzun süreli bağımlılıkları daha iyi öğrenmelerini sağlar.

### 4. Üretici Çekişmeli Ağlar (Generative Adversarial Networks - GAN)

*   İki sinir ağından oluşur: bir **Üretici (Generator)** ve bir **Ayırt Edici (Discriminator)**.
*   Üretici, gerçekçi görünen sahte veriler (örneğin, görüntüler, metinler) üretmeye çalışır.
*   Ayırt Edici, gerçek verilerle üreticinin oluşturduğu sahte verileri ayırt etmeye çalışır.
*   Bu iki ağ, birbirleriyle rekabet ederek (çekişerek) eğitilir. Üretici daha gerçekçi veriler üretmeye, Ayırt Edici ise sahteleri daha iyi tespit etmeye çalışır.
*   Görüntü üretimi, stil aktarımı ve veri artırma gibi alanlarda kullanılırlar.

## Öğrenme Süreci (Eğitim)

Sinir ağları genellikle **gözetimli öğrenme** adı verilen bir süreçle eğitilir:

1.  **Veri Hazırlığı:** Büyük miktarda etiketli veri toplanır (girdiler ve karşılık gelen doğru çıktılar).
2.  **Başlatma:** Ağın ağırlıkları genellikle rastgele değerlerle başlatılır.
3.  **İleri Yayılım (Forward Propagation):** Eğitim verisinden bir girdi alınır ve ağ üzerinden geçirilerek bir çıktı üretilir.
4.  **Kayıp Fonksiyonu (Loss Function):** Ağın ürettiği çıktı ile gerçek (beklenen) çıktı arasındaki fark (hata) hesaplanır.
5.  **Geri Yayılım (Backpropagation):** Hesaplanan hata, ağ üzerinden geriye doğru yayılır. Bu süreçte, her bir ağırlığın hataya olan katkısı belirlenir (gradyanlar hesaplanır).
6.  **Ağırlık Güncelleme:** Ağırlıklar, hatayı azaltacak yönde (genellikle gradyan inişi gibi bir optimizasyon algoritması kullanılarak) güncellenir.
7.  Bu adımlar (3-6), ağın performansı istenen seviyeye ulaşana kadar veya belirli bir eğitim süresi boyunca tekrarlanır.

## Sonuç

Temel sinir ağı mimarileri, yapay zekanın birçok alanında devrim yaratmıştır. Farklı mimari türleri, belirli problem türleri için optimize edilmiştir ve sürekli olarak yeni ve daha gelişmiş mimariler geliştirilmektedir. Bu temel kavramları anlamak, modern yapay zeka uygulamalarını ve araştırmalarını takip etmek için önemlidir. 