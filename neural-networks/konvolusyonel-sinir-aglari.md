[Üst Sayfaya Dön](../../topics/neural-networks/)

# Konvolüsyonel Sinir Ağları (CNN)

Konvolüsyonel Sinir Ağları (Convolutional Neural Networks - CNN veya ConvNet), özellikle görüntü tanıma ve işleme görevlerinde olağanüstü başarılar elde etmiş derin öğrenme mimarileridir. İnsan görsel korteksinden esinlenerek tasarlanan CNN'ler, piksellerden karmaşık özellikleri hiyerarşik bir şekilde öğrenme yeteneğine sahiptir.

## CNN Mimarisi ve Temel Katmanları

Bir CNN tipik olarak birkaç temel katman türünün birleşiminden oluşur:

### 1. Konvolüsyon Katmanı (Convolutional Layer)

*   **Amaç:** Girdi görüntüsünden (veya bir önceki katmanın özellik haritasından) özellikleri çıkarmak.
*   **Çalışma Prensibi:** Küçük filtreler (kernel veya çekirdek olarak da adlandırılır) girdi üzerinde kaydırılır. Her konumda, filtre ile girdi arasındaki noktasal çarpım hesaplanır ve bu, özellik haritasındaki (feature map) bir pikseli oluşturur.
*   **Parametreler:**
    *   **Filtre Sayısı:** Kaç farklı özellik haritası üretileceğini belirler. Her filtre farklı bir özelliği (kenarlar, köşeler, dokular vb.) tanımak için eğitilir.
    *   **Filtre Boyutu (Kernel Size):** Genellikle küçük karelerdir (örn: 3x3, 5x5).
    *   **Adım (Stride):** Filtrenin girdi üzerinde ne kadar kaydırılacağını belirler. Daha büyük adım, daha küçük özellik haritası demektir.
    *   **Doldurma (Padding):** Girdinin kenarlarına sıfırlar eklenerek çıktı özellik haritasının boyutunun korunması veya ayarlanması sağlanır. "Same" padding, çıktı boyutunu girdiyle aynı tutar; "valid" padding ise doldurma yapmaz.
*   **Örnek:** Bir 3x3'lük filtre, bir görüntüdeki dikey kenarları tespit etmek için eğitilebilir.

![CNN Konvolüsyon Katmanı Örneği](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Conv_layer.png/600px-Conv_layer.png)
*Resim Kaynağı: Wikipedia*

### 2. Aktivasyon Katmanı (Activation Layer)

*   **Amaç:** Modele doğrusal olmayanlık (non-linearity) katmak. Bu, ağın daha karmaşık ilişkileri öğrenebilmesi için kritik öneme sahiptir.
*   **Çalışma Prensibi:** Genellikle konvolüsyon katmanından sonra uygulanır. En yaygın kullanılan aktivasyon fonksiyonu **ReLU (Rectified Linear Unit)**'dir (`f(x) = max(0, x)`). ReLU, hesaplama açısından verimlidir ve gradyanların kaybolması sorununu azaltmaya yardımcı olur. Diğer aktivasyon fonksiyonları arasında Sigmoid ve Tanh bulunur, ancak derin ağlarda ReLU ve varyantları (Leaky ReLU, Parametric ReLU) daha sık tercih edilir.

### 3. Havuzlama Katmanı (Pooling Layer / Subsampling)

*   **Amaç:** Özellik haritalarının boyutunu küçülterek hesaplama yükünü azaltmak, parametre sayısını düşürmek ve bir miktar öteleme değişmezliği (translation invariance) sağlamak.
*   **Çalışma Prensibi:** Özellik haritasındaki küçük bir pencere (örn: 2x2) üzerinde işlem yapar ve bu penceredeki değerleri tek bir değerle temsil eder.
*   **Türleri:**
    *   **Maksimum Havuzlama (Max Pooling):** Penceredeki maksimum değeri alır. En yaygın kullanılanıdır ve en belirgin özelliği korur.
    *   **Ortalama Havuzlama (Average Pooling):** Penceredeki değerlerin ortalamasını alır.
*   **Örnek:** 2x2'lik bir maksimum havuzlama filtresi, bir özellik haritasının genişliğini ve yüksekliğini yarıya indirir.

![CNN Maksimum Havuzlama Örneği](https://computersciencewiki.org/images/8/8a/MaxpoolSample2.png)
*Resim Kaynağı: ComputerScienceWiki.org*

### 4. Tam Bağlantılı Katman (Fully Connected Layer / Dense Layer)

*   **Amaç:** Konvolüsyon ve havuzlama katmanlarından elde edilen yüksek seviyeli özellikleri kullanarak sınıflandırma veya regresyon gibi nihai görevi gerçekleştirmek.
*   **Çalışma Prensibi:** Bu katmandaki her nöron, bir önceki katmandaki tüm aktivasyonlara bağlıdır (geleneksel çok katmanlı algılayıcılarda olduğu gibi).
*   Genellikle CNN mimarisinin sonuna doğru bir veya daha fazla tam bağlantılı katman bulunur. Son tam bağlantılı katmanın çıktı sayısı, sınıflandırma problemindeki sınıf sayısına (örneğin, 10 rakam için 10 çıktı) veya regresyon için tek bir değere karşılık gelir.
*   Son sınıflandırma katmanında genellikle **Softmax** aktivasyon fonksiyonu kullanılır (çok sınıflı sınıflandırma için olasılık dağılımı üretir).

## Tipik Bir CNN Mimarisi

Basit bir CNN mimarisi genellikle şu sırayı izler:

1.  **GİRİŞ GÖRÜNTÜSÜ**
2.  **[KONVOLÜSYON + ReLU]** katmanı (birkaç kez tekrarlanabilir)
3.  **HAVUZLAMA** katmanı
4.  **[KONVOLÜSYON + ReLU]** katmanı (birkaç kez tekrarlanabilir)
5.  **HAVUZLAMA** katmanı
6.  ... (Daha fazla konvolüsyon ve havuzlama bloğu)
7.  **DÜZLEŞTİRME (Flattening):** Son havuzlama katmanının çıktısı olan 2D özellik haritaları, tam bağlantılı katmana girdi olabilmesi için 1D vektöre dönüştürülür.
8.  **TAM BAĞLANTILI + ReLU** katmanı
9.  **TAM BAĞLANTILI (ÇIKIŞ)** katmanı (örneğin, Softmax ile)

![Tipik CNN Mimarisi (LeNet-5 Örneği)](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/LeNet.png/500px-LeNet.png)
*Resim Kaynağı: Wikipedia (LeNet-5 mimarisi)*

## CNN'lerin Öğrenme Süreci

*   **Özellik Hiyerarşisi:** CNN'ler, verilerden hiyerarşik bir şekilde özellikler öğrenir. İlk katmanlar genellikle düşük seviyeli özellikleri (kenarlar, köşeler, renkler) yakalarken, daha derin katmanlar bu özellikleri birleştirerek daha karmaşık ve soyut özellikleri (nesne parçaları, nesneler) öğrenir.
*   **Ağırlık Paylaşımı (Weight Sharing):** Konvolüsyon katmanlarındaki filtreler, girdi görüntüsünün tamamında aynı ağırlıkları kullanır. Bu, modelin parametre sayısını önemli ölçüde azaltır ve öteleme değişmezliği sağlar (bir nesnenin görüntüdeki konumu değişse bile tanınabilmesi).
*   **Eğitim:** CNN'ler, genellikle "geri yayılım" (backpropagation) algoritması ve SGD (Stochastic Gradient Descent) gibi optimizasyon yöntemleri kullanılarak etiketli verilerle eğitilir. Kayıp fonksiyonu (loss function), modelin tahminleri ile gerçek etiketler arasındaki farkı ölçer ve ağ bu kaybı minimize etmeye çalışır.

## Popüler CNN Mimarileri

Yıllar içinde birçok etkili CNN mimarisi geliştirilmiştir. Bazı önemli örnekler:

*   **LeNet-5 (1998):** İlk başarılı CNN'lerden biri, el yazısı rakam tanıma için kullanıldı.
*   **AlexNet (2012):** ImageNet yarışmasını kazanarak derin öğrenmenin popülaritesini artırdı. ReLU ve Dropout gibi teknikleri kullandı.
*   **VGGNet (2014):** Çok sayıda küçük (3x3) filtre kullanarak derinliği artırdı. Basit ve homojen bir yapıya sahiptir.
*   **GoogLeNet / Inception (2014):** "Inception modülleri" kullanarak farklı boyutlardaki konvolüsyonları paralel olarak uyguladı ve hesaplama verimliliğini artırdı.
*   **ResNet (Residual Networks) (2015):** "Artık bloklar" (residual blocks) kullanarak çok derin ağların (100+ katman) eğitilmesini mümkün kıldı ve kaybolan gradyan sorununu hafifletti.
*   **DenseNet (Densely Connected Convolutional Networks) (2016):** Her katmanı, önceki tüm katmanlara doğrudan bağlayarak özellik yayılımını iyileştirdi.
*   **EfficientNet (2019):** Model ölçeklendirmeyi (derinlik, genişlik, çözünürlük) dengeli bir şekilde yaparak yüksek doğruluk ve verimlilik elde etti.

## CNN Uygulama Alanları

CNN'ler, özellikle görsel verilerle ilgili çok çeşitli görevlerde devrim yaratmıştır:

*   **Görüntü Sınıflandırma:** Bir görüntüye etiket atama (kedi, köpek, araba vb.).
*   **Nesne Tespiti (Object Detection):** Bir görüntüdeki nesnelerin yerini belirleme ve sınıflandırma (örneğin, Yolo, SSD, Faster R-CNN).
*   **Görüntü Segmentasyonu:** Bir görüntüyü piksel düzeyinde farklı bölgelere ayırma (anlamsal segmentasyon, örnek segmentasyonu).
*   **Yüz Tanıma ve Doğrulama**
*   **Sahne Anlama**
*   **Tıbbi Görüntü Analizi:** Kanser tespiti, organ segmentasyonu vb.
*   **Otonom Araçlar:** Yol, şerit, trafik işaretleri ve diğer araçların tespiti.
*   **Video Analizi:** Eylem tanıma, nesne takibi.
*   **Doğal Dil İşleme (NLP):** Metin sınıflandırma gibi bazı NLP görevlerinde de konvolüsyonel yapılar kullanılabilmektedir.
*   **Sanat Üretimi (Style Transfer)**

## CNN'lerin Avantajları

*   **Yüksek Doğruluk:** Özellikle görüntü tabanlı görevlerde son teknoloji (state-of-the-art) sonuçlar elde ederler.
*   **Özellik Öğrenme:** Manuel özellik mühendisliğine olan ihtiyacı azaltır; ağ, verilerden ilgili özellikleri otomatik olarak öğrenir.
*   **Ağırlık Paylaşımı:** Parametre sayısını azaltır, bu da daha az veriyle daha iyi genelleme yapmaya ve aşırı öğrenmeyi (overfitting) azaltmaya yardımcı olur.
*   **Hiyerarşik Özellik Temsili:** Basit özelliklerden karmaşık özelliklere doğru bir hiyerarşi oluşturarak verinin yapısını daha iyi anlarlar.
*   **Öteleme Değişmezliği:** Havuzlama katmanları sayesinde nesnelerin konumundaki küçük değişikliklere karşı daha dirençlidirler.

## CNN'lerin Dezavantajları

*   **Hesaplama Yoğunluğu:** Derin CNN'ler, büyük miktarda veri ve önemli hesaplama kaynakları (GPU'lar) gerektirebilir.
*   **Büyük Veri İhtiyacı:** İyi performans için genellikle büyük etiketli veri kümelerine ihtiyaç duyarlar.
*   **Yorumlanabilirlik Zorluğu:** Derin ağların karar verme süreçleri genellikle "kara kutu" gibidir ve neden belirli bir tahminde bulunduklarını anlamak zor olabilir.
*   **Dönme ve Ölçek Değişimlerine Karşı Hassasiyet:** Standart CNN'ler, nesnelerin dönmesine veya ölçeklenmesine karşı tamamen değişmez değildir (veri artırma teknikleri bu sorunu hafifletebilir).
*   **Hiperparametre Ayarlama:** Filtre sayısı, filtre boyutu, adım, öğrenme oranı gibi birçok hiperparametrenin ayarlanması zaman alıcı ve zor olabilir.

## Sonuç

Konvolüsyonel Sinir Ağları, bilgisayarlı görü alanında bir paradigma değişikliği yaratmış ve yapay zekanın birçok pratik uygulamasının önünü açmıştır. Sürekli gelişen mimariler ve tekniklerle birlikte, CNN'lerin yetenekleri ve uygulama alanları genişlemeye devam etmektedir. Görüntü ve video verilerinin anlaşılması ve işlenmesinde temel bir araç haline gelmişlerdir. 