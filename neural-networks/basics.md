# Yapay Sinir Ağları Temelleri

![Yapay Sinir Ağları Temelleri](/images/neural_networks_basics.jpg)

## Yapay Sinir Ağları Nedir?

Yapay Sinir Ağları (YSA), insan beyninin çalışma prensiplerinden esinlenerek geliştirilmiş, karmaşık veri setlerinden öğrenebilen ve bu öğrenme sonucunda tahminler yapabilen hesaplama modelleridir. Birbirine bağlı nöronlardan (yapay sinir hücreleri) oluşan bu ağlar, günümüzde yapay zeka ve makine öğrenmesinin temel yapı taşlarından biridir.

## Biyolojik İlham: İnsan Beyninden YSA'ya

İnsan beyni, yaklaşık 86 milyar nörondan oluşan karmaşık bir ağdır. Her nöron, dendritler aracılığıyla diğer nöronlardan sinyaller alır, bu sinyalleri işler ve aksonlar aracılığıyla diğer nöronlara iletir. Yapay sinir ağları, bu biyolojik yapıyı basitleştirerek modellemektedir:

- **Biyolojik Nöron** → **Yapay Nöron**
- **Dendritler** → **Girdiler (Inputs)**
- **Hücre Gövdesi** → **Aktivasyon Fonksiyonu**
- **Akson** → **Çıktı (Output)**
- **Sinaptik Bağlantılar** → **Ağırlıklar (Weights)**

## Yapay Nöron Yapısı

Bir yapay nöron (perceptron), temel olarak şu bileşenlerden oluşur:

1. **Girdiler (x₁, x₂, ..., xₙ)**: Nörona gelen veri noktaları
2. **Ağırlıklar (w₁, w₂, ..., wₙ)**: Her girdinin önemini belirleyen değerler
3. **Toplama Fonksiyonu**: Genellikle ağırlıklı toplam (Σ wᵢxᵢ + b)
4. **Bias (b)**: Aktivasyon eşiğini ayarlayan sabit değer
5. **Aktivasyon Fonksiyonu**: Toplam değeri dönüştüren fonksiyon (sigmoid, ReLU, tanh vb.)
6. **Çıktı (y)**: Nöronun ürettiği sonuç

## Aktivasyon Fonksiyonları

Aktivasyon fonksiyonları, nöronun toplam girdisini işleyerek çıktıya dönüştüren matematiksel fonksiyonlardır. En yaygın kullanılan aktivasyon fonksiyonları:

- **Sigmoid**: f(x) = 1 / (1 + e^(-x))
  - Çıktı aralığı: (0, 1)
  - Avantaj: Olasılık değerleri için uygun
  - Dezavantaj: Gradyan yok olması problemi

- **Tanh (Hiperbolik Tanjant)**: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
  - Çıktı aralığı: (-1, 1)
  - Avantaj: Sıfır merkezli
  - Dezavantaj: Gradyan yok olması problemi

- **ReLU (Rectified Linear Unit)**: f(x) = max(0, x)
  - Çıktı aralığı: [0, ∞)
  - Avantaj: Hesaplama verimliliği, gradyan yok olması problemini azaltır
  - Dezavantaj: "Ölü ReLU" problemi

- **Leaky ReLU**: f(x) = max(αx, x), α küçük bir sabit (örn. 0.01)
  - Avantaj: Ölü ReLU problemini çözer
  - Dezavantaj: Ek bir hiperparametre

- **Softmax**: Çok sınıflı sınıflandırma problemleri için çıktıları olasılık dağılımına dönüştürür

## Yapay Sinir Ağı Mimarisi

Yapay sinir ağları genellikle katmanlı bir yapıya sahiptir:

1. **Girdi Katmanı**: Veri noktalarını ağa alan ilk katman
2. **Gizli Katmanlar**: Verileri işleyen ara katmanlar (derin öğrenmede birden fazla olabilir)
3. **Çıktı Katmanı**: Ağın tahminini veya kararını üreten son katman

Katmanlar arası bağlantı türlerine göre farklı ağ mimarileri bulunur:

- **İleri Beslemeli Ağlar (Feedforward Neural Networks)**: Bilgi sadece ileri yönde akar
- **Tekrarlayan Sinir Ağları (Recurrent Neural Networks)**: Geri bildirim bağlantıları içerir
- **Evrişimli Sinir Ağları (Convolutional Neural Networks)**: Görüntü işleme için özelleşmiş
- **Transformerlar**: Dikkat mekanizması kullanan modern mimari

## Öğrenme Süreci

Yapay sinir ağlarının öğrenme süreci, temel olarak şu adımlardan oluşur:

1. **İleri Yayılım (Forward Propagation)**: Girdiler ağ boyunca ilerletilir ve çıktı hesaplanır
2. **Hata Hesaplama**: Gerçek değer ile tahmin arasındaki fark hesaplanır
3. **Geri Yayılım (Backpropagation)**: Hata, ağ boyunca geriye doğru yayılır
4. **Ağırlık Güncelleme**: Gradyan iniş algoritması ile ağırlıklar güncellenir

Bu süreç, ağ yeterince iyi performans gösterene kadar tekrarlanır.

## Optimizasyon Algoritmaları

Yapay sinir ağlarının eğitiminde kullanılan başlıca optimizasyon algoritmaları:

- **Stokastik Gradyan İniş (SGD)**: Her iterasyonda rastgele bir veri örneği kullanılır
- **Mini-Batch Gradyan İniş**: Her iterasyonda küçük bir veri alt kümesi kullanılır
- **Momentum**: Gradyan güncellemelerini hızlandırır ve yerel minimumlara takılmayı önler
- **Adam**: Adaptif öğrenme oranları kullanır, momentum ve RMSprop'u birleştirir

## Yapay Sinir Ağlarının Avantajları

- **Doğrusal Olmayan İlişkileri Öğrenebilme**: Karmaşık veri yapılarını modelleyebilir
- **Genelleme Yeteneği**: Yeni, daha önce görülmemiş verilere uyum sağlayabilir
- **Paralel İşleme**: Dağıtık hesaplama için uygundur
- **Hata Toleransı**: Kısmi bilgi kaybında bile çalışabilir

## Yapay Sinir Ağlarının Zorlukları

- **Kara Kutu Problemi**: Kararların arkasındaki mantık açıklanması zor olabilir
- **Aşırı Öğrenme (Overfitting)**: Eğitim verisine aşırı uyum sağlama riski
- **Hesaplama Maliyeti**: Büyük ağların eğitimi yüksek hesaplama gücü gerektirebilir
- **Hiperparametre Ayarlama**: Optimal performans için birçok parametrenin ayarlanması gerekir

## Yapay Sinir Ağlarının Uygulama Alanları

- **Bilgisayarlı Görü**: Nesne tanıma, görüntü sınıflandırma, yüz tanıma
- **Doğal Dil İşleme**: Metin sınıflandırma, duygu analizi, makine çevirisi
- **Ses İşleme**: Konuşma tanıma, müzik üretimi
- **Oyun Oynama**: AlphaGo, Dota 2, Starcraft II
- **Sağlık**: Hastalık teşhisi, ilaç keşfi
- **Finans**: Hisse senedi tahmini, kredi risk değerlendirmesi
- **Otonom Araçlar**: Sürüş sistemleri, çevre algılama

## Sonuç

Yapay sinir ağları, makine öğrenmesi ve yapay zeka alanındaki en güçlü tekniklerden biridir. İnsan beyninden esinlenen bu hesaplama modelleri, karmaşık problemleri çözmede ve verilerden anlamlı desenler çıkarmada büyük başarı göstermektedir. Temel prensiplerini anlamak, daha ileri düzey yapay zeka uygulamaları geliştirmek için sağlam bir temel oluşturur.

---

## İleri Okuma

- [Deep Learning](https://www.deeplearningbook.org/) - Ian Goodfellow, Yoshua Bengio ve Aaron Courville
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Michael Nielsen
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [PyTorch Tutorials](https://pytorch.org/tutorials/) 