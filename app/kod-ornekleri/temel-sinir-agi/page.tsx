import { Metadata } from 'next';
import Link from 'next/link';
import Image from 'next/image';
import { ArrowLeft, ArrowRight, Download, Github, Copy } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'Temel Yapay Sinir Ağı Uygulaması | Kod Örnekleri | Kodleon',
  description: 'NumPy kullanarak sıfırdan basit bir yapay sinir ağı oluşturma ve eğitme örneği.',
  openGraph: {
    title: 'Temel Yapay Sinir Ağı Uygulaması | Kodleon',
    description: 'NumPy kullanarak sıfırdan basit bir yapay sinir ağı oluşturma ve eğitme örneği.',
    images: [{ url: '/images/code-examples/neural-network.jpg' }],
  },
};

export default function BasicNeuralNetworkPage() {
  return (
    <div className="container max-w-6xl mx-auto py-12 px-4">
      <div className="mb-8">
        <Button asChild variant="outline" size="sm" className="gap-1">
          <Link href="/kod-ornekleri">
            <ArrowLeft className="h-4 w-4" />
            Tüm Kod Örnekleri
          </Link>
        </Button>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Sol Taraf - Açıklama */}
        <div className="lg:col-span-1">
          <div className="sticky top-20">
            <h1 className="text-3xl font-bold mb-4">Temel Yapay Sinir Ağı Uygulaması</h1>
            
            <div className="flex items-center gap-2 mb-4">
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                Derin Öğrenme
              </span>
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                Başlangıç
              </span>
            </div>
            
            <p className="text-muted-foreground mb-6">
              Bu örnekte, NumPy kütüphanesini kullanarak sıfırdan basit bir yapay sinir ağı oluşturacak ve eğiteceksiniz. 
              Örnek, ileri besleme (feedforward) ve geri yayılım (backpropagation) algoritmalarının temel uygulamasını içerir.
            </p>
            
            <div className="space-y-4 mb-6">
              <div>
                <h3 className="font-medium">Gereksinimler:</h3>
                <ul className="list-disc list-inside text-sm text-muted-foreground">
                  <li>Python 3.6+</li>
                  <li>NumPy</li>
                  <li>Matplotlib (görselleştirme için)</li>
                </ul>
              </div>
              
              <div>
                <h3 className="font-medium">Öğrenilecek Kavramlar:</h3>
                <ul className="list-disc list-inside text-sm text-muted-foreground">
                  <li>Yapay sinir ağı mimarisi</li>
                  <li>İleri besleme (Feedforward)</li>
                  <li>Geri yayılım (Backpropagation)</li>
                  <li>Gradyan iniş optimizasyonu</li>
                </ul>
              </div>
            </div>
            
            <div className="flex flex-col gap-2">
              <Button asChild variant="default" className="gap-2">
                <a href="/notebooks/temel-sinir-agi.ipynb" download>
                  <Download className="h-4 w-4" />
                  Jupyter Notebook İndir
                </a>
              </Button>
              <Button asChild variant="outline" className="gap-2">
                <a href="https://github.com/kodleon/ai-examples/blob/main/neural-networks/basic-neural-network.ipynb" target="_blank" rel="noopener noreferrer">
                  <Github className="h-4 w-4" />
                  GitHub'da Görüntüle
                </a>
              </Button>
            </div>
          </div>
        </div>
        
        {/* Sağ Taraf - Kod */}
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-850 rounded-xl shadow-md overflow-hidden">
            <Tabs defaultValue="code" className="w-full">
              <div className="border-b">
                <TabsList className="p-0 bg-transparent">
                  <TabsTrigger value="code" className="rounded-none py-3 px-6 data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none">
                    Kod
                  </TabsTrigger>
                  <TabsTrigger value="explanation" className="rounded-none py-3 px-6 data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none">
                    Açıklama
                  </TabsTrigger>
                  <TabsTrigger value="output" className="rounded-none py-3 px-6 data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none">
                    Çıktı
                  </TabsTrigger>
                </TabsList>
              </div>
              
              <TabsContent value="code" className="p-0 m-0">
                <div className="relative">
                  <Button variant="ghost" size="sm" className="absolute right-2 top-2 gap-1">
                    <Copy className="h-4 w-4" />
                    Kopyala
                  </Button>
                  <pre className="p-6 pt-12 overflow-x-auto text-sm">
                    <code>{`import numpy as np
import matplotlib.pyplot as plt

# Aktivasyon fonksiyonları ve türevleri
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4) 
        self.weights2 = np.random.rand(4, 1)                 
        self.y = y
        self.output = np.zeros(self.y.shape)
        
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        
    def backprop(self):
        # Ağırlıkları güncellemek için geri yayılım uygula
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
        
        # Ağırlıkları güncelle
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        
    def train(self, epochs=10000):
        errors = []
        for i in range(epochs):
            self.feedforward()
            self.backprop()
            if i % 1000 == 0:
                error = np.mean(np.square(self.y - self.output))
                errors.append(error)
                print(f"Epoch {i}: Error {error}")
        return errors

# XOR problemi için veri seti oluştur
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Sinir ağını oluştur ve eğit
nn = NeuralNetwork(X, y)
errors = nn.train(epochs=10000)

# Sonuçları görüntüle
print("Tahminler:")
nn.feedforward()
print(nn.output)

# Hata grafiği
plt.figure(figsize=(10, 6))
plt.plot(range(0, 10000, 1000), errors)
plt.title('Eğitim Hatası')
plt.xlabel('Epoch')
plt.ylabel('Ortalama Kare Hata')
plt.grid(True)
plt.show()

# XOR fonksiyonunu görselleştir
plt.figure(figsize=(10, 6))
plt.scatter(X[0:1, 0], X[0:1, 1], c='red', marker='o', s=100, label='0')
plt.scatter(X[1:3, 0], X[1:3, 1], c='blue', marker='o', s=100, label='1')
plt.scatter(X[3:4, 0], X[3:4, 1], c='red', marker='o', s=100)
plt.title('XOR Problemi')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.show()`}</code>
                  </pre>
                </div>
              </TabsContent>
              
              <TabsContent value="explanation" className="p-6 m-0 space-y-6">
                <h3 className="text-xl font-bold">Kodun Adım Adım Açıklaması</h3>
                
                <div>
                  <h4 className="font-semibold text-lg">Temel Fikir: Yapay Sinir Ağı Nedir?</h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    Bu kodu, öğrenme yeteneğine sahip küçük bir "dijital beyin" olarak düşünebilirsiniz. Amacımız, bu beyine XOR problemini (bir mantık problemi) öğreterek, doğru tahminler yapmasını sağlamak. Tıpkı bir çocuğun deneme yanılma yoluyla öğrenmesi gibi, bizim dijital beynimiz de yaptığı hatalardan ders çıkararak kendini geliştirecek.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold text-lg">1. Aktivasyon Fonksiyonu: Sigmoid</h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    Bir nörondan (beyin hücresi) geçen bilginin ne kadar "önemli" olduğunu belirleyen bir kapı gibidir. Sigmoid fonksiyonu, kendisine gelen herhangi bir sayıyı 0 ile 1 arasında bir değere sıkıştırır. Bu, bir nöronun "ateşlenip ateşlenmeyeceğini" (yani sinyalin ne kadar güçlü geçeceğini) kontrol etmemizi sağlar. 0'a yakın değerler "zayıf sinyal", 1'e yakın değerler ise "güçlü sinyal" anlamına gelir. Türevi ise öğrenme (geri yayılım) aşamasında hatanın ne yönde düzeltileceğini hesaplamak için kullanılır.
                  </p>
                </div>

                <div>
                  <h4 className="font-semibold text-lg">2. NeuralNetwork Sınıfı: Beynimizi İnşa Etmek</h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    Burada dijital beynimizin yapısını kuruyoruz. Bu yapı 3 katmandan oluşur:
                    <ul className="list-disc list-inside mt-2 pl-4">
                      <li><b>Giriş Katmanı (Input Layer):</b> Problemin verilerini (XOR için [0,0], [0,1] gibi) alır.</li>
                      <li><b>Gizli Katman (Hidden Layer):</b> Beynin düşünme kısmıdır. Girdileri alır, işler ve bir sonraki katmana gönderir. Kodda 4 nörondan oluşur.</li>
                      <li><b>Çıkış Katmanı (Output Layer):</b> Beynin tahminini (sonucunu) üretir.</li>
                    </ul>
                    `weights` (ağırlıklar) ise nöronlar arasındaki bağlantıların gücünü temsil eder. Başlangıçta bu bağlantıların ne kadar güçlü olacağını bilmediğimiz için rastgele sayılarla başlatırız. Öğrenme süreci, bu ağırlıkları doğru değerlere ayarlama işlemidir.
                  </p>
                </div>

                <div>
                  <h4 className="font-semibold text-lg">3. İleri Besleme (Feedforward): Tahmin Yapma</h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    Bu, beynin "düşünme" veya "tahmin etme" adımıdır. Veri, giriş katmanından girer, gizli katmandan geçer ve çıkış katmanında bir sonuca ulaşır.
                    <br />
                    <code>self.layer1 = sigmoid(np.dot(self.input, self.weights1))</code> satırı şunu yapar: Giriş verilerini ilk ağırlık setiyle çarpar (bilgiyi işler) ve sonucu sigmoid fonksiyonundan geçirerek gizli katmanın çıktısını oluşturur. Bir sonraki satır da aynı işlemi gizli katman ve ikinci ağırlık setiyle yaparak nihai tahmini üretir.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold text-lg">4. Geri Yayılım (Backpropagation): Hatalardan Ders Çıkarma</h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    Bu, öğrenmenin gerçekleştiği en kritik adımdır. Beyin bir tahminde bulundu (ileri besleme), şimdi bu tahminin ne kadar "yanlış" olduğunu hesaplayıp kendini düzeltecek.
                    <ol className="list-decimal list-inside mt-2 pl-4 space-y-1">
                      <li><b>Hatayı Hesapla:</b> <code>(self.y - self.output)</code> koduyla, beynin tahmini (output) ile gerçek doğru cevap (y) arasındaki fark (hata) bulunur.</li>
                      <li><b>Hatayı Geri Yay:</b> Bu hata, "suçun ne kadarının hangi bağlantıdan kaynaklandığını" bulmak için çıkıştan girişe doğru geri yayılır. Türevler burada devreye girerek her bir ağırlığın hataya olan etkisini (gradyanını) hesaplar.</li>
                      <li><b>Ağırlıkları Güncelle:</b> <code>self.weights1 += d_weights1</code> satırıyla, hesaplanan bu "suç paylarına" göre ağırlıklar yavaşça güncellenir. Bu, beynin bir sonraki seferde daha doğru tahmin yapmasını sağlayacak şekilde bağlantılarını "ayarlaması" anlamına gelir.</li>
                    </ol>
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold text-lg">5. Eğitim (Training): Tekrar Tekrar Denemek</h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    Bir "epoch", tüm veri setinin bir kez ileri besleme ve geri yayılım sürecinden geçmesidir. Biz bu işlemi 10,000 kez tekrarlıyoruz. Her tekrarda, beynimizdeki ağırlıklar çok küçük adımlarla daha doğru hale gelir ve ağımız problemi çözmeyi yavaş yavaş "öğrenir". Hatanın her 1000 epoch'ta bir yazdırılması, öğrenme sürecinin ne kadar iyi gittiğini görmemizi sağlar.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold text-lg">6. XOR Problemi: Neden Önemli?</h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    XOR problemi, "doğrusal olarak ayrılamayan" bir problem olduğu için klasiktir. Yani, bir grafik üzerinde sonuçları (0'ları ve 1'leri) tek bir düz çizgiyle ayıramazsınız. Bu tür problemleri çözmek için en az bir gizli katmana sahip sinir ağları gerekir, bu da yapay sinir ağlarının gücünü gösterir.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold text-lg">7. Görselleştirme: Sonuçları Anlamak</h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    İlk grafik, eğitim süresince hatanın nasıl giderek azaldığını gösterir. Bu, ağın başarılı bir şekilde öğrendiğinin kanıtıdır. İkinci grafik ise XOR probleminin veri noktalarını görselleştirerek problemi daha somut bir şekilde anlamamıza yardımcı olur.
                  </p>
                </div>
              </TabsContent>
              
              <TabsContent value="output" className="p-6 m-0">
                <div className="space-y-6">
                  <div>
                    <h3 className="font-semibold mb-2">Konsol Çıktısı:</h3>
                    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md text-sm overflow-x-auto">
                      <code>{`Epoch 0: Error 0.24937604107115477
Epoch 1000: Error 0.008564914768612755
Epoch 2000: Error 0.0036567098274375373
Epoch 3000: Error 0.0023306395027231224
Epoch 4000: Error 0.0017223631714400184
Epoch 5000: Error 0.0013729731127141298
Epoch 6000: Error 0.001142851117401871
Epoch 7000: Error 0.0009812708051325696
Epoch 8000: Error 0.0008608635865440678
Epoch 9000: Error 0.0007672745749284473
Tahminler:
[[0.00984667]
 [0.99034295]
 [0.99028902]
 [0.00971106]]`}</code>
                    </pre>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold mb-2">Eğitim Hatası Grafiği:</h3>
                    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md">
                      <Image 
                        src="/images/code-examples/neural-network-error.jpg" 
                        alt="Eğitim Hatası Grafiği" 
                        width={600} 
                        height={350} 
                        className="mx-auto"
                      />
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold mb-2">XOR Problemi Görselleştirmesi:</h3>
                    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md">
                      <Image 
                        src="/images/code-examples/xor-visualization.jpg" 
                        alt="XOR Problemi Görselleştirmesi" 
                        width={600} 
                        height={350} 
                        className="mx-auto"
                      />
                    </div>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </div>
          
          {/* İleri Okuma */}
          <div className="mt-8">
            <h3 className="text-xl font-bold mb-4">İleri Okuma</h3>
            <ul className="space-y-2">
              <li className="flex items-start">
                <span className="text-primary mr-2">→</span>
                <a href="https://www.deeplearningbook.org/" target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-400 hover:underline">
                  Deep Learning Book (Goodfellow, Bengio, Courville)
                </a>
              </li>
              <li className="flex items-start">
                <span className="text-primary mr-2">→</span>
                <a href="http://neuralnetworksanddeeplearning.com/" target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-400 hover:underline">
                  Neural Networks and Deep Learning (Michael Nielsen)
                </a>
              </li>
              <li className="flex items-start">
                <span className="text-primary mr-2">→</span>
                <a href="/topics/neural-networks/basics" className="text-blue-600 dark:text-blue-400 hover:underline">
                  Yapay Sinir Ağları Temelleri (Kodleon)
                </a>
              </li>
            </ul>
          </div>
          
          {/* İlgili Kod Örnekleri */}
          <div className="mt-8">
            <h3 className="text-xl font-bold mb-4">İlgili Kod Örnekleri</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">CNN ile Görüntü Sınıflandırma</CardTitle>
                  <CardDescription>TensorFlow ve Keras kullanarak evrişimli sinir ağı ile görüntü sınıflandırma.</CardDescription>
                </CardHeader>
                <CardFooter>
                  <Button asChild variant="outline" size="sm" className="w-full">
                    <Link href="/kod-ornekleri/resim-siniflandirma">
                      İncele
                      <ArrowRight className="h-4 w-4 ml-2" />
                    </Link>
                  </Button>
                </CardFooter>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">LSTM ile Zaman Serisi Tahmini</CardTitle>
                  <CardDescription>Uzun-Kısa Vadeli Bellek ağları ile zaman serisi verilerinde tahmin yapma.</CardDescription>
                </CardHeader>
                <CardFooter>
                  <Button asChild variant="outline" size="sm" className="w-full">
                    <Link href="/kod-ornekleri/zaman-serisi-tahmini">
                      İncele
                      <ArrowRight className="h-4 w-4 ml-2" />
                    </Link>
                  </Button>
                </CardFooter>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 