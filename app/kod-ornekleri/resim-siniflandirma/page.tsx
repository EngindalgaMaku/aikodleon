import { Metadata } from 'next';
import Link from 'next/link';
import Image from 'next/image';
import { ArrowLeft, ArrowRight, Download, Github, Copy } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'CNN ile Görüntü Sınıflandırma | Kod Örnekleri | Kodleon',
  description: 'TensorFlow ve Keras kullanarak evrişimli sinir ağı (CNN) ile görüntü sınıflandırma örneği.',
  openGraph: {
    title: 'CNN ile Görüntü Sınıflandırma | Kodleon',
    description: 'TensorFlow ve Keras kullanarak evrişimli sinir ağı (CNN) ile görüntü sınıflandırma örneği.',
    images: [{ url: '/images/code-examples/image-classification.jpg' }],
  },
};

export default function ImageClassificationPage() {
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
            <h1 className="text-3xl font-bold mb-4">CNN ile Görüntü Sınıflandırma</h1>
            
            <div className="flex items-center gap-2 mb-4">
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
                Bilgisayarlı Görü
              </span>
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
                Orta
              </span>
            </div>
            
            <p className="text-muted-foreground mb-6">
              Bu örnekte, TensorFlow ve Keras kütüphanelerini kullanarak bir Evrişimli Sinir Ağı (CNN) modeli oluşturacak 
              ve CIFAR-10 veri seti üzerinde görüntü sınıflandırma yapacaksınız. CNN'ler, özellikle görüntü işleme 
              görevlerinde yüksek performans gösteren derin öğrenme mimarileridir.
            </p>
            
            <div className="space-y-4 mb-6">
              <div>
                <h3 className="font-medium">Gereksinimler:</h3>
                <ul className="list-disc list-inside text-sm text-muted-foreground">
                  <li>Python 3.6+</li>
                  <li>TensorFlow 2.x</li>
                  <li>Keras</li>
                  <li>NumPy</li>
                  <li>Matplotlib</li>
                </ul>
              </div>
              
              <div>
                <h3 className="font-medium">Öğrenilecek Kavramlar:</h3>
                <ul className="list-disc list-inside text-sm text-muted-foreground">
                  <li>Evrişimli Sinir Ağları (CNN) mimarisi</li>
                  <li>Evrişim (Convolution) ve Havuzlama (Pooling) katmanları</li>
                  <li>Görüntü veri ön işleme</li>
                  <li>Model eğitimi ve değerlendirme</li>
                  <li>Veri artırma (Data Augmentation)</li>
                </ul>
              </div>
            </div>
            
            <div className="flex flex-col gap-2">
              <Button asChild variant="default" className="gap-2">
                <a href="/notebooks/resim-siniflandirma.ipynb" download>
                  <Download className="h-4 w-4" />
                  Jupyter Notebook İndir
                </a>
              </Button>
              <Button asChild variant="outline" className="gap-2">
                <a href="https://github.com/kodleon/ai-examples/blob/main/computer-vision/cnn-image-classification.ipynb" target="_blank" rel="noopener noreferrer">
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
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# CIFAR-10 veri setini yükle
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Veri setini normalize et (0-1 aralığına getir)
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Sınıf isimlerini tanımla
class_names = ['Uçak', 'Araba', 'Kuş', 'Kedi', 'Geyik', 
               'Köpek', 'Kurbağa', 'At', 'Gemi', 'Kamyon']

# Etiketleri kategorik hale getir
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Veri artırma için ImageDataGenerator kullan
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# CNN modelini oluştur
model = models.Sequential()

# Evrişim katmanları
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.4))

# Yoğun (tam bağlantılı) katmanlar
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# Modeli derle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model mimarisini özetle
model.summary()

# Eğitim için erken durdurma ve öğrenme oranı azaltma callback'leri
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001
)

# Modeli eğit
batch_size = 64
epochs = 50

history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=batch_size),
    epochs=epochs,
    validation_data=(test_images, test_labels),
    callbacks=[early_stopping, reduce_lr]
)

# Eğitim ve doğrulama metriklerini görselleştir
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.tight_layout()
plt.show()

# Test veri seti üzerinde modeli değerlendir
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test doğruluğu: {test_acc:.4f}')

# Rastgele örnekler üzerinde tahminler yap ve görselleştir
def plot_predictions(images, true_labels, predictions, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        
        predicted_label = np.argmax(predictions[i])
        true_label = np.argmax(true_labels[i])
        
        if predicted_label == true_label:
            color = 'green'
        else:
            color = 'red'
            
        plt.xlabel(f"{class_names[predicted_label]} ({class_names[true_label]})", color=color)
    plt.tight_layout()
    plt.show()

# Rastgele 25 test görüntüsü seç
indices = np.random.choice(test_images.shape[0], 25)
sample_images = test_images[indices]
sample_labels = test_labels[indices]

# Tahminler yap
predictions = model.predict(sample_images)

# Tahminleri görselleştir
plot_predictions(sample_images, sample_labels, predictions, class_names)

# Modeli kaydet
model.save('cifar10_cnn_model.h5')
print("Model kaydedildi: cifar10_cnn_model.h5")`}</code>
                  </pre>
                </div>
              </TabsContent>
              
              <TabsContent value="explanation" className="p-6 m-0 space-y-4">
                <h3 className="text-xl font-bold">Kod Açıklaması</h3>
                
                <div>
                  <h4 className="font-semibold">1. Veri Yükleme ve Ön İşleme</h4>
                  <p className="text-sm text-muted-foreground">
                    CIFAR-10 veri seti, 10 farklı sınıfa ait 60.000 renkli görüntü içerir (50.000 eğitim, 10.000 test). 
                    Görüntüler 32x32 piksel boyutundadır. Kod, veriyi yükler ve 0-1 aralığında normalize eder.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">2. Veri Artırma (Data Augmentation)</h4>
                  <p className="text-sm text-muted-foreground">
                    <code>ImageDataGenerator</code> kullanarak eğitim verilerini çeşitlendiriyoruz. Bu, modelin daha iyi genelleme 
                    yapmasına yardımcı olur. Dönüşümler arasında döndürme, kaydırma, çevirme ve yakınlaştırma bulunur.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">3. CNN Mimarisi</h4>
                  <p className="text-sm text-muted-foreground">
                    Model, üç evrişim bloğundan oluşur. Her blok şunları içerir:
                    <ul className="list-disc list-inside ml-4 mt-2">
                      <li>İki evrişim katmanı (Conv2D)</li>
                      <li>Batch normalizasyon katmanları</li>
                      <li>Bir maksimum havuzlama katmanı (MaxPooling2D)</li>
                      <li>Bir dropout katmanı (aşırı öğrenmeyi önlemek için)</li>
                    </ul>
                    Son olarak, tam bağlantılı katmanlar (Dense) sınıflandırma için kullanılır.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">4. Model Derleme ve Eğitim</h4>
                  <p className="text-sm text-muted-foreground">
                    Model, Adam optimizasyonu ve categorical_crossentropy kayıp fonksiyonu ile derlenir. Eğitim sırasında, 
                    erken durdurma (early stopping) ve öğrenme oranı azaltma (learning rate reduction) teknikleri kullanılır.
                    Bu, modelin daha iyi performans göstermesine ve aşırı öğrenmeyi önlemeye yardımcı olur.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">5. Değerlendirme ve Görselleştirme</h4>
                  <p className="text-sm text-muted-foreground">
                    Eğitim tamamlandıktan sonra, doğruluk ve kayıp grafikleri çizilir. Ayrıca, model test veri seti üzerinde 
                    değerlendirilir ve rastgele seçilen 25 görüntü üzerinde tahminler yapılır. Doğru tahminler yeşil, 
                    yanlış tahminler kırmızı olarak işaretlenir.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">6. Model Kaydetme</h4>
                  <p className="text-sm text-muted-foreground">
                    Eğitilen model, daha sonra kullanılmak üzere HDF5 formatında kaydedilir.
                  </p>
                </div>
              </TabsContent>
              
              <TabsContent value="output" className="p-6 m-0">
                <div className="space-y-6">
                  <div>
                    <h3 className="font-semibold mb-2">Model Özeti:</h3>
                    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md text-sm overflow-x-auto">
                      <code>{`Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 32, 32, 32)        896       
                                                                 
 batch_normalization (Batch  (None, 32, 32, 32)        128       
 Normalization)                                                  
                                                                 
 conv2d_1 (Conv2D)           (None, 32, 32, 32)        9248      
                                                                 
 batch_normalization_1 (Bat  (None, 32, 32, 32)        128       
 chNormalization)                                                
                                                                 
 max_pooling2d (MaxPooling2  (None, 16, 16, 32)        0         
 D)                                                              
                                                                 
 dropout (Dropout)           (None, 16, 16, 32)        0         
                                                                 
 conv2d_2 (Conv2D)           (None, 16, 16, 64)        18496     
                                                                 
 batch_normalization_2 (Bat  (None, 16, 16, 64)        256       
 chNormalization)                                                
                                                                 
 conv2d_3 (Conv2D)           (None, 16, 16, 64)        36928     
                                                                 
 batch_normalization_3 (Bat  (None, 16, 16, 64)        256       
 chNormalization)                                                
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 8, 8, 64)          0         
 g2D)                                                            
                                                                 
 dropout_1 (Dropout)         (None, 8, 8, 64)          0         
                                                                 
 conv2d_4 (Conv2D)           (None, 8, 8, 128)         73856     
                                                                 
 batch_normalization_4 (Bat  (None, 8, 8, 128)         512       
 chNormalization)                                                
                                                                 
 conv2d_5 (Conv2D)           (None, 8, 8, 128)         147584    
                                                                 
 batch_normalization_5 (Bat  (None, 8, 8, 128)         512       
 chNormalization)                                                
                                                                 
 max_pooling2d_2 (MaxPoolin  (None, 4, 4, 128)         0         
 g2D)                                                            
                                                                 
 dropout_2 (Dropout)         (None, 4, 4, 128)         0         
                                                                 
 flatten (Flatten)           (None, 2048)              0         
                                                                 
 dense (Dense)               (None, 128)               262272    
                                                                 
 batch_normalization_6 (Bat  (None, 128)               512       
 chNormalization)                                                
                                                                 
 dropout_3 (Dropout)         (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 552,874
Trainable params: 551,722
Non-trainable params: 1,152
_________________________________________________________________`}</code>
                    </pre>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold mb-2">Eğitim Sonuçları:</h3>
                    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md text-sm overflow-x-auto">
                      <code>{`Epoch 25/50
782/782 [==============================] - 8s 10ms/step - loss: 0.4712 - accuracy: 0.8357 - val_loss: 0.5101 - val_accuracy: 0.8302 - lr: 1.0000e-04
Epoch 26/50
782/782 [==============================] - 8s 10ms/step - loss: 0.4687 - accuracy: 0.8362 - val_loss: 0.5094 - val_accuracy: 0.8312 - lr: 1.0000e-04
...
Epoch 36/50
782/782 [==============================] - 8s 10ms/step - loss: 0.4431 - accuracy: 0.8451 - val_loss: 0.5092 - val_accuracy: 0.8318 - lr: 1.0000e-05

313/313 [==============================] - 1s 3ms/step - loss: 0.5092 - accuracy: 0.8318
Test doğruluğu: 0.8318`}</code>
                    </pre>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold mb-2">Eğitim ve Doğrulama Grafikleri:</h3>
                    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md">
                      <Image 
                        src="/images/code-examples/cnn-training-curves.jpg" 
                        alt="Eğitim ve Doğrulama Grafikleri" 
                        width={600} 
                        height={250} 
                        className="mx-auto"
                      />
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold mb-2">Tahmin Sonuçları:</h3>
                    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md">
                      <Image 
                        src="/images/code-examples/cnn-predictions.jpg" 
                        alt="Tahmin Sonuçları" 
                        width={600} 
                        height={600} 
                        className="mx-auto"
                      />
                      <p className="text-center text-sm mt-2 text-muted-foreground">
                        Yeşil etiketler doğru tahminleri, kırmızı etiketler yanlış tahminleri gösterir.
                        Parantez içindeki değerler gerçek sınıfı belirtir.
                      </p>
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
                <a href="https://www.tensorflow.org/tutorials/images/cnn" target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-400 hover:underline">
                  TensorFlow: Evrişimli Sinir Ağları
                </a>
              </li>
              <li className="flex items-start">
                <span className="text-primary mr-2">→</span>
                <a href="https://cs231n.github.io/" target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-400 hover:underline">
                  Stanford CS231n: Görsel Tanıma için Evrişimli Sinir Ağları
                </a>
              </li>
              <li className="flex items-start">
                <span className="text-primary mr-2">→</span>
                <a href="/topics/neural-networks/konvolusyonel-sinir-aglari" className="text-blue-600 dark:text-blue-400 hover:underline">
                  Evrişimli Sinir Ağları Detaylı İnceleme (Kodleon)
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
                  <CardTitle className="text-base">Temel Yapay Sinir Ağı</CardTitle>
                  <CardDescription>NumPy kullanarak sıfırdan basit bir yapay sinir ağı oluşturma ve eğitme.</CardDescription>
                </CardHeader>
                <CardFooter>
                  <Button asChild variant="outline" size="sm" className="w-full">
                    <Link href="/kod-ornekleri/temel-sinir-agi">
                      İncele
                      <ArrowRight className="h-4 w-4 ml-2" />
                    </Link>
                  </Button>
                </CardFooter>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">NLP ile Duygu Analizi</CardTitle>
                  <CardDescription>NLTK ve scikit-learn kullanarak metin tabanlı duygu analizi uygulaması.</CardDescription>
                </CardHeader>
                <CardFooter>
                  <Button asChild variant="outline" size="sm" className="w-full">
                    <Link href="/kod-ornekleri/nlp-duygu-analizi">
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