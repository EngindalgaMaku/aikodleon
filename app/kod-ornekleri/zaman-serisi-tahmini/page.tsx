import { Metadata } from 'next';
import Link from 'next/link';
import Image from 'next/image';
import { ArrowLeft, ArrowRight, Download, Github, Copy } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'LSTM ile Zaman Serisi Tahmini | Kod Örnekleri | Kodleon',
  description: 'Uzun-Kısa Vadeli Bellek (LSTM) ağları ile zaman serisi verilerinde tahmin yapma örneği.',
  openGraph: {
    title: 'LSTM ile Zaman Serisi Tahmini | Kodleon',
    description: 'Uzun-Kısa Vadeli Bellek (LSTM) ağları ile zaman serisi verilerinde tahmin yapma örneği.',
    images: [{ url: '/images/code-examples/time-series.jpg' }],
  },
};

export default function TimeSeriesForecastingPage() {
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
            <h1 className="text-3xl font-bold mb-4">LSTM ile Zaman Serisi Tahmini</h1>
            
            <div className="flex items-center gap-2 mb-4">
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                Derin Öğrenme
              </span>
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200">
                İleri
              </span>
            </div>
            
            <p className="text-muted-foreground mb-6">
              Bu örnekte, TensorFlow ve Keras kullanarak Uzun-Kısa Vadeli Bellek (LSTM) ağları ile zaman serisi verilerinde tahmin yapacaksınız. 
              Örnek, finansal verilerde fiyat tahmini yapmak için bir LSTM modelinin nasıl oluşturulacağını ve eğitileceğini gösterir.
            </p>
            
            <div className="space-y-4 mb-6">
              <div>
                <h3 className="font-medium">Gereksinimler:</h3>
                <ul className="list-disc list-inside text-sm text-muted-foreground">
                  <li>Python 3.6+</li>
                  <li>TensorFlow 2.x</li>
                  <li>Keras</li>
                  <li>NumPy</li>
                  <li>Pandas</li>
                  <li>Matplotlib</li>
                  <li>scikit-learn</li>
                </ul>
              </div>
              
              <div>
                <h3 className="font-medium">Öğrenilecek Kavramlar:</h3>
                <ul className="list-disc list-inside text-sm text-muted-foreground">
                  <li>Zaman serisi verilerinin ön işlemesi</li>
                  <li>LSTM mimarisi ve çalışma prensibi</li>
                  <li>Zaman adımlarının (time steps) yapılandırılması</li>
                  <li>Sekans-sekans (sequence-to-sequence) modelleme</li>
                  <li>Model değerlendirme metrikleri</li>
                </ul>
              </div>
            </div>
            
            <div className="flex flex-col gap-2">
              <Button asChild variant="default" className="gap-2">
                <a href="/notebooks/zaman-serisi-tahmini.ipynb" download>
                  <Download className="h-4 w-4" />
                  Jupyter Notebook İndir
                </a>
              </Button>
              <Button asChild variant="outline" className="gap-2">
                <a href="https://github.com/kodleon/ai-examples/blob/main/time-series/lstm-forecasting.ipynb" target="_blank" rel="noopener noreferrer">
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
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import yfinance as yf

# Veri setini indir (örnek olarak Bitcoin fiyat verileri)
df = yf.download('BTC-USD', start='2020-01-01', end='2023-01-01')

# Sadece kapanış fiyatlarını al
data = df['Close'].values.reshape(-1, 1)

# Veriyi normalize et (0-1 arasına ölçeklendir)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Eğitim ve test setlerini oluştur
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data = scaled_data[0:train_size, :]
test_data = scaled_data[train_size:len(scaled_data), :]

# Zaman serisi veri setini oluştur (X=t, y=t+1)
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Zaman adımı (kaç gün önceki verileri kullanarak tahmin yapacağız)
time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# LSTM modeli için veriyi yeniden şekillendir [samples, time_steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# LSTM modelini oluştur
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Modeli derle
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğit
history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                    validation_data=(X_test, y_test), verbose=1)

# Test seti üzerinde tahminler yap
y_pred = model.predict(X_test)

# Tahminleri orijinal ölçeğe geri dönüştür
y_pred = scaler.inverse_transform(y_pred)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Performans metriklerini hesapla
mse = mean_squared_error(y_test_scaled, y_pred)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y_test_scaled, y_pred)
print(f'Test RMSE: {rmse:.2f}')
print(f'Test MAE: {mae:.2f}')

# Eğitim kaybını görselleştir
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Kayıp Değerleri')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Tahminleri görselleştir
plt.figure(figsize=(12, 6))
plt.plot(y_test_scaled, label='Gerçek Fiyat')
plt.plot(y_pred, label='Tahmin Edilen Fiyat')
plt.title('Bitcoin Fiyat Tahmini')
plt.xlabel('Zaman')
plt.ylabel('Fiyat (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Gelecek 30 gün için tahmin yap
last_60_days = scaled_data[-60:]
X_future = []
X_future.append(last_60_days[:, 0])
X_future = np.array(X_future)
X_future = X_future.reshape(X_future.shape[0], X_future.shape[1], 1)

# İlk tahmin
future_pred = []
current_batch = X_future[0]

# 30 gün için tahmin yap
for i in range(30):
    # Mevcut batch ile tahmin yap
    current_pred = model.predict(current_batch.reshape(1, time_step, 1))[0]
    
    # Tahmin sonucunu kaydet
    future_pred.append(current_pred[0])
    
    # Batch'i güncelle (en eski değeri çıkar, yeni tahmini ekle)
    current_batch = np.append(current_batch[1:], current_pred)
    current_batch = current_batch.reshape(time_step, 1)

# Tahminleri orijinal ölçeğe dönüştür
future_pred = np.array(future_pred).reshape(-1, 1)
future_pred = scaler.inverse_transform(future_pred)

# Son 60 gün + gelecek 30 gün tahminini görselleştir
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(y_test_scaled)), y_test_scaled, label='Geçmiş Fiyat')
plt.plot(np.arange(len(y_test_scaled), len(y_test_scaled) + 30), future_pred, label='Gelecek Tahmin', color='red')
plt.title('Bitcoin Fiyat Tahmini (Gelecek 30 Gün)')
plt.xlabel('Zaman')
plt.ylabel('Fiyat (USD)')
plt.legend()
plt.grid(True)
plt.show()`}</code>
                  </pre>
                </div>
              </TabsContent>
              
              <TabsContent value="explanation" className="p-6 m-0 space-y-4">
                <h3 className="text-xl font-bold">Kod Açıklaması</h3>
                
                <div>
                  <h4 className="font-semibold">1. Veri Hazırlama</h4>
                  <p className="text-sm text-muted-foreground">
                    Yahoo Finance API (yfinance) kullanarak Bitcoin fiyat verilerini indiriyoruz. Daha sonra bu verileri 0-1 arasına normalize ediyoruz. Normalizasyon, LSTM ağlarının daha iyi performans göstermesine yardımcı olur.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">2. Zaman Serisi Veri Seti Oluşturma</h4>
                  <p className="text-sm text-muted-foreground">
                    <code>create_dataset()</code> fonksiyonu, zaman serisi verilerini LSTM modeli için uygun formata dönüştürür. Her bir girdi, belirtilen zaman adımı kadar geçmiş veriyi içerir ve çıktı, bir sonraki zaman adımındaki değerdir. Bu durumda, 60 günlük geçmiş veriler kullanılarak bir sonraki günün fiyatı tahmin edilecektir.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">3. LSTM Mimarisi</h4>
                  <p className="text-sm text-muted-foreground">
                    Model, üç LSTM katmanından oluşur, her biri 50 birimden oluşur ve aşırı öğrenmeyi önlemek için Dropout katmanları ile ayrılır. LSTM katmanları, zaman serisi verilerindeki uzun vadeli bağımlılıkları yakalamak için tasarlanmıştır. Son katman, tek bir değer (bir sonraki gün için fiyat) tahmin eden bir yoğun (Dense) katmandır.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">4. Model Eğitimi</h4>
                  <p className="text-sm text-muted-foreground">
                    Model, Adam optimizer ve ortalama kare hata (MSE) kaybı kullanılarak eğitilir. Eğitim 50 epoch boyunca devam eder ve 32'lik batch boyutu kullanılır. Eğitim sırasında, doğrulama seti olarak test verisi kullanılır ve eğitim/doğrulama kayıpları kaydedilir.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">5. Performans Değerlendirme</h4>
                  <p className="text-sm text-muted-foreground">
                    Model performansı, ortalama kare hata (MSE), kök ortalama kare hata (RMSE) ve ortalama mutlak hata (MAE) metrikleri kullanılarak değerlendirilir. Bu metrikler, tahminlerin gerçek değerlerden ne kadar uzak olduğunu ölçer.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">6. Gelecek Tahminleri</h4>
                  <p className="text-sm text-muted-foreground">
                    Son olarak, model gelecekteki 30 gün için fiyat tahmini yapar. Bu, son 60 günlük veriyi kullanarak başlar ve her tahmin, bir sonraki tahminin girdisi olarak kullanılır (yinelemeli tahmin). Bu yaklaşım, gerçek dünya uygulamalarında gelecekteki fiyat hareketlerini tahmin etmek için kullanılabilir.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">7. Görselleştirme</h4>
                  <p className="text-sm text-muted-foreground">
                    Kod, eğitim ve doğrulama kayıplarını, gerçek ve tahmin edilen fiyatları ve gelecek 30 gün için yapılan tahminleri görselleştiren grafikler oluşturur. Bu görselleştirmeler, modelin performansını ve tahminlerin doğruluğunu değerlendirmek için önemlidir.
                  </p>
                </div>
              </TabsContent>
              
              <TabsContent value="output" className="p-6 m-0">
                <div className="space-y-6">
                  <div>
                    <h3 className="font-semibold mb-2">Konsol Çıktısı:</h3>
                    <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md text-sm overflow-x-auto">
                      <code>{`[*********************100%***********************]  1 of 1 completed
Epoch 1/50
24/24 [==============================] - 2s 43ms/step - loss: 0.0278 - val_loss: 0.0068
Epoch 2/50
24/24 [==============================] - 1s 30ms/step - loss: 0.0093 - val_loss: 0.0045
...
Epoch 49/50
24/24 [==============================] - 1s 29ms/step - loss: 0.0018 - val_loss: 0.0025
Epoch 50/50
24/24 [==============================] - 1s 29ms/step - loss: 0.0018 - val_loss: 0.0024
5/5 [==============================] - 0s 6ms/step
Test RMSE: 1254.87
Test MAE: 941.23
1/1 [==============================] - 0s 24ms/step
1/1 [==============================] - 0s 23ms/step
...
1/1 [==============================] - 0s 22ms/step`}</code>
                    </pre>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold mb-2">Eğitim ve Doğrulama Kaybı:</h3>
                    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md">
                      <Image 
                        src="/images/code-examples/time-series-loss.jpg" 
                        alt="Eğitim ve Doğrulama Kaybı Grafiği" 
                        width={600} 
                        height={350} 
                        className="mx-auto"
                      />
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold mb-2">Tahmin Sonuçları:</h3>
                    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md">
                      <Image 
                        src="/images/code-examples/time-series-prediction.jpg" 
                        alt="Bitcoin Fiyat Tahmini" 
                        width={600} 
                        height={350} 
                        className="mx-auto"
                      />
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold mb-2">Gelecek 30 Gün Tahmini:</h3>
                    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md">
                      <Image 
                        src="/images/code-examples/time-series-future.jpg" 
                        alt="Gelecek 30 Gün Bitcoin Fiyat Tahmini" 
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
                <a href="https://www.tensorflow.org/tutorials/structured_data/time_series" target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-400 hover:underline">
                  TensorFlow Time Series Forecasting Tutorial
                </a>
              </li>
              <li className="flex items-start">
                <span className="text-primary mr-2">→</span>
                <a href="https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/" target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-400 hover:underline">
                  Time Series Forecasting Methods in Python (Machine Learning Mastery)
                </a>
              </li>
              <li className="flex items-start">
                <span className="text-primary mr-2">→</span>
                <a href="/topics/deep-learning/lstm-networks" className="text-blue-600 dark:text-blue-400 hover:underline">
                  LSTM Ağları ve Uygulamaları (Kodleon)
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
                  <CardTitle className="text-base">Anomali Tespiti Algoritması</CardTitle>
                  <CardDescription>Denetimsiz öğrenme ile veri setindeki anormallikleri tespit etme.</CardDescription>
                </CardHeader>
                <CardFooter>
                  <Button asChild variant="outline" size="sm" className="w-full">
                    <Link href="/kod-ornekleri/anomali-tespiti">
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