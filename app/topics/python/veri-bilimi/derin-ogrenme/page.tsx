import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python ile Derin Öğrenme | Python Veri Bilimi | Kodleon',
  description: 'Python kullanarak derin öğrenme temellerini, sinir ağlarını ve modern derin öğrenme uygulamalarını öğrenin.',
};

const content = `
# Python ile Derin Öğrenme

Derin öğrenme, yapay sinir ağlarını kullanarak karmaşık örüntüleri öğrenebilen bir makine öğrenmesi alt dalıdır. Bu bölümde, Python ile derin öğrenme uygulamalarını öğreneceğiz.

## Derin Öğrenme Temelleri

### Yapay Sinir Ağları

\`\`\`python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Basit bir sinir ağı örneği
model = Sequential([
    Dense(4, activation='relu', input_shape=(2,)),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Model özeti
model.summary()

# Görselleştirme için örnek veri
X = np.random.randn(1000, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Model eğitimi
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, epochs=10, validation_split=0.2)

# Eğitim sürecini görselleştirme
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()
\`\`\`

### Aktivasyon Fonksiyonları

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

# Aktivasyon fonksiyonları
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Görselleştirme
x = np.linspace(-5, 5, 100)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, relu(x))
plt.title('ReLU')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, tanh(x))
plt.title('Tanh')
plt.grid(True)

plt.tight_layout()
plt.show()
\`\`\`

## Evrişimli Sinir Ağları (CNN)

### MNIST Örneği

\`\`\`python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# Veri yükleme
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Veri ön işleme
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# CNN modeli
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Model derleme
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model eğitimi
history = model.fit(X_train, y_train, epochs=5, 
                   validation_data=(X_test, y_test))

# Sonuçları görselleştirme
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()
\`\`\`

## Tekrarlayan Sinir Ağları (RNN)

### LSTM ile Metin Sınıflandırma

\`\`\`python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Örnek veri
metinler = [
    "Bu film çok güzeldi",
    "Hayal kırıklığına uğradım",
    "Muhteşem bir deneyimdi",
    "Zamanımı boşa harcadım",
    # ... daha fazla örnek
]
etiketler = [1, 0, 1, 0]  # 1: Pozitif, 0: Negatif

# Metin ön işleme
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(metinler)
X = tokenizer.texts_to_sequences(metinler)
X = pad_sequences(X, maxlen=20)

# LSTM modeli
model = Sequential([
    Embedding(1000, 16, input_length=20),
    LSTM(32, return_sequences=True),
    LSTM(16),
    Dense(1, activation='sigmoid')
])

# Model derleme ve eğitim
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X, np.array(etiketler), 
                   epochs=10, validation_split=0.2)
\`\`\`

## Transfer Öğrenme

### ResNet50 ile Görüntü Sınıflandırma

\`\`\`python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Önceden eğitilmiş model
base_model = ResNet50(weights='imagenet', 
                     include_top=False, 
                     input_shape=(224, 224, 3))

# Transfer öğrenme modeli
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Temel modeli dondurma
base_model.trainable = False

# Veri artırma
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Model derleme
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
\`\`\`

## Otokodlayıcılar

### Gürültü Giderme Örneği

\`\`\`python
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model

# Otokodlayıcı model
input_img = Input(shape=(28, 28, 1))

# Kodlayıcı
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Kod çözücü
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Model oluşturma
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Gürültülü veri oluşturma
noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(
    loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(
    loc=0.0, scale=1.0, size=X_test.shape)

# Model eğitimi
autoencoder.fit(X_train_noisy, X_train,
                epochs=10,
                batch_size=128,
                validation_data=(X_test_noisy, X_test))
\`\`\`

## Üretici Çekişmeli Ağlar (GAN)

### Basit GAN Örneği

\`\`\`python
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.models import Sequential

# Üretici model
def build_generator():
    model = Sequential([
        Dense(256, input_dim=100),
        LeakyReLU(alpha=0.2),
        Dense(512),
        LeakyReLU(alpha=0.2),
        Dense(784, activation='tanh')
    ])
    return model

# Ayırt edici model
def build_discriminator():
    model = Sequential([
        Dense(512, input_dim=784),
        LeakyReLU(alpha=0.2),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ])
    return model

# GAN modeli
generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

gan_input = Input(shape=(100,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')
\`\`\`

## Alıştırmalar

1. **Sinir Ağı Tasarımı**
   - Farklı aktivasyon fonksiyonları deneyin
   - Katman sayısı ve nöron sayısını değiştirin
   - Dropout ve batch normalization ekleyin

2. **CNN Uygulamaları**
   - Kendi veri setinizle görüntü sınıflandırma yapın
   - Farklı CNN mimarileri deneyin
   - Veri artırma teknikleri uygulayın

3. **RNN ve LSTM**
   - Metin üretme modeli geliştirin
   - Duygu analizi yapın
   - Zaman serisi tahmini yapın

## Sonraki Adımlar

1. [Doğal Dil İşleme](/topics/python/veri-bilimi/dogal-dil-isleme)
2. [Bilgisayarlı Görü](/topics/python/veri-bilimi/bilgisayarli-goru)
3. [Pekiştirmeli Öğrenme](/topics/python/veri-bilimi/pekistirmeli-ogrenme)

## Faydalı Kaynaklar

- [TensorFlow Dokümantasyonu](https://www.tensorflow.org/guide)
- [PyTorch Öğreticileri](https://pytorch.org/tutorials/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
`;

export default function DeepLearningPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild className="mb-6">
            <Link href="/topics/python/veri-bilimi" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Veri Bilimi
            </Link>
          </Button>
        </div>

        <div className="prose prose-lg dark:prose-invert">
          <MarkdownContent content={content} />
        </div>

        <div className="mt-16 text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} Kodleon | Python Eğitim Platformu</p>
        </div>
      </div>
    </div>
  );
} 