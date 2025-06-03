import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export const metadata: Metadata = {
  title: 'Derin Öğrenme Frameworkleri | Kodleon',
  description: 'PyTorch ve TensorFlow gibi popüler derin öğrenme kütüphanelerini öğrenin.',
};

const content = `
# Derin Öğrenme Frameworkleri

Derin öğrenme frameworkleri, karmaşık sinir ağlarını kolayca oluşturmamızı ve eğitmemizi sağlayan güçlü araçlardır. Bu bölümde, en popüler iki framework olan PyTorch ve TensorFlow'u inceleyeceğiz.

## 1. PyTorch

PyTorch, dinamik hesaplama grafikleri ve Python'a yakın bir programlama deneyimi sunan modern bir derin öğrenme frameworküdür.

### Temel Kavramlar

\`\`\`python
import torch
import torch.nn as nn
import torch.optim as optim

# Tensör oluşturma
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.zeros(2, 2)

# Basit bir sinir ağı
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model oluşturma ve eğitim
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Eğitim döngüsü
for epoch in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
\`\`\`

### Veri Yükleme ve İşleme

\`\`\`python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# DataLoader kullanımı
dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_data, batch_labels in dataloader:
    # Eğitim işlemleri
    pass
\`\`\`

## 2. TensorFlow ve Keras

TensorFlow, Google tarafından geliştirilen ve Keras yüksek seviye API'sini içeren güçlü bir derin öğrenme frameworküdür.

### Model Oluşturma

\`\`\`python
import tensorflow as tf
from tensorflow import keras

# Sequential API ile model oluşturma
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Model derleme
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model özeti
model.summary()
\`\`\`

### Özel Katmanlar ve Modeller

\`\`\`python
class CustomLayer(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# Özel model
class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = CustomLayer(32)
        self.dense2 = keras.layers.Dense(1)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
\`\`\`

## 3. Model Eğitimi ve Değerlendirme

### PyTorch ile Eğitim

\`\`\`python
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = correct / len(test_loader.dataset)
    return accuracy
\`\`\`

### TensorFlow/Keras ile Eğitim

\`\`\`python
# Veri hazırlama
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(buffer_size=1024).batch(32)

# Model eğitimi
history = model.fit(
    train_data,
    epochs=10,
    validation_data=(x_val, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3),
        keras.callbacks.ModelCheckpoint('best_model.h5')
    ]
)

# Model değerlendirme
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
\`\`\`

## 4. Model Kaydetme ve Yükleme

### PyTorch

\`\`\`python
# Model kaydetme
torch.save(model.state_dict(), 'model.pth')

# Model yükleme
model = SimpleNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()
\`\`\`

### TensorFlow/Keras

\`\`\`python
# Model kaydetme
model.save('model.h5')

# Model yükleme
loaded_model = keras.models.load_model('model.h5')
\`\`\`

## Alıştırmalar

1. **PyTorch ile Başlangıç**
   - Tensör işlemleri
   - Basit sinir ağı oluşturma
   - Veri yükleme ve ön işleme

2. **TensorFlow/Keras ile Başlangıç**
   - Keras Sequential API kullanımı
   - Özel katman oluşturma
   - Callback fonksiyonları

3. **Model Geliştirme**
   - Farklı optimizasyon algoritmalarını deneme
   - Hiperparametre optimizasyonu
   - Model performans analizi

## Kaynaklar

- [PyTorch Dokümantasyonu](https://pytorch.org/docs/stable/index.html)
- [TensorFlow Dokümantasyonu](https://www.tensorflow.org/api_docs)
- [Keras Rehberi](https://keras.io/guides/)
`;

export default function DeepLearningFrameworksPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <Button asChild variant="ghost" size="sm" className="gap-1">
          <Link href="/topics/python/derin-ogrenme">
            <ArrowLeft className="h-4 w-4" />
            Derin Öğrenmeye Dön
          </Link>
        </Button>
      </div>
      
      <div className="prose prose-lg dark:prose-invert max-w-none">
        <MarkdownContent content={content} />
      </div>
      
      {/* Interactive Examples */}
      <div className="my-12">
        <h2 className="text-3xl font-bold mb-8">İnteraktif Örnekler</h2>
        <Tabs defaultValue="pytorch">
          <TabsList>
            <TabsTrigger value="pytorch">PyTorch</TabsTrigger>
            <TabsTrigger value="tensorflow">TensorFlow</TabsTrigger>
            <TabsTrigger value="comparison">Karşılaştırma</TabsTrigger>
          </TabsList>
          
          <TabsContent value="pytorch">
            <Card>
              <CardHeader>
                <CardTitle>PyTorch Örneği</CardTitle>
                <CardDescription>
                  PyTorch ile basit bir sinir ağı oluşturma ve eğitme
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`import torch
import torch.nn as nn

# Basit bir sinir ağı
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model oluşturma
model = Net()

# Örnek veri
x = torch.randn(32, 10)  # 32 örnek, 10 özellik
y = torch.randn(32, 1)   # 32 hedef değer

# Kayıp fonksiyonu ve optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Eğitim döngüsü
for epoch in range(100):
    # İleri yayılım
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # Geri yayılım
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="tensorflow">
            <Card>
              <CardHeader>
                <CardTitle>TensorFlow/Keras Örneği</CardTitle>
                <CardDescription>
                  Keras ile model oluşturma ve eğitim
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`import tensorflow as tf
from tensorflow import keras

# Model oluşturma
model = keras.Sequential([
    keras.layers.Dense(5, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])

# Model derleme
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Örnek veri
x = tf.random.normal((32, 10))
y = tf.random.normal((32, 1))

# Model eğitimi
history = model.fit(
    x, y,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Model değerlendirme
test_x = tf.random.normal((10, 10))
predictions = model.predict(test_x)
print("Tahminler:", predictions[:5])`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="comparison">
            <Card>
              <CardHeader>
                <CardTitle>Framework Karşılaştırması</CardTitle>
                <CardDescription>
                  PyTorch ve TensorFlow arasındaki temel farklar
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`# PyTorch - Dinamik Hesaplama Grafı
import torch

x = torch.tensor([1.0], requires_grad=True)
y = x * 2
y.backward()
print("PyTorch gradient:", x.grad)

# TensorFlow - Statik Hesaplama Grafı
import tensorflow as tf

x = tf.Variable([1.0])
with tf.GradientTape() as tape:
    y = x * 2
grad = tape.gradient(y, x)
print("TensorFlow gradient:", grad.numpy())

# Karşılaştırma Notları:
# 1. PyTorch: Daha Pythonic, araştırma için ideal
# 2. TensorFlow: Üretim için güçlü araçlar
# 3. PyTorch: Dinamik graflar
# 4. TensorFlow: Statik graflar (Eager execution ile dinamik de olabilir)
# 5. PyTorch: Daha kolay hata ayıklama
# 6. TensorFlow: Daha iyi üretim araçları ve TensorBoard`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Navigation */}
      <div className="mt-12 flex flex-col sm:flex-row justify-between gap-4">
        <Button asChild variant="outline" className="gap-2">
          <Link href="/topics/python/derin-ogrenme/yapay-sinir-aglari-temelleri">
            <ArrowLeft className="h-4 w-4" />
            Önceki Konu: Yapay Sinir Ağları Temelleri
          </Link>
        </Button>
        
        <Button asChild variant="default" className="gap-2">
          <Link href="/topics/python/derin-ogrenme/evrisimli-sinir-aglari">
            Sonraki Konu: Evrişimli Sinir Ağları
            <ArrowRight className="h-4 w-4" />
          </Link>
        </Button>
      </div>
      
      <div className="mt-16 text-center text-sm text-muted-foreground">
        <p>© {new Date().getFullYear()} Kodleon | Yapay Zeka Eğitim Platformu</p>
      </div>
    </div>
  );
} 