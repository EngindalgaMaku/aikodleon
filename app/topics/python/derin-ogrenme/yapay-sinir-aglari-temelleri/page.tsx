import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export const metadata: Metadata = {
  title: 'Yapay Sinir Ağları Temelleri | Kodleon',
  description: 'Yapay sinir ağlarının temel kavramlarını, yapısını ve çalışma prensiplerini öğrenin.',
};

const content = `
# Yapay Sinir Ağları Temelleri

Yapay sinir ağları, insan beyninin çalışma prensibinden esinlenerek geliştirilmiş matematiksel modellerdir. Bu bölümde, yapay sinir ağlarının temel bileşenlerini ve çalışma prensiplerini öğreneceğiz.

## 1. Yapay Nöron (Perceptron)

Yapay sinir ağlarının temel yapı taşı olan perceptron, biyolojik nöronların matematiksel modelidir.

\`\`\`python
import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
    
    def activation(self, x):
        return 1 if x > 0 else 0
    
    def predict(self, inputs):
        sum_value = np.dot(inputs, self.weights) + self.bias
        return self.activation(sum_value)
    
    def train(self, inputs, target):
        prediction = self.predict(inputs)
        error = target - prediction
        
        # Ağırlıkları ve bias'ı güncelle
        self.weights += self.learning_rate * error * inputs
        self.bias += self.learning_rate * error

# Kullanım
# XOR problemi için perceptron
p = Perceptron(2)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

# Eğitim
for _ in range(100):
    for inputs, target in zip(X, y):
        p.train(inputs, target)
\`\`\`

## 2. Aktivasyon Fonksiyonları

Aktivasyon fonksiyonları, nöronların çıktısını belirleyen matematiksel fonksiyonlardır.

### Sigmoid Fonksiyonu

\`\`\`python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
\`\`\`

### ReLU (Rectified Linear Unit)

\`\`\`python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)
\`\`\`

### Tanh Fonksiyonu

\`\`\`python
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2
\`\`\`

## 3. İleri Yayılım (Forward Propagation)

İleri yayılım, giriş verilerinin ağ boyunca ilerleyerek çıktıya dönüşme sürecidir.

\`\`\`python
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        
        # Ağırlıkları ve bias'ları başlat
        for i in range(len(layers)-1):
            w = np.random.randn(layers[i], layers[i+1])
            b = np.random.randn(layers[i+1])
            self.weights.append(w)
            self.biases.append(b)
    
    def forward_propagation(self, inputs):
        activations = [inputs]
        current_input = inputs
        
        for i in range(len(self.weights)):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            current_input = sigmoid(z)
            activations.append(current_input)
        
        return activations

# Kullanım
# 2-3-1 mimarisinde bir ağ oluştur
nn = NeuralNetwork([2, 3, 1])
sample_input = np.array([0.5, 0.8])
output = nn.forward_propagation(sample_input)
\`\`\`

## 4. Geri Yayılım (Backpropagation)

Geri yayılım, ağın çıktısındaki hatanın geriye doğru yayılarak ağırlıkların güncellenmesi sürecidir.

\`\`\`python
def backward_propagation(self, x, y, activations):
    m = x.shape[0]
    delta = activations[-1] - y
    
    dW = []
    db = []
    
    for i in range(len(self.weights) - 1, -1, -1):
        dw = np.dot(activations[i].T, delta) / m
        db = np.sum(delta, axis=0) / m
        
        if i > 0:
            delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(activations[i])
        
        dW.insert(0, dw)
        db.insert(0, db)
    
    return dW, db
\`\`\`

## 5. Optimizasyon Algoritmaları

### Gradient Descent

\`\`\`python
def gradient_descent(self, x, y, learning_rate, epochs):
    for _ in range(epochs):
        # İleri yayılım
        activations = self.forward_propagation(x)
        
        # Geri yayılım
        dW, db = self.backward_propagation(x, y, activations)
        
        # Ağırlıkları güncelle
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * db[i]
\`\`\`

### Stochastic Gradient Descent (SGD)

\`\`\`python
def sgd(self, x, y, learning_rate, epochs, batch_size):
    m = x.shape[0]
    
    for _ in range(epochs):
        # Veriyi karıştır
        indices = np.random.permutation(m)
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        
        # Mini-batch'ler üzerinde eğitim
        for i in range(0, m, batch_size):
            x_batch = x_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            activations = self.forward_propagation(x_batch)
            dW, db = self.backward_propagation(x_batch, y_batch, activations)
            
            # Ağırlıkları güncelle
            for j in range(len(self.weights)):
                self.weights[j] -= learning_rate * dW[j]
                self.biases[j] -= learning_rate * db[j]
\`\`\`

## Alıştırmalar

1. **Temel Perceptron**
   - AND kapısı implementasyonu
   - OR kapısı implementasyonu
   - NOT kapısı implementasyonu

2. **Aktivasyon Fonksiyonları**
   - Farklı aktivasyon fonksiyonlarının karşılaştırılması
   - Gradyan hesaplama
   - Aktivasyon fonksiyonu seçiminin etkisi

3. **İleri ve Geri Yayılım**
   - El ile gradyan hesaplama
   - Zincir kuralı uygulaması
   - Hata fonksiyonu optimizasyonu

## Kaynaklar

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Stanford CS231n](http://cs231n.stanford.edu/)
`;

export default function NeuralNetworkFundamentalsPage() {
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
        <Tabs defaultValue="perceptron">
          <TabsList>
            <TabsTrigger value="perceptron">Perceptron</TabsTrigger>
            <TabsTrigger value="activation">Aktivasyon</TabsTrigger>
            <TabsTrigger value="backprop">Geri Yayılım</TabsTrigger>
          </TabsList>
          
          <TabsContent value="perceptron">
            <Card>
              <CardHeader>
                <CardTitle>Perceptron Örneği</CardTitle>
                <CardDescription>
                  Basit bir perceptron implementasyonu ve eğitimi
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`import numpy as np

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
    
    def predict(self, inputs):
        return 1 if np.dot(inputs, self.weights) + self.bias > 0 else 0

# AND kapısı için perceptron
p = Perceptron(2)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

# Test
for inputs in X:
    prediction = p.predict(inputs)
    print(f"Giriş: {inputs}, Çıkış: {prediction}")`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="activation">
            <Card>
              <CardHeader>
                <CardTitle>Aktivasyon Fonksiyonları</CardTitle>
                <CardDescription>
                  Farklı aktivasyon fonksiyonlarının karşılaştırması
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Test
x = np.linspace(-5, 5, 100)
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)

print("Sigmoid çıktısı:", y_sigmoid[:5])
print("ReLU çıktısı:", y_relu[:5])
print("Tanh çıktısı:", y_tanh[:5])`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="backprop">
            <Card>
              <CardHeader>
                <CardTitle>Geri Yayılım Örneği</CardTitle>
                <CardDescription>
                  Basit bir sinir ağında geri yayılım
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
    
    def forward(self, x):
        self.z1 = np.dot(x, self.w1)
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2)
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, x, y, learning_rate):
        m = x.shape[0]
        delta2 = self.a2 - y
        dw2 = np.dot(self.a1.T, delta2) / m
        delta1 = np.dot(delta2, self.w2.T) * sigmoid_derivative(self.a1)
        dw1 = np.dot(x.T, delta1) / m
        
        self.w2 -= learning_rate * dw2
        self.w1 -= learning_rate * dw1

# Test
nn = SimpleNeuralNetwork(2, 3, 1)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Eğitim
for _ in range(1000):
    output = nn.forward(X)
    nn.backward(X, y, 0.1)

print("Final output:", output)`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Navigation */}
      <div className="mt-12 flex flex-col sm:flex-row justify-between gap-4">
        <Button variant="outline" disabled className="gap-2">
          <ArrowLeft className="h-4 w-4" />
          Önceki Konu
        </Button>
        
        <Button asChild variant="default" className="gap-2">
          <Link href="/topics/python/derin-ogrenme/derin-ogrenme-frameworkleri">
            Sonraki Konu: Derin Öğrenme Frameworkleri
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