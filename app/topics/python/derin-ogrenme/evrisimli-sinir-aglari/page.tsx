import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export const metadata: Metadata = {
  title: 'Evrişimli Sinir Ağları (CNN) | Kodleon',
  description: 'Görüntü işleme ve bilgisayarlı görü için CNN modellerini öğrenin.',
};

const content = `
# Evrişimli Sinir Ağları (CNN)

Evrişimli Sinir Ağları (CNN), özellikle görüntü işleme ve bilgisayarlı görü alanında kullanılan özel bir yapay sinir ağı türüdür. Bu bölümde, CNN'lerin temel yapısını ve uygulamalarını öğreneceğiz.

## 1. CNN Mimarisi

CNN'ler temel olarak şu katmanlardan oluşur:
- Evrişim (Convolution) Katmanları
- Havuzlama (Pooling) Katmanları
- Tam Bağlantılı (Fully Connected) Katmanları

### Evrişim Katmanı

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Basit bir evrişim katmanı
conv_layer = nn.Conv2d(
    in_channels=1,    # Giriş kanalı sayısı (gri tonlamalı görüntü için 1)
    out_channels=16,  # Çıkış özellik haritası sayısı
    kernel_size=3,    # Filtre boyutu (3x3)
    stride=1,         # Kaydırma miktarı
    padding=1         # Kenar dolgulama
)

# Örnek görüntü (1 kanal, 28x28 piksel)
input_image = torch.randn(1, 1, 28, 28)

# Evrişim işlemi
output_feature_map = conv_layer(input_image)
\`\`\`

### Havuzlama Katmanı

\`\`\`python
# Maksimum havuzlama katmanı
max_pool = nn.MaxPool2d(
    kernel_size=2,  # 2x2 havuzlama penceresi
    stride=2        # 2 birim kaydırma
)

# Ortalama havuzlama katmanı
avg_pool = nn.AvgPool2d(
    kernel_size=2,
    stride=2
)

# Havuzlama işlemi
pooled_features = max_pool(output_feature_map)
\`\`\`

## 2. Temel CNN Modeli

\`\`\`python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Evrişim katmanları
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Havuzlama katmanı
        self.pool = nn.MaxPool2d(2)
        
        # Tam bağlantılı katmanlar
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # İlk evrişim bloğu
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        # İkinci evrişim bloğu
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Vektörleştirme
        x = x.view(-1, 64 * 7 * 7)
        
        # Tam bağlantılı katmanlar
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

# Model oluşturma
model = SimpleCNN()
\`\`\`

## 3. Transfer Öğrenme

Transfer öğrenme, önceden eğitilmiş modelleri kullanarak yeni görevler için model geliştirmeyi sağlar.

\`\`\`python
import torchvision.models as models

# ResNet18 modelini yükle
model = models.resnet18(pretrained=True)

# Son katmanı değiştir
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Özellik çıkarıcı katmanları dondur
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True
\`\`\`

## 4. Veri Artırma (Data Augmentation)

\`\`\`python
from torchvision import transforms

# Veri artırma dönüşümleri
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Test dönüşümleri
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
\`\`\`

## 5. Model Eğitimi

\`\`\`python
def train_cnn(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Gradyanları sıfırla
        optimizer.zero_grad()
        
        # İleri yayılım
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Geri yayılım
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

def evaluate_cnn(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total
\`\`\`

## 6. Görselleştirme ve Analiz

### Özellik Haritaları

\`\`\`python
def visualize_feature_maps(model, image, layer_name):
    # Belirli bir katmanın çıktısını al
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Hook'u kaydet
    model._modules[layer_name].register_forward_hook(get_activation(layer_name))
    
    # İleri yayılım
    output = model(image)
    
    # Özellik haritalarını döndür
    return activation[layer_name]

# Görselleştirme örneği
feature_maps = visualize_feature_maps(model, sample_image, 'conv1')
\`\`\`

### Sınıf Aktivasyon Haritaları (CAM)

\`\`\`python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_cam(model, image, target_layer):
    cam = GradCAM(model=model, target_layer=target_layer)
    grayscale_cam = cam(input_tensor=image)
    visualization = show_cam_on_image(image.squeeze().numpy(), grayscale_cam[0])
    return visualization
\`\`\`

## Alıştırmalar

1. **Temel CNN Uygulamaları**
   - MNIST el yazısı rakam tanıma
   - CIFAR-10 nesne sınıflandırma
   - Özel veri seti oluşturma

2. **Transfer Öğrenme**
   - ResNet ile transfer öğrenme
   - Feature extraction vs. fine-tuning
   - Özel veri setine adaptasyon

3. **İleri Seviye Teknikler**
   - Veri artırma stratejileri
   - Hiperparametre optimizasyonu
   - Model analizi ve görselleştirme

## Kaynaklar

- [CNN Mimarileri](https://arxiv.org/abs/1610.02915)
- [Transfer Öğrenme Rehberi](https://cs231n.github.io/transfer-learning/)
- [PyTorch CNN Örnekleri](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
`;

export default function CNNPage() {
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
        <Tabs defaultValue="basic">
          <TabsList>
            <TabsTrigger value="basic">Temel CNN</TabsTrigger>
            <TabsTrigger value="transfer">Transfer Öğrenme</TabsTrigger>
            <TabsTrigger value="visualization">Görselleştirme</TabsTrigger>
          </TabsList>
          
          <TabsContent value="basic">
            <Card>
              <CardHeader>
                <CardTitle>Temel CNN Örneği</CardTitle>
                <CardDescription>
                  MNIST veri seti üzerinde basit bir CNN modeli
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# MNIST veri setini yükle
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model tanımı
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Model eğitimi
model = MNISTNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="transfer">
            <Card>
              <CardHeader>
                <CardTitle>Transfer Öğrenme Örneği</CardTitle>
                <CardDescription>
                  ResNet18 ile transfer öğrenme uygulaması
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`import torchvision.models as models
import torch.nn as nn

# ResNet18 modelini yükle
model = models.resnet18(pretrained=True)

# Son katmanı değiştir
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Özellik çıkarıcı katmanları dondur
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True

# Veri artırma
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Model eğitimi
optimizer = torch.optim.Adam(model.fc.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="visualization">
            <Card>
              <CardHeader>
                <CardTitle>Görselleştirme Örneği</CardTitle>
                <CardDescription>
                  Özellik haritaları ve CAM görselleştirme
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM

def visualize_layer(model, layer_name, image):
    # Hook tanımla
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Hook'u kaydet
    model._modules[layer_name].register_forward_hook(get_activation(layer_name))
    
    # İleri yayılım
    output = model(image)
    
    # Özellik haritalarını görselleştir
    act = activation[layer_name].squeeze()
    fig, axs = plt.subplots(4, 8, figsize=(20, 10))
    for idx in range(min(32, act.shape[0])):
        row = idx // 8
        col = idx % 8
        axs[row, col].imshow(act[idx])
        axs[row, col].axis('off')
    plt.show()

# Grad-CAM görselleştirme
def visualize_cam(model, image, target_class):
    cam = GradCAM(model=model, target_layer=model.layer4[-1])
    grayscale_cam = cam(input_tensor=image, target_category=target_class)
    visualization = show_cam_on_image(image.squeeze().numpy(), grayscale_cam[0])
    plt.imshow(visualization)
    plt.axis('off')
    plt.show()`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Navigation */}
      <div className="mt-12 flex flex-col sm:flex-row justify-between gap-4">
        <Button asChild variant="outline" className="gap-2">
          <Link href="/topics/python/derin-ogrenme/derin-ogrenme-frameworkleri">
            <ArrowLeft className="h-4 w-4" />
            Önceki Konu: Derin Öğrenme Frameworkleri
          </Link>
        </Button>
        
        <Button asChild variant="default" className="gap-2">
          <Link href="/topics/python/derin-ogrenme/tekrarlayan-sinir-aglari">
            Sonraki Konu: Tekrarlayan Sinir Ağları
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