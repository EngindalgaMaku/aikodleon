import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, Download, Github, Copy } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export const metadata: Metadata = {
  title: 'GANs ile Görüntü Sentezi | Kod Örnekleri | Kodleon',
  description: 'PyTorch ve Üretici Çekişmeli Ağlar (GANs) kullanarak sentetik görüntüler oluşturma örneği.',
  openGraph: {
    title: 'GANs ile Görüntü Sentezi | Kodleon',
    description: 'PyTorch ve Üretici Çekişmeli Ağlar (GANs) kullanarak sentetik görüntüler oluşturma örneği.',
    images: [{ url: '/images/code-examples/gans.jpg' }], // Bu resmin eklenmesi gerekiyor
  },
};

export default function GANsPage() {
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
            <h1 className="text-3xl font-bold mb-4">GANs ile Görüntü Sentezi</h1>
            
            <div className="flex items-center gap-2 mb-4">
               <span className="px-3 py-1 rounded-full text-xs font-medium bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-200">
                Bilgisayarlı Görü
              </span>
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200">
                İleri
              </span>
            </div>
            
            <p className="text-muted-foreground mb-6">
              Bu örnekte, PyTorch kullanarak basit bir Üretici Çekişmeli Ağ (GAN) modeli ile MNIST veri setindeki el yazısı rakamlarına benzer yeni, sentetik görüntüler üreteceksiniz.
            </p>
            
            <div className="space-y-4 mb-6">
              <div>
                <h3 className="font-medium">Gereksinimler:</h3>
                <ul className="list-disc list-inside text-sm text-muted-foreground">
                  <li>Python 3.7+</li>
                  <li>PyTorch</li>
                  <li>TorchVision</li>
                  <li>Matplotlib</li>
                </ul>
              </div>
              
              <div>
                <h3 className="font-medium">Öğrenilecek Kavramlar:</h3>
                <ul className="list-disc list-inside text-sm text-muted-foreground">
                  <li>Üretici Çekişmeli Ağlar (GANs)</li>
                  <li>Üretici (Generator) ve Ayırt Edici (Discriminator)</li>
                  <li>Kayıp Fonksiyonları (Loss Functions)</li>
                  <li>Sinir Ağı Eğitimi</li>
                </ul>
              </div>
            </div>
            
             <div className="flex flex-col gap-2">
               <Button asChild variant="default" className="gap-2" disabled>
                <a href="#">
                  <Download className="h-4 w-4" />
                  Jupyter Notebook (Yakında)
                </a>
              </Button>
              <Button asChild variant="outline" className="gap-2" disabled>
                <a href="#" target="_blank" rel="noopener noreferrer">
                  <Github className="h-4 w-4" />
                  GitHub'da Görüntüle (Yakında)
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
                </TabsList>
              </div>
              
              <TabsContent value="code" className="p-0 m-0">
                <div className="relative">
                  <Button variant="ghost" size="sm" className="absolute right-2 top-2 gap-1">
                    <Copy className="h-4 w-4" />
                    Kopyala
                  </Button>
                  <pre className="p-6 pt-12 overflow-x-auto text-sm">
                    <code>{`import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hiperparametreler
lr = 0.0002
batch_size = 128
image_size = 64
channels_img = 1
noise_dim = 100
num_epochs = 5

# Ayırt Edici Model
class Discriminator(nn.Module):
    # ... model mimarisi ...
    pass

# Üretici Model
class Generator(nn.Module):
    # ... model mimarisi ...
    pass

# Veri setini yükleme
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]),
])
dataset = datasets.MNIST(root="dataset/", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Modelleri, optimizatörleri ve kayıp fonksiyonunu başlatma
# ...

# Eğitim döngüsü
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(dataloader):
        # Ayırt ediciyi eğitme
        # ...

        # Üreticiyi eğitme
        # ...
    print(f"Epoch [{epoch+1}/{num_epochs}] tamamlandı.")

# Üretilen görüntüleri gösterme
# ...
`}</code>
                  </pre>
                </div>
              </TabsContent>
              
              <TabsContent value="explanation" className="p-6 m-0 space-y-4">
                <h3 className="text-xl font-bold">Kod Açıklaması</h3>
                
                <div>
                  <h4 className="font-semibold">1. Modellerin Tanımlanması</h4>
                  <p className="text-sm text-muted-foreground">
                    `Generator` (Üretici) ve `Discriminator` (Ayırt Edici) adında iki ana sinir ağı modeli tanımlanır. Üretici, rastgele gürültüden (noise) sahte görüntüler üretmeye çalışırken, Ayırt Edici gerçek ve sahte görüntüleri birbirinden ayırmaya çalışır.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">2. Veri Yükleme</h4>
                  <p className="text-sm text-muted-foreground">
                    `torchvision` kütüphanesi kullanılarak MNIST veri seti (el yazısı rakamlar) indirilir ve eğitim için hazırlanır. Görüntüler yeniden boyutlandırılır ve normalize edilir.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">3. Eğitim Döngüsü</h4>
                  <p className="text-sm text-muted-foreground">
                    Eğitim, iki aşamalı bir süreçtir. Önce Ayırt Edici, bir grup gerçek ve Üretici'nin yarattığı bir grup sahte görüntü ile eğitilir. Ardından, Üretici, Ayırt Edici'yi kandıracak daha iyi sahte görüntüler üretmesi için eğitilir.
                  </p>
                </div>

                <div>
                  <h4 className="font-semibold">4. Kayıp Fonksiyonu</h4>
                  <p className="text-sm text-muted-foreground">
                    Genellikle `Binary Cross-Entropy Loss` kullanılır. Ayırt Edici, gerçek görüntüler için 1'e, sahte görüntüler için 0'a yakın bir sonuç vermeye çalışır. Üretici ise Ayırt Edici'nin kendi ürettiği sahte görüntüler için 1'e yakın bir sonuç vermesini sağlamaya çalışır.
                  </p>
                </div>

                <div>
                  <h4 className="font-semibold">5. Sonuç</h4>
                  <p className="text-sm text-muted-foreground">
                    Eğitim tamamlandığında, Üretici model, rastgele bir gürültü vektöründen MNIST veri setindekilere oldukça benzeyen yeni ve orijinal rakam görüntüleri üretebilir hale gelir.
                  </p>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </div>
  );
} 