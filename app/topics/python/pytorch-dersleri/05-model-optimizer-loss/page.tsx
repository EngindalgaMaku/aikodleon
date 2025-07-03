'use client';

import { CodeBlock, dracula } from "react-code-blocks";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ArrowRight } from "lucide-react";
import Link from "next/link";

const Page = () => {
  const code = `
import torch
import torch.nn as nn

# ----- 1. Veri Seti (Öncekiyle aynı) -----
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
Y = torch.tensor([[3.1], [4.9], [7.2], [8.8], [11.1]])

# ----- 2. Modeli Tanımlama (torch.nn.Module ile) -----
# Bu, standart PyTorch model oluşturma yöntemidir.
# Modelimiz, bir giriş ve bir çıkış özelliğine sahip basit bir doğrusal katmandan oluşur.
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)

# Modelimizi oluşturalım. PyTorch ağırlıkları (w) ve sapmayı (b) otomatik başlatır.
model = LinearRegression()

# ----- 3. Kayıp Fonksiyonu ve Optimizatör -----
# PyTorch'un hazır Kayıp (Loss) ve Optimizatörlerini kullanıyoruz.
learning_rate = 0.01
n_epochs = 100

loss_function = nn.MSELoss() # Ortalama Kare Hata (Mean Squared Error)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Stokastik Gradyan İnişi

# ----- 4. Geliştirilmiş Eğitim Döngüsü -----
for epoch in range(n_epochs):
    # a. İleri Besleme (Tahmin)
    y_pred = model(X)
    
    # b. Kayıp (Loss) Hesaplama
    loss = loss_function(y_pred, Y)
    
    # c. Geri Yayılım (Gradyan Hesaplama)
    loss.backward()
    
    # d. Parametre Güncelleme (Öğrenme)
    # optimizer.step() bizim için 'w -= lr * w.grad' işlemini otomatik yapar.
    optimizer.step()
    
    # e. Gradyanları Sıfırlama
    # Bir sonraki epoch için gradyanları temizliyoruz.
    optimizer.zero_grad()
    
    if (epoch + 1) % 10 == 0:
        # model.parameters()'dan w ve b'yi alıp yazdıralım
        [w, b] = model.parameters()
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}, w: {w[0][0].item():.3f}')

# ----- 5. Sonuçları Görüntüleme -----
print("\\nEğitimden sonra öğrenilen parametreler:")
# Eğitilmiş modelden tahmin yapma
predicted = model(torch.tensor([6.0])).item()
print(f'x = 6.0 için modelin tahmini: {predicted:.3f}')
`;

  return (
    <div className="container mx-auto p-4 md:p-8 lg:p-12">
      <h1 className="text-3xl md:text-4xl font-bold mb-4 text-center">
        Model, Optimizasyon ve Kayıp Fonksiyonu
      </h1>
      <p className="text-muted-foreground text-center mb-8">
        PyTorch'un gücünü keşfedin: Modelleri `nn.Module` ile yapılandırın, `torch.optim` ile öğrenmeyi otomatikleştirin.
      </p>

      <div className="w-full">
        <Tabs defaultValue="explanation" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="code">Kod Örneği</TabsTrigger>
            <TabsTrigger value="explanation">Açıklama</TabsTrigger>
          </TabsList>
          <TabsContent value="code" className="p-0 m-0">
            <CodeBlock
              text={code}
              language="python"
              showLineNumbers={true}
              theme={dracula}
            />
          </TabsContent>
          <TabsContent value="explanation" className="p-6 m-0 space-y-6">
            <h3 className="text-xl font-bold">Kodun Profesyonel Yapısı</h3>
            
            <p className="text-sm text-muted-foreground mt-1">
              Bu kod, bir önceki dersimizdeki manuel yaklaşımı, PyTorch'un standart ve daha güçlü bileşenleriyle yeniden yapılandırır. Bu yapı, karmaşık modeller oluştururken kodunuzu düzenli ve ölçeklenebilir tutmanızı sağlar.
            </p>

            <div>
              <h4 className="font-semibold text-lg">1. `nn.Module`: Modeller için İskelet</h4>
              <p className="text-sm text-muted-foreground mt-1">
                <code>torch.nn.Module</code>, tüm sinir ağı modelleri için temel sınıftır. Kendi modelimizi bu sınıftan türetiriz.
                <ul className="list-disc list-inside mt-2 pl-4">
                  <li><b>`__init__()`</b>: Modelin katmanlarını (layers) tanımladığımız yerdir. Burada `nn.Linear(1, 1)` ile tek girişli, tek çıkışlı bir doğrusal katman oluşturuyoruz. Bu katman, bizim önceki derste manuel olarak oluşturduğumuz `w` ve `b`'yi kendi içinde barındırır.</li>
                  <li><b>`forward()`</b>: Verinin model içinde nasıl akacağını (ileri besleme) tanımladığımız yerdir. Girdiyi alır ve katmanlardan geçirip çıktıyı döndürür.</li>
                </ul>
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold text-lg">2. `nn.MSELoss`: Hazır Kayıp Fonksiyonları</h4>
              <p className="text-sm text-muted-foreground mt-1">
                Bir önceki derste ortalama kare hatayı `torch.mean((y_pred - Y) ** 2)` şeklinde manuel olarak hesaplamıştık. PyTorch, `nn.MSELoss` gibi birçok standart kayıp fonksiyonunu hazır olarak sunar. Bu hem kodumuzu temizler hem de sayısal kararlılığı artırır.
              </p>
            </div>

            <div>
              <h4 className="font-semibold text-lg">3. `torch.optim`: Otomatik Parametre Güncelleme</h4>
              <p className="text-sm text-muted-foreground mt-1">
                Bu, en büyük kolaylıklardan biridir. Artık `with torch.no_grad(): w -= ...` gibi manuel güncelleme işlemleri yapmamıza gerek yok. 
                <br />
                Bir <b>optimizatör</b> (örneğin `torch.optim.SGD`), modelimizin tüm parametrelerini (`model.parameters()`) ve bir öğrenme oranını alır. Eğitim döngüsünde sadece iki basit komut kullanırız:
                <ul className="list-disc list-inside mt-2 pl-4">
                  <li><b>`optimizer.step()`</b>: Arka planda tüm parametreler için `param -= lr * param.grad` işlemini otomatik olarak yapar.</li>
                  <li><b>`optimizer.zero_grad()`</b>: Modelin tüm parametrelerinin gradyanlarını tek seferde sıfırlar.</li>
                </ul>
              </p>
            </div>
            
            <p className="text-sm font-medium text-primary mt-4">
              Bu <b>Model -> Kayıp -> Optimizatör</b> üçlüsü, PyTorch ile model eğitmenin standart yoludur. Bu yapıyı benimsemek, basit bir doğrusal regresyondan devasa bir dil modeline kadar her türlü projede size yol gösterecektir.
            </p>

          </TabsContent>
        </Tabs>
      </div>

      <div className="mt-8 flex justify-between">
        <Link href="/topics/python/pytorch-dersleri/04-basit-dogrusal-regresyon">
          <Button variant="outline">
            <ArrowLeft className="h-4 w-4 mr-2" /> Önceki Ders: Doğrusal Regresyon
          </Button>
        </Link>
        {/* <Link href="/topics/python/pytorch-dersleri/06-dataset-dataloader">
          <Button>
            Sonraki Ders: Dataset & DataLoader <ArrowRight className="h-4 w-4 ml-2" />
          </Button>
        </Link> */}
      </div>
    </div>
  );
};

export default Page; 