'use client';

import { CodeBlock, dracula } from "react-code-blocks";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ArrowRight } from "lucide-react";
import Link from "next/link";

const Page = () => {
  const code = `
import torch

# ----- 1. Veri Setini Oluşturma -----
# Gerçek dünyadaki gibi, bir miktar gürültü içeren bir veri seti hazırlayalım.
# Hedefimiz y = 2 * x + 1 denklemini öğrenmek olacak.
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
Y = torch.tensor([[3.1], [4.9], [7.2], [8.8], [11.1]]) # Yaklaşık y = 2*x + 1

# ----- 2. Model Parametrelerini Başlatma -----
# Öğrenmek istediğimiz iki değer var: ağırlık (w) ve sapma (b).
# Bunları rastgele değerlerle başlatıyoruz ve gradyan takibini açıyoruz.
w = torch.tensor([0.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

# ----- 3. Eğitim Ayarları -----
learning_rate = 0.01
n_epochs = 100

# ----- 4. Eğitim Döngüsü -----
for epoch in range(n_epochs):
    # a. İleri Besleme (Tahmin)
    # Modelin mevcut w ve b değerleriyle yaptığı tahmin.
    y_pred = w * X + b
    
    # b. Kayıp (Loss) Hesaplama
    # Tahminler ile gerçek değerler arasındaki hatayı ölçüyoruz.
    # Ortalama Kare Hata (Mean Squared Error) kullanıyoruz.
    loss = torch.mean((y_pred - Y) ** 2)
    
    # c. Geri Yayılım (Gradyan Hesaplama)
    # Autograd kullanarak kaybın w ve b'ye göre gradyanlarını hesaplıyoruz.
    loss.backward()
    
    # d. Parametre Güncelleme (Öğrenme)
    # Gradyanları kullanarak w ve b'yi hatayı azaltacak yönde güncelliyoruz.
    # torch.no_grad() ile bu güncelleme işleminin kendisinin takip edilmesini engelliyoruz.
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
    # e. Gradyanları Sıfırlama
    # Bir sonraki epoch için gradyanları temizliyoruz.
    w.grad.zero_()
    b.grad.zero_()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# ----- 5. Sonuçları Görüntüleme -----
print("\\nEğitimden sonra öğrenilen parametreler:")
print(f'Ağırlık (w): {w.item():.3f}') # Hedef: 2.0
print(f'Sapma (b): {b.item():.3f}')   # Hedef: 1.0
`;

  return (
    <div className="container mx-auto p-4 md:p-8 lg:p-12">
      <h1 className="text-3xl md:text-4xl font-bold mb-4 text-center">
        PyTorch ile Basit Doğrusal Regresyon
      </h1>
      <p className="text-muted-foreground text-center mb-8">
        Teoriyi pratiğe dökme zamanı! Şimdiye kadar öğrendiklerimizle ilk çalışan makine öğrenmesi modelimizi oluşturalım.
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
            <h3 className="text-xl font-bold">Modelin Adım Adım Açıklaması</h3>
            
            <p className="text-sm text-muted-foreground mt-1">
              Bu kod, bir makine öğrenmesi modelinin en temel eğitim döngüsünü gösterir. Amacımız, <code>y = w * x + b</code> formülündeki <code>w</code> (ağırlık) ve <code>b</code> (sapma) değerlerini, verdiğimiz veri setine en uygun olacak şekilde otomatik olarak bulmaktır.
            </p>

            <div>
              <h4 className="font-semibold text-lg">1. İleri Besleme (Forward Pass)</h4>
              <p className="text-sm text-muted-foreground mt-1">
                Modelin mevcut parametreleriyle (başlangıçta rastgele) bir tahminde bulunduğu adımdır. Kodumuzdaki <code>y_pred = w * X + b</code> satırı tam olarak bunu yapar.
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold text-lg">2. Kayıp Hesaplama (Loss Calculation)</h4>
              <p className="text-sm text-muted-foreground mt-1">
                Modelin tahminlerinin (<code>y_pred</code>) gerçek değerlerden (<code>Y</code>) ne kadar saptığını ölçtüğümüz adımdır. Bu "hata" değerine kayıp (loss) denir. Popüler bir yöntem olan Ortalama Kare Hata'yı (MSE) kullanıyoruz. Amacımız bu kayıp değerini mümkün olduğunca düşürmektir.
              </p>
            </div>

            <div>
              <h4 className="font-semibold text-lg">3. Geri Yayılım (Backward Pass)</h4>
              <p className="text-sm text-muted-foreground mt-1">
                İşte Autograd'ın sahneye çıktığı yer! Hesapladığımız kayıp (loss) üzerinden <code>.backward()</code> çağırarak, kaybın model parametrelerine (<code>w</code> ve <code>b</code>) göre gradyanlarını (türevlerini) otomatik olarak hesaplatırız. Bu gradyanlar, her bir parametrenin hataya olan etkisini gösterir.
              </p>
            </div>

            <div>
              <h4 className="font-semibold text-lg">4. Parametre Güncelleme (Parameter Update)</h4>
              <p className="text-sm text-muted-foreground mt-1">
                "Öğrenmenin" gerçekleştiği adımdır. Hesapladığımız gradyanları kullanarak parametreleri güncelleriz. Formül şöyledir: <code>parametre = parametre - öğrenme_oranı * gradyan</code>. Bu, parametreleri kaybı azaltacak yönde küçük adımlarla hareket ettirir. Bu güncelleme işleminin kendisinin bir sonraki gradyan hesaplamasını etkilememesi için <code>torch.no_grad()</code> bloğu içinde yaparız.
              </p>
            </div>

             <div>
              <h4 className="font-semibold text-lg">5. Gradyanları Sıfırlama</h4>
              <p className="text-sm text-muted-foreground mt-1">
                PyTorch, <code>.backward()</code> çağrıldığında gradyanları mevcut değerlerin üzerine ekler. Bu bazı gelişmiş modeller için faydalı olsa da, bizim durumumuzda her eğitim adımında (epoch) gradyanları sıfırdan hesaplamak isteriz. Bu yüzden her döngünün sonunda <code>.grad.zero_()</code> ile gradyanları temizleriz.
              </p>
            </div>

            <p className="text-sm font-medium text-primary mt-4">
              Bu 5 adımlık döngü, en karmaşık sinir ağlarının bile temel eğitim mantığıdır. Bu yapıyı anlamak, derin öğrenmeyi anlamanın anahtarıdır.
            </p>

          </TabsContent>
        </Tabs>
      </div>

      <div className="mt-8 flex justify-between">
        <Link href="/topics/python/pytorch-dersleri/03-autograd-ile-otomatik-turev">
          <Button variant="outline">
            <ArrowLeft className="h-4 w-4 mr-2" /> Önceki Ders: Autograd
          </Button>
        </Link>
        <Link href="/topics/python/pytorch-dersleri/05-model-optimizer-loss">
          <Button>
            Sonraki Ders: Model, Optimizer, Loss <ArrowRight className="h-4 w-4 ml-2" />
          </Button>
        </Link>
      </div>
    </div>
  );
};

export default Page; 