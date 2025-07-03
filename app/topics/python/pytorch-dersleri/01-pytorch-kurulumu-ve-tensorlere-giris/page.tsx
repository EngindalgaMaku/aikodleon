'use client';

import { CodeBlock, dracula } from "react-code-blocks";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";
import Link from "next/link";

const Page = () => {
  const code = `
# PyTorch'u kurmak için terminal veya komut istemcisine aşağıdaki komutu yazın:
# pip install torch torchvision torchaudio

import torch

# ----- 1. Tensör Oluşturma -----

# Boş bir tensör (5x3 boyutunda)
x_empty = torch.empty(5, 3)
print("Boş Tensör:\\n", x_empty)

# Rastgele sayılarla dolu bir tensör
x_rand = torch.rand(5, 3)
print("\\nRastgele Tensör:\\n", x_rand)

# Sıfırlarla dolu bir tensör
x_zeros = torch.zeros(5, 3, dtype=torch.long)
print("\\nSıfırlarla Dolu Tensör:\\n", x_zeros)

# Doğrudan veriden tensör oluşturma
x_data = torch.tensor([5.5, 3])
print("\\nVeriden Tensör:\\n", x_data)


# ----- 2. Tensör Özellikleri -----

# Bir tensörün boyutunu (shape) öğrenme
print("\\nTensör Boyutu:", x_rand.size())

# Bir tensörün veri tipini öğrenme
print("Tensör Veri Tipi:", x_rand.dtype)


# ----- 3. Tensör Operasyonları -----

# İki tensörü toplama
y = torch.rand(5, 3)
result_add = x_rand + y
# Alternatif yazım: torch.add(x_rand, y)
print("\\nToplama Sonucu:\\n", result_add)

# Yerinde (in-place) toplama
# y tensörünün değerini x_rand ile toplayarak günceller
y.add_(x_rand)
print("\\nYerinde Toplama (y'nin yeni hali):\\n", y)


# ----- 4. NumPy ve Tensörler Arası Dönüşüm -----

# Tensörü NumPy dizisine çevirme
a = torch.ones(5)
b = a.numpy()
print("\\nNumPy Dizisi (b):", b)

# NumPy dizisini Tensöre çevirme
import numpy as np
c = np.ones(5)
d = torch.from_numpy(c)
print("NumPy'den Tensör (d):", d)

# Önemli Not: CPU üzerindeki tensör ve NumPy dizisi aynı bellek alanını paylaşır.
# Birini değiştirmek diğerini de değiştirir.
a.add_(1)
print("\\n'a' değiştirildi, 'b' de değişti:", b)
`;

  return (
    <div className="container mx-auto p-4 md:p-8 lg:p-12">
      <h1 className="text-3xl md:text-4xl font-bold mb-4 text-center">
        PyTorch Kurulumu ve Tensörlere Giriş
      </h1>
      <p className="text-muted-foreground text-center mb-8">
        Bu derste, modern yapay zeka uygulamalarının temel taşı olan PyTorch kütüphanesini kuracak ve en temel veri yapısı olan tensörleri tanıyacağız.
      </p>

      <div className="w-full">
        <Tabs defaultValue="code" className="w-full">
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
            <h3 className="text-xl font-bold">Kodun Adım Adım Açıklaması</h3>

            <div>
              <h4 className="font-semibold text-lg">PyTorch Nedir ve Neden Önemlidir?</h4>
              <p className="text-sm text-muted-foreground mt-1">
                PyTorch, Facebook'un AI Araştırma laboratuvarı (FAIR) tarafından geliştirilen açık kaynaklı bir makine öğrenmesi kütüphanesidir. Özellikle iki temel özelliği ile öne çıkar:
                <ul className="list-disc list-inside mt-2 pl-4">
                  <li><b>Tensör Hesaplamaları:</b> NumPy'a çok benzeyen ama güçlü GPU desteği sunan tensör işlemleri.</li>
                  <li><b>Otomatik Türev (Autograd):</b> Derin sinir ağları oluşturmak ve eğitmek için otomatik türev alabilen bir sistem.</li>
                </ul>
                Esnek yapısı ve Python'a olan yakınlığı sayesinde hem araştırmacılar hem de geliştiriciler tarafından sıkça tercih edilir.
              </p>
            </div>

            <div>
              <h4 className="font-semibold text-lg">1. PyTorch Kurulumu</h4>
              <p className="text-sm text-muted-foreground mt-1">
                Kodun en başındaki yorum satırında belirtildiği gibi, PyTorch'u bilgisayarınıza kurmak için <code>pip</code> paket yöneticisini kullanabilirsiniz. Genellikle <code>torchvision</code> (görüntü işleme) ve <code>torchaudio</code> (ses işleme) kütüphaneleriyle birlikte kurulur.
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold text-lg">2. Tensör: PyTorch'un Yapı Taşı</h4>
              <p className="text-sm text-muted-foreground mt-1">
                Tensör, çok boyutlu bir dizi veya matris olarak düşünülebilir. PyTorch'taki tüm işlemlerin temelini tensörler oluşturur. Kodda farklı şekillerde tensörler oluşturuyoruz:
                <ul className="list-disc list-inside mt-2 pl-4">
                  <li><code>torch.empty()</code>: Bellekte yer ayırır ama içini herhangi bir değerle doldurmaz. Hızlıdır ama başlangıç değerleri anlamsızdır.</li>
                  <li><code>torch.rand()</code>: 0 ile 1 arasında rastgele sayılarla doldurur.</li>
                  <li><code>torch.zeros()</code>: Tüm elemanları 0 olan bir tensör oluşturur.</li>
                  <li><code>torch.tensor()</code>: Mevcut bir Python listesi veya dizisinden doğrudan bir tensör oluşturur.</li>
                </ul>
              </p>
            </div>

            <div>
              <h4 className="font-semibold text-lg">3. Tensör Özellikleri ve Operasyonları</h4>
              <p className="text-sm text-muted-foreground mt-1">
                Her tensörün bir boyutu (<code>.size()</code>) ve bir veri tipi (<code>.dtype</code>) vardır. Tıpkı sayılarla işlem yapar gibi tensörlerle de toplama, çarpma gibi matematiksel işlemler yapabilirsiniz. Koddaki <code>y.add_(x_rand)</code> gibi sonunda alt çizgi (<code>_</code>) olan operasyonlar, işlemi "yerinde" (in-place) yapar. Yani yeni bir tensör oluşturmak yerine, mevcut tensörün değerini günceller.
              </p>
            </div>

            <div>
              <h4 className="font-semibold text-lg">4. NumPy Köprüsü</h4>
              <p className="text-sm text-muted-foreground mt-1">
                PyTorch, popüler bilimsel hesaplama kütüphanesi NumPy ile sorunsuz bir şekilde çalışabilir. Bir PyTorch tensörünü kolayca bir NumPy dizisine (<code>.numpy()</code>) ve tam tersini (<code>torch.from_numpy()</code>) yapabilirsiniz. Buradaki en önemli nokta, CPU üzerinde bu iki yapının aynı belleği paylaşmasıdır. Bu, birini değiştirdiğinizde diğerinin de anında değişeceği anlamına gelir, bu da performansı artırır ama dikkatli olmayı gerektirir.
              </p>
            </div>

          </TabsContent>
        </Tabs>
      </div>

      <div className="mt-8 flex justify-end">
        <Link href="/topics/python/pytorch-dersleri/02-tensor-operasyonlari">
          <Button>
            Sonraki Ders: Tensör Operasyonları <ArrowRight className="h-4 w-4 ml-2" />
          </Button>
        </Link>
      </div>
    </div>
  );
};

export default Page; 