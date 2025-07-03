'use client';

import { CodeBlock, dracula } from "react-code-blocks";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowLeft, ArrowRight } from "lucide-react";

const Page = () => {
  const code = `
import torch

# ----- Örnek bir tensör oluşturalım -----
# 4x3 boyutunda, 0'dan 11'e kadar sayılarla dolu bir tensör
x = torch.arange(12).view(4, 3)
print("Başlangıç Tensörü (x):\\n", x)


# ----- 1. İndeksleme ve Dilimleme (Indexing & Slicing) -----

# NumPy'dakine çok benzer şekilde çalışır.

# İlk satırı seçme
print("\\nİlk Satır:", x[0])

# İlk sütunu seçme
print("İlk Sütun:", x[:, 0])

# Belirli bir elemanı seçme (1. satır, 2. sütun)
print("Eleman (1, 2):", x[1, 2].item()) # .item() tek elemanlı tensörü Python sayısına çevirir

# Alt tensör oluşturma (ilk iki satır, 1. ve 2. sütunlar)
sub_tensor = x[:2, 1:]
print("Alt Tensör:\\n", sub_tensor)


# ----- 2. Yeniden Şekillendirme (Reshaping) -----

# 4x3'lük bir tensörü 2x6'lık bir tensöre çevirme
y = x.view(2, 6)
print("\\nview(2, 6) ile yeniden şekillendirilmiş (y):\\n", y)

# view() metodunun önemli bir özelliği, orijinal tensörle aynı veriyi paylaşmasıdır.
# x'i değiştirmek y'yi de değiştirir.
x[0, 0] = 99
print("x değiştirildi, y de değişti:\\n", y)

# Boyutlardan birini -1 olarak belirleyerek otomatik hesaplatma
# Örneğin, 12 elemanlı bir tensörü 3 sütunlu yap, satır sayısını kendi bulsun.
z = x.view(-1, 3) # Sonuç 4x3 olacaktır
print("\\nview(-1, 3) ile yeniden şekillendirilmiş (z):\\n", z)


# ----- 3. Matematiksel Operasyonlar -----

# Element-wise (eleman bazında) çarpma
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a * b # veya torch.mul(a, b)
print("\\nEleman bazında çarpma:", c) # Sonuç: [4, 10, 18]

# Matris Çarpımı (Dot Product)
matrix1 = torch.randn(2, 3)
matrix2 = torch.randn(3, 4)
matrix_mul = torch.matmul(matrix1, matrix2)
print("\\nMatris Çarpımı (2x3 * 3x4):\\n", matrix_mul)
print("Sonuç Boyutu:", matrix_mul.shape) # Sonuç 2x4 olacak


# ----- 4. Tensörleri Birleştirme (Concatenating) -----

# Sütun bazında birleştirme (dim=1)
t1 = torch.randn(2, 3)
t2 = torch.randn(2, 3)
cat_dim1 = torch.cat((t1, t2), dim=1)
print("\\nSütun bazında birleştirme (dim=1):\\n", cat_dim1)
print("Boyut:", cat_dim1.shape) # Sonuç 2x6 olacak

# Satır bazında birleştirme (dim=0)
cat_dim0 = torch.cat((t1, t2), dim=0)
print("\\nSatır bazında birleştirme (dim=0):\\n", cat_dim0)
print("Boyut:", cat_dim0.shape) # Sonuç 4x3 olacak
`;

  return (
    <div className="container mx-auto p-4 md:p-8 lg:p-12">
      <h1 className="text-3xl md:text-4xl font-bold mb-4 text-center">
        Detaylı Tensör Operasyonları
      </h1>
      <p className="text-muted-foreground text-center mb-8">
        Tensörler, derin öğrenmenin temel veri yapılarıdır. Bu derste, tensörleri nasıl etkili bir şekilde manipüle edeceğimizi öğreneceğiz.
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
              <h4 className="font-semibold text-lg">1. İndeksleme ve Dilimleme</h4>
              <p className="text-sm text-muted-foreground mt-1">
                Eğer daha önce NumPy kullandıysanız, bu bölüm size çok tanıdık gelecektir. Tensörler de aynı şekilde indekslenir ve dilimlenir. Köşeli parantezler <code>[]</code> kullanarak belirli satır, sütun veya elemanlara erişebilirsiniz. <code>:</code> notasyonu, bir boyutun tamamını veya belirli bir aralığını seçmenizi sağlar.
                <br />
                <code>.item()</code> metodu, tek bir eleman içeren bir tensörün içindeki Python sayısını (int, float vb.) almak için kullanılır. Bu, özellikle kayıp (loss) değeri gibi tek bir sonucu yazdırmak için kullanışlıdır.
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold text-lg">2. Yeniden Şekillendirme: `view()` Metodu</h4>
              <p className="text-sm text-muted-foreground mt-1">
                <code>view()</code>, bir tensörün eleman sayısını değiştirmeden boyutlarını yeniden düzenlemenizi sağlayan çok güçlü bir metottur. Örneğin, 12 elemanlı bir tensörü 4x3, 2x6 veya 6x2 gibi farklı şekillerde "görüntüleyebilirsiniz".
                <br />
                <b>Önemli Not:</b> <code>view()</code> metodu yeni bir kopya oluşturmaz. Yeni tensör, orijinal tensörle aynı bellek alanını paylaşır. Bu, yüksek performans sağlar ama dikkatli olmayı gerektirir; birini değiştirmek diğerini de etkiler. Eğer boyutlardan birinden emin değilseniz, o boyuta <code>-1</code> vererek PyTorch'un geri kalan boyutlara göre doğru sayıyı otomatik olarak hesaplamasını sağlayabilirsiniz.
              </p>
            </div>

            <div>
              <h4 className="font-semibold text-lg">3. Matematiksel Operasyonlar</h4>
              <p className="text-sm text-muted-foreground mt-1">
                PyTorch, zengin bir matematiksel operasyon seti sunar.
                <ul className="list-disc list-inside mt-2 pl-4">
                  <li><b>Eleman Bazında (Element-wise) İşlemler:</b> Standart <code>+</code>, <code>-</code>, <code>*</code>, <code>/</code> operatörleri, iki tensörün karşılıklı elemanları arasında işlem yapar.</li>
                  <li><b>Matris Çarpımı:</b> Lineer cebirin temel taşı olan matris çarpımı için <code>torch.matmul()</code> veya <code>@</code> operatörü kullanılır. Bu işlem, sinir ağlarındaki katmanlar arası geçişlerin temelini oluşturur.</li>
                </ul>
              </p>
            </div>

            <div>
              <h4 className="font-semibold text-lg">4. Tensörleri Birleştirme: `torch.cat()`</h4>
              <p className="text-sm text-muted-foreground mt-1">
                <code>torch.cat()</code> fonksiyonu, bir dizi tensörü belirli bir boyut (dimension) boyunca birleştirmenizi sağlar. Bu, farklı kaynaklardan gelen veri gruplarını veya bir sinir ağının farklı dallarından gelen çıktıları birleştirmek için sıkça kullanılır.
                <br />
                <code>dim=0</code>, tensörleri satır bazında (üst üste) birleştirir.
                <br />
                <code>dim=1</code>, tensörleri sütun bazında (yan yana) birleştirir.
              </p>
            </div>

          </TabsContent>
        </Tabs>
      </div>

      <div className="mt-8 flex justify-between">
        <Link href="/topics/python/pytorch-dersleri/01-pytorch-kurulumu-ve-tensorlere-giris">
          <Button variant="outline">
            <ArrowLeft className="h-4 w-4 mr-2" /> Önceki Ders: Kurulum ve Tensörler
          </Button>
        </Link>
        <Link href="/topics/python/pytorch-dersleri/03-autograd-ile-otomatik-turev">
          <Button>
            Sonraki Ders: Autograd <ArrowRight className="h-4 w-4 ml-2" />
          </Button>
        </Link>
      </div>
    </div>
  );
};

export default Page; 