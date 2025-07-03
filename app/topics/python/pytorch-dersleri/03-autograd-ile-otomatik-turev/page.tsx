'use client';

import { CodeBlock, dracula } from "react-code-blocks";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ArrowRight } from "lucide-react";
import Link from "next/link";

const Page = () => {
  const code = `
import torch

# ----- 1. Gradyan Takibini Başlatma -----

# Bir tensör oluştururken requires_grad=True parametresini verirsek,
# PyTorch bu tensör üzerindeki tüm işlemleri takip etmeye başlar.
x = torch.tensor([3.0], requires_grad=True)
print("Başlangıç Tensörü (x):", x)


# ----- 2. Bir Hesaplama Grafiği Oluşturma -----

# Basit bir işlem yapalım. Bu işlem bir "hesaplama grafiği" oluşturur.
# Grafik: x -> y -> z
y = x ** 2
z = y * 5  # z = 5 * x^2

# z tensörü, x'e bağlı olduğu için bir gradyan fonksiyonuna (.grad_fn) sahiptir.
print("\\ny tensörü:", y)
print("z tensörü:", z)
print("z'nin gradyan fonksiyonu:", z.grad_fn)


# ----- 3. Gradyanları Hesaplama (Geri Yayılım) -----

# z'nin x'e göre türevini (dz/dx) hesaplayalım.
# Bu matematiksel olarak 10*x'e eşittir. x=3.0 olduğu için sonuç 30 olmalı.
z.backward()


# ----- 4. Gradyan Sonucunu Görüntüleme -----

# Hesaplanan gradyan, orijinal tensörün .grad özelliğinde saklanır.
print("\\nHesaplanan Gradyan (dz/dx) [x=3 için]:", x.grad)


# ----- Gradyan Takibini Durdurma -----

print("\\n--- Gradyan Takibini Durdurma ---")
with torch.no_grad():
    a = torch.tensor([2.0], requires_grad=True)
    b = a * 2
    # Bu blok içindeki işlemler takip edilmez.
    print("b'nin gradyan takibi var mı?", b.requires_grad) # Sonuç: False

# Bir tensörün takibini kalıcı olarak da durdurabilirsiniz.
c = x.detach()
print("c'nin gradyan takibi var mı?", c.requires_grad) # Sonuç: False
`;

  return (
    <div className="container mx-auto p-4 md:p-8 lg:p-12">
      <h1 className="text-3xl md:text-4xl font-bold mb-4 text-center">
        Autograd ile Otomatik Türev
      </h1>
      <p className="text-muted-foreground text-center mb-8">
        PyTorch'un sihirli değneği: Autograd. Sinir ağlarının öğrenme mekanizmasının temelini oluşturan otomatik türev almayı öğrenin.
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
              <h4 className="font-semibold text-lg">1. `requires_grad=True`: Sihrin Başlangıcı</h4>
              <p className="text-sm text-muted-foreground mt-1">
                Bir sinir ağını eğittiğimizde, temelde modelin tahmin hatasını azaltmak için ağırlıklarını (parametrelerini) ayarlamamız gerekir. Bu ayarlamanın ne yönde ve ne kadar olacağını bilmek için "gradyanlara" (türevlere) ihtiyaç duyarız.
                <br />
                Bir tensör oluştururken <code>requires_grad=True</code> bayrağını ayarlamak, PyTorch'a "Bu tensörle ilgili tüm işlemleri kaydet, çünkü daha sonra bununla ilgili gradyanları hesaplamanı isteyeceğim." demek gibidir. Bu, genellikle bir modelin öğrenilebilir parametreleri için kullanılır.
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold text-lg">2. Hesaplama Grafiği (Computational Graph)</h4>
              <p className="text-sm text-muted-foreground mt-1">
                <code>requires_grad=True</code> olan bir tensörle işlem yaptığınızda, PyTorch arka planda bir "hesaplama grafiği" oluşturur. Bu, işlemlerin birbiriyle nasıl bağlantılı olduğunu gösteren bir yapıdır. Örneğimizdeki grafiğimiz çok basittir: <code>x</code> bir kare alma işlemine girer ve <code>y</code>'yi oluşturur, <code>y</code> de 5 ile çarpma işlemine girerek <code>z</code>'yi oluşturur. Bu grafik, gradyanların nasıl hesaplanacağını belirlemek için kullanılır.
              </p>
            </div>

            <div>
              <h4 className="font-semibold text-lg">3. `.backward()`: Geri Yayılımı Başlatmak</h4>
              <p className="text-sm text-muted-foreground mt-1">
                Bu, tüm sihrin gerçekleştiği yerdir. Bir skaler (tek elemanlı) tensör üzerinde <code>.backward()</code> metodunu çağırdığınızda, PyTorch hesaplama grafiğinde geriye doğru gider ve zincir kuralını (chain rule) kullanarak grafikteki <code>requires_grad=True</code> olarak ayarlanmış tüm "yaprak" tensörlerin (bizim örneğimizde <code>x</code>) gradyanlarını hesaplar.
                <br />
                Matematiksel olarak, <code>z = 5 * x^2</code> fonksiyonunun <code>x</code>'e göre türevi <code>dz/dx = 10 * x</code>'dir. Kodda <code>x=3.0</code> olduğu için, beklediğimiz gradyan <code>10 * 3.0 = 30.0</code>'dır.
              </p>
            </div>

            <div>
              <h4 className="font-semibold text-lg">4. `.grad`: Sonuçları Okumak</h4>
              <p className="text-sm text-muted-foreground mt-1">
                <code>.backward()</code> çağrısı tamamlandıktan sonra, hesaplanan gradyanlar orijinal tensörün <code>.grad</code> özelliğinde birikir (toplanır). Kodda <code>x.grad</code>'ı yazdırdığımızda, PyTorch'un hesapladığı <code>30.0</code> değerini görürüz. Bu, bir sinir ağında "hatayı azaltmak için bu ağırlığı ne kadar değiştirmeliyim?" sorusunun cevabını verir.
              </p>
            </div>

             <div>
              <h4 className="font-semibold text-lg">Gradyan Takibini Durdurma</h4>
              <p className="text-sm text-muted-foreground mt-1">
                Bazen, özellikle bir modeli eğittikten sonra sadece tahmin yapmak (inference) istediğimizde, gradyan hesaplamalarına ihtiyacımız olmaz. Bu durumlarda, gereksiz hesaplamaları önlemek için gradyan takibini <code>torch.no_grad()</code> bloğu ile geçici olarak veya <code>.detach()</code> metodu ile kalıcı olarak durdurabiliriz. Bu, performansı artırır ve bellek kullanımını azaltır.
              </p>
            </div>

          </TabsContent>
        </Tabs>
      </div>

      <div className="mt-8 flex justify-between">
        <Link href="/topics/python/pytorch-dersleri/02-tensor-operasyonlari">
          <Button variant="outline">
            <ArrowLeft className="h-4 w-4 mr-2" /> Önceki Ders: Tensör Operasyonları
          </Button>
        </Link>
        <Link href="/topics/python/pytorch-dersleri/04-basit-dogrusal-regresyon">
          <Button>
            Sonraki Ders: Basit Doğrusal Regresyon <ArrowRight className="h-4 w-4 ml-2" />
          </Button>
        </Link>
      </div>
    </div>
  );
};

export default Page; 