import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, Download, Github, Copy } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export const metadata: Metadata = {
  title: 'Transformers ile Metin Üretimi | Kod Örnekleri | Kodleon',
  description: 'Hugging Face Transformers kütüphanesi kullanarak GPT-2 ile metin üretme örneği.',
  openGraph: {
    title: 'Transformers ile Metin Üretimi | Kodleon',
    description: 'Hugging Face Transformers kütüphanesi kullanarak GPT-2 ile metin üretme örneği.',
    images: [{ url: '/images/code-examples/text-generation.jpg' }], // Bu resmin eklenmesi gerekiyor
  },
};

export default function TextGenerationPage() {
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
            <h1 className="text-3xl font-bold mb-4">Transformers ile Metin Üretimi</h1>
            
            <div className="flex items-center gap-2 mb-4">
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
                Doğal Dil İşleme
              </span>
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
                Orta
              </span>
            </div>
            
            <p className="text-muted-foreground mb-6">
              Bu örnekte, Hugging Face'in popüler `transformers` kütüphanesini kullanarak önceden eğitilmiş bir GPT-2 modeli ile nasıl metin üreteceğinizi öğreneceksiniz.
            </p>
            
            <div className="space-y-4 mb-6">
              <div>
                <h3 className="font-medium">Gereksinimler:</h3>
                <ul className="list-disc list-inside text-sm text-muted-foreground">
                  <li>Python 3.7+</li>
                  <li>PyTorch</li>
                  <li>Transformers (Hugging Face)</li>
                </ul>
              </div>
              
              <div>
                <h3 className="font-medium">Öğrenilecek Kavramlar:</h3>
                <ul className="list-disc list-inside text-sm text-muted-foreground">
                  <li>Transformer mimarisi temelleri</li>
                  <li>Önceden eğitilmiş dil modelleri (GPT-2)</li>
                  <li>Tokenization (Metni sayısallaştırma)</li>
                  <li>Metin üretimi (Text Generation)</li>
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
                    <code>{`from transformers import pipeline

# Metin üretimi için bir pipeline oluşturun
# Model olarak 'gpt2' veya Türkçe için 'dbmdz/gpt2-small-turkish-cased' kullanabilirsiniz
generator = pipeline('text-generation', model='gpt2')

# Başlangıç metnini verin
prompt = "Yapay zeka, günümüz teknolojisinin en heyecan verici alanlarından biridir ve"

# Metni üretin
generated_text = generator(prompt, max_length=100, num_return_sequences=1)

# Üretilen metni yazdırın
print(generated_text[0]['generated_text'])
`}</code>
                  </pre>
                </div>
              </TabsContent>
              
              <TabsContent value="explanation" className="p-6 m-0 space-y-4">
                <h3 className="text-xl font-bold">Kod Açıklaması</h3>
                
                <div>
                  <h4 className="font-semibold">1. Kütüphaneyi İçe Aktarma</h4>
                  <p className="text-sm text-muted-foreground">
                    Hugging Face `transformers` kütüphanesinden `pipeline` fonksiyonu içe aktarılır. Bu fonksiyon, karmaşık görevleri birkaç satır kodla yapmayı kolaylaştırır.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">2. Pipeline Oluşturma</h4>
                  <p className="text-sm text-muted-foreground">
                    `pipeline('text-generation', model='gpt2')` komutu ile metin üretimi için bir pipeline oluşturulur. Burada, OpenAI'nin popüler GPT-2 modeli kullanılır. Farklı veya daha büyük modeller de (örn. 'gpt2-medium') seçilebilir.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">3. Başlangıç Metni (Prompt)</h4>
                  <p className="text-sm text-muted-foreground">
                    Modele, metin üretimine başlaması için bir başlangıç cümlesi veya "prompt" verilir. Model bu metni devam ettirmeye çalışacaktır.
                  </p>
                </div>

                <div>
                  <h4 className="font-semibold">4. Metin Üretimi</h4>
                  <p className="text-sm text-muted-foreground">
                    `generator()` fonksiyonu çağrılarak metin üretilir. `max_length` parametresi ile üretilecek metnin maksimum uzunluğu (token sayısı) belirlenir. `num_return_sequences` ile kaç farklı sonuç üretileceği ayarlanır.
                  </p>
                </div>

                <div>
                  <h4 className="font-semibold">5. Sonucu Yazdırma</h4>
                  <p className="text-sm text-muted-foreground">
                    Üretilen metin, bir sözlük listesi olarak döner. İlk sonucun `'generated_text'` anahtarındaki değer ekrana yazdırılır. Bu, hem başlangıç metnini hem de modelin ürettiği devamını içerir.
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