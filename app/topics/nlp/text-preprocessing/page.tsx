import { Metadata } from 'next';
import { createPageMetadata } from '@/lib/seo';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';

export const metadata: Metadata = createPageMetadata({
  title: 'Metin Ön İşleme',
  description: "Kodleon'da metin ön işleme tekniklerini ve metin verilerini analiz için nasıl hazırlayacağınızı öğrenin.",
  path: '/topics/nlp/text-preprocessing',
  keywords: ['metin ön işleme', 'text preprocessing', 'nlp', 'doğal dil işleme', 'tokenizasyon', 'kök bulma', 'lemmatizasyon', 'metin temizleme', 'kodleon', 'türkçe ai eğitimi'],
});

export default function TextPreprocessingPage() {
  return (
    <div className="container max-w-6xl mx-auto py-12">
      <div className="mb-8">
        <Button asChild variant="ghost" size="sm" className="gap-1">
          <Link href="/topics/nlp" aria-label="Doğal Dil İşleme konusuna geri dön">
            <ArrowLeft className="h-4 w-4" aria-hidden="true" />
            Doğal Dil İşleme
          </Link>
        </Button>
      </div>
      <h1 className="text-4xl font-bold mb-6">Metin Ön İşleme</h1>
      <p className="text-xl text-muted-foreground mb-8">
        Metin ön işleme (Text Preprocessing), doğal dil işleme (NLP) görevleri için ham metin verilerini temizleme, dönüştürme ve standart hale getirme sürecidir. Gerçek dünya metin verileri genellikle gürültülü, tutarsız ve yapısal olmayan bir yapıya sahiptir. Etkili bir NLP modeli oluşturmak için bu verilerin uygun şekilde ön işlenmesi kritik öneme sahiptir.
      </p>

      <div className="prose prose-lg dark:prose-invert max-w-none">
        <h2>Temel Ön İşleme Teknikleri</h2>
        <p>Yaygın olarak kullanılan metin ön işleme teknikleri şunlardır:</p>
        <ol>
          <li>**Tokenizasyon (Tokenization):** Metni kelimeler, noktalama işaretleri gibi daha küçük birimlere (tokenlara) ayırma işlemidir.</li>
          <li>**Kök Bulma (Stemming):** Kelimelerin eklerini atarak kök formunu bulma tekniğidir (Örn: 'koşan', 'koşuyor' -&gt; 'koş'). Genellikle dilbilgisel olarak doğru kök olmayabilir.</li>
          <li>**Lemmatizasyon (Lemmatization):** Kelimeleri anlamlı ve sözcük dağarcığında bulunan temel formuna (lemma) indirme işlemidir (Örn: 'koşan', 'koşuyor' -&gt; 'koşmak'). Kök bulmaya göre daha karmaşıktır ve kelimenin türünü dikkate alır.</li>
          <li>**Durak Kelimeleri Kaldırma (Stop Word Removal):** Anlam taşımayan veya çok sık geçen kelimelerin (ve, ile, bir vb.) metinden çıkarılması işlemidir.</li>
          <li>**Küçük Harfe Dönüştürme (Lowercasing):** Tüm metni küçük harfe dönüştürerek aynı kelimenin farklı yazımlarının (Apple, apple) aynı kabul edilmesini sağlar.</li>
          <li>**Noktalama İşaretlerini ve Özel Karakterleri Kaldırma:** Metin analizini etkileyebilecek noktalama işaretleri, sayılar veya özel karakterlerin metinden temizlenmesi.</li>
          <li>**Boşlukları Standartlaştırma:** Birden fazla boşluğu tek boşluğa indirgeme veya baştaki/sondaki boşlukları kaldırma.</li>
        </ol>

        <h2>Neden Metin Ön İşleme Yapmalıyız?</h2>
        <p>Metin ön işlemenin temel amaçları:</p>
        <ul>
          <li>Veri Gürültüsünü Azaltma</li>
          <li>Veri Setini Küçültme (özellikle kelime haznesi boyutu)</li>
          <li>Model Performansını Artırma</li>
          <li>Metin Verisini Standartlaştırma</li>
        </ul>

        <p>Doğru ön işleme adımları, NLP modellerinin veriyi daha iyi anlamasına ve daha doğru sonuçlar üretmesine yardımcı olur.</p>
      </div>
    </div>
  );
} 