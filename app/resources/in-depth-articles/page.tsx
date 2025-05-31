import { ArrowRight, BookOpen, Brain, Atom, FileText } from 'lucide-react';
import Link from 'next/link';
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'İleri Seviye Yapay Zeka: Makaleler, Bloglar ve Araştırmalar | Kodleon',
  description: 'Kodleon ile yapay zeka, makine öğrenmesi ve derin öğrenmede uzmanlaşın. Sinir ağları, Transformer modelleri, GAN\'ler ve en son AI araştırmaları üzerine teknik makaleler ve uzman blog yazılarını keşfedin.',
  keywords: ['derinlemesine AI makaleleri', 'yapay zeka blogları Türkiye', 'ileri seviye makine öğrenmesi', 'derin öğrenme teknikleri', 'sinir ağı mimarileri', 'Transformer modelleri nedir', 'GAN nasıl çalışır', 'teknik AI blogları', 'yapay zeka araştırma makaleleri', 'Kodleon AI uzman analizleri', 'CNN', 'RNN'],
  openGraph: {
    title: 'İleri Seviye Yapay Zeka: Makaleler, Bloglar ve Araştırmalar | Kodleon',
    description: 'Yapay zeka ve makine öğrenimi alanındaki en son gelişmeler, derinlemesine teknik makaleler ve uzman analizleri Kodleon\'da.',
    url: 'https://kodleon.com/resources/in-depth-articles',
    images: [
      {
        url: '/images/in-depth-articles-og.png',
        width: 1200,
        height: 630,
        alt: 'Kodleon İleri Seviye Yapay Zeka Makaleleri'
      }
    ]
  }
};

interface Article {
  id: string;
  title: string;
  source: string;
  description: string;
  link: string;
  icon: React.ReactNode;
  tags: string[];
}

const articles: Article[] = [
  {
    id: 'nn-article-1',
    title: 'Evrişimli Sinir Ağı Mimarilerine Dayalı Türkçe Duygu Analizi',
    source: 'Dergipark - Aytuğ Onan',
    description: 'Türkçe metinler üzerinde duygu analizi için evrişimli sinir ağı (CNN) tabanlı üç farklı derin öğrenme mimarisinin etkinliğini değerlendiren bir akademik makale. Kelime gömme yöntemleri ve farklı CNN mimarilerinin karşılaştırmalı sonuçlarını sunar.',
    link: 'https://dergipark.org.tr/en/download/article-file/1240865',
    icon: <FileText className="h-8 w-8 text-blue-500" />,
    tags: ['Akademik Makale', 'CNN', 'Duygu Analizi', 'Türkçe'],
  },
  {
    id: 'nn-article-2',
    title: 'Derin Öğrenme ve Yapay Sinir Ağı Modelleri Üzerine Bir İnceleme',
    source: 'Dergipark - Ercan Akın, Mustafa Ergin Şahin',
    description: 'Derin öğrenmenin tarihçesi, çalışma prensibi, uygulama alanları ve bu alanlarda kullanılan yapay sinir ağları modelleri (CNN, RNN, LSTM, RBM, Autoencoder\'lar dahil) hakkında genel bir bakış sunan bir derleme makalesi.',
    link: 'https://dergipark.org.tr/en/download/article-file/3309608',
    icon: <FileText className="h-8 w-8 text-blue-500" />,
    tags: ['Akademik Makale', 'Derin Öğrenme', 'YSA', 'CNN', 'RNN', 'LSTM', 'Türkçe'],
  },
  {
    id: 'nn-article-3',
    title: 'Açıklanabilir Evrişimsel Sinir Ağları ile Beyin Tümörü Tespiti',
    source: 'Dergipark - Abdullah Orman, Utku Köse, Tuncay Yiğit',
    description: 'Beyin tümörü tespiti için bir Evrişimsel Sinir Ağı (CNN) modeli kullanan ve modelin güvenilirliğini Sınıf Aktivasyon Haritalama (CAM) ile değerlendiren bir çalışma. CNN\'lerin pratik bir uygulamasını ve "açıklanabilir yapay zeka" kavramını ele alır.',
    link: 'https://dergipark.org.tr/tr/download/article-file/1724370',
    icon: <FileText className="h-8 w-8 text-blue-500" />,
    tags: ['Akademik Makale', 'CNN', 'Beyin Tümörü Tespiti', 'Açıklanabilir YZ', 'Türkçe'],
  },
  {
    id: 'transformer-article-1',
    title: 'Transformer Modeli Nedir?',
    source: 'OpenZeka Blog',
    description: 'Transformer modelini, sıralı verilerdeki ilişkileri izleyerek bağlamı ve anlamı öğrenen bir sinir ağı olarak tanıtan bir blog yazısı. "İlgi" (attention) mekanizmasına, uygulama alanlarına ve önemli Transformer modellerine (BERT, GPT-3) değinir.',
    link: 'https://blog.openzeka.com/ai/transformer-modeli-nedir/',
    icon: <Brain className="h-8 w-8 text-green-500" />,
    tags: ['Blog Yazısı', 'Transformer', 'Attention', 'Doğal Dil İşleme', 'Türkçe'],
  },
  {
    id: 'transformer-article-2',
    title: 'Transformatörler: Tüm İhtiyacınız Olan Dikkat',
    source: 'Medium - Cahit Barkin Ozer',
    description: 'Popüler "Attention is all you need" makalesinin Türkçe bir özeti ve çevirisi. Transformer mimarisinin dikkat mekanizmasına dayalı yapısını ve NLP dışındaki alanlara uzanan potansiyelini vurgular.',
    link: 'https://cbarkinozer.medium.com/transformat%C3%B6rler-t%C3%BCm-i%CC%87htiyac%C4%B1n%C4%B1z-olan-dikkat-ee4ff66723b1',
    icon: <Brain className="h-8 w-8 text-green-500" />,
    tags: ['Blog Yazısı', 'Transformer', 'Attention', 'Makale Özeti', 'Türkçe'],
  },
  {
    id: 'gan-article-1',
    title: 'Generative Adversarial Networks (GAN) nedir?',
    source: 'Medium - Cihan Öngün',
    description: 'GAN\'leri "Çekişmeli Üretici Ağlar" olarak tanımlayan, çalışma prensibini "kalpazan ve polis" analojisiyle açıklayan bir blog yazısı. Uygulama alanlarına, etik problemlere ve temel matematiksel fonksiyona değinir.',
    link: 'https://cihanongun.medium.com/generative-adversarial-networks-gan-nedir-5cc6a48a6870',
    icon: <Atom className="h-8 w-8 text-purple-500" />,
    tags: ['Blog Yazısı', 'GAN', 'Derin Öğrenme', 'Yapay Sinir Ağları', 'Türkçe'],
  },
  {
    id: 'gan-article-2',
    title: 'Generative Adversarial Networks (GAN) Nedir?',
    source: 'YapayZekaTR Blog',
    description: 'GAN\'ı Üretici ve Ayırt Edici ağların rekabetine dayanan bir model olarak açıklayan, üretken ve ayırt edici algoritmalar farkına değinen bir yazı. GAN\'ların çalışma adımlarını ve geniş uygulama alanlarını listeler.',
    link: 'https://www.yapayzekatr.com/tr/blog/detay/generative-adversarial-networks-gan-nedir/4/20/0',
    icon: <Atom className="h-8 w-8 text-purple-500" />,
    tags: ['Blog Yazısı', 'GAN', 'Derin Öğrenme', 'Uygulama Alanları', 'Türkçe'],
  },
  {
    id: 'gan-article-3',
    title: 'Generative Adversarial Networks — GAN nedir ? ( Türkçe )',
    source: 'Medium - Muhammed Buyukkinaci',
    description: 'GAN\'ların çalışma prensibini "kalpazan" ve "dedektif" metaforuyla açıklayan, gradient hesaplamalarına ve Generator\'ın kendini nasıl güncellediğine değinen bir yazı. DCGAN gibi popüler GAN çeşitlerinden ve pratik uygulama örneklerinden bahseder.',
    link: 'https://medium.com/@muhammedbuyukkinaci/generative-adversarial-networks-gan-nedir-t%C3%BCrk%C3%A7e-5819fe9c1fa7',
    icon: <Atom className="h-8 w-8 text-purple-500" />,
    tags: ['Blog Yazısı', 'GAN', 'DCGAN', 'Derin Öğrenme', 'Türkçe'],
  },
];

const ArticleCard: React.FC<{ article: Article }> = ({ article }) => (
  <div className="bg-gray-800 shadow-lg rounded-lg overflow-hidden transform transition-all hover:scale-105 duration-300 ease-in-out">
    <div className="p-6">
      <div className="flex items-center mb-4">
        <div className="p-2 bg-gray-700 rounded-full mr-4">
          {article.icon}
        </div>
        <h3 className="text-xl font-semibold text-sky-400 group-hover:text-sky-300 transition-colors duration-300">{article.title}</h3>
      </div>
      <p className="text-gray-400 text-sm mb-1">Kaynak: {article.source}</p>
      <p className="text-gray-300 mb-4 text-base leading-relaxed">{article.description}</p>
      <div className="mb-4">
        {article.tags.map(tag => (
          <span key={tag} className="text-xs bg-gray-700 text-sky-300 px-2 py-1 rounded-full mr-2 mb-2 inline-block">{tag}</span>
        ))}
      </div>
      <Link href={article.link} target="_blank" rel="noopener noreferrer"
        className="inline-flex items-center text-amber-400 hover:text-amber-300 font-medium group transition-colors duration-300">
        Kaynağa Git
        <ArrowRight className="ml-2 h-5 w-5 transform group-hover:translate-x-1 transition-transform duration-300" />
      </Link>
    </div>
  </div>
);

export default function InDepthArticlesPage() {
  const nnArticles = articles.filter(a => a.id.startsWith('nn-'));
  const transformerArticles = articles.filter(a => a.id.startsWith('transformer-'));
  const ganArticles = articles.filter(a => a.id.startsWith('gan-'));

  return (
    <div className="bg-gray-900 text-white min-h-screen">
      {/* Hero Section */}
      <section className="py-16 md:py-24 bg-gradient-to-br from-gray-900 via-blue-900/30 to-gray-900">
        <div className="container mx-auto px-4 text-center">
          <BookOpen className="h-16 w-16 text-sky-400 mx-auto mb-6" />
          <h1 className="text-4xl md:text-5xl font-bold mb-6 tracking-tight">
            Derinlemesine Makaleler ve Bloglar
          </h1>
          <p className="text-lg md:text-xl text-gray-300 max-w-3xl mx-auto mb-8">
            Yapay zeka, makine öğrenmesi ve derin öğrenme dünyasındaki karmaşık konuları ve en son araştırma bulgularını keşfedin. Bu bölümde, alanında uzman kişiler tarafından yazılmış teknik makalelere ve kapsamlı blog yazılarına ulaşabilirsiniz.
          </p>
          <Link href="/resources"
            className="inline-flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-sky-600 hover:bg-sky-700 transition-colors duration-300">
            Diğer Kaynak Kategorileri
          </Link>
        </div>
      </section>

      {/* Articles Section */}
      <section className="py-12 md:py-20">
        <div className="container mx-auto px-4">

          {/* Neural Network Architectures */}
          <div className="mb-16">
            <div className="flex items-center mb-8">
              <Brain className="h-10 w-10 text-pink-500 mr-4" />
              <h2 className="text-3xl font-semibold text-pink-400">Yapay Sinir Ağı Mimarileri</h2>
            </div>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
              {nnArticles.map(article => (
                <ArticleCard key={article.id} article={article} />
              ))}
            </div>
          </div>

          {/* Transformer Models */}
          <div className="mb-16">
            <div className="flex items-center mb-8">
              <Brain className="h-10 w-10 text-green-500 mr-4" />
              <h2 className="text-3xl font-semibold text-green-400">Transformer Modelleri</h2>
            </div>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
              {transformerArticles.map(article => (
                <ArticleCard key={article.id} article={article} />
              ))}
            </div>
          </div>

          {/* GANs */}
          <div>
            <div className="flex items-center mb-8">
              <Atom className="h-10 w-10 text-purple-500 mr-4" />
              <h2 className="text-3xl font-semibold text-purple-400">Çekişmeli Üretici Ağlar (GAN)</h2>
            </div>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
              {ganArticles.map(article => (
                <ArticleCard key={article.id} article={article} />
              ))}
            </div>
          </div>

        </div>
      </section>

      {/* CTA to other categories */}
      <section className="py-16 bg-gray-800">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-3xl font-semibold mb-6 text-sky-400">Daha Fazla Kaynak Keşfedin</h2>
          <p className="text-gray-300 max-w-xl mx-auto mb-8">
            Yapay zeka öğrenme yolculuğunuzda size yardımcı olacak diğer kategorilerimize de göz atın.
          </p>
          <Link href="/resources"
            className="inline-block bg-amber-500 hover:bg-amber-600 text-gray-900 font-semibold py-3 px-8 rounded-lg shadow-md hover:shadow-lg transition-all duration-300 transform hover:scale-105">
            Tüm Kategoriler
          </Link>
        </div>
      </section>
    </div>
  );
} 