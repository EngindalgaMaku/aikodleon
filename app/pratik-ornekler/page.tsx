import Link from 'next/link';
import { Metadata } from 'next';
import { ChevronRight, Code2, BrainCircuit, Zap } from 'lucide-react'; // İkonlar eklendi
import { siteConfig } from '@/config/site';

export const metadata: Metadata = {
  title: 'Kod Örnekleri | Kodleon',
  description: 'Yapay zeka, makine öğrenmesi ve metasezgisel optimizasyon gibi konularda pratik kod örnekleri ve uygulamalar. Python ile gerçek dünya problemlerine çözümler.',
  keywords: 'kod örnekleri, pratik uygulamalar, python örnekleri, yapay zeka projeleri, makine öğrenmesi uygulamaları, metasezgisel optimizasyon kodları, kodleon',
  alternates: {
    canonical: `${siteConfig.url}/pratik-ornekler`,
  },
  openGraph: {
    title: 'Kod Örnekleri | Kodleon',
    description: 'Çeşitli yapay zeka konularında Python ile pratik kod örnekleri.',
    url: `${siteConfig.url}/pratik-ornekler`,
    images: [
      {
        url: `${siteConfig.url}/images/og-pratik-ornekler.jpg`,
        width: 1200,
        height: 630,
        alt: 'Kodleon - Kod Örnekleri',
      }
    ],
  },
};

interface ExampleCategory {
  id: string;
  name: string;
  description: string;
  icon: JSX.Element;
  examples: ExampleLink[];
}

interface ExampleLink {
  title: string;
  href: string;
  language?: string;
}

const exampleCategories: ExampleCategory[] = [
  {
    id: 'metasezgisel-optimizasyon',
    name: 'Metasezgisel Optimizasyon Örnekleri',
    description: 'Genetik algoritmalar, parçacık sürüsü optimizasyonu gibi metasezgisel yöntemlerin pratik uygulamaları.',
    icon: <BrainCircuit className="h-8 w-8 text-purple-500" />,
    examples: [
      {
        title: 'Genetik Algoritma - Python Örnekleri',
        href: '/topics/metasezgisel-optimizasyon/genetik-algoritmalar/genetik-algoritma-ornekleri',
        language: 'Python',
      },
      // Gelecekte eklenecek diğer metasezgisel örnekler buraya gelebilir
      // { title: 'Parçacık Sürüsü Optimizasyonu - Python', href: '#', language: 'Python' },
    ],
  },
  {
    id: 'derin-ogrenme',
    name: 'Derin Öğrenme Uygulamaları',
    description: 'Görüntü işleme, doğal dil işleme gibi alanlarda derin öğrenme modelleriyle geliştirilmiş örnek projeler.',
    icon: <Zap className="h-8 w-8 text-orange-500" />,
    examples: [
      // Gelecekte eklenecek derin öğrenme örnekleri
      // { title: 'Basit Görüntü Sınıflandırma (CNN) - TensorFlow/Keras', href: '#', language: 'Python' },
    ],
  },
  // Gelecekte başka kategoriler eklenebilir
];

export default function PracticalExamplesPage() {
  return (
    <div className="container max-w-6xl mx-auto px-4 py-12 md:py-20">
      <div className="text-center mb-12 md:mb-16">
        <div className="inline-block p-3 bg-primary/10 rounded-full mb-4">
          <Code2 className="h-10 w-10 text-primary" />
        </div>
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
          Kod Örnekleri
        </h1>
        <p className="mt-4 md:mt-6 text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto">
          Farklı yapay zeka ve optimizasyon konularında Python ile geliştirilmiş uygulanabilir kod örneklerini keşfedin.
        </p>
      </div>

      <div className="space-y-10">
        {exampleCategories.map((category) => (
          category.examples.length > 0 && (
            <section key={category.id} aria-labelledby={category.id + '-heading'}>
              <div className="flex items-center gap-3 mb-6">
                {category.icon}
                <h2 id={category.id + '-heading'} className="text-2xl sm:text-3xl font-bold text-gray-800 dark:text-gray-200">
                  {category.name}
                </h2>
              </div>
              <p className="text-gray-600 dark:text-gray-400 mb-6 ml-11">
                {category.description}
              </p>
              <ul className="space-y-4 ml-11">
                {category.examples.map((example) => (
                  <li key={example.href} className="bg-white dark:bg-gray-800/50 shadow-lg rounded-lg overflow-hidden transition-all hover:shadow-xl">
                    <Link href={example.href} className="block hover:bg-gray-50 dark:hover:bg-gray-700/50">
                      <div className="p-5 sm:p-6">
                        <div className="flex items-center justify-between">
                          <h3 className="text-lg font-semibold text-primary dark:text-sky-400">
                            {example.title}
                          </h3>
                          <ChevronRight className="h-5 w-5 text-gray-400 dark:text-gray-500" />
                        </div>
                        {example.language && (
                          <span className="mt-1 inline-block bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 text-xs font-medium px-2.5 py-0.5 rounded-full">
                            {example.language}
                          </span>
                        )}
                      </div>
                    </Link>
                  </li>
                ))}
              </ul>
            </section>
          )
        ))}

        {exampleCategories.every(cat => cat.examples.length === 0) && (
            <div className="text-center py-10">
                <p className="text-xl text-gray-500 dark:text-gray-400">Şu anda gösterilecek pratik örnek bulunmamaktadır.</p>
                <p className="text-gray-400 dark:text-gray-500 mt-2">Yakında eklenecek yeni örnekler için takipte kalın!</p>
            </div>
        )}
      </div>
    </div>
  );
} 