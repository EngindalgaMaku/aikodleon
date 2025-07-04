export type SitemapEntry = {
  url: string;
  lastModified?: Date;
  changeFrequency?: 'always' | 'hourly' | 'daily' | 'weekly' | 'monthly' | 'yearly' | 'never';
  priority?: number;
  alternateRefs?: {
    hreflang: string;
    href: string;
  }[];
};

export function generateSitemapEntries(): SitemapEntry[] {
  const baseUrl = 'https://kodleon.com';
  const now = new Date();

  // Ana sayfalar
  const mainPages = [
    {
      url: '/',
      changeFrequency: 'weekly',
      priority: 1.0,
    },
    {
      url: '/topics',
      changeFrequency: 'weekly',
      priority: 0.9,
    },
    {
      url: '/about',
      changeFrequency: 'monthly',
      priority: 0.8,
    },
    {
      url: '/contact',
      changeFrequency: 'monthly',
      priority: 0.7,
    },
  ];

  // Python OOP sayfaları
  const pythonOOPPages = [
    '/topics/python/nesne-tabanli-programlama',
    '/topics/python/nesne-tabanli-programlama/siniflar-ve-nesneler',
    '/topics/python/nesne-tabanli-programlama/kalitim',
    '/topics/python/nesne-tabanli-programlama/kapsulleme',
    '/topics/python/nesne-tabanli-programlama/cok-bicimlilk',
    '/topics/python/nesne-tabanli-programlama/soyut-siniflar-ve-arayuzler',
    '/topics/python/nesne-tabanli-programlama/tasarim-desenleri',
    '/topics/python/nesne-tabanli-programlama/pratik-ornekler',
    '/topics/python/nesne-tabanli-programlama/terimler-sozlugu',
    // Yeni eklenen örnek sayfaları
    '/topics/python/nesne-tabanli-programlama/pratik-ornekler/temel-ornekler',
    '/topics/python/nesne-tabanli-programlama/pratik-ornekler/temel-ornekler/ogrenci-sistemi',
    '/topics/python/nesne-tabanli-programlama/pratik-ornekler/temel-ornekler/arac-kiralama',
    '/topics/python/nesne-tabanli-programlama/pratik-ornekler/tasarim-desenleri',
    '/topics/python/nesne-tabanli-programlama/pratik-ornekler/gercek-dunya'
  ].map(url => ({
    url,
    changeFrequency: 'weekly',
    priority: 0.8,
  }));

  // Diğer konu sayfaları
  const topicPages = [
    '/topics/machine-learning',
    '/topics/nlp',
    '/topics/computer-vision',
    '/topics/generative-ai',
    '/topics/neural-networks',
    '/topics/ai-ethics',
  ].map(url => ({
    url,
    changeFrequency: 'monthly',
    priority: 0.8,
  }));

  // Yasal sayfalar
  const legalPages = [
    '/privacy',
    '/terms',
  ].map(url => ({
    url,
    changeFrequency: 'yearly',
    priority: 0.5,
  }));

  // Tüm sayfaları birleştir
  const allPages = [...mainPages, ...pythonOOPPages, ...topicPages, ...legalPages];

  // Her sayfa için tam URL ve alternatif dil referansı ekle
  return allPages.map(page => ({
    ...page,
    url: `${baseUrl}${page.url}`,
    lastModified: now,
    alternateRefs: [
      {
        hreflang: 'tr-TR',
        href: `${baseUrl}${page.url}`,
      }
    ]
  })) as SitemapEntry[];
}

// Blog ve kurs sayfaları gibi dinamik içerikler için ileride bu fonksiyon genişletilebilir
export async function getDynamicPages(): Promise<SitemapEntry[]> {
  const baseUrl = 'https://kodleon.com';
  
  // Manuel olarak eklenen güncel blog yazıları
  const manualBlogPosts: SitemapEntry[] = [
    {
      url: `${baseUrl}/blog/guncel-ai-modelleri-2025`,
      lastModified: new Date('2025-06-01'),
      changeFrequency: 'weekly',
      priority: 0.9,
      alternateRefs: [
        {
          hreflang: 'tr-TR',
          href: `${baseUrl}/blog/guncel-ai-modelleri-2025`,
        }
      ]
    }
  ];
  
  return manualBlogPosts;
}

// Tüm sitemap girdilerini oluştur
export async function getAllSitemapEntries(): Promise<SitemapEntry[]> {
  const staticPages = generateSitemapEntries();
  const dynamicPages = await getDynamicPages();
  
  return [...staticPages, ...dynamicPages];
} 