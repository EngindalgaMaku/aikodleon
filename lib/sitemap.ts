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

  // Konu sayfaları
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
  const allPages = [...mainPages, ...topicPages, ...legalPages];

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
  // Örnek: Blog yazıları veya kurs içerikleri gibi dinamik içerikleri veritabanından çekip
  // sitemap girdilerine dönüştürmek için kullanılabilir
  
  // Şu an için boş bir array dönüyoruz
  return [];
}

// Tüm sitemap girdilerini oluştur
export async function getAllSitemapEntries(): Promise<SitemapEntry[]> {
  const staticPages = generateSitemapEntries();
  const dynamicPages = await getDynamicPages();
  
  return [...staticPages, ...dynamicPages];
} 