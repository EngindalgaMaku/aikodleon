import { MetadataRoute } from 'next';

type ChangeFrequency = 'daily' | 'weekly' | 'monthly' | 'yearly' | 'always' | 'hourly' | 'never';

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = 'https://kodleon.com'
  const lastModified = new Date()

  const mainPages = [
    {
      url: baseUrl,
      lastModified,
      changeFrequency: 'daily' as ChangeFrequency,
      priority: 1
    },
    {
      url: `${baseUrl}/about`,
      lastModified,
      changeFrequency: 'monthly' as ChangeFrequency,
      priority: 0.8
    },
    {
      url: `${baseUrl}/contact`,
      lastModified,
      changeFrequency: 'monthly' as ChangeFrequency,
      priority: 0.7
    }
  ]

  const pythonOOPPages = [
    {
      url: `${baseUrl}/topics/python/nesne-tabanli-programlama`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.9
    },
    {
      url: `${baseUrl}/topics/python/nesne-tabanli-programlama/siniflar-ve-nesneler`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.9
    },
    {
      url: `${baseUrl}/topics/python/nesne-tabanli-programlama/siniflar-ve-nesneler/nesne-dizileri`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.8
    },
    {
      url: `${baseUrl}/topics/python/nesne-tabanli-programlama/siniflar-ve-nesneler/nesne-olusturma`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.8
    },
    {
      url: `${baseUrl}/topics/python/nesne-tabanli-programlama/siniflar-ve-nesneler/sinif-kavrami`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.8
    },
    {
      url: `${baseUrl}/topics/python/nesne-tabanli-programlama/siniflar-ve-nesneler/instance-metodlari`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.8
    },
    {
      url: `${baseUrl}/topics/python/nesne-tabanli-programlama/siniflar-ve-nesneler/iyi-pratikler`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.8
    },
    {
      url: `${baseUrl}/topics/python/nesne-tabanli-programlama/siniflar-ve-nesneler/alistirmalar`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.8
    },
    {
      url: `${baseUrl}/topics/python/nesne-tabanli-programlama/kalitim`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.8
    },
    {
      url: `${baseUrl}/topics/python/nesne-tabanli-programlama/kapsulleme`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.8
    },
    {
      url: `${baseUrl}/topics/python/nesne-tabanli-programlama/cok-bicimlilik`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.8
    },
    {
      url: `${baseUrl}/topics/python/nesne-tabanli-programlama/soyut-siniflar-ve-arayuzler`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.8
    },
    {
      url: `${baseUrl}/topics/python/nesne-tabanli-programlama/tasarim-desenleri`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.8
    },
    {
      url: `${baseUrl}/topics/python/nesne-tabanli-programlama/terimler-sozlugu`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.8
    },
    {
      url: `${baseUrl}/topics/python/nesne-tabanli-programlama/method-overriding`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.8
    }
  ]

  const metasezgiselPages = [
    {
      url: `${baseUrl}/topics/metasezgisel-optimizasyon`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.9
    },
    {
      url: `${baseUrl}/topics/metasezgisel-optimizasyon/guguk-kusu-aramasi`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.8
    },
    {
      url: `${baseUrl}/topics/metasezgisel-optimizasyon/yapay-ari-kolonisi-optimizasyonu`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.8
    }
  ]

  const nlpPages = [
    {
      url: `${baseUrl}/topics/nlp`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.9
    }
  ]

  const neuralNetworksPages = [
    {
      url: `${baseUrl}/topics/neural-networks`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.9
    }
  ]

  const aiFundamentalsPages = [
    {
      url: `${baseUrl}/topics/ai-fundamentals`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.9
    }
  ]

  const codeExamplesPages = [
    {
      url: `${baseUrl}/kod-ornekleri`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.9
    },
    {
      url: `${baseUrl}/kod-ornekleri/temel-sinir-agi`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.8
    },
    {
      url: `${baseUrl}/kod-ornekleri/resim-siniflandirma`,
      lastModified,
      changeFrequency: 'weekly' as ChangeFrequency,
      priority: 0.8
    }
  ]

  const blogPages = [
    {
      url: `${baseUrl}/blog`,
      lastModified,
      changeFrequency: 'daily' as ChangeFrequency,
      priority: 0.9
    }
  ]

  const legalPages = [
    {
      url: `${baseUrl}/privacy`,
      lastModified,
      changeFrequency: 'monthly' as ChangeFrequency,
      priority: 0.5
    },
    {
      url: `${baseUrl}/terms`,
      lastModified,
      changeFrequency: 'monthly' as ChangeFrequency,
      priority: 0.5
    }
  ]

  return [
    ...mainPages,
    ...pythonOOPPages,
    ...metasezgiselPages,
    ...nlpPages,
    ...neuralNetworksPages,
    ...aiFundamentalsPages,
    ...codeExamplesPages,
    ...blogPages,
    ...legalPages
  ]
} 