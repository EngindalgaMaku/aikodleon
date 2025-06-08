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
      url: `${baseUrl}/topics/python/nesne-tabanli-programlama/cok-bicimlilk`,
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

  return [...mainPages, ...pythonOOPPages, ...blogPages, ...legalPages]
} 