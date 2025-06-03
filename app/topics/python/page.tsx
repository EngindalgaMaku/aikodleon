import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, Brain, Code2, Database } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'Python Dersleri | Kodleon',
  description: 'Python programlama dilini sıfırdan öğrenin. Nesne yönelimli programlama, derin öğrenme ve daha fazlası.',
};

const topics = [
  {
    title: 'Nesneye Yönelik Programlama',
    description: 'Python\'da OOP kavramlarını, sınıfları, kalıtımı ve daha fazlasını öğrenin.',
    icon: <Code2 className="h-8 w-8 text-primary" />,
    href: '/topics/python/nesneye-yonelik-programlama'
  },
  {
    title: 'Veri Yapıları ve Algoritmalar',
    description: 'Python ile temel veri yapılarını ve algoritmaları keşfedin.',
    icon: <Database className="h-8 w-8 text-primary" />,
    href: '/topics/python/veri-yapilari-ve-algoritmalar'
  },
  {
    title: 'Derin Öğrenme',
    description: 'Python kullanarak derin öğrenme modellerini nasıl oluşturacağınızı ve eğiteceğinizi öğrenin.',
    icon: <Brain className="h-8 w-8 text-primary" />,
    href: '/topics/python/derin-ogrenme'
  }
];

export default function PythonTopicsPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <Button asChild variant="ghost" size="sm" className="gap-1">
          <Link href="/topics">
            <ArrowLeft className="h-4 w-4" />
            Tüm Konulara Dön
          </Link>
        </Button>
      </div>

      <div className="max-w-3xl mx-auto">
        <h1 className="text-4xl font-bold mb-6">Python Dersleri</h1>
        <p className="text-xl text-muted-foreground mb-12">
          Python programlama dilini sıfırdan öğrenin. Temel kavramlardan ileri seviye konulara kadar kapsamlı bir eğitim içeriği.
        </p>

        <div className="grid gap-6">
          {topics.map((topic, index) => (
            <Card key={index} className="transition-all hover:shadow-lg">
              <Link href={topic.href} className="block">
                <CardHeader>
                  <div className="flex items-center gap-4">
                    <div className="p-2 rounded-lg bg-primary/10">
                      {topic.icon}
                    </div>
                    <div>
                      <CardTitle className="text-xl">{topic.title}</CardTitle>
                      <CardDescription className="mt-1">{topic.description}</CardDescription>
                    </div>
                  </div>
                </CardHeader>
              </Link>
            </Card>
          ))}
        </div>
      </div>

      <div className="mt-16 text-center text-sm text-muted-foreground">
        <p>© {new Date().getFullYear()} Kodleon | Yapay Zeka Eğitim Platformu</p>
      </div>
    </div>
  );
} 