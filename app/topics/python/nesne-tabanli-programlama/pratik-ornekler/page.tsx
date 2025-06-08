import { Metadata } from 'next';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowRight, Code, Blocks, Factory, Globe } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python OOP Pratik Örnekler | Kodleon',
  description: 'Python nesne tabanlı programlama pratik örnekleri: Temel örnekler, tasarım desenleri ve gerçek dünya uygulamaları.',
};

const examples = [
  {
    title: "Temel Örnekler",
    description: "Nesne tabanlı programlamanın temel kavramlarını gösteren örnek uygulamalar",
    icon: <Code className="h-6 w-6" />,
    href: "/topics/python/nesne-tabanli-programlama/pratik-ornekler/temel-ornekler",
    topics: [
      "Öğrenci Bilgi Sistemi",
      "Araç Kiralama Sistemi",
      "Kütüphane Yönetimi",
      "Banka Hesap Sistemi"
    ]
  },
  {
    title: "Tasarım Desenleri",
    description: "Yaygın tasarım desenlerinin Python implementasyonları",
    icon: <Blocks className="h-6 w-6" />,
    href: "/topics/python/nesne-tabanli-programlama/pratik-ornekler/tasarim-desenleri",
    topics: [
      "Creational Patterns",
      "Structural Patterns",
      "Behavioral Patterns",
      "Best Practices"
    ]
  },
  {
    title: "Gerçek Dünya Uygulamaları",
    description: "Profesyonel yazılım geliştirmede kullanılan örnek uygulamalar",
    icon: <Globe className="h-6 w-6" />,
    href: "/topics/python/nesne-tabanli-programlama/pratik-ornekler/gercek-dunya",
    topics: [
      "REST API Servisi",
      "ORM Sistemi",
      "GUI Uygulaması",
      "Web Scraping"
    ]
  }
];

export default function PratikOrneklerPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="prose dark:prose-invert max-w-none mb-12">
          <h1>Python OOP Pratik Örnekler</h1>
          <p>
            Bu bölümde, nesne tabanlı programlama kavramlarını pekiştirmek için çeşitli pratik örnekler bulacaksınız.
            Her örnek, gerçek dünya senaryolarını temel alarak hazırlanmış ve detaylı açıklamalarla sunulmuştur.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {examples.map((example, index) => (
            <Card key={index} className="bg-yellow-50 hover:bg-yellow-100 dark:bg-yellow-950/50 dark:hover:bg-yellow-950/70 transition-all duration-300">
              <CardHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="p-2 rounded-lg text-yellow-600 dark:text-yellow-400">
                    {example.icon}
                  </div>
                  <CardTitle>{example.title}</CardTitle>
                </div>
                <CardDescription className="dark:text-gray-300">{example.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground dark:text-gray-400">
                  {example.topics.map((topic, i) => (
                    <li key={i}>{topic}</li>
                  ))}
                </ul>
              </CardContent>
              <CardFooter>
                <Button asChild variant="outline" className="w-full group">
                  <Link href={example.href}>
                    İncele
                    <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
                  </Link>
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>

        {/* Back to OOP Topics Button */}
        <div className="mt-12 flex justify-end">
          <Button asChild variant="outline" className="group">
            <Link href="/topics/python/nesne-tabanli-programlama">
              OOP Konularına Dön
              <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 