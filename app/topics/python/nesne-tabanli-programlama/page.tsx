import { Metadata } from 'next';
import Link from 'next/link';
import { 
    ArrowLeft, 
    Code2, 
    GitBranch, 
    Lock, 
    Shapes, 
    Component, 
    Factory, 
    BookOpen,
    Lightbulb
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python ile Nesne Tabanlı Programlama (OOP)',
  description: "Python'da nesne tabanlı programlamanın (OOP) temellerini, sınıfları, nesneleri, kalıtımı ve daha fazlasını öğrenin.",
};

const introContent = `
# Python ile Nesne Tabanlı Programlama (OOP)

Python'da nesne tabanlı programlama (OOP), kodunuzu daha modüler, okunabilir ve yeniden kullanılabilir hale getiren güçlü bir programlama paradigmasıdır. Bu bölümde, OOP'nin temel kavramlarını ve Python'da nasıl uygulandığını detaylı örneklerle öğreneceksiniz.
`;

const cardColors = [
  'bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 border-blue-200 dark:border-blue-800',
  'bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 border-green-200 dark:border-green-800',
  'bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 border-purple-200 dark:border-purple-800',
  'bg-gradient-to-br from-amber-50 to-amber-100 dark:from-amber-900/20 dark:to-amber-800/20 border-amber-200 dark:border-amber-800',
  'bg-gradient-to-br from-rose-50 to-rose-100 dark:from-rose-900/20 dark:to-rose-800/20 border-rose-200 dark:border-rose-800',
  'bg-gradient-to-br from-cyan-50 to-cyan-100 dark:from-cyan-900/20 dark:to-cyan-800/20 border-cyan-200 dark:border-cyan-800',
  'bg-gradient-to-br from-lime-50 to-lime-100 dark:from-lime-900/20 dark:to-lime-800/20 border-lime-200 dark:border-lime-800',
  'bg-gradient-to-br from-teal-50 to-teal-100 dark:from-teal-900/20 dark:to-teal-800/20 border-teal-200 dark:border-teal-800',
];

const sections = [
  {
    title: "1. Sınıflar ve Nesneler",
    description: "Python'da sınıf ve nesne kavramları, metodlar ve özellikler.",
    icon: <Code2 className="h-6 w-6" />,
    href: "/topics/python/nesne-tabanli-programlama/siniflar-ve-nesneler",
    topics: ["Sınıf tanımlama", "Nesne oluşturma", "__init__ metodu", "Instance metodları"]
  },
  {
    title: "2. Kalıtım",
    description: "Sınıflar arası kalıtım ilişkileri ve kodun yeniden kullanımı.",
    icon: <GitBranch className="h-6 w-6" />,
    href: "/topics/python/nesne-tabanli-programlama/kalitim",
    topics: ["Temel ve türetilmiş sınıflar", "super() kullanımı", "Çoklu kalıtım", "Method overriding"]
  },
  {
    title: "3. Kapsülleme",
    description: "Veri gizleme (encapsulation) ve kontrollü erişim.",
    icon: <Lock className="h-6 w-6" />,
    href: "/topics/python/nesne-tabanli-programlama/kapsulleme",
    topics: ["Public, private, protected", "Getter ve setter", "Property dekoratörü", "Name mangling"]
  },
  {
    title: "4. Çok Biçimlilik",
    description: "Nesnelerin farklı durumlarda farklı davranma yeteneği (polymorphism).",
    icon: <Shapes className="h-6 w-6" />,
    href: "/topics/python/nesne-tabanli-programlama/cok-bicimlilik",
    topics: ["Method overriding", "Method overloading", "Duck typing", "Operatörlerin yeniden yüklenmesi"]
  },
  {
    title: "5. Soyut Sınıflar",
    description: "Soyutlama (abstraction) ile arayüz ve standartlar oluşturma.",
    icon: <Component className="h-6 w-6" />,
    href: "/topics/python/nesne-tabanli-programlama/soyut-siniflar",
    topics: ["ABC modülü", "Soyut metodlar", "Interface tasarımı"]
  },
  {
    title: "6. Tasarım Desenleri",
    description: "Sık karşılaşılan sorunlar için kanıtlanmış OOP çözüm desenleri.",
    icon: <Factory className="h-6 w-6" />,
    href: "/topics/python/nesne-tabanli-programlama/tasarim-desenleri",
    topics: ["Singleton", "Factory", "Observer", "Decorator"]
  },
  {
    title: "7. Pratik Projeler",
    description: "Gerçek dünya senaryoları ile OOP becerilerinizi geliştirin.",
    icon: <BookOpen className="h-6 w-6" />,
    href: "/topics/python/nesne-tabanli-programlama/pratik-projeler",
    topics: ["Kütüphane Yönetim Sistemi", "Banka Uygulaması", "Basit E-ticaret Sistemi"]
  },
  {
    title: "8. Karışık OOP Örnekleri",
    description: "Tüm OOP konularını içeren alıştırmalarla bilginizi pekiştirin.",
    icon: <Lightbulb className="h-6 w-6" />,
    href: "/topics/python/nesne-tabanli-programlama/karisik-ornekler",
    topics: ["Çeşitli zorluklarda problemler", "Tasarım desenleri uygulamaları", "Kodlama mülakatı soruları"]
  }
];

export default function PythonOOPPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-6xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild className="mb-6">
            <Link href="/topics/python" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Python Eğitimleri
            </Link>
          </Button>
        </div>

        <div className="prose prose-lg dark:prose-invert mb-12">
          <MarkdownContent content={introContent} />
        </div>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            {sections.map((section, index) => (
            <Card key={index} className={`flex flex-col ${cardColors[index % cardColors.length]} hover:shadow-lg transition-shadow duration-300`}>
                <CardHeader>
                    <div className="flex items-center gap-2">
                        {section.icon}
                        <CardTitle>{section.title}</CardTitle>
                    </div>
                    <CardDescription>{section.description}</CardDescription>
                </CardHeader>
                <CardContent className="flex-1">
                    <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                        {section.topics.map((topic, i) => (
                            <li key={i}>{topic}</li>
                        ))}
                    </ul>
                </CardContent>
                <div className="p-6 pt-0">
                    <Button asChild className="w-full">
                        <Link href={section.href}>Derse Gir</Link>
                    </Button>
                </div>
            </Card>
          ))}
        </div>

        <div className="mt-16 text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} Kodleon | Python Eğitim Platformu</p>
        </div>
      </div>
    </div>
  );
} 