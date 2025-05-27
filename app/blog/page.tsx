import Link from "next/link";
import Image from "next/image";
import { ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";

const blogPosts = [
  {
    title: "Yapay Zekanın Geleceği: 2024 Trendleri",
    description: "Yapay zeka teknolojilerinin gelecek yıl nasıl şekilleneceğini ve hangi alanlarda öne çıkacağını inceliyoruz.",
    author: "Dr. Ahmet Yılmaz",
    date: "15 Mart 2024",
    imageUrl: "https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "Teknoloji Trendleri"
  },
  {
    title: "Makine Öğrenmesi ile Veri Analizi",
    description: "Büyük veri setlerinden anlamlı içgörüler çıkarmak için makine öğrenmesi tekniklerinin kullanımı.",
    author: "Zeynep Aksoy",
    date: "12 Mart 2024",
    imageUrl: "https://images.pexels.com/photos/7412095/pexels-photo-7412095.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "Veri Bilimi"
  },
  {
    title: "ChatGPT ve Eğitimde AI Kullanımı",
    description: "Yapay zeka destekli dil modellerinin eğitim alanında nasıl kullanılabileceğini keşfediyoruz.",
    author: "Mehmet Kaya",
    date: "10 Mart 2024",
    imageUrl: "https://images.pexels.com/photos/8438922/pexels-photo-8438922.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "Eğitim Teknolojileri"
  },
  {
    title: "Bilgisayarlı Görü Uygulamaları",
    description: "Görüntü işleme ve nesne tanıma teknolojilerinin endüstriyel uygulamaları.",
    author: "Ayşe Demir",
    date: "8 Mart 2024",
    imageUrl: "https://images.pexels.com/photos/8386434/pexels-photo-8386434.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "Bilgisayarlı Görü"
  }
];

export default function BlogPage() {
  return (
    <div className="container max-w-6xl mx-auto py-12">
      <div className="max-w-3xl mx-auto text-center mb-12">
        <h1 className="text-4xl font-bold tracking-tight mb-4">Blog</h1>
        <p className="text-xl text-muted-foreground">
          Yapay zeka dünyasındaki son gelişmeler, teknoloji trendleri ve uzman görüşleri.
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {blogPosts.map((post, index) => (
          <Card key={index} className="overflow-hidden">
            <div className="relative h-48">
              <Image 
                src={post.imageUrl}
                alt={post.title}
                fill
                className="object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent" />
              <div className="absolute top-4 left-4">
                <span className="px-3 py-1 rounded-full text-xs font-medium bg-background/80 backdrop-blur-sm">
                  {post.category}
                </span>
              </div>
            </div>
            <CardHeader>
              <CardTitle>{post.title}</CardTitle>
              <CardDescription>{post.description}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between text-sm text-muted-foreground">
                <span>{post.author}</span>
                <span>{post.date}</span>
              </div>
            </CardContent>
            <CardFooter>
              <Button asChild variant="ghost" className="gap-1 ml-auto">
                <Link href="#">
                  Devamını Oku
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
}