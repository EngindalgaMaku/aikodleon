import Link from "next/link";
import Image from "next/image";
import { ArrowRight, Brain, Ship as Chip, Database, Eye, FileText, Lightbulb, Rocket, Shapes, Users } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const allTopics = [
  {
    title: "Makine Öğrenmesi",
    description: "Algoritmaların veri kullanarak nasıl öğrendiğini ve tahminlerde bulunduğunu keşfedin.",
    icon: <Database className="h-8 w-8 text-chart-1" />,
    href: "/topics/machine-learning",
    imageUrl: "https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "core",
    subtopics: ["Denetimli Öğrenme", "Denetimsiz Öğrenme", "Pekiştirmeli Öğrenme", "Derin Öğrenme Temelleri"]
  },
  {
    title: "Doğal Dil İşleme",
    description: "Makinelerin insan dilini nasıl anlayıp işlediğini ve ürettiğini öğrenin.",
    icon: <FileText className="h-8 w-8 text-chart-2" />,
    href: "/topics/nlp",
    imageUrl: "https://images.pexels.com/photos/7412095/pexels-photo-7412095.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "core",
    subtopics: ["Metin Analizi", "Dil Modelleri", "Duygu Analizi", "Makine Çevirisi"]
  },
  {
    title: "Bilgisayarlı Görü",
    description: "Bilgisayarların görüntüleri nasıl algıladığını ve işlediğini anlayın.",
    icon: <Eye className="h-8 w-8 text-chart-3" />,
    href: "/topics/computer-vision",
    imageUrl: "https://images.pexels.com/photos/8438922/pexels-photo-8438922.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "core",
    subtopics: ["Görüntü Sınıflandırma", "Nesne Tespiti", "Yüz Tanıma", "Görüntü Segmentasyonu"]
  },
  {
    title: "Üretken AI",
    description: "Metin, görüntü ve ses üretebilen yapay zeka modellerini keşfedin.",
    icon: <Lightbulb className="h-8 w-8 text-chart-4" />,
    href: "/topics/generative-ai",
    imageUrl: "https://images.pexels.com/photos/8386434/pexels-photo-8386434.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "advanced",
    subtopics: ["Üretken Çekişmeli Ağlar (GAN)", "Diffusion Modelleri", "Büyük Dil Modelleri", "Text-to-Image Modelleri"]
  },
  {
    title: "Sinir Ağları",
    description: "Beynin çalışma prensibinden esinlenen yapay sinir ağları hakkında bilgi edinin.",
    icon: <Brain className="h-8 w-8 text-chart-5" />,
    href: "/topics/neural-networks",
    imageUrl: "https://images.pexels.com/photos/8386421/pexels-photo-8386421.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "core",
    subtopics: ["Temel Sinir Ağı Mimarileri", "Konvolüsyonel Sinir Ağları (CNN)", "Tekrarlayan Sinir Ağları (RNN)", "Transformerlar"]
  },
  {
    title: "AI Etiği",
    description: "Yapay zekanın etik kullanımı ve toplumsal etkileri üzerine tartışmalar.",
    icon: <Users className="h-8 w-8 text-chart-1" />,
    href: "/topics/ai-ethics",
    imageUrl: "https://images.pexels.com/photos/8386422/pexels-photo-8386422.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "applications",
    subtopics: ["Adil ve Tarafsız AI", "Gizlilik ve Güvenlik", "AI Düzenlemeleri", "Etik Yapay Zeka Tasarımı"]
  },
  {
    title: "Robotik",
    description: "Yapay zekanın robotik sistemlerde uygulanmasını ve robotların dünyayla etkileşimini öğrenin.",
    icon: <Chip className="h-8 w-8 text-chart-2" />,
    href: "/topics/robotics",
    imageUrl: "https://images.pexels.com/photos/2599244/pexels-photo-2599244.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "applications",
    subtopics: ["Robot Hareket Planlaması", "Sensör Füzyonu", "İnsan-Robot Etkileşimi", "Otonom Sistemler"]
  },
  {
    title: "AI'nin Geleceği",
    description: "Yapay zekanın gelecekteki yönelimlerini ve gelişimini inceleyin.",
    icon: <Rocket className="h-8 w-8 text-chart-3" />,
    href: "/topics/future-of-ai",
    imageUrl: "https://images.pexels.com/photos/2007647/pexels-photo-2007647.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "advanced",
    subtopics: ["Genel Yapay Zeka (AGI)", "Süper Zeka", "Kuantum AI", "Beyin-Bilgisayar Arayüzleri"]
  },
  {
    title: "AI ve Veri Bilimi",
    description: "Veri bilimi ve yapay zeka arasındaki ilişkiyi ve veri analizinde AI kullanımını keşfedin.",
    icon: <Shapes className="h-8 w-8 text-chart-4" />,
    href: "/topics/ai-data-science",
    imageUrl: "https://images.pexels.com/photos/669615/pexels-photo-669615.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "applications",
    subtopics: ["Büyük Veri Analizi", "Tahmine Dayalı Modelleme", "Anomali Tespiti", "Veri Görselleştirme"]
  },
];

export default function TopicsPage() {
  return (
    <div className="container py-12">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold tracking-tight mb-4">AI Eğitim Konuları</h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Yapay zeka teknolojilerinin farklı alanlarını keşfedin ve uzmanlaşmak istediğiniz konuları seçin.
        </p>
      </div>
      
      <Tabs defaultValue="all" className="mb-12">
        <div className="flex justify-center mb-8">
          <TabsList>
            <TabsTrigger value="all">Tümü</TabsTrigger>
            <TabsTrigger value="core">Temel Konular</TabsTrigger>
            <TabsTrigger value="applications">Uygulamalar</TabsTrigger>
            <TabsTrigger value="advanced">İleri Düzey</TabsTrigger>
          </TabsList>
        </div>
        
        <TabsContent value="all" className="mt-0">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {allTopics.map((topic, index) => (
              <TopicCard key={index} topic={topic} />
            ))}
          </div>
        </TabsContent>
        
        <TabsContent value="core" className="mt-0">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {allTopics.filter(topic => topic.category === "core").map((topic, index) => (
              <TopicCard key={index} topic={topic} />
            ))}
          </div>
        </TabsContent>
        
        <TabsContent value="applications" className="mt-0">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {allTopics.filter(topic => topic.category === "applications").map((topic, index) => (
              <TopicCard key={index} topic={topic} />
            ))}
          </div>
        </TabsContent>
        
        <TabsContent value="advanced" className="mt-0">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {allTopics.filter(topic => topic.category === "advanced").map((topic, index) => (
              <TopicCard key={index} topic={topic} />
            ))}
          </div>
        </TabsContent>
      </Tabs>
      
      <div className="mt-16 py-10 px-6 md:p-10 bg-muted rounded-xl text-center">
        <h2 className="text-2xl md:text-3xl font-bold mb-4">Özel İçerik Önerileri Mi İstiyorsunuz?</h2>
        <p className="text-muted-foreground mb-6 max-w-2xl mx-auto">
          İlgi alanlarınıza ve öğrenme hedeflerinize göre özelleştirilmiş içerik önerileri için bizimle iletişime geçin.
        </p>
        <Button asChild className="rounded-full">
          <Link href="/contact">İletişime Geçin</Link>
        </Button>
      </div>
    </div>
  );
}

function TopicCard({ topic }: { topic: any }) {
  return (
    <Card className="overflow-hidden transition-all duration-300 hover:shadow-lg hover:-translate-y-1">
      <div className="relative h-48">
        <Image 
          src={topic.imageUrl}
          alt={topic.title}
          fill
          className="object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent" />
        <div className="absolute bottom-4 left-4 p-2 rounded-full bg-background/80 backdrop-blur-sm">
          {topic.icon}
        </div>
      </div>
      <CardHeader>
        <CardTitle>{topic.title}</CardTitle>
        <CardDescription>{topic.description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-1">
          <p className="text-sm font-medium">Alt Konular:</p>
          <ul className="text-sm text-muted-foreground space-y-1">
            {topic.subtopics.map((subtopic: string, index: number) => (
              <li key={index} className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-primary flex-shrink-0" />
                {subtopic}
              </li>
            ))}
          </ul>
        </div>
      </CardContent>
      <CardFooter>
        <Button asChild variant="ghost" className="gap-1 ml-auto">
          <Link href={topic.href}>
            Daha Fazla
            <ArrowRight className="h-4 w-4" />
          </Link>
        </Button>
      </CardFooter>
    </Card>
  );
}