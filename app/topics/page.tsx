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
    subtopics: [
      { title: "Denetimli Öğrenme", href: "/topics/machine-learning/supervised-learning" },
      { title: "Denetimsiz Öğrenme", href: "/topics/machine-learning/unsupervised-learning" },
      { title: "Pekiştirmeli Öğrenme", href: "/topics/machine-learning/reinforcement-learning" },
      { title: "Derin Öğrenme Temelleri", href: "/topics/machine-learning/deep-learning-basics" }
    ]
  },
  {
    title: "Doğal Dil İşleme",
    description: "Makinelerin insan dilini nasıl anlayıp işlediğini ve ürettiğini öğrenin.",
    icon: <FileText className="h-8 w-8 text-chart-2" />,
    href: "/topics/nlp",
    imageUrl: "https://images.pexels.com/photos/7412095/pexels-photo-7412095.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "core",
    subtopics: [
      { title: "Metin Ön İşleme", href: "/topics/nlp/text-preprocessing" },
      { title: "Metin Analizi", href: "/topics/nlp/text-analysis" },
      { title: "Dil Modelleri", href: "/topics/nlp/language-models" },
      { title: "Duygu Analizi", href: "/topics/nlp/sentiment-analysis" },
      { title: "Makine Çevirisi", href: "/topics/nlp/machine-translation" }
    ]
  },
  {
    title: "Bilgisayarlı Görü",
    description: "Bilgisayarların görüntüleri nasıl algıladığını ve işlediğini anlayın.",
    icon: <Eye className="h-8 w-8 text-chart-3" />,
    href: "/topics/computer-vision",
    imageUrl: "https://images.pexels.com/photos/8438922/pexels-photo-8438922.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "core",
    subtopics: [
      { title: "Görüntü Sınıflandırma", href: "/topics/computer-vision/image-classification" },
      { title: "Nesne Tespiti", href: "/topics/computer-vision/object-detection" },
      { title: "Yüz Tanıma", href: "/topics/computer-vision/face-recognition" },
      { title: "Görüntü Segmentasyonu", href: "/topics/computer-vision/image-segmentation" }
    ]
  },
  {
    title: "Üretken AI",
    description: "Metin, görüntü ve ses üretebilen yapay zeka modellerini keşfedin.",
    icon: <Lightbulb className="h-8 w-8 text-chart-4" />,
    href: "/topics/generative-ai",
    imageUrl: "https://images.pexels.com/photos/8386434/pexels-photo-8386434.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "advanced",
    subtopics: [
      { title: "Üretken Çekişmeli Ağlar (GAN)", href: "/topics/generative-ai/gans" },
      { title: "Diffusion Modelleri", href: "/topics/generative-ai/diffusion-models" },
      { title: "Büyük Dil Modelleri", href: "/topics/generative-ai/large-language-models" },
      { title: "Text-to-Image Modelleri", href: "/topics/generative-ai/text-to-image" }
    ]
  },
  {
    title: "Sinir Ağları",
    description: "Beynin çalışma prensibinden esinlenen yapay sinir ağları hakkında bilgi edinin.",
    icon: <Brain className="h-8 w-8 text-chart-5" />,
    href: "/topics/neural-networks",
    imageUrl: "https://images.pexels.com/photos/8386421/pexels-photo-8386421.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "core",
    subtopics: [
      { title: "Temel Sinir Ağı Mimarileri", href: "/topics/neural-networks/basic-architectures" },
      { title: "Konvolüsyonel Sinir Ağları (CNN)", href: "/topics/neural-networks/cnns" },
      { title: "Tekrarlayan Sinir Ağları (RNN)", href: "/topics/neural-networks/rnns" },
      { title: "Transformerlar", href: "/topics/neural-networks/transformers" }
    ]
  },
  {
    title: "AI Etiği",
    description: "Yapay zekanın etik kullanımı ve toplumsal etkileri üzerine tartışmalar.",
    icon: <Users className="h-8 w-8 text-chart-1" />,
    href: "/topics/ai-ethics",
    imageUrl: "https://images.pexels.com/photos/8386422/pexels-photo-8386422.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "applications",
    subtopics: [
      { title: "Adil ve Tarafsız AI", href: "/topics/ai-ethics/fairness-bias" },
      { title: "Gizlilik ve Güvenlik", href: "/topics/ai-ethics/privacy-security" },
      { title: "AI Düzenlemeleri", href: "/topics/ai-ethics/regulations" },
      { title: "Etik Yapay Zeka Tasarımı", href: "/topics/ai-ethics/ethical-design" }
    ]
  },
  {
    title: "Robotik",
    description: "Yapay zekanın robotik sistemlerde uygulanmasını ve robotların dünyayla etkileşimini öğrenin.",
    icon: <Chip className="h-8 w-8 text-chart-2" />,
    href: "/topics/robotics",
    imageUrl: "https://images.pexels.com/photos/2599244/pexels-photo-2599244.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "applications",
    subtopics: [
      { title: "Robot Hareket Planlaması", href: "/topics/robotics/motion-planning" },
      { title: "Sensör Füzyonu", href: "/topics/robotics/sensor-fusion" },
      { title: "İnsan-Robot Etkileşimi", href: "/topics/robotics/human-robot-interaction" },
      { title: "Otonom Sistemler", href: "/topics/robotics/autonomous-systems" }
    ]
  },
  {
    title: "AI'nin Geleceği",
    description: "Yapay zekanın gelecekteki yönelimlerini ve gelişimini inceleyin.",
    icon: <Rocket className="h-8 w-8 text-chart-3" />,
    href: "/topics/future-of-ai",
    imageUrl: "https://images.pexels.com/photos/2007647/pexels-photo-2007647.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "advanced",
    subtopics: [
      { title: "Genel Yapay Zeka (AGI)", href: "/topics/future-of-ai/agi" },
      { title: "Süper Zeka", href: "/topics/future-of-ai/superintelligence" },
      { title: "Kuantum AI", href: "/topics/future-of-ai/quantum-ai" },
      { title: "Beyin-Bilgisayar Arayüzleri", href: "/topics/future-of-ai/bci" }
    ]
  },
  {
    title: "AI ve Veri Bilimi",
    description: "Veri bilimi ve yapay zeka arasındaki ilişkiyi ve veri analizinde AI kullanımını keşfedin.",
    icon: <Shapes className="h-8 w-8 text-chart-4" />,
    href: "/topics/ai-data-science",
    imageUrl: "https://images.pexels.com/photos/669615/pexels-photo-669615.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    category: "applications",
    subtopics: [
      { title: "Büyük Veri Analizi", href: "/topics/ai-data-science/big-data-analytics" },
      { title: "Tahmine Dayalı Modelleme", href: "/topics/ai-data-science/predictive-modeling" },
      { title: "Anomali Tespiti", href: "/topics/ai-data-science/anomaly-detection" },
      { title: "Veri Görselleştirme", href: "/topics/ai-data-science/data-visualization" }
    ]
  },
];

export default function TopicsPage() {
  return (
    <div className="container max-w-6xl mx-auto py-12">
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
          <div className="w-full flex justify-center">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {allTopics.map((topic, index) => (
                <TopicCard key={index} topic={topic} />
              ))}
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="core" className="mt-0">
          <div className="w-full flex justify-center">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {allTopics.filter(topic => topic.category === "core").map((topic, index) => (
                <TopicCard key={index} topic={topic} />
              ))}
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="applications" className="mt-0">
          <div className="w-full flex justify-center">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {allTopics.filter(topic => topic.category === "applications").map((topic, index) => (
                <TopicCard key={index} topic={topic} />
              ))}
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="advanced" className="mt-0">
          <div className="w-full flex justify-center">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {allTopics.filter(topic => topic.category === "advanced").map((topic, index) => (
                <TopicCard key={index} topic={topic} />
              ))}
            </div>
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
    <Card className="w-80 overflow-hidden transition-all duration-300 hover:shadow-lg hover:-translate-y-1">
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
            {topic.subtopics.map((subtopic: { title: string; href: string }, index: number) => (
              <li key={index} className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-primary flex-shrink-0" />
                <Link href={subtopic.href} className="hover:underline">
                  {subtopic.title}
                </Link>
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