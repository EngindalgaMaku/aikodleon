import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Bot, GraduationCap, ShieldCheck, Users, BookOpen, MessageCircle, Zap, CreditCard, Lock } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";

const faqs = [
  {
    icon: <Bot className="h-6 w-6 text-primary" />, 
    question: "Kodleon platformunda hangi yapay zeka araçları mevcut?",
    answer: "Kodleon, makine öğrenmesi, derin öğrenme, doğal dil işleme ve veri bilimi için interaktif eğitimler, uygulama araçları ve projeler sunar. Ayrıca, kendi AI modellerinizi geliştirebileceğiniz bir çalışma alanı sağlar."
  },
  {
    icon: <BookOpen className="h-6 w-6 text-primary" />, 
    question: "Eğitim içerikleri kimler için uygundur?",
    answer: "Başlangıç seviyesinden ileri düzeye kadar herkes için içerik sunuyoruz. Temel programlama bilgisi önerilir, ancak sıfırdan başlayanlar için de rehberlerimiz mevcut."
  },
  {
    icon: <GraduationCap className="h-6 w-6 text-primary" />, 
    question: "Sertifika alabilir miyim?",
    answer: "Evet, her eğitimi başarıyla tamamlayanlara blockchain doğrulamalı dijital sertifika veriyoruz. Sertifikalarınızı LinkedIn'de paylaşabilirsiniz."
  },
  {
    icon: <Zap className="h-6 w-6 text-primary" />, 
    question: "Canlı destek ve mentorluk var mı?",
    answer: "Haftalık canlı soru-cevap oturumları, topluluk forumu ve birebir mentorluk imkanlarımız mevcuttur. Sorularınızı uzman eğitmenlere iletebilirsiniz."
  },
  {
    icon: <Users className="h-6 w-6 text-primary" />, 
    question: "Topluluğa nasıl katılabilirim?",
    answer: "Kodleon topluluğuna ücretsiz katılabilir, diğer katılımcılarla projeler geliştirebilir ve deneyimlerinizi paylaşabilirsiniz. Katılım için profilinizi oluşturmanız yeterli."
  },
  {
    icon: <CreditCard className="h-6 w-6 text-primary" />, 
    question: "Ödeme ve abonelik seçenekleri nelerdir?",
    answer: "Kredi kartı, banka havalesi ve kripto para ile ödeme yapabilirsiniz. Aylık ve yıllık abonelik seçenekleri ile esnek ödeme imkanları sunuyoruz."
  },
  {
    icon: <ShieldCheck className="h-6 w-6 text-primary" />, 
    question: "Veri gizliliği ve güvenliği nasıl sağlanıyor?",
    answer: "Kullanıcı verileriniz şifrelenerek saklanır ve üçüncü şahıslarla paylaşılmaz. Platformumuz GDPR ve KVKK uyumludur."
  },
  {
    icon: <Lock className="h-6 w-6 text-primary" />, 
    question: "Eğitim materyallerine erişim süresi nedir?",
    answer: "Satın aldığınız eğitimlere süresiz erişim hakkınız olur. İçerikler düzenli olarak güncellenir ve yeni kaynaklar eklenir."
  },
  {
    icon: <MessageCircle className="h-6 w-6 text-primary" />, 
    question: "Daha fazla sorum olursa ne yapmalıyım?",
    answer: "Bize iletişim sayfasından ulaşabilir veya topluluk forumunda yeni bir başlık açabilirsiniz. Size en kısa sürede dönüş yaparız."
  }
];

export default function FAQPage() {
  return (
    <div className="container max-w-4xl mx-auto py-12">
      <div className="max-w-2xl mx-auto text-center mb-12">
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-4 text-primary">Sık Sorulan Sorular</h1>
        <p className="text-lg md:text-xl text-muted-foreground">
          Kodleon platformu, eğitimler ve topluluk hakkında en çok merak edilenleri burada bulabilirsiniz.
        </p>
      </div>
      <div className="space-y-4">
        <Accordion type="single" collapsible className="w-full">
          {faqs.map((faq, index) => (
            <AccordionItem key={index} value={`item-${index}`} className="bg-card/80 border border-border rounded-xl shadow-md">
              <AccordionTrigger className="flex items-center gap-3 text-left text-lg font-semibold px-6 py-4">
                {faq.icon}
                <span>{faq.question}</span>
              </AccordionTrigger>
              <AccordionContent className="px-6 pb-6 text-base text-muted-foreground">
                {faq.answer}
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </div>
      <div className="mt-12 text-center">
        <p className="text-lg mb-4">Sorunuzun cevabını bulamadınız mı?</p>
        <Button asChild size="lg" className="rounded-full shadow-lg text-lg px-8 py-6">
          <Link href="/contact">
            İletişime Geçin
          </Link>
        </Button>
      </div>
    </div>
  );
}