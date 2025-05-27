import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

const faqs = [
  {
    question: "Eğitimlere başlamak için ön koşul var mı?",
    answer: "Temel programlama bilgisi ve matematik anlayışı yeterlidir. Her konu için gerekli ön bilgiler ilgili eğitim sayfasında belirtilmiştir."
  },
  {
    question: "Eğitimler ne kadar sürüyor?",
    answer: "Eğitim süreleri konuya göre değişmektedir. Temel konular 4-6 hafta, ileri düzey konular 8-12 hafta sürebilmektedir. Kendi hızınızda ilerleyebilirsiniz."
  },
  {
    question: "Eğitimler canlı mı veriliyor?",
    answer: "Eğitimlerimiz hem kayıtlı video dersler hem de haftalık canlı oturumlar şeklinde yapılmaktadır. Canlı oturumlarda sorularınızı sorabilir ve uygulamalı örnekler yapabilirsiniz."
  },
  {
    question: "Eğitim materyallerine ne kadar süre erişebilirim?",
    answer: "Satın aldığınız eğitimlerin materyallerine süresiz erişim hakkına sahip olursunuz. İçerikler düzenli olarak güncellenmektedir."
  },
  {
    question: "Sertifika veriliyor mu?",
    answer: "Evet, eğitimi başarıyla tamamlayan katılımcılara dijital sertifika verilmektedir. Sertifikalar blockchain üzerinde doğrulanabilir."
  },
  {
    question: "Ödeme seçenekleri nelerdir?",
    answer: "Kredi kartı, banka havalesi ve kripto para ile ödeme kabul edilmektedir. Ayrıca taksit seçenekleri de mevcuttur."
  },
  {
    question: "Grup indirimleri var mı?",
    answer: "Evet, kurumsal eğitimler ve 5+ kişilik grup kayıtları için özel indirimler sunulmaktadır. İletişim sayfasından bizimle iletişime geçebilirsiniz."
  },
  {
    question: "İade politikanız nedir?",
    answer: "İlk 14 gün içinde, eğitimin %25'inden fazlasını tamamlamamış iseniz iade talep edebilirsiniz. İade talepleri 5 iş günü içinde değerlendirilir."
  }
];

export default function FAQPage() {
  return (
    <div className="container py-12">
      <div className="max-w-3xl mx-auto text-center mb-12">
        <h1 className="text-4xl font-bold tracking-tight mb-4">Sık Sorulan Sorular</h1>
        <p className="text-xl text-muted-foreground">
          Eğitimlerimiz hakkında sık sorulan soruları ve cevaplarını burada bulabilirsiniz.
        </p>
      </div>
      
      <div className="max-w-3xl mx-auto">
        <Accordion type="single" collapsible className="w-full">
          {faqs.map((faq, index) => (
            <AccordionItem key={index} value={`item-${index}`}>
              <AccordionTrigger className="text-left">
                {faq.question}
              </AccordionTrigger>
              <AccordionContent>
                {faq.answer}
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </div>
    </div>
  );
}