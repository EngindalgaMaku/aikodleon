import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowRight, Factory, Layers, Brain, Code } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python OOP Tasarım Desenleri Örnekleri | Kodleon',
  description: 'Python nesne tabanlı programlama ile tasarım desenleri örnekleri ve detaylı açıklamalar.',
};

const content = `
# Tasarım Desenleri Örnekleri

Bu bölümde, nesne tabanlı programlamada sıkça kullanılan tasarım desenlerinin Python implementasyonlarını inceleyeceğiz. Her desen için gerçek dünya örnekleri ve kullanım senaryoları sunulacaktır.

## 1. Yaratımsal (Creational) Desenler

### Singleton Deseni

Bir sınıfın yalnızca bir örneğinin olmasını sağlar. Örneğin, bir veritabanı bağlantısı veya konfigürasyon yöneticisi için kullanılabilir.

\`\`\`python
class DatabaseConnection:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Veritabanı bağlantısı kurulur
            cls._instance.connected = True
        return cls._instance
    
    def query(self, sql: str) -> list:
        if self.connected:
            print(f"Executing query: {sql}")
            return []  # Gerçek uygulamada sorgu sonuçları döner

# Kullanım
db1 = DatabaseConnection()
db2 = DatabaseConnection()
print(db1 is db2)  # True - aynı örnek
\`\`\`

### Factory Method Deseni

Nesne oluşturma işlemini alt sınıflara devreder. Örneğin, farklı türde belge oluşturan bir sistem için kullanılabilir.

\`\`\`python
from abc import ABC, abstractmethod

class Document(ABC):
    @abstractmethod
    def create(self) -> str:
        pass

class PDFDocument(Document):
    def create(self) -> str:
        return "PDF belgesi oluşturuldu"

class WordDocument(Document):
    def create(self) -> str:
        return "Word belgesi oluşturuldu"

class DocumentFactory:
    @staticmethod
    def create_document(doc_type: str) -> Document:
        if doc_type.lower() == "pdf":
            return PDFDocument()
        elif doc_type.lower() == "word":
            return WordDocument()
        raise ValueError(f"Bilinmeyen belge tipi: {doc_type}")

# Kullanım
factory = DocumentFactory()
pdf_doc = factory.create_document("pdf")
word_doc = factory.create_document("word")
print(pdf_doc.create())   # PDF belgesi oluşturuldu
print(word_doc.create())  # Word belgesi oluşturuldu
\`\`\`

### Builder Deseni

Karmaşık nesnelerin adım adım oluşturulmasını sağlar. Örneğin, özelleştirilebilir bir rapor oluşturma sistemi için kullanılabilir.

\`\`\`python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Report:
    title: str
    header: Optional[str] = None
    content: List[str] = None
    footer: Optional[str] = None
    
    def display(self) -> str:
        parts = []
        if self.header:
            parts.append(f"=== {self.header} ===\\n")
        parts.append(f"# {self.title}\\n")
        if self.content:
            parts.extend(self.content)
        if self.footer:
            parts.append(f"\\n--- {self.footer} ---")
        return "\\n".join(parts)

class ReportBuilder:
    def __init__(self, title: str):
        self._report = Report(title=title, content=[])
    
    def add_header(self, header: str) -> "ReportBuilder":
        self._report.header = header
        return self
    
    def add_content(self, content: str) -> "ReportBuilder":
        if self._report.content is None:
            self._report.content = []
        self._report.content.append(content)
        return self
    
    def add_footer(self, footer: str) -> "ReportBuilder":
        self._report.footer = footer
        return self
    
    def build(self) -> Report:
        return self._report

# Kullanım
report = (ReportBuilder("Satış Raporu")
          .add_header("AYLIK RAPOR")
          .add_content("Toplam Satış: 150.000₺")
          .add_content("Müşteri Sayısı: 45")
          .add_footer("Rapor Sonu")
          .build())
print(report.display())
\`\`\`

## 2. Yapısal (Structural) Desenler

### Adapter Deseni

Uyumsuz arayüzleri birlikte çalışabilir hale getirir. Örneğin, farklı ödeme sistemlerini tek bir arayüz altında birleştirmek için kullanılabilir.

\`\`\`python
from abc import ABC, abstractmethod

class PaymentProcessor(ABC):
    @abstractmethod
    def process_payment(self, amount: float) -> bool:
        pass

class PayPalAPI:
    def send_payment(self, amount: float, currency: str = "USD") -> bool:
        print(f"PayPal ile {amount} {currency} ödeme yapıldı")
        return True

class StripeAPI:
    def charge(self, amount_cents: int) -> dict:
        amount = amount_cents / 100
        print(f"Stripe ile {amount} USD ödeme yapıldı")
        return {"status": "success", "amount": amount}

class PayPalAdapter(PaymentProcessor):
    def __init__(self, paypal: PayPalAPI):
        self.paypal = paypal
    
    def process_payment(self, amount: float) -> bool:
        return self.paypal.send_payment(amount)

class StripeAdapter(PaymentProcessor):
    def __init__(self, stripe: StripeAPI):
        self.stripe = stripe
    
    def process_payment(self, amount: float) -> bool:
        cents = int(amount * 100)
        result = self.stripe.charge(cents)
        return result["status"] == "success"

# Kullanım
paypal = PayPalAdapter(PayPalAPI())
stripe = StripeAdapter(StripeAPI())

paypal.process_payment(100.0)  # PayPal ile 100.0 USD ödeme yapıldı
stripe.process_payment(50.0)   # Stripe ile 50.0 USD ödeme yapıldı
\`\`\`

### Decorator Deseni

Nesnelere dinamik olarak yeni davranışlar ekler. Örneğin, bir metin işleme sisteminde farklı formatlamalar eklemek için kullanılabilir.

\`\`\`python
from abc import ABC, abstractmethod

class Text(ABC):
    @abstractmethod
    def render(self) -> str:
        pass

class PlainText(Text):
    def __init__(self, content: str):
        self._content = content
    
    def render(self) -> str:
        return self._content

class TextDecorator(Text):
    def __init__(self, text: Text):
        self._text = text
    
    def render(self) -> str:
        return self._text.render()

class BoldDecorator(TextDecorator):
    def render(self) -> str:
        return f"<b>{self._text.render()}</b>"

class ItalicDecorator(TextDecorator):
    def render(self) -> str:
        return f"<i>{self._text.render()}</i>"

class UnderlineDecorator(TextDecorator):
    def render(self) -> str:
        return f"<u>{self._text.render()}</u>"

# Kullanım
text = PlainText("Merhaba Dünya")
bold_text = BoldDecorator(text)
italic_bold_text = ItalicDecorator(bold_text)
decorated_text = UnderlineDecorator(italic_bold_text)

print(decorated_text.render())  # <u><i><b>Merhaba Dünya</b></i></u>
\`\`\`

## 3. Davranışsal (Behavioral) Desenler

### Observer Deseni

Nesneler arasında one-to-many bağımlılık ilişkisi kurar. Örneğin, bir haber yayın sistemi için kullanılabilir.

\`\`\`python
from abc import ABC, abstractmethod
from typing import List

class Observer(ABC):
    @abstractmethod
    def update(self, message: str):
        pass

class Subject(ABC):
    @abstractmethod
    def attach(self, observer: Observer):
        pass
    
    @abstractmethod
    def detach(self, observer: Observer):
        pass
    
    @abstractmethod
    def notify(self):
        pass

class NewsAgency(Subject):
    def __init__(self):
        self._observers: List[Observer] = []
        self._news: str = ""
    
    def attach(self, observer: Observer):
        self._observers.append(observer)
    
    def detach(self, observer: Observer):
        self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self._news)
    
    def add_news(self, news: str):
        self._news = news
        self.notify()

class NewsChannel(Observer):
    def __init__(self, name: str):
        self.name = name
    
    def update(self, message: str):
        print(f"{self.name} yayınlıyor: {message}")

# Kullanım
agency = NewsAgency()
channel1 = NewsChannel("Kanal D")
channel2 = NewsChannel("CNN")

agency.attach(channel1)
agency.attach(channel2)

agency.add_news("Önemli gelişme!")
# Kanal D yayınlıyor: Önemli gelişme!
# CNN yayınlıyor: Önemli gelişme!
\`\`\`

### Strategy Deseni

Algoritmaları kapsüller ve değiştirilebilir hale getirir. Örneğin, farklı indirim hesaplama stratejileri için kullanılabilir.

\`\`\`python
from abc import ABC, abstractmethod
from typing import List

class DiscountStrategy(ABC):
    @abstractmethod
    def calculate(self, amount: float) -> float:
        pass

class RegularDiscount(DiscountStrategy):
    def calculate(self, amount: float) -> float:
        return amount * 0.1  # %10 indirim

class PremiumDiscount(DiscountStrategy):
    def calculate(self, amount: float) -> float:
        return amount * 0.2  # %20 indirim

class VIPDiscount(DiscountStrategy):
    def calculate(self, amount: float) -> float:
        return amount * 0.3  # %30 indirim

class ShoppingCart:
    def __init__(self, discount_strategy: DiscountStrategy):
        self.items: List[float] = []
        self.discount_strategy = discount_strategy
    
    def add_item(self, price: float):
        self.items.append(price)
    
    def calculate_total(self) -> float:
        amount = sum(self.items)
        discount = self.discount_strategy.calculate(amount)
        return amount - discount

# Kullanım
regular_cart = ShoppingCart(RegularDiscount())
regular_cart.add_item(100)
regular_cart.add_item(50)
print(f"Regular total: {regular_cart.calculate_total()}")  # 135.0

vip_cart = ShoppingCart(VIPDiscount())
vip_cart.add_item(100)
vip_cart.add_item(50)
print(f"VIP total: {vip_cart.calculate_total()}")  # 105.0
\`\`\`

## İyi Pratikler

1. **SOLID Prensipleri**
   - Single Responsibility: Her sınıf tek bir sorumluluğa sahip olmalı
   - Open/Closed: Genişletmeye açık, değişikliğe kapalı olmalı
   - Liskov Substitution: Alt sınıflar üst sınıfların yerine geçebilmeli
   - Interface Segregation: Arayüzler küçük ve özel olmalı
   - Dependency Inversion: Yüksek seviye modüller düşük seviye modüllere bağımlı olmamalı

2. **Tasarım Deseni Seçimi**
   - İhtiyaca uygun desen seçilmeli
   - Aşırı karmaşıklıktan kaçınılmalı
   - Kodun okunabilirliği korunmalı
   - Performans göz önünde bulundurulmalı

3. **Kod Organizasyonu**
   - Modüler yapı kullanılmalı
   - Tekrar kullanılabilir bileşenler oluşturulmalı
   - Uygun isimlendirme yapılmalı
   - Dokümantasyon eklenmelı
`;

const sections = [
  {
    title: "Yaratımsal Desenler",
    description: "Nesne oluşturma mekanizmaları",
    icon: <Factory className="h-6 w-6" />,
    topics: [
      "Singleton",
      "Factory Method",
      "Builder",
      "Abstract Factory"
    ]
  },
  {
    title: "Yapısal Desenler",
    description: "Sınıf ve nesne kompozisyonu",
    icon: <Layers className="h-6 w-6" />,
    topics: [
      "Adapter",
      "Decorator",
      "Composite",
      "Bridge"
    ]
  },
  {
    title: "Davranışsal Desenler",
    description: "Nesneler arası iletişim",
    icon: <Brain className="h-6 w-6" />,
    topics: [
      "Observer",
      "Strategy",
      "Command",
      "State"
    ]
  },
  {
    title: "İyi Pratikler",
    description: "Tasarım desenleri kullanım prensipleri",
    icon: <Code className="h-6 w-6" />,
    topics: [
      "SOLID Prensipleri",
      "Desen Seçimi",
      "Kod Organizasyonu",
      "Performans"
    ]
  }
];

export default function TasarimDesenleriPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Feature Cards */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Desen Kategorileri</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {sections.map((section, index) => (
              <Card key={index} className="bg-yellow-50 hover:bg-yellow-100 dark:bg-yellow-950/50 dark:hover:bg-yellow-950/70 transition-all duration-300">
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg text-yellow-600 dark:text-yellow-400">
                      {section.icon}
                    </div>
                    <CardTitle>{section.title}</CardTitle>
                  </div>
                  <CardDescription className="dark:text-gray-300">{section.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground dark:text-gray-400">
                    {section.topics.map((topic, i) => (
                      <li key={i}>{topic}</li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Back to Examples Button */}
        <div className="mt-12 flex justify-end">
          <Button asChild variant="outline" className="group">
            <Link href="/topics/python/nesne-tabanli-programlama/pratik-ornekler">
              Örneklere Dön
              <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 