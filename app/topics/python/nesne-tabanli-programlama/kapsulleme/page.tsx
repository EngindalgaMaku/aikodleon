import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import Image from "next/image";
import { ArrowRight, Lock, Key, Shield, Eye } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python Kapsülleme (Encapsulation) | AIKOD',
  description: 'Python\'da kapsülleme (encapsulation) kavramı, private ve protected üyeler, getter ve setter metodları, property dekoratörü ve daha fazlası.',
};

const content = `
# Python'da Kapsülleme (Encapsulation)

Kapsülleme, nesne yönelimli programlamanın temel prensiplerinden biridir. Bu kavram, bir sınıfın içindeki verilerin ve metodların dış dünyadan gizlenmesini ve sadece belirlenen arayüzler üzerinden erişilmesini sağlar.

## Kapsülleme Nedir?

Kapsülleme, bir sınıfın içindeki verilerin ve bu veriler üzerinde işlem yapan metodların bir arada tutulması ve dış dünyadan gizlenmesi prensibidir. Bu sayede:

- Veri güvenliği sağlanır
- Kodun karmaşıklığı azalır
- Bakım kolaylaşır
- Sınıfın iç yapısı değiştiğinde dış kod etkilenmez

## Python'da Erişim Belirleyiciler

Python'da diğer dillerdeki gibi katı erişim belirleyiciler (private, protected, public) yoktur. Bunun yerine isimlendirme kuralları kullanılır:

\`\`\`python
class BankaHesabi:
    def __init__(self):
        self.bakiye = 0          # public
        self._limit = 1000       # protected
        self.__pin = "1234"      # private
\`\`\`

### Public Üyeler
- Herhangi bir özel işaret olmadan tanımlanan üyeler
- Dışarıdan doğrudan erişilebilir
- Örnek: \`self.bakiye\`

### Protected Üyeler
- Tek alt çizgi (_) ile başlayan üyeler
- Alt sınıflardan erişilebilir
- Dışarıdan erişim önerilmez
- Örnek: \`self._limit\`

### Private Üyeler
- Çift alt çizgi (__) ile başlayan üyeler
- Name mangling ile gizlenir
- Dışarıdan doğrudan erişilemez
- Örnek: \`self.__pin\`

## Getter ve Setter Metodları

Kapsülleme için sıkça kullanılan bir yöntem, private değişkenlere erişim için getter ve setter metodları tanımlamaktır:

\`\`\`python
class BankaHesabi:
    def __init__(self):
        self.__bakiye = 0
    
    # Getter metodu
    def get_bakiye(self):
        return self.__bakiye
    
    # Setter metodu
    def set_bakiye(self, miktar):
        if miktar >= 0:
            self.__bakiye = miktar
        else:
            raise ValueError("Bakiye negatif olamaz!")
\`\`\`

## Property Dekoratörü

Python'da \`@property\` dekoratörü, getter ve setter metodlarını daha elegant bir şekilde tanımlamamızı sağlar:

\`\`\`python
class BankaHesabi:
    def __init__(self):
        self.__bakiye = 0
    
    @property
    def bakiye(self):
        return self.__bakiye
    
    @bakiye.setter
    def bakiye(self, miktar):
        if miktar >= 0:
            self.__bakiye = miktar
        else:
            raise ValueError("Bakiye negatif olamaz!")

# Kullanımı
hesap = BankaHesabi()
hesap.bakiye = 1000  # setter çağrılır
print(hesap.bakiye)  # getter çağrılır
\`\`\`

## Name Mangling

Python'da private üyelerin nasıl gizlendiğini anlamak için name mangling kavramını bilmek önemlidir:

\`\`\`python
class Sinif:
    def __init__(self):
        self.__gizli = "Gizli veri"

nesne = Sinif()
print(nesne._Sinif__gizli)  # Name mangling ile erişim
\`\`\`

## Pratik Örnek: Öğrenci Sınıfı

Aşağıda kapsüllemenin pratik bir örneğini görebilirsiniz:

\`\`\`python
class Ogrenci:
    def __init__(self, ad, soyad):
        self.__ad = ad
        self.__soyad = soyad
        self.__notlar = []
    
    @property
    def ad_soyad(self):
        return f"{self.__ad} {self.__soyad}"
    
    @property
    def ortalama(self):
        if not self.__notlar:
            return 0
        return sum(self.__notlar) / len(self.__notlar)
    
    def not_ekle(self, not_):
        if 0 <= not_ <= 100:
            self.__notlar.append(not_)
        else:
            raise ValueError("Not 0-100 arasında olmalıdır!")

# Kullanımı
ogrenci = Ogrenci("Ahmet", "Yılmaz")
ogrenci.not_ekle(85)
ogrenci.not_ekle(90)
print(ogrenci.ad_soyad)  # Ahmet Yılmaz
print(ogrenci.ortalama)  # 87.5
\`\`\`

## İyi Kapsülleme Pratikleri

1. Veriyi her zaman private yapın ve gerektiğinde public metodlar üzerinden erişim sağlayın
2. Validation işlemlerini setter metodlarında yapın
3. Property dekoratörünü kullanarak temiz bir API sunun
4. Dokümantasyon yazın ve hangi metodların public API'nin parçası olduğunu belirtin
5. Gereksiz getter/setter metodları oluşturmaktan kaçının
`;

const sections = [
  {
    title: "Erişim Belirleyiciler",
    description: "Python'da public, protected ve private üyelerin kullanımı",
    icon: <Lock className="h-6 w-6" />,
    topics: [
      "Public üyeler",
      "Protected üyeler (_)",
      "Private üyeler (__)",
      "Name mangling"
    ]
  },
  {
    title: "Getter ve Setter",
    description: "Veri erişim ve değiştirme metodları",
    icon: <Key className="h-6 w-6" />,
    topics: [
      "Getter metodları",
      "Setter metodları",
      "Veri doğrulama",
      "Örnek uygulamalar"
    ]
  },
  {
    title: "Property Dekoratörü",
    description: "@property kullanımı ve avantajları",
    icon: <Shield className="h-6 w-6" />,
    topics: [
      "@property dekoratörü",
      "Getter property",
      "Setter property",
      "Computed properties"
    ]
  },
  {
    title: "İyi Pratikler",
    description: "Kapsülleme için önerilen yaklaşımlar",
    icon: <Eye className="h-6 w-6" />,
    topics: [
      "Veri gizleme",
      "API tasarımı",
      "Dokümantasyon",
      "Yaygın hatalar"
    ]
  }
];

export default function KapsullemePage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Concept Cards */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Önemli Kavramlar</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {sections.map((section, index) => (
              <Card key={index} className="bg-purple-50 hover:bg-purple-100 dark:bg-purple-950/50 dark:hover:bg-purple-950/70 transition-all duration-300">
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg text-purple-600 dark:text-purple-400">
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

        {/* Next Topic Button */}
        <div className="mt-12 flex justify-end">
          <Button asChild className="group">
            <Link href="/topics/python/nesne-tabanli-programlama/cok-bicimlilk">
              Sonraki Konu: Çok Biçimlilik
              <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 