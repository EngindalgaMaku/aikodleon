import Image from "next/image";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Brain, BookOpen, Award, Users, Lightbulb, Rocket } from "lucide-react";

export default function AboutPage() {
  return (
    <div className="flex flex-col">
      {/* Hero section */}
      <section className="relative py-20 md:py-32 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-primary/10 to-secondary/10 dark:from-primary/5 dark:to-secondary/5" />
        <div 
          className="absolute inset-0 opacity-30 dark:opacity-20"
          style={{
            backgroundImage: 'url("data:image/svg+xml,%3Csvg width="20" height="20" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="%239C92AC" fill-opacity="0.2"%3E%3Cpath d="M0 0h20L0 20z"/%3E%3C/g%3E%3C/svg%3E")',
            backgroundSize: '20px 20px'
          }}
        />
        <div className="container relative z-10">
          <div className="max-w-3xl mx-auto text-center">
            <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6">
              Hakkımızda
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground mb-8">
              Yapay zeka eğitiminde lider olan ekibimiz ve misyonumuzla tanışın.
            </p>
          </div>
        </div>
      </section>
      
      {/* Mission section */}
      <section className="py-16">
        <div className="container">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div className="relative aspect-square rounded-2xl overflow-hidden">
              <Image 
                src="https://images.pexels.com/photos/3183150/pexels-photo-3183150.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
                alt="Our mission"
                fill
                className="object-cover"
              />
            </div>
            
            <div>
              <div className="inline-flex items-center gap-2 bg-primary/10 rounded-full px-4 py-1 mb-6">
                <Brain className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">Misyonumuz</span>
              </div>
              <h2 className="text-3xl font-bold tracking-tight mb-6">
                Yapay Zeka Eğitimini Herkes İçin Erişilebilir Kılmak
              </h2>
              <p className="text-lg text-muted-foreground mb-6">
                AI Eğitim platformu olarak, yapay zeka teknolojilerini öğrenmek isteyen herkese kapsamlı, anlaşılır ve güncel eğitimler sunmayı amaçlıyoruz. Misyonumuz, yapay zeka eğitimini demokratikleştirerek daha geniş kitlelere ulaştırmak ve geleceğin teknoloji liderlerini yetiştirmektir.
              </p>
              <p className="text-lg text-muted-foreground mb-6">
                Eğitim içeriklerimizi, sektör profesyonelleri ve akademisyenlerle iş birliği içinde hazırlıyor, teorik bilgiyi pratik uygulamalarla destekliyoruz. Öğrencilerimizin gerçek dünya problemlerini çözebilecek becerilere sahip olmalarını sağlamak için sürekli kendimizi yeniliyoruz.
              </p>
              <div className="grid grid-cols-2 gap-4 mt-8">
                <div className="flex flex-col items-center p-4 bg-muted rounded-lg">
                  <span className="text-3xl font-bold">1000+</span>
                  <span className="text-sm text-muted-foreground">Öğrenci</span>
                </div>
                <div className="flex flex-col items-center p-4 bg-muted rounded-lg">
                  <span className="text-3xl font-bold">50+</span>
                  <span className="text-sm text-muted-foreground">Eğitim İçeriği</span>
                </div>
                <div className="flex flex-col items-center p-4 bg-muted rounded-lg">
                  <span className="text-3xl font-bold">15+</span>
                  <span className="text-sm text-muted-foreground">Uzman Eğitmen</span>
                </div>
                <div className="flex flex-col items-center p-4 bg-muted rounded-lg">
                  <span className="text-3xl font-bold">8+</span>
                  <span className="text-sm text-muted-foreground">Yıllık Deneyim</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
      
      {/* Values section */}
      <section className="py-16 bg-muted/50">
        <div className="container">
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 bg-primary/10 rounded-full px-4 py-1 mb-6">
              <Lightbulb className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium">Değerlerimiz</span>
            </div>
            <h2 className="text-3xl font-bold tracking-tight mb-4">Neden Biz?</h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Eğitim yaklaşımımızı şekillendiren temel değerlerimiz ve bizi farklı kılan özelliklerimiz.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="bg-card rounded-xl p-6 shadow-sm">
              <div className="p-3 rounded-full bg-primary/10 w-fit mb-4">
                <BookOpen className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-bold mb-3">Kapsamlı İçerik</h3>
              <p className="text-muted-foreground">
                Temel kavramlardan ileri düzey uygulamalara kadar yapay zekanın tüm alanlarını kapsayan içerikler sunuyoruz.
              </p>
            </div>
            
            <div className="bg-card rounded-xl p-6 shadow-sm">
              <div className="p-3 rounded-full bg-primary/10 w-fit mb-4">
                <Award className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-bold mb-3">Uzman Eğitmenler</h3>
              <p className="text-muted-foreground">
                Akademi ve endüstri deneyimine sahip uzman eğitmenlerle çalışıyor, güncel ve pratik bilgiler sunuyoruz.
              </p>
            </div>
            
            <div className="bg-card rounded-xl p-6 shadow-sm">
              <div className="p-3 rounded-full bg-primary/10 w-fit mb-4">
                <Rocket className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-bold mb-3">Uygulamalı Öğrenim</h3>
              <p className="text-muted-foreground">
                Teorik bilgiyi pratik uygulamalarla pekiştirerek öğrencilerin gerçek dünya problemlerini çözmelerini sağlıyoruz.
              </p>
            </div>
            
            <div className="bg-card rounded-xl p-6 shadow-sm">
              <div className="p-3 rounded-full bg-primary/10 w-fit mb-4">
                <Users className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-bold mb-3">Topluluk Desteği</h3>
              <p className="text-muted-foreground">
                Öğrenciler ve eğitmenler arasında sürekli etkileşim sağlayan canlı bir öğrenme topluluğu oluşturuyoruz.
              </p>
            </div>
            
            <div className="bg-card rounded-xl p-6 shadow-sm">
              <div className="p-3 rounded-full bg-primary/10 w-fit mb-4">
                <Brain className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-bold mb-3">Sürekli Güncelleme</h3>
              <p className="text-muted-foreground">
                Yapay zeka alanındaki hızlı gelişmeleri takip ederek içeriklerimizi sürekli güncelliyoruz.
              </p>
            </div>
            
            <div className="bg-card rounded-xl p-6 shadow-sm">
              <div className="p-3 rounded-full bg-primary/10 w-fit mb-4">
                <Lightbulb className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-bold mb-3">İnovasyon Odaklı</h3>
              <p className="text-muted-foreground">
                Yenilikçi öğretim yöntemleri ve teknolojileri kullanarak eğitim deneyimini sürekli iyileştiriyoruz.
              </p>
            </div>
          </div>
        </div>
      </section>
      
      {/* Team section */}
      <section className="py-16">
        <div className="container">
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 bg-primary/10 rounded-full px-4 py-1 mb-6">
              <Users className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium">Ekibimiz</span>
            </div>
            <h2 className="text-3xl font-bold tracking-tight mb-4">Uzman Kadromuz</h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Eğitim içeriklerimizi hazırlayan ve sürekli geliştiren uzman ekibimizle tanışın.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                name: "Dr. Ahmet Yılmaz",
                role: "Kurucu & Eğitim Direktörü",
                bio: "Makine öğrenmesi ve derin öğrenme alanlarında 10+ yıl deneyim.",
                image: "https://images.pexels.com/photos/2379004/pexels-photo-2379004.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
              },
              {
                name: "Ayşe Demir",
                role: "Doğal Dil İşleme Uzmanı",
                bio: "NLP ve büyük dil modelleri konusunda uzman akademisyen.",
                image: "https://images.pexels.com/photos/1181686/pexels-photo-1181686.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
              },
              {
                name: "Mehmet Kaya",
                role: "Bilgisayarlı Görü Uzmanı",
                bio: "Bilgisayarlı görü ve görüntü işleme konularında endüstri deneyimi.",
                image: "https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
              },
              {
                name: "Zeynep Aksoy",
                role: "AI Etiği Danışmanı",
                bio: "Yapay zeka etiği ve düzenlemeleri konusunda uzmanlaşmış hukuk danışmanı.",
                image: "https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
              }
            ].map((member, index) => (
              <div key={index} className="group relative overflow-hidden rounded-xl">
                <div className="aspect-[3/4] relative">
                  <Image 
                    src={member.image}
                    alt={member.name}
                    fill
                    className="object-cover transition-transform group-hover:scale-105"
                  />
                </div>
                <div className="absolute inset-0 bg-gradient-to-t from-background via-background/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-end p-6">
                  <h3 className="text-xl font-bold mb-1">{member.name}</h3>
                  <p className="text-primary font-medium mb-2">{member.role}</p>
                  <p className="text-sm text-muted-foreground">{member.bio}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
      
      {/* CTA section */}
      <section className="py-16 bg-primary text-primary-foreground">
        <div className="container">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-3xl font-bold tracking-tight mb-4">
              Yapay Zeka Yolculuğunuza Başlamaya Hazır Mısınız?
            </h2>
            <p className="text-xl mb-8 text-primary-foreground/80">
              Kapsamlı eğitim içeriklerimizle yapay zeka dünyasını keşfedin ve geleceğin teknolojilerine hakim olun.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button asChild size="lg" variant="secondary" className="rounded-full">
                <Link href="/topics">Konuları Keşfet</Link>
              </Button>
              <Button asChild size="lg" variant="outline" className="rounded-full border-primary-foreground/20 hover:bg-primary-foreground/10">
                <Link href="/contact">İletişime Geç</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}