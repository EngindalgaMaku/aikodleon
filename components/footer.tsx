import Link from "next/link";
import { Brain } from "lucide-react";

export default function Footer() {
  return (
    <footer className="w-full border-t bg-background">
      <div className="container max-w-6xl mx-auto py-10">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="flex flex-col space-y-4">
            <Link href="/" className="flex items-center gap-2">
              <Brain className="h-6 w-6 text-primary" />
              <span className="font-bold text-xl">AI Eğitim</span>
            </Link>
            <p className="text-sm text-muted-foreground">
              Yapay zeka dünyasını keşfedin ve geleceğin teknolojilerini öğrenin.
            </p>
          </div>
          
          <div>
            <h3 className="font-medium text-lg mb-4">Konular</h3>
            <ul className="space-y-2">
              <li>
                <Link href="/topics/machine-learning" className="text-sm text-muted-foreground hover:text-primary transition-colors">
                  Makine Öğrenmesi
                </Link>
              </li>
              <li>
                <Link href="/topics/nlp" className="text-sm text-muted-foreground hover:text-primary transition-colors">
                  Doğal Dil İşleme
                </Link>
              </li>
              <li>
                <Link href="/topics/computer-vision" className="text-sm text-muted-foreground hover:text-primary transition-colors">
                  Bilgisayarlı Görü
                </Link>
              </li>
              <li>
                <Link href="/topics/generative-ai" className="text-sm text-muted-foreground hover:text-primary transition-colors">
                  Üretken AI
                </Link>
              </li>
            </ul>
          </div>
          
          <div>
            <h3 className="font-medium text-lg mb-4">Kaynaklar</h3>
            <ul className="space-y-2">
              <li>
                <Link href="/blog" className="text-sm text-muted-foreground hover:text-primary transition-colors">
                  Blog
                </Link>
              </li>
              <li>
                <Link href="/resources" className="text-sm text-muted-foreground hover:text-primary transition-colors">
                  Eğitim Materyalleri
                </Link>
              </li>
              <li>
                <Link href="/faq" className="text-sm text-muted-foreground hover:text-primary transition-colors">
                  Sık Sorulan Sorular
                </Link>
              </li>
            </ul>
          </div>
          
          <div>
            <h3 className="font-medium text-lg mb-4">İletişim</h3>
            <ul className="space-y-2">
              <li>
                <Link href="/contact" className="text-sm text-muted-foreground hover:text-primary transition-colors">
                  Bize Ulaşın
                </Link>
              </li>
              <li>
                <Link href="/about" className="text-sm text-muted-foreground hover:text-primary transition-colors">
                  Hakkımızda
                </Link>
              </li>
            </ul>
          </div>
        </div>
        
        <div className="border-t mt-8 pt-6 flex flex-col sm:flex-row justify-between items-center">
          <p className="text-sm text-muted-foreground">
            &copy; {new Date().getFullYear()} AI Eğitim. Tüm hakları saklıdır.
          </p>
          <div className="flex gap-4 mt-4 sm:mt-0">
            <Link href="#" className="text-sm text-muted-foreground hover:text-primary transition-colors">
              Gizlilik Politikası
            </Link>
            <Link href="#" className="text-sm text-muted-foreground hover:text-primary transition-colors">
              Kullanım Şartları
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
}