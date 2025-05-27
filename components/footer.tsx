import Link from "next/link";
import { Brain } from "lucide-react";

export default function Footer() {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="w-full border-t bg-background" aria-labelledby="footer-heading">
      <h2 id="footer-heading" className="sr-only">Kodleon site alt bölümü</h2>
      <div className="container max-w-6xl mx-auto py-10">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="flex flex-col space-y-4">
            <Link href="/" className="flex items-center gap-2" aria-label="Kodleon ana sayfaya dön">
              <Brain className="h-6 w-6 text-primary" aria-hidden="true" />
              <span className="font-bold text-xl">Kodleon</span>
            </Link>
            <p className="text-sm text-muted-foreground">
              Yapay zeka dünyasını keşfedin ve geleceğin teknolojilerini öğrenin.
            </p>
            <p className="text-sm text-muted-foreground">
              <a href="mailto:info@kodleon.com" className="hover:text-primary transition-colors">info@kodleon.com</a>
            </p>
          </div>
          
          <nav aria-labelledby="footer-konular">
            <h3 id="footer-konular" className="font-medium text-lg mb-4">Konular</h3>
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
          </nav>
          
          <nav aria-labelledby="footer-kaynaklar">
            <h3 id="footer-kaynaklar" className="font-medium text-lg mb-4">Kaynaklar</h3>
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
          </nav>
          
          <nav aria-labelledby="footer-iletisim">
            <h3 id="footer-iletisim" className="font-medium text-lg mb-4">İletişim</h3>
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
              <li className="flex gap-4 mt-4">
                <a href="https://twitter.com/kodleon" target="_blank" rel="noopener noreferrer" aria-label="Kodleon Twitter sayfası" className="text-muted-foreground hover:text-primary transition-colors">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M22 4s-.7 2.1-2 3.4c1.6 10-9.4 17.3-18 11.6 2.2.1 4.4-.6 6-2C3 15.5.5 9.6 3 5c2.2 2.6 5.6 4.1 9 4-.9-4.2 4-6.6 7-3.8 1.1 0 3-1.2 3-1.2z"></path></svg>
                </a>
                <a href="https://www.linkedin.com/company/kodleon" target="_blank" rel="noopener noreferrer" aria-label="Kodleon LinkedIn sayfası" className="text-muted-foreground hover:text-primary transition-colors">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path><rect x="2" y="9" width="4" height="12"></rect><circle cx="4" cy="4" r="2"></circle></svg>
                </a>
                <a href="https://www.youtube.com/kodleon" target="_blank" rel="noopener noreferrer" aria-label="Kodleon YouTube kanalı" className="text-muted-foreground hover:text-primary transition-colors">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M22.54 6.42a2.78 2.78 0 0 0-1.94-2C18.88 4 12 4 12 4s-6.88 0-8.6.46a2.78 2.78 0 0 0-1.94 2A29 29 0 0 0 1 11.75a29 29 0 0 0 .46 5.33A2.78 2.78 0 0 0 3.4 19c1.72.46 8.6.46 8.6.46s6.88 0 8.6-.46a2.78 2.78 0 0 0 1.94-2 29 29 0 0 0 .46-5.25 29 29 0 0 0-.46-5.33z"></path><polygon points="9.75 15.02 15.5 11.75 9.75 8.48 9.75 15.02"></polygon></svg>
                </a>
              </li>
            </ul>
          </nav>
        </div>
        
        <div className="border-t mt-8 pt-6 flex flex-col sm:flex-row justify-between items-center">
          <p className="text-sm text-muted-foreground">
            &copy; {currentYear} Kodleon. Tüm hakları saklıdır.
          </p>
          <div className="flex gap-4 mt-4 sm:mt-0">
            <Link href="/privacy" className="text-sm text-muted-foreground hover:text-primary transition-colors">
              Gizlilik Politikası
            </Link>
            <Link href="/terms" className="text-sm text-muted-foreground hover:text-primary transition-colors">
              Kullanım Şartları
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
}