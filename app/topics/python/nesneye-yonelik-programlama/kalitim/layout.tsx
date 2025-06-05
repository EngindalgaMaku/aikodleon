import { Metadata } from 'next';
import { Button } from '@/components/ui/button';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import Link from 'next/link';

export const metadata: Metadata = {
  title: 'Python OOP: Kalıtım (Inheritance) | Kodleon',
  description: 'Python\'da kalıtım kavramını, türetilmiş sınıfları ve çoklu kalıtımı öğrenin.',
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="container mx-auto">
      <nav className="flex flex-col sm:flex-row justify-between items-center gap-4 mb-8 bg-muted/30 p-4 rounded-lg">
        <Link href="/topics/python/nesneye-yonelik-programlama/siniflar-ve-nesneler" className="w-full sm:w-auto">
          <Button variant="outline" className="w-full">
            <ArrowLeft className="mr-2 h-4 w-4" />
            <div className="flex flex-col items-start">
              <span className="text-xs text-muted-foreground">Önceki Konu</span>
              <span>Sınıflar ve Nesneler</span>
            </div>
          </Button>
        </Link>
        <Link href="/topics/python/nesneye-yonelik-programlama/kapsulleme" className="w-full sm:w-auto">
          <Button variant="outline" className="w-full">
            <div className="flex flex-col items-end">
              <span className="text-xs text-muted-foreground">Sonraki Konu</span>
              <span>Kapsülleme</span>
            </div>
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </Link>
      </nav>

      {children}

      <footer className="mt-16 text-center text-sm text-muted-foreground pb-8">
        <p>© {new Date().getFullYear()} Kodleon | Yapay Zeka Eğitim Platformu</p>
      </footer>
    </div>
  );
} 