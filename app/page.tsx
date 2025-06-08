import type { Metadata } from 'next'
import HomePageClientContent from "@/components/HomePageClientContent"

export const metadata: Metadata = {
  title: "Kodleon | Türkiye'nin Lider Yapay Zeka Eğitim Platformu",
  description: "Yapay zeka, makine öğrenmesi, veri bilimi ve yazılım geliştirme konularında Türkçe eğitim içerikleri sunan modern eğitim platformu.",
  keywords: [
    'yapay zeka', 
    'makine öğrenmesi', 
    'derin öğrenme', 
    'yapay zeka eğitimi', 
    'kodleon', 
    'türkçe ai eğitimi',
    'veri bilimi',
    'nlp',
    'doğal dil işleme',
    'bilgisayarlı görü',
    'python programlama',
    'yazılım geliştirme'
  ],
  openGraph: {
    type: 'website',
    locale: 'tr_TR',
    url: 'https://kodleon.com',
    title: "Kodleon | Türkiye'nin Lider Yapay Zeka Eğitim Platformu",
    description: "Yapay zeka, makine öğrenmesi ve veri bilimi konularında Türkçe eğitim içerikleri sunan modern eğitim platformu.",
    siteName: 'Kodleon',
  },
  twitter: {
    card: 'summary_large_image',
    title: "Kodleon | Türkiye'nin Lider Yapay Zeka Eğitim Platformu",
    description: "Yapay zeka, makine öğrenmesi ve veri bilimi konularında Türkçe eğitim içerikleri.",
    creator: '@kodleon',
  },
  robots: {
    index: true,
    follow: true,
  },
  icons: {
    icon: '/favicon.ico',
    apple: '/apple-icon.png',
  },
  verification: {
    google: 'google-site-verification-code',
  },
  category: 'education',
  formatDetection: {
    telephone: false,
  },
  viewport: {
    width: 'device-width',
    initialScale: 1,
  },
  applicationName: 'Kodleon',
  colorScheme: 'dark light',
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: 'white' },
    { media: '(prefers-color-scheme: dark)', color: 'black' }
  ]
}

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col">
      <HomePageClientContent />
    </main>
  )
}