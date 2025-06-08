import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Python Eğitimleri | Kodleon - Yapay Zeka Eğitim Platformu',
  description: 'Python programlama dilini baştan sona öğrenin. Yapay zeka, veri bilimi, web geliştirme ve otomasyon alanlarında uzmanlaşın. Kodleon ile Python öğrenmeye başlayın.',
  keywords: 'python eğitimi, python dersleri, python programlama, yapay zeka python, veri bilimi python, web geliştirme python, python öğrenme, python kursu, python tutorial',
  openGraph: {
    title: 'Python Eğitimleri | Kodleon - Yapay Zeka Eğitim Platformu',
    description: 'Python programlama dilini baştan sona öğrenin. Yapay zeka, veri bilimi, web geliştirme ve otomasyon alanlarında uzmanlaşın.',
    images: ['/images/python-banner.jpg'],
    type: 'website',
    locale: 'tr_TR',
    url: 'https://kodleon.com/topics/python',
    siteName: 'Kodleon',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Python Eğitimleri | Kodleon',
    description: 'Python programlama dilini baştan sona öğrenin. Yapay zeka, veri bilimi, web geliştirme ve otomasyon alanlarında uzmanlaşın.',
    images: ['/images/python-banner.jpg'],
    site: '@kodleon',
    creator: '@kodleon'
  },
  alternates: {
    canonical: 'https://kodleon.com/topics/python'
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    google: 'your-google-verification-code',
  },
  category: 'education'
}; 